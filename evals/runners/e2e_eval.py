"""End-to-end relevance eval with Claude Opus as judge.

For each query in judge_validation.yaml:
- Runs the full router pipeline → AssistantResponse
- Scores with Opus judge 3 times per query (three-run variance check)
- Aggregates: median per dimension, variance flag
- Computes judge-human agreement on queries where human_rating_* fields are populated
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from app.agents.router import get_default_router
from app.llm.judge_client import OpusJudgeClient
from app.models.schemas import (
    AssistantResponse,
    E2EEvalResult,
    JudgmentScore,
    QueryJudgment,
)

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "judge_validation.yaml"

JUDGE_SYSTEM_PROMPT = """\
You evaluate the quality of responses from a multilingual shopping assistant for PAYBACK, a German loyalty program.

The assistant returns one of three response types:
- recommendations: a list of products
- clarification: a clarifying question with suggested options
- navigation: a redirect to a specific partner

You score each response on three dimensions, each 0-3:

1. query_satisfaction — Did the response address what the user asked?
Keep in mind catalog is limited to as long as some products are relevant 
keeping in mind additional parameters like target partner etc, the query satisfaction can be lenient even if the query is not fully addressed.

   0 = completely off-topic or wrong response type
   1 = partially relevant but missing the main intent
   2 = mostly addresses the query with minor gaps
   3 = fully addresses what the user wanted

2. result_quality — Quality of the actual content returned:
   - For recommendations: are the products appropriate? Mixed-partner variety? Reasonable matches?
   - For clarification: is the question grounded in real options? Are the suggested options useful?
   - For navigation: is the partner choice correct?
   However a big point to be noted: the catalog that the respond model would have access to is limited and may not have access to variety
   of products, so if the response is reasonable given a limited catalog, it should not be penalized. Like for Tiramisu Ingredient query,
   diverese product list like Mascarpone, Coffee, Ladyfingers etc might be expected in general sense but for the model to surface not so relevant items makes it catalog
   grounded since its one of the main constraints.
   So in terms of finding helpful product in the recommendation list - leniency is expected given the limitation
   of the synthetic data.
   In such cases: User's query may not be fully addresses but the result quality is great since its grounded in catalog data.
   However, keep your eyes open for target partner. Overall use your judgment to balance these factors for the final score.
   0 = poor or hallucinated
   1 = some issues but somewhat useful
   2 = solid with minor flaws
   3 = excellent

3. language_tone_match — Did the response language match the user's query language?
   The system has mostly German products with descriptions. If general search about items surfaces the right product but
   in a different language than the query, you should be lenient on language tone as long as the
   response is surfacing relevant products. 
   But for navigation or support cases replying the same tone and language as the user is more important since its more about the interaction quality. 
   So use your judgment to balance these factors for the final score.
   0 = wrong language entirely
   1 = mixed or partly wrong
   2 = correct with minor inconsistency
   3 = consistently matches user's language

Be strict but fair. A 3 means "I would ship this to production." A 1 means "this would frustrate a user."

You MUST respond by calling the `respond` tool with a JudgmentScore object.\
"""

JUDGE_USER_PROMPT_TEMPLATE = """
Evaluate the following exchange.

USER QUERY: "{query}"

ASSISTANT RESPONSE TYPE: {response_type}

ASSISTANT RESPONSE CONTENT:
{response_content}

Score each dimension 0-3 with a one-sentence reasoning per dimension.\
"""

_NUM_JUDGE_RUNS = 3


def _load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def _format_response_content(response: AssistantResponse) -> str:
    """Serialize the relevant response payload for the judge prompt."""
    if response.response_type == "recommendations" and response.recommendations:
        items = [
            f"{i+1}. [{r.product.partner.value}] {r.product.name} — {r.product.description[:80]}"
            for i, r in enumerate(response.recommendations[:5])
        ]
        return "\n".join(items)
    elif response.response_type == "clarification" and response.clarification:
        return (
            f"Question: {response.clarification.question}\n"
            f"Options: {', '.join(response.clarification.suggested_options)}"
        )
    elif response.response_type == "navigation":
        return f"Navigate to: {response.navigation_target}"
    return "(empty response)"


async def _judge_once(
    judge: OpusJudgeClient,
    query: str,
    response: AssistantResponse,
) -> JudgmentScore:
    content = _format_response_content(response)
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        query=query,
        response_type=response.response_type,
        response_content=content,
    )
    result = await judge.structured_completion(
        prompt=user_prompt,
        schema=JudgmentScore,
        system=JUDGE_SYSTEM_PROMPT,
        max_tokens=512,
    )
    return result  # type: ignore[return-value]


def _compute_judge_human_agreement(
    per_query: list[QueryJudgment],
    dataset: list[dict],
) -> dict | None:
    """Compute within-1-point agreement for queries with human ratings."""
    rated = [
        entry for entry in dataset
        if entry.get("human_rating_query_satisfaction") is not None
        and entry.get("human_rating_result_quality") is not None
        and entry.get("human_rating_language_tone") is not None
    ]
    if not rated:
        return None

    query_to_judgment = {j.query: j for j in per_query}
    dims = ["query_satisfaction", "result_quality", "language_tone"]
    human_keys = [
        "human_rating_query_satisfaction",
        "human_rating_result_quality",
        "human_rating_language_tone",
    ]
    judge_attrs = [
        "median_query_satisfaction",
        "median_result_quality",
        "median_language_tone",
    ]

    agreement: dict[str, float] = {}
    for dim, hkey, jattr in zip(dims, human_keys, judge_attrs):
        matches = 0
        count = 0
        for entry in rated:
            q = entry["query"]
            if q not in query_to_judgment:
                continue
            human_score = entry[hkey]
            judge_score = getattr(query_to_judgment[q], jattr)
            if abs(judge_score - human_score) <= 1:
                matches += 1
            count += 1
        agreement[dim] = matches / count if count else 0.0

    total_matches = sum(
        1
        for entry in rated
        for dim, hkey, jattr in zip(dims, human_keys, judge_attrs)
        if entry["query"] in query_to_judgment
        and abs(getattr(query_to_judgment[entry["query"]], jattr) - entry[hkey]) <= 1
    )
    total_pairs = sum(
        1
        for entry in rated
        if entry["query"] in query_to_judgment
        for _ in dims
    )
    agreement["overall"] = total_matches / total_pairs if total_pairs else 0.0
    return agreement


async def run_e2e_eval(
    dataset: list[dict] | None = None,
    router=None,
    judge: OpusJudgeClient | None = None,
) -> E2EEvalResult:
    """Run the E2E eval. Pass router/judge for testing."""
    if dataset is None:
        dataset = _load_dataset()
    if router is None:
        router = get_default_router()
    if judge is None:
        judge = OpusJudgeClient()

    start = time.perf_counter()
    per_query: list[QueryJudgment] = []

    for entry in dataset:
        query = entry["query"]
        response: AssistantResponse = await router.handle(query=query)

        runs: list[JudgmentScore] = []
        for _ in range(_NUM_JUDGE_RUNS):
            score = await _judge_once(judge, query, response)
            runs.append(score)

        qs_scores = [r.query_satisfaction for r in runs]
        rq_scores = [r.result_quality for r in runs]
        lt_scores = [r.language_tone_match for r in runs]

        med_qs = int(statistics.median(qs_scores))
        med_rq = int(statistics.median(rq_scores))
        med_lt = int(statistics.median(lt_scores))

        variance_flag = (
            max(qs_scores) - min(qs_scores) > 1
            or max(rq_scores) - min(rq_scores) > 1
            or max(lt_scores) - min(lt_scores) > 1
        )

        per_query.append(QueryJudgment(
            query=query,
            response_type=response.response_type,
            runs=runs,
            median_query_satisfaction=med_qs,
            median_result_quality=med_rq,
            median_language_tone=med_lt,
            total_median=med_qs + med_rq + med_lt,
            variance_flag=variance_flag,
        ))

    duration = time.perf_counter() - start

    mean_qs = sum(j.median_query_satisfaction for j in per_query) / len(per_query)
    mean_rq = sum(j.median_result_quality for j in per_query) / len(per_query)
    mean_lt = sum(j.median_language_tone for j in per_query) / len(per_query)
    mean_total = sum(j.total_median for j in per_query) / len(per_query)

    high_variance = [j.query for j in per_query if j.variance_flag]
    agreement = _compute_judge_human_agreement(per_query, dataset)

    return E2EEvalResult(
        total_queries=len(per_query),
        mean_total=mean_total,
        mean_query_satisfaction=mean_qs,
        mean_result_quality=mean_rq,
        mean_language_tone=mean_lt,
        high_variance_queries=high_variance,
        per_query=per_query,
        judge_human_agreement=agreement,
        duration_seconds=duration,
    )
