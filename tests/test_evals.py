"""Unit tests for eval runner logic.

Tests the math and aggregation logic only — no real LLM calls.
All LLM calls are mocked.
"""
from __future__ import annotations

import asyncio
import pytest

from app.models.schemas import (
    JudgmentScore,
    Language,
    Intent,
    Specificity,
    Partner,
)
from evals.runners.intent_eval import IntentEvalResult, _compare
from evals.runners.retrieval_eval import RetrievalEvalResult
from evals.runners.e2e_eval import _compute_judge_human_agreement, QueryJudgment


# ---------------------------------------------------------------------------
# 1. Intent eval aggregation
# ---------------------------------------------------------------------------


class MockIntentResultObj:
    """Minimal mock for IntentResult."""
    def __init__(self, language, intent, specificity, is_basket_query, target_partner):
        self.language = language
        self.intent = intent
        self.specificity = specificity
        self.is_basket_query = is_basket_query
        self.target_partner = target_partner


@pytest.mark.asyncio
async def test_intent_eval_aggregation():
    """Feed a 4-entry dataset with a predictable mock agent, assert per-dimension accuracy."""

    dataset = [
        {
            "query": "wireless mouse",
            "expected": {
                "language": "en",
                "intent": "search",
                "specificity": "specific",
                "is_basket_query": False,
                "target_partner": None,
            },
            "notes": "",
        },
        {
            "query": "open Amazon",
            "expected": {
                "language": "en",
                "intent": "search",
                "specificity": "navigational",
                "is_basket_query": False,
                "target_partner": "amazon",
            },
            "notes": "",
        },
        {
            "query": "Windeln",
            "expected": {
                "language": "de",
                "intent": "search",
                "specificity": "specific",
                "is_basket_query": False,
                "target_partner": None,
            },
            "notes": "",
        },
        {
            "query": "pasta dinner",
            "expected": {
                "language": "en",
                "intent": "search",
                "specificity": "specific",
                "is_basket_query": True,
                "target_partner": None,
            },
            "notes": "",
        },
    ]

    # Mock: first 3 queries are all-correct, last has wrong specificity (vague instead of specific)
    mock_results = [
        MockIntentResultObj(Language.en, Intent.search, Specificity.specific, False, None),
        MockIntentResultObj(Language.en, Intent.search, Specificity.navigational, False, Partner.amazon),
        MockIntentResultObj(Language.de, Intent.search, Specificity.specific, False, None),
        MockIntentResultObj(Language.en, Intent.search, Specificity.vague, True, None),  # wrong specificity
    ]

    class MockAgent:
        def __init__(self, results):
            self._results = iter(results)
        async def classify(self, query):
            return next(self._results)

    from evals.runners.intent_eval import run_intent_eval as _run
    import evals.runners.intent_eval as _mod

    original_factory = _mod.get_default_intent_agent
    _mod.get_default_intent_agent = lambda: MockAgent(mock_results)
    try:
        result = await _run(dataset=dataset)
    finally:
        _mod.get_default_intent_agent = original_factory

    assert result.total_queries == 4
    assert result.language_accuracy == 1.0          # all correct
    assert result.intent_accuracy == 1.0            # all correct
    assert result.specificity_accuracy == 3 / 4     # last is wrong
    assert result.is_basket_accuracy == 1.0         # all correct
    assert result.target_partner_accuracy == 1.0    # all correct
    assert result.overall_accuracy == 3 / 4         # last has one wrong field
    assert len(result.failures) == 1
    assert "specificity" in result.failures[0]["wrong_fields"]


# ---------------------------------------------------------------------------
# 2. Retrieval eval precision/recall math
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_eval_precision_recall_math():
    """Assert precision@5 and recall@5 compute correctly on edge cases."""

    dataset = [
        # Case 1: 3 of 5 retrieved are relevant → P=0.6, R=3/5=0.6
        {
            "query": "spaghetti",
            "relevant_product_ids": ["a", "b", "c", "d", "e"],
            "notes": "",
        },
        # Case 2: none retrieved are relevant → P=0.0, R=0.0
        {
            "query": "nothing",
            "relevant_product_ids": ["x", "y", "z"],
            "notes": "",
        },
        # Case 3: all 5 retrieved are relevant (only 3 total relevant) → P=1.0, R=1.0
        # Actually: if top-5 retrieved = [a,b,c,d,e] and relevant=[a,b,c], then
        # relevant_retrieved = 3, P=3/5=0.6, R=3/3=1.0
        {
            "query": "all_relevant",
            "relevant_product_ids": ["a", "b", "c"],
            "notes": "",
        },
    ]

    # Build products with .product_id attribute
    class FakeProduct:
        def __init__(self, pid):
            self.product_id = pid

    class MockStore:
        async def search(self, query, top_k=5, partner_filter=None):
            if query == "spaghetti":
                return [(FakeProduct(pid), 0.9) for pid in ["a", "b", "c", "f", "g"]]
            elif query == "nothing":
                return [(FakeProduct(pid), 0.9) for pid in ["p", "q", "r", "s", "t"]]
            elif query == "all_relevant":
                return [(FakeProduct(pid), 0.9) for pid in ["a", "b", "c", "d", "e"]]
            return []

    from evals.runners.retrieval_eval import run_retrieval_eval
    result = await run_retrieval_eval(dataset=dataset, vector_store=MockStore())

    assert result.total_queries == 3
    # Case 1: spaghetti — 3 of ["a","b","c","f","g"] in relevant {"a","b","c","d","e"} → 3
    # P = 3/5 = 0.6, R = 3/5 = 0.6
    assert abs(result.per_query[0]["precision_at_5"] - 0.6) < 1e-6
    assert abs(result.per_query[0]["recall_at_5"] - 0.6) < 1e-6

    # Case 2: nothing — 0 relevant → P=0, R=0
    assert result.per_query[1]["precision_at_5"] == 0.0
    assert result.per_query[1]["recall_at_5"] == 0.0

    # Case 3: all_relevant — top-5 = ["a","b","c","d","e"], relevant = {"a","b","c"}
    # relevant_retrieved = 3, P = 3/5 = 0.6, R = 3/3 = 1.0
    assert abs(result.per_query[2]["precision_at_5"] - 0.6) < 1e-6
    assert abs(result.per_query[2]["recall_at_5"] - 1.0) < 1e-6

    expected_mean_p = (0.6 + 0.0 + 0.6) / 3
    expected_mean_r = (0.6 + 0.0 + 1.0) / 3
    assert abs(result.mean_precision_at_5 - expected_mean_p) < 1e-6
    assert abs(result.mean_recall_at_5 - expected_mean_r) < 1e-6


# ---------------------------------------------------------------------------
# 3. E2E eval median aggregation + variance flag
# ---------------------------------------------------------------------------


def test_e2e_eval_median_aggregation():
    """Assert median computation and variance flag logic."""
    runs_no_variance = [
        JudgmentScore(query_satisfaction=2, result_quality=2, language_tone_match=3, reasoning="good"),
        JudgmentScore(query_satisfaction=2, result_quality=3, language_tone_match=3, reasoning="good"),
        JudgmentScore(query_satisfaction=2, result_quality=2, language_tone_match=3, reasoning="good"),
    ]
    # medians: qs=2, rq=2, lt=3, total=7
    import statistics
    qs = [r.query_satisfaction for r in runs_no_variance]
    rq = [r.result_quality for r in runs_no_variance]
    lt = [r.language_tone_match for r in runs_no_variance]
    assert int(statistics.median(qs)) == 2
    assert int(statistics.median(rq)) == 2
    assert int(statistics.median(lt)) == 3
    # No variance: all ranges ≤ 1
    assert max(qs) - min(qs) <= 1
    assert max(rq) - min(rq) <= 1
    assert max(lt) - min(lt) <= 1

    runs_high_variance = [
        JudgmentScore(query_satisfaction=1, result_quality=0, language_tone_match=3, reasoning="bad"),
        JudgmentScore(query_satisfaction=3, result_quality=3, language_tone_match=3, reasoning="ok"),
        JudgmentScore(query_satisfaction=2, result_quality=1, language_tone_match=3, reasoning="mid"),
    ]
    qs2 = [r.query_satisfaction for r in runs_high_variance]
    rq2 = [r.result_quality for r in runs_high_variance]
    lt2 = [r.language_tone_match for r in runs_high_variance]
    # qs range = 3-1 = 2 > 1 → variance flag
    assert max(qs2) - min(qs2) > 1
    # rq range = 3-0 = 3 > 1 → variance flag
    assert max(rq2) - min(rq2) > 1
    # lt range = 0 → no variance
    assert max(lt2) - min(lt2) <= 1

    # Overall variance_flag = True if ANY dimension has range > 1
    variance_flag = (
        max(qs2) - min(qs2) > 1
        or max(rq2) - min(rq2) > 1
        or max(lt2) - min(lt2) > 1
    )
    assert variance_flag is True


# ---------------------------------------------------------------------------
# 4. Judge-human agreement calculation
# ---------------------------------------------------------------------------


def test_judge_human_agreement_calculation():
    """Assert within-1-point agreement computes correctly."""
    # Build fake QueryJudgment objects
    def _qj(query, med_qs, med_rq, med_lt):
        return QueryJudgment(
            query=query,
            response_type="recommendations",
            runs=[
                JudgmentScore(query_satisfaction=med_qs, result_quality=med_rq, language_tone_match=med_lt, reasoning="r")
                for _ in range(3)
            ],
            median_query_satisfaction=med_qs,
            median_result_quality=med_rq,
            median_language_tone=med_lt,
            total_median=med_qs + med_rq + med_lt,
            variance_flag=False,
        )

    per_query = [
        _qj("q1", 2, 2, 3),
        _qj("q2", 1, 1, 1),
        _qj("q3", 3, 3, 3),
    ]

    dataset = [
        {
            "query": "q1",
            "human_rating_query_satisfaction": 2,   # exact match → within-1
            "human_rating_result_quality": 1,        # |2-1|=1 → within-1
            "human_rating_language_tone": 1,         # |3-1|=2 → NOT within-1
        },
        {
            "query": "q2",
            "human_rating_query_satisfaction": 1,    # exact match
            "human_rating_result_quality": 2,        # |1-2|=1 → within-1
            "human_rating_language_tone": 1,         # exact match
        },
        {
            "query": "q3",
            "human_rating_query_satisfaction": None,  # no rating → skip
            "human_rating_result_quality": None,
            "human_rating_language_tone": None,
        },
    ]

    agreement = _compute_judge_human_agreement(per_query, dataset)
    assert agreement is not None

    # q1: qs=within-1 (2==2), rq=within-1 (|2-1|=1), lt=NOT within-1 (|3-1|=2)
    # q2: qs=within-1 (1==1), rq=within-1 (|1-2|=1), lt=within-1 (1==1)
    # q3: skipped (null ratings)

    # query_satisfaction: q1=yes, q2=yes → 2/2 = 1.0
    assert abs(agreement["query_satisfaction"] - 1.0) < 1e-6
    # result_quality: q1=yes, q2=yes → 2/2 = 1.0
    assert abs(agreement["result_quality"] - 1.0) < 1e-6
    # language_tone: q1=no, q2=yes → 1/2 = 0.5
    assert abs(agreement["language_tone"] - 0.5) < 1e-6
    # overall: 5/6 = 0.833...
    assert abs(agreement["overall"] - 5/6) < 1e-6
