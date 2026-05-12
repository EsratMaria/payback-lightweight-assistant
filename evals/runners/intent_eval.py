"""Intent classification accuracy eval.

For each query in intent_labels.yaml, runs the intent agent and compares the result
to the expected ground truth. Computes per-dimension accuracy.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.agents.intent_agent import get_default_intent_agent
from app.models.schemas import Intent, Language, Partner, Specificity

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "intent_labels.yaml"


class IntentEvalResult(BaseModel):
    total_queries: int
    language_accuracy: float
    intent_accuracy: float
    specificity_accuracy: float
    is_basket_accuracy: float
    target_partner_accuracy: float
    overall_accuracy: float
    failures: list[dict] = Field(default_factory=list)
    duration_seconds: float


def _load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def _compare(result: Any, expected: dict) -> dict[str, bool]:
    """Compare intent result fields to expected values. Returns per-field correctness."""
    exp_language = Language(expected["language"])
    exp_intent = Intent(expected["intent"])
    exp_specificity = Specificity(expected["specificity"])
    exp_basket = expected["is_basket_query"]
    exp_partner = Partner(expected["target_partner"]) if expected.get("target_partner") else None

    return {
        "language": result.language == exp_language,
        "intent": result.intent == exp_intent,
        "specificity": result.specificity == exp_specificity,
        "is_basket_query": result.is_basket_query == exp_basket,
        "target_partner": result.target_partner == exp_partner,
    }


async def run_intent_eval(dataset: list[dict] | None = None) -> IntentEvalResult:
    """Run the intent eval. Pass a pre-loaded dataset for testing."""
    if dataset is None:
        dataset = _load_dataset()

    agent = get_default_intent_agent()
    start = time.perf_counter()

    counts: dict[str, int] = {
        "language": 0,
        "intent": 0,
        "specificity": 0,
        "is_basket_query": 0,
        "target_partner": 0,
        "overall": 0,
    }
    failures: list[dict] = []
    total = len(dataset)

    for entry in dataset:
        query = entry["query"]
        expected = entry["expected"]
        try:
            result = await agent.classify(query)
            correctness = _compare(result, expected)

            for field, correct in correctness.items():
                if correct:
                    counts[field] += 1

            all_correct = all(correctness.values())
            if all_correct:
                counts["overall"] += 1
            else:
                wrong_fields = [f for f, ok in correctness.items() if not ok]
                failures.append({
                    "query": query,
                    "wrong_fields": wrong_fields,
                    "expected": expected,
                    "actual": {
                        "language": result.language.value,
                        "intent": result.intent.value,
                        "specificity": result.specificity.value,
                        "is_basket_query": result.is_basket_query,
                        "target_partner": result.target_partner.value if result.target_partner else None,
                    },
                    "notes": entry.get("notes", ""),
                })
        except Exception as exc:
            failures.append({
                "query": query,
                "error": str(exc),
                "expected": expected,
            })

    duration = time.perf_counter() - start
    return IntentEvalResult(
        total_queries=total,
        language_accuracy=counts["language"] / total,
        intent_accuracy=counts["intent"] / total,
        specificity_accuracy=counts["specificity"] / total,
        is_basket_accuracy=counts["is_basket_query"] / total,
        target_partner_accuracy=counts["target_partner"] / total,
        overall_accuracy=counts["overall"] / total,
        failures=failures,
        duration_seconds=duration,
    )
