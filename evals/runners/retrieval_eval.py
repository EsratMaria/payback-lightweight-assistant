"""Retrieval precision@5 and recall@5 eval.

For each query in retrieval_labels.yaml, runs vector store search with top_k=5
(no partner filter) and computes precision@5 and recall@5.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.config import settings
from app.retrieval.local_store import LocalChromaStore

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "retrieval_labels.yaml"


class RetrievalEvalResult(BaseModel):
    total_queries: int
    mean_precision_at_5: float
    mean_recall_at_5: float
    per_query: list[dict] = Field(default_factory=list)
    duration_seconds: float


def _load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


async def run_retrieval_eval(
    dataset: list[dict] | None = None,
    vector_store=None,
) -> RetrievalEvalResult:
    """Run retrieval eval. Pass dataset and/or vector_store for testing."""
    if dataset is None:
        dataset = _load_dataset()

    if vector_store is None:
        vector_store = LocalChromaStore(persist_dir=settings.chroma_persist_dir)

    start = time.perf_counter()
    per_query: list[dict] = []
    total_precision = 0.0
    total_recall = 0.0
    total = len(dataset)

    for entry in dataset:
        query = entry["query"]
        relevant_ids: set[str] = set(entry["relevant_product_ids"])
        notes = entry.get("notes", "")

        results = await vector_store.search(query, top_k=5, partner_filter=None)
        retrieved_ids = [p.product_id for p, _ in results]

        relevant_retrieved = sum(1 for pid in retrieved_ids if pid in relevant_ids)
        precision = relevant_retrieved / 5 if results else 0.0
        recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0

        missed = [pid for pid in relevant_ids if pid not in retrieved_ids]

        per_query.append({
            "query": query,
            "precision_at_5": precision,
            "recall_at_5": recall,
            "retrieved_ids": retrieved_ids,
            "missed_ids": missed[:5],  # cap to avoid huge output
            "notes": notes,
        })
        total_precision += precision
        total_recall += recall

    duration = time.perf_counter() - start
    return RetrievalEvalResult(
        total_queries=total,
        mean_precision_at_5=total_precision / total if total else 0.0,
        mean_recall_at_5=total_recall / total if total else 0.0,
        per_query=per_query,
        duration_seconds=duration,
    )
