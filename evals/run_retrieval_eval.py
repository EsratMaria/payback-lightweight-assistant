"""CLI entry point for retrieval precision@5 / recall@5 eval.

Prereq: catalogs ingested via `python -m app.retrieval.ingest --rebuild`
Run: python evals/run_retrieval_eval.py
Results land in evals/results/retrieval_<timestamp>/
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.runners.retrieval_eval import run_retrieval_eval


def _render_report(result, timestamp: str) -> str:
    lines = [
        f"# Retrieval Eval (Precision@5 / Recall@5) — {timestamp}",
        "",
        f"**Dataset:** {result.total_queries} queries  ",
        f"**Duration:** {result.duration_seconds:.1f}s  ",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mean Precision@5 | {result.mean_precision_at_5:.3f} |",
        f"| Mean Recall@5 | {result.mean_recall_at_5:.3f} |",
        "",
        "## Per-Query Results",
        "",
        "| Query | P@5 | R@5 | Missed |",
        "|---|---|---|---|",
    ]

    cross_lingual = []
    for q in result.per_query:
        missed_str = ", ".join(q["missed_ids"][:3]) if q["missed_ids"] else "—"
        lines.append(
            f"| `{q['query']}` | {q['precision_at_5']:.2f} | {q['recall_at_5']:.2f} | {missed_str} |"
        )
        notes = q.get("notes", "")
        if "cross-lingual" in notes.lower() or "cross_lingual" in notes.lower():
            cross_lingual.append((q["query"], q["precision_at_5"]))

    lines.append("")

    cross_avg = (
        sum(p for _, p in cross_lingual) / len(cross_lingual) if cross_lingual else None
    )
    overall_avg = result.mean_precision_at_5

    lines += [
        "## Interpretation",
        "",
        (
            f"Mean precision@5 is {overall_avg:.3f}. "
            + (
                f"Cross-lingual queries average {cross_avg:.3f} precision@5 "
                f"({'above' if cross_avg >= overall_avg else 'below'} the overall mean), "
                "indicating the multilingual embeddings handle language mismatch "
                + ("well." if cross_avg >= overall_avg - 0.1 else "with some degradation.")
            )
            if cross_avg is not None
            else f"Mean precision@5 is {overall_avg:.3f}."
        ),
        "",
    ]

    return "\n".join(lines)


async def main() -> None:
    print("Running retrieval eval...")
    result = await run_retrieval_eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"retrieval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_md = _render_report(result, timestamp)
    (out_dir / "report.md").write_text(report_md)
    (out_dir / "result.json").write_text(result.model_dump_json(indent=2))

    print(
        f"  precision@5={result.mean_precision_at_5:.3f} | "
        f"recall@5={result.mean_recall_at_5:.3f} | "
        f"{result.duration_seconds:.0f}s"
    )
    print(f"  Report: {out_dir}/report.md")


if __name__ == "__main__":
    asyncio.run(main())
