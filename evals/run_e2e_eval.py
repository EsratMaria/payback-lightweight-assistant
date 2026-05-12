"""CLI entry point for E2E LLM-as-judge eval.

Run: python evals/run_e2e_eval.py
Results land in evals/results/e2e_<timestamp>/
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.runners.e2e_eval import run_e2e_eval

JUDGE_MODEL = "claude-opus-4-7"


def _score_dist(scores: list[int]) -> str:
    """Textual histogram for 0-3 scores."""
    counts = [scores.count(i) for i in range(4)]
    return " | ".join(f"{i}:{counts[i]}" for i in range(4))


def _render_report(result, timestamp: str) -> str:
    lines = [
        f"# E2E Eval (LLM-as-Judge) — {timestamp}",
        "",
        f"**Judge model:** {JUDGE_MODEL}  ",
        f"**Production model:** claude-sonnet-4-6  ",
        f"**Queries:** {result.total_queries}  ",
        f"**Runs per query:** 3  ",
        f"**Duration:** {result.duration_seconds:.1f}s  ",
        "",
        "## Mean Scores (out of 3)",
        "",
        "| Dimension | Mean Score |",
        "|---|---|",
        f"| Query Satisfaction | {result.mean_query_satisfaction:.2f} |",
        f"| Result Quality | {result.mean_result_quality:.2f} |",
        f"| Language & Tone Match | {result.mean_language_tone:.2f} |",
        f"| **Total (out of 9)** | **{result.mean_total:.2f}** |",
        "",
        "## Score Distribution",
        "",
        "| Dimension | 0 | 1 | 2 | 3 |",
        "|---|---|---|---|---|",
    ]

    qs_scores = [j.median_query_satisfaction for j in result.per_query]
    rq_scores = [j.median_result_quality for j in result.per_query]
    lt_scores = [j.median_language_tone for j in result.per_query]

    for label, scores in [
        ("Query Satisfaction", qs_scores),
        ("Result Quality", rq_scores),
        ("Language Tone", lt_scores),
    ]:
        counts = [scores.count(i) for i in range(4)]
        lines.append(f"| {label} | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} |")

    lines.append("")

    if result.high_variance_queries:
        lines += [
            "## High-Variance Queries",
            "",
            "These queries had judge disagreement (range > 1) across 3 runs:",
            "",
        ]
        for q_str in result.high_variance_queries:
            judgment = next(j for j in result.per_query if j.query == q_str)
            lines.append(f"### `{q_str}`")
            lines.append("")
            for i, run in enumerate(judgment.runs, 1):
                lines.append(
                    f"- Run {i}: qs={run.query_satisfaction} rq={run.result_quality} lt={run.language_tone_match}"
                )
                lines.append(f"  {run.reasoning}")
            lines.append("")

    if result.judge_human_agreement:
        lines += [
            "## Judge–Human Agreement",
            "",
            "Agreement = fraction of queries where |judge_score − human_score| ≤ 1",
            "",
            "| Dimension | Agreement |",
            "|---|---|",
        ]
        for dim, val in result.judge_human_agreement.items():
            lines.append(f"| {dim} | {val:.1%} |")
        lines.append("")
    else:
        lines += [
            "## Judge–Human Agreement",
            "",
            "_No human ratings available yet. Fill in `human_rating_*` fields in_",
            "_`evals/datasets/judge_validation.yaml`, then re-run this eval._",
            "",
        ]

    lines += [
        "## Per-Query Detail",
        "",
        "| Query | Type | QS | RQ | LT | Total | Variance |",
        "|---|---|---|---|---|---|---|",
    ]
    for j in result.per_query:
        flag = "⚠️" if j.variance_flag else "✓"
        lines.append(
            f"| `{j.query[:40]}` | {j.response_type} | "
            f"{j.median_query_satisfaction} | {j.median_result_quality} | "
            f"{j.median_language_tone} | {j.total_median} | {flag} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        (
            f"Mean total score is {result.mean_total:.2f}/9 across {result.total_queries} queries. "
            f"Query satisfaction ({result.mean_query_satisfaction:.2f}) measures response-type correctness; "
            f"result quality ({result.mean_result_quality:.2f}) measures content appropriateness; "
            f"language tone match ({result.mean_language_tone:.2f}) measures bilingual consistency. "
            + (
                f"{len(result.high_variance_queries)} queries showed judge disagreement — "
                "review those manually before drawing conclusions."
                if result.high_variance_queries
                else "No high-variance queries — judge agreement was consistent across runs."
            )
        ),
        "",
        "---",
        "",
        "<!-- TODO: Once Maria has filled in human_rating_* fields in judge_validation.yaml,",
        "re-run this eval to populate the Judge–Human Agreement section above.",
        "Target: ≥75% within-1-point agreement per dimension. Below 70% means",
        "the judge prompt needs revision. -->",
    ]

    return "\n".join(lines)


async def main() -> None:
    print(f"Running E2E eval (judge: {JUDGE_MODEL}, 3 runs/query)...")
    result = await run_e2e_eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"e2e_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_md = _render_report(result, timestamp)
    (out_dir / "report.md").write_text(report_md)
    (out_dir / "result.json").write_text(result.model_dump_json(indent=2))

    print(
        f"  total={result.mean_total:.2f}/9 | "
        f"qs={result.mean_query_satisfaction:.2f} | "
        f"rq={result.mean_result_quality:.2f} | "
        f"lt={result.mean_language_tone:.2f} | "
        f"{len(result.high_variance_queries)} high-variance | "
        f"{result.duration_seconds:.0f}s"
    )
    print(f"  Report: {out_dir}/report.md")


if __name__ == "__main__":
    asyncio.run(main())
