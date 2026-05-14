"""CLI entry point for intent classification eval.

Run: python evals/run_intent_eval.py
Results land in evals/results/intent_<timestamp>/
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.runners.intent_eval import run_intent_eval


def _render_report(result, timestamp: str, model: str) -> str:
    lines = [
        f"# Intent Classification Eval — {timestamp}",
        "",
        f"**Model:** {model}  ",
        f"**Dataset:** {result.total_queries} queries  ",
        f"**Duration:** {result.duration_seconds:.1f}s  ",
        "",
        "## Per-Dimension Accuracy",
        "",
        "| Dimension | Accuracy |",
        "|---|---|",
        f"| Language | {result.language_accuracy:.1%} |",
        f"| Intent | {result.intent_accuracy:.1%} |",
        f"| Specificity | {result.specificity_accuracy:.1%} |",
        f"| Is Basket Query | {result.is_basket_accuracy:.1%} |",
        f"| Target Partner | {result.target_partner_accuracy:.1%} |",
        f"| **Overall (all correct)** | **{result.overall_accuracy:.1%}** |",
        "",
    ]

    if result.failures:
        lines += [
            "## Failures",
            "",
            f"{len(result.failures)} queries failed (partial or complete mismatch):",
            "",
        ]
        for f in result.failures:
            if "error" in f:
                lines.append(f"- **`{f['query']}`** — ERROR: {f['error']}")
            else:
                wrong = ", ".join(f["wrong_fields"])
                exp = f["expected"]
                act = f["actual"]
                lines.append(f"- **`{f['query']}`** — wrong fields: `{wrong}`")
                lines.append(f"  - expected: `{exp}`")
                lines.append(f"  - actual:   `{act}`")
                if f.get("notes"):
                    lines.append(f"  - notes: {f['notes']}")
            lines.append("")
    else:
        lines += ["## Failures", "", "None — all queries classified correctly.", ""]

    # Interpretation
    weakest = min(
        [
            ("language", result.language_accuracy),
            ("intent", result.intent_accuracy),
            ("specificity", result.specificity_accuracy),
            ("is_basket_query", result.is_basket_accuracy),
            ("target_partner", result.target_partner_accuracy),
        ],
        key=lambda x: x[1],
    )
    lines += [
        "## Interpretation",
        "",
        (
            f"Overall accuracy is {result.overall_accuracy:.1%} across {result.total_queries} queries. "
            f"The weakest dimension is **{weakest[0]}** at {weakest[1]:.1%}. "
            f"There are {len(result.failures)} failing queries. "
            "Review the failures section above."
        ),
        "",
    ]

    return "\n".join(lines)


async def main() -> None:
    print("Running intent eval...")
    result = await run_intent_eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results" / f"intent_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = "claude-sonnet-4-6"
    report_md = _render_report(result, timestamp, model)
    (out_dir / "report.md").write_text(report_md)
    (out_dir / "result.json").write_text(result.model_dump_json(indent=2))

    print(
        f"  overall={result.overall_accuracy:.1%} | "
        f"intent={result.intent_accuracy:.1%} | "
        f"specificity={result.specificity_accuracy:.1%} | "
        f"partner={result.target_partner_accuracy:.1%} | "
        f"basket={result.is_basket_accuracy:.1%} | "
        f"{result.duration_seconds:.0f}s"
    )
    print(f"  Report: {out_dir}/report.md")


if __name__ == "__main__":
    asyncio.run(main())
