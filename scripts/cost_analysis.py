"""Estimate monthly LLM inference cost based on token usage.

Usage:
    python scripts/cost_analysis.py --requests-per-day 10000 --provider claude
"""
from __future__ import annotations

import argparse

PRICING = {
    "claude": {
        "model": "claude-3-5-haiku-20241022",
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
        "avg_input_tokens": 400,
        "avg_output_tokens": 150,
    },
}

EUR_PER_USD = 0.92


def estimate(provider: str, requests_per_day: int) -> None:
    p = PRICING[provider]
    daily_input_tokens = requests_per_day * p["avg_input_tokens"]
    daily_output_tokens = requests_per_day * p["avg_output_tokens"]

    daily_cost_usd = (
        daily_input_tokens / 1_000_000 * p["input_per_1m"]
        + daily_output_tokens / 1_000_000 * p["output_per_1m"]
    )
    monthly_cost_usd = daily_cost_usd * 30
    monthly_cost_eur = monthly_cost_usd * EUR_PER_USD

    print(f"\nProvider:          {provider} ({p['model']})")
    print(f"Requests/day:      {requests_per_day:,}")
    print(f"Avg input tokens:  {p['avg_input_tokens']}")
    print(f"Avg output tokens: {p['avg_output_tokens']}")
    print(f"Daily cost:        ${daily_cost_usd:.4f} USD")
    print(f"Monthly cost:      ${monthly_cost_usd:.2f} USD / €{monthly_cost_eur:.2f} EUR")
    print(f"Cost per request:  €{monthly_cost_eur / (requests_per_day * 30) * 1000:.4f} per 1k reqs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-per-day", type=int, default=10_000)
    parser.add_argument("--provider", choices=["claude"], default="claude")
    args = parser.parse_args()
    estimate(args.provider, args.requests_per_day)


if __name__ == "__main__":
    main()
