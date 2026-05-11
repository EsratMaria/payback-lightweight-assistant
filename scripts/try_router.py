"""Manual end-to-end sanity check for Router against real Claude API + real catalog.

Prereq: catalogs ingested via `python -m app.retrieval.ingest --rebuild`.
Run: python -m scripts.try_router
"""
import asyncio

from app.agents.router import get_default_router

QUERIES = [
    "I need stuff for a pasta dinner",
    "Bitte zeige mir Angebote für günstige Windeln",
    "open Amazon for me",
    "something for my dog",
    "etwas Schönes für meinen Mann",
    "pasta dinner from edeka",
    "pasta from REWE",
    "how do I redeem my points",
    "wireless mouse",
    "Schokolade",
]


async def main() -> None:
    router = get_default_router()
    for q in QUERIES:
        print("\n" + "=" * 70)
        print(f"Query: {q!r}")
        response = await router.handle(query=q)
        print(f"  type:    {response.response_type}")
        print(
            f"  intent:  {response.intent_result.intent.value}/"
            f"{response.intent_result.specificity.value}/"
            f"conf={response.intent_result.confidence:.2f}"
        )
        print(f"  partner: {response.intent_result.target_partner}")
        print(f"  latency: {response.latency_ms:.0f}ms")
        print(f"  cost:    €{response.estimated_cost_eur:.5f}")

        if response.response_type == "recommendations":
            print(f"  results: {len(response.recommendations)} products")
            for rec in response.recommendations[:5]:
                print(
                    f"    {rec.rank}. [{rec.product.partner.value}] "
                    f"{rec.product.name} (score={rec.final_score:.2f})"
                )
        elif response.response_type == "clarification":
            print(f"  question: {response.clarification.question}")
            print(f"  options:  {response.clarification.suggested_options}")
        elif response.response_type == "navigation":
            print(f"  navigate to: {response.navigation_target}")


if __name__ == "__main__":
    asyncio.run(main())
