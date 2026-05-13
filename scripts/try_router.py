"""Manual end-to-end sanity check for Router against real Claude API + real catalog.

Prereq: catalogs ingested via `python -m app.retrieval.ingest --rebuild`.
Run: python -m scripts.try_router
"""
import asyncio
import json
import pathlib

from app.agents.router import get_default_router
from app.models.schemas import UserContext

QUERIES = [
    # "I need stuff for a pasta dinner",
    # "open Amazon for me",
    # "something for my dog",
    # "etwas Schönes für meinen Mann",
    # "pasta dinner from edeka",
    # "pasta from REWE",
    # "how do I redeem my points",
    # "wireless mouse",
    # "Schokolade",
    # "take me to the shop",
    # "stuff for a pasta dinner",
    # "ingredients for tiramisu",
    # "alles für ein Wochenende am See",
    # "Geburtstagsparty Snacks",
    # "ingredient to cook a meal like spaghetti or chicken for a dinner date with girlfriend",
    # "stuffs for glowing skin",
    # "USB-C charger",
    "Bitte zeige mir Angebote für günstige Windeln",
    "günstige Bio Tomaten",
    "cheap wireless mouse",
]

_profiles_raw = json.loads(
    (pathlib.Path(__file__).parent.parent / "data" / "user_profiles.json").read_text()
)
PROFILES: dict[str, UserContext] = {
    name: UserContext.model_validate(data)
    for name, data in _profiles_raw.items()
}


async def main() -> None:
    router = get_default_router()
    for profile_name, user_ctx in PROFILES.items():
        print(f"\n{'#' * 70}")
        print(f"# Profile: {profile_name}  (new_user={user_ctx.is_new_user})")
        print(f"# Affinity: { {p.value: round(v, 2) for p, v in user_ctx.partner_affinity.items()} }")
        for q in QUERIES:
            print("\n" + "=" * 70)
            print(f"Query: {q!r}")
            response = await router.handle(query=q, user_context=user_ctx)
            print(f"  type:    {response.response_type}")
            print(
                f"  intent:  {response.intent_result.intent.value}/"
                f"{response.intent_result.specificity.value}/"
                f"conf={response.intent_result.confidence:.2f}"
            )
            print(f"  partner: {response.intent_result.target_partner}")
            print(f"  latency: {response.latency_ms:.0f}ms")
            print(f"  cost:    €{response.estimated_cost_eur:.5f}")
            print(f"  basket:      {response.intent_result.is_basket_query}")
            print(f"  deals:       {response.intent_result.prefers_deals}")
            if response.response_type == "recommendations":
                if response.promo_fallback:
                    print("  promo_fallback: True (no promo products matched; showing all)")
                if response.intent_result.is_basket_query and response.debug_expanded_queries:
                    print(f"  proposed: {response.debug_expanded_queries}")
                    if response.debug_dropped_queries:
                        print(f"  dropped:  {response.debug_dropped_queries}")
                print(f"  results: {len(response.recommendations)} products")
                for rec in response.recommendations[:5]:
                    print(
                        f"    {rec.rank}. [{rec.product.partner.value}] "
                        f"{rec.product.name} "
                        f"(sem={rec.semantic_score:.2f} boost={rec.loyalty_boost:.2f} "
                        f"div={rec.diversity_bonus:.2f} final={rec.final_score:.2f})"
                    )
            elif response.response_type == "clarification":
                print(f"  question: {response.clarification.question}")
                print(f"  options:  {response.clarification.suggested_options}")
            elif response.response_type == "navigation":
                print(f"  navigate to: {response.navigation_target}")


if __name__ == "__main__":
    asyncio.run(main())
