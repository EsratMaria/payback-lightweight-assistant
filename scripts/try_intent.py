"""Manual sanity check for IntentAgent against real Claude API.

Run: python scripts/try_intent.py
"""
import asyncio

from app.agents.intent_agent import get_default_intent_agent

QUERIES = [
    "I need stuff for a pasta dinner",
    "Bitte zeige mir Angebote für günstige Windeln",
    "open Amazon for me",
    "something for my dog",
    "wireless mouse vs trackpad which is better",
    "etwas Schönes für meinen Mann",
    "how do I redeem my points",
]


async def main() -> None:
    agent = get_default_intent_agent()
    for q in QUERIES:
        result = await agent.classify(q)
        print(f"\nQuery: {q!r}")
        print(f"  language     = {result.language.value}")
        print(f"  intent       = {result.intent.value}")
        print(f"  specificity  = {result.specificity.value}")
        print(f"  confidence   = {result.confidence:.2f}")
        print(f"  target       = {result.target_partner.value if result.target_partner else None}")
        print(f"  extracted    = {result.extracted_query!r}")
        print(f"  reasoning    = {result.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
