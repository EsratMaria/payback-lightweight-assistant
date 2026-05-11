"""Manual sanity check for IntentAgent against real Claude API.

Run: python scripts/try_intent.py
"""
import asyncio

from app.agents.intent_agent import get_default_intent_agent

"""
"open Amazon for me" came back as intent=discovery, specificity=navigational. 
That's slightly inconsistent — discovery implies "browsing without target" but navigational implies "specific target." 
Claude is mixing the dimensions a bit.
It doesn't break anything because the router switches on specificity, not intent. Navigational wins.

In the schema, intent and specificity are independent dimensions. 
The router uses specificity. 
Intent is more of a behavioral signal — useful for analytics later, not for routing. 

"""

QUERIES = [
    "I need stuff for a pasta dinner",
    "I need stuff for a pasta dinner from dm",
    "Bitte zeige mir Angebote für günstige Windeln",
    "open Amazon for me",
    "something for my dog",
    "wireless mouse vs trackpad which is better",
    "etwas Schönes für meinen Mann",
    "how do I redeem my points",
    "pasta dinner from edeka",
    "Windeln bei dm",
    "pasta dinner from dm or edeka",
    "pasta from REWE",
    "Schokolade von Lidl",
]


async def main() -> None:
    agent = get_default_intent_agent()
    for q in QUERIES:
        result = await agent.classify(q)
        print(f"\nQuery: {q!r}")
        # print(result)
        print(f"  language     = {result.language.value}")
        print(f"  intent       = {result.intent.value}")
        print(f"  specificity  = {result.specificity.value}")
        print(f"  confidence   = {result.confidence:.2f}")
        print(f"  target       = {result.target_partner.value if result.target_partner else None}")
        print(f"  extracted    = {result.extracted_query!r}")
        print(f"  reasoning    = {result.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
