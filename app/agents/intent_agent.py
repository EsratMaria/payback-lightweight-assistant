"""Intent classification agent.

Sends the raw user query to the configured LLM and returns an IntentResult.
The agent's only job is classification — it does not route, clarify, or search.
Routing decisions live in app/agents/router.py (Step 5).
"""
from __future__ import annotations

from app.config import settings
from app.llm.base import LLMClient
from app.llm.claude_client import ClaudeClient
from app.models.schemas import IntentResult

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier for a multilingual shopping assistant for PAYBACK, "
    "a German loyalty program. PAYBACK users shop across three partners:\n"
    "- dm (drugstore: cosmetics, baby products, household, personal care)\n"
    "- EDEKA (grocery: fresh food, dairy, pantry, beverages)\n"
    "- Amazon (long-tail: electronics, home, books, tools, etc.)\n\n"
    "Your job is to classify each user query so the system knows what to do next. "
    "You MUST respond by calling the `respond` tool with valid IntentResult fields. "
    "Always provide a brief reasoning trace."
)

INTENT_USER_PROMPT_TEMPLATE = """Classify the following query.

QUERY: "{query}"

Definitions:
- language: "de" if query is mostly German, "en" if mostly English. Mixed -> pick the dominant language.
- intent:
    - "search": user knows what they want, wants matching products.
    - "discovery": user is browsing or exploring without a target ("show me deals", "what's on sale").
    - "comparison": user wants to compare options ("X vs Y", "which is better").
    - "support": user has a problem or question about points, account, or how something works.
- specificity:
    - "specific": query is concrete enough to retrieve useful products (e.g. "pasta dinner", "guenstige Windeln", "wireless mouse under 30 euros").
    - "vague": query is too broad to retrieve useful results (e.g. "something for my dog", "a gift", "etwas Schoenes fuer meinen Mann").
    - "navigational": user explicitly wants a specific partner (e.g. "open Amazon", "EDEKA Angebote", "go to dm shop").
- target_partner: set ONLY if specificity is "navigational" - one of "dm", "edeka", "amazon". Otherwise null.
- extracted_query: a cleaned version of the query suitable for semantic search. Strip filler words; preserve intent. For navigational queries this can be empty.
- confidence: 0.0-1.0 - how certain you are about the classification. Use lower confidence when the query is borderline.
- reasoning: ONE short sentence explaining your decision. This is logged for debugging."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class IntentAgent:
    """Classifies a raw user query into a structured IntentResult.

    The agent is LLM-provider-agnostic: it accepts any LLMClient implementation
    via constructor injection. This means tests can pass a MockLLMClient, and
    swapping Claude for Gemini in production requires no changes here.

    The router (Step 5) uses BOTH `specificity` and `confidence` together:
    - Routes to clarification if specificity == "vague".
    - Also routes to clarification if specificity == "specific" but confidence < 0.6
      (configurable threshold — the agent reports confidence, the router enforces it).
    The agent itself does not enforce any threshold — it just classifies and reports.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def classify(self, query: str) -> IntentResult:
        """Classify `query` and return a fully validated IntentResult."""
        user_prompt = INTENT_USER_PROMPT_TEMPLATE.format(query=query)
        result = await self.llm.structured_completion(
            prompt=user_prompt,
            schema=IntentResult,
            system=INTENT_SYSTEM_PROMPT,
            max_tokens=512,
        )
        return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_default_intent_agent() -> IntentAgent:
    """Return an IntentAgent wired to the configured default LLM provider.

    Reads settings.default_llm_provider. Currently only 'claude' is implemented;
    'gemini' raises NotImplementedError pointing to the GeminiClient stub.
    """
    provider = settings.default_llm_provider

    if provider == "claude":
        return IntentAgent(ClaudeClient())

    if provider == "gemini":
        raise NotImplementedError(
            "Gemini client deferred — see app/llm/gemini_client.py"
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")
