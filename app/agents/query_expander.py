"""Query expansion agent.

Decomposes basket-style queries into 3-5 related sub-queries so the retrieval
layer can surface a full shopping-basket result rather than only literal matches.

Only invoked when the intent agent flags is_basket_query=True. For single-item
queries the router skips this module entirely.
"""
from __future__ import annotations

from app.config import settings
from app.llm.base import LLMClient
from app.llm.claude_client import ClaudeClient
from app.models.schemas import ExpandedQueries, Language

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUERY_EXPANSION_SYSTEM_PROMPT = (
    "You decompose multi-item shopping queries into related sub-queries for semantic "
    "retrieval. PAYBACK users shop across three partners — dm (drugstore), EDEKA (grocery), "
    "Amazon (long-tail).\n\n"
    "When a user asks for a 'basket' of items — a meal, an event, a project — you produce "
    "3-5 related search queries that together cover what they'd actually buy. Each sub-query "
    "should be retrievable on its own.\n\n"
    "Rules:\n"
    "- Always include the user's original query as one of the sub-queries.\n"
    "- Sub-queries should be short — 1-3 words each, retrieval-friendly.\n"
    "- Stay grounded in what our three partners realistically sell. Don't suggest fresh fish "
    "for dm, don't suggest cosmetics for EDEKA.\n"
    "- Match the user's language. If the query is German, return German sub-queries. "
    "If English, English.\n"
    "- For ambiguous queries, choose the most common interpretation (e.g. 'pasta dinner' "
    "= Italian-style, not Asian noodle dish).\n\n"
    "You MUST respond by calling the `respond` tool."
)

QUERY_EXPANSION_USER_PROMPT_TEMPLATE = (
    "Decompose the following shopping query into 3-5 retrieval-friendly sub-queries.\n\n"
    'QUERY: "{query}"\n'
    "LANGUAGE: {language}\n\n"
    "Return:\n"
    "- sub_queries: list with the original query first, then 2-4 related sub-queries\n"
    "- reasoning: one sentence explaining the decomposition"
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class QueryExpander:
    """Decomposes basket-style queries into 3-5 related sub-queries for broader retrieval.

    Used when the intent agent flags is_basket_query=True. For "pasta dinner" the expander
    might produce ["pasta dinner", "pasta", "tomato sauce", "olive oil", "parmesan"], enabling
    retrieval to surface a full shopping-basket result rather than only literal pasta.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def expand(self, query: str, language: Language) -> list[str]:
        """Return 3-5 sub-queries with the original at index 0."""
        user_prompt = QUERY_EXPANSION_USER_PROMPT_TEMPLATE.format(
            query=query,
            language=language.value,
        )
        result = await self.llm.structured_completion(
            prompt=user_prompt,
            schema=ExpandedQueries,
            system=QUERY_EXPANSION_SYSTEM_PROMPT,
            max_tokens=256,
        )
        expanded: ExpandedQueries = result  # type: ignore[assignment]
        # Guarantee original query is first
        sub_queries = list(expanded.sub_queries)
        if sub_queries and sub_queries[0] != query:
            if query in sub_queries:
                sub_queries.remove(query)
            sub_queries.insert(0, query)
        return sub_queries


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_default_query_expander() -> QueryExpander:
    """Returns a QueryExpander backed by the default LLM provider (currently Claude)."""
    provider = settings.default_llm_provider
    if provider == "claude":
        return QueryExpander(ClaudeClient())
    if provider == "gemini":
        raise NotImplementedError("Gemini client deferred — see app/llm/gemini_client.py")
    raise ValueError(f"Unknown LLM provider: {provider!r}")
