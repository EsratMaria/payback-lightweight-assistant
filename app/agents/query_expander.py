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
from app.retrieval.local_store import LocalChromaStore
from app.retrieval.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUERY_EXPANSION_SYSTEM_PROMPT = (
    "You decompose multi-item shopping queries into related sub-queries for semantic "
    "retrieval. PAYBACK users shop across three partners — dm (drugstore), EDEKA (grocery), "
    "Amazon (long-tail).\n\n"
    "When a user asks for a 'basket' of items — a meal, an event, a project — you produce "
    "3-5 related search queries that together cover what they'd actually buy from these partners.\n\n"
    "Rules:\n"
    "- Always include the user's original query as one of the sub-queries.\n"
    "- Sub-queries should be short — 1-3 words each, retrieval-friendly.\n"
    "- STAY WITHIN THE AVAILABLE CATALOG CATEGORIES provided in the user prompt. "
    "Do not invent sub-queries for categories not listed.\n"
    "- If the available categories list is empty or limited, produce best-effort sub-queries "
    "but stay close to the user's original intent rather than inventing aspirational items.\n"
    "- Match the user's language (German or English).\n"
    "- For ambiguous queries, choose the most common interpretation.\n\n"
    "You MUST respond by calling the `respond` tool."
)

QUERY_EXPANSION_USER_PROMPT_TEMPLATE = (
    "Decompose the following shopping query into 3-5 retrieval-friendly sub-queries.\n\n"
    'QUERY: "{query}"\n'
    "LANGUAGE: {language}\n"
    "AVAILABLE CATALOG CATEGORIES FOR THIS QUERY: {available_categories}\n\n"
    "Return:\n"
    "- sub_queries: list with the original query first, then 2-4 related sub-queries "
    "that STAY WITHIN the available categories\n"
    "- reasoning: one sentence explaining the decomposition and how it respects the available categories"
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class QueryExpander:
    """Decomposes basket-style queries into 3-5 related sub-queries for broader retrieval.

    Catalog-grounded: a pre-flight retrieval discovers available categories before
    the LLM expansion call, so sub-queries stay within catalog scope rather than
    proposing items that don't exist (e.g. "glitter eyeshadow" when only "cosmetics"
    and "personal_care" are available).
    """

    def __init__(self, llm: LLMClient, vector_store: VectorStore) -> None:
        self.llm = llm
        self.vector_store = vector_store

    async def _discover_available_categories(self, query: str) -> list[str]:
        """Pre-flight retrieval to discover what categories the catalog has for this query.

        Returns a sorted list of unique category strings from the top-K results.
        If retrieval returns nothing, returns an empty list — the prompt will note
        'no clear matches in catalog' and let the LLM produce a best-effort expansion.
        """
        results = await self.vector_store.search(
            query, top_k=settings.expansion_preflight_topk
        )
        categories = sorted({product.category for product, _ in results})
        return categories

    async def expand(self, query: str, language: Language) -> ExpandedQueries:
        """Return catalog-grounded ExpandedQueries with the original query at index 0."""
        available_categories = await self._discover_available_categories(query)
        category_hint = (
            ", ".join(available_categories)
            if available_categories
            else "no clear category matches found in the catalog for this query"
        )

        user_prompt = QUERY_EXPANSION_USER_PROMPT_TEMPLATE.format(
            query=query,
            language=language.value,
            available_categories=category_hint,
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

        return ExpandedQueries(sub_queries=sub_queries, reasoning=expanded.reasoning)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_default_query_expander() -> QueryExpander:
    """Returns a QueryExpander backed by the default LLM provider and vector store."""
    return QueryExpander(
        llm=ClaudeClient(),
        vector_store=LocalChromaStore(persist_dir=settings.chroma_persist_dir),
    )
