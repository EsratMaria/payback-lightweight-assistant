"""Router — orchestrates intent classification → retrieval → ranking → response.

This is the top-level coordinator. It holds no business logic itself: it reads
the IntentResult produced by the IntentAgent and dispatches to the right branch,
then assembles the final AssistantResponse.

Branches (in order):
  A   navigational + known partner → navigation_target, no retrieval
  A.5 support → hardcoded out-of-scope message, no retrieval
  A.6 navigational + no partner → hardcoded "which partner?" question, no retrieval
  B   specific + high-confidence → retrieve, rank, recommendations
  C   vague / low-confidence → retrieve for grounding + clarification agent
"""
from __future__ import annotations

import time
from typing import Optional

from app.agents.clarification import ClarificationAgent
from app.agents.intent_agent import IntentAgent, get_default_intent_agent
from app.agents.query_expander import QueryExpander, get_default_query_expander
from app.config import settings
from app.llm.claude_client import ClaudeClient
from app.models.schemas import (
    AssistantResponse,
    ClarifyingQuestion,
    Intent,
    Language,
    Specificity,
    UserContext,
)
from app.ranking.loyalty_ranker import LoyaltyRanker
from app.retrieval.local_store import LocalChromaStore
from app.retrieval.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Cost estimation constants (rough Claude Sonnet pricing, USD→EUR)
# ---------------------------------------------------------------------------

COST_PER_1K_INPUT_TOKENS_EUR = 0.0028
COST_PER_1K_OUTPUT_TOKENS_EUR = 0.0139

TOKENS_PER_INTENT_CALL_INPUT = 400
TOKENS_PER_INTENT_CALL_OUTPUT = 150
TOKENS_PER_CLARIFICATION_CALL_INPUT = 600
TOKENS_PER_CLARIFICATION_CALL_OUTPUT = 100
TOKENS_PER_EXPANSION_CALL_INPUT = 300
TOKENS_PER_EXPANSION_CALL_OUTPUT = 80


def estimate_cost(
    intent_calls: int,
    clarification_calls: int,
    expansion_calls: int = 0,
) -> float:
    """Return estimated total cost in EUR for this request's LLM calls."""
    intent_cost = intent_calls * (
        TOKENS_PER_INTENT_CALL_INPUT / 1000 * COST_PER_1K_INPUT_TOKENS_EUR
        + TOKENS_PER_INTENT_CALL_OUTPUT / 1000 * COST_PER_1K_OUTPUT_TOKENS_EUR
    )
    clarification_cost = clarification_calls * (
        TOKENS_PER_CLARIFICATION_CALL_INPUT / 1000 * COST_PER_1K_INPUT_TOKENS_EUR
        + TOKENS_PER_CLARIFICATION_CALL_OUTPUT / 1000 * COST_PER_1K_OUTPUT_TOKENS_EUR
    )
    expansion_cost = expansion_calls * (
        TOKENS_PER_EXPANSION_CALL_INPUT / 1000 * COST_PER_1K_INPUT_TOKENS_EUR
        + TOKENS_PER_EXPANSION_CALL_OUTPUT / 1000 * COST_PER_1K_OUTPUT_TOKENS_EUR
    )
    return intent_cost + clarification_cost + expansion_cost


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Router:
    """Stateless orchestrator: classify → route → respond.

    All dependencies are injected — the router never instantiates anything.
    This makes it trivially testable with mocks and decoupled from provider choices.
    """

    def __init__(
        self,
        intent_agent: IntentAgent,
        clarification_agent: ClarificationAgent,
        vector_store: VectorStore,
        ranker: LoyaltyRanker,
        query_expander: QueryExpander | None = None,
    ) -> None:
        self.intent_agent = intent_agent
        self.clarification_agent = clarification_agent
        self.vector_store = vector_store
        self.ranker = ranker
        self.query_expander = query_expander

    async def handle(
        self,
        query: str,
        user_context: Optional[UserContext] = None,
    ) -> AssistantResponse:
        """Classify the query and return a fully populated AssistantResponse."""
        start = time.perf_counter()

        intent = await self.intent_agent.classify(query)

        # ------------------------------------------------------------------
        # Branch A: navigational — user named a specific partner to open
        # ------------------------------------------------------------------
        if (
            intent.specificity == Specificity.navigational
            and intent.target_partner is not None
        ):
            latency_ms = (time.perf_counter() - start) * 1000
            return AssistantResponse(
                response_type="navigation",
                intent_result=intent,
                navigation_target=str(intent.target_partner),
                recommendations=None,
                clarification=None,
                latency_ms=latency_ms,
                estimated_cost_eur=estimate_cost(intent_calls=1, clarification_calls=0),
            )

        # ------------------------------------------------------------------
        # Branch A.5: support intent — out-of-scope, hardcoded honest message.
        # No retrieval, no second LLM call. The assistant focuses on product
        # discovery; account/points questions belong in PAYBACK support.
        # ------------------------------------------------------------------
        if intent.intent == Intent.support:
            if intent.language == Language.de:
                out_of_scope = ClarifyingQuestion(
                    question=(
                        "Ich kann dir helfen, Produkte bei dm, EDEKA und Amazon zu finden. "
                        "Für Fragen zu Punkten, deinem Konto oder Service besuche bitte "
                        "https://www.payback.group/de/kontakt oder den PAYBACK-Support."
                    ),
                    suggested_options=[
                        "Stattdessen ein Produkt suchen",
                        "Angebote durchsuchen",
                        "Zu https://www.payback.group/de/kontakt",
                    ],
                )
            else:
                out_of_scope = ClarifyingQuestion(
                    question=(
                        "I can help you find products across dm, EDEKA, and Amazon — "
                        "but for points, account, or service questions, please visit "
                        "https://www.payback.group/en/contact or contact PAYBACK support."
                    ),
                    suggested_options=[
                        "Find a product instead",
                        "Browse deals",
                        "Go to https://www.payback.group/en/contact",
                    ],
                )
            latency_ms = (time.perf_counter() - start) * 1000
            return AssistantResponse(
                response_type="clarification",
                intent_result=intent,
                clarification=out_of_scope,
                recommendations=None,
                navigation_target=None,
                latency_ms=latency_ms,
                estimated_cost_eur=estimate_cost(intent_calls=1, clarification_calls=0),
            )

        # ------------------------------------------------------------------
        # Branch A.6: navigational without a recognised partner.
        # Ask the user which of our three partners they want. No retrieval.
        # ------------------------------------------------------------------
        if (
            intent.specificity == Specificity.navigational
            and intent.target_partner is None
        ):
            if intent.language == Language.de:
                partner_question = ClarifyingQuestion(
                    question="Welchen Partner möchtest du besuchen?",
                    suggested_options=["dm", "EDEKA", "Amazon"],
                )
            else:
                partner_question = ClarifyingQuestion(
                    question="Which partner would you like to visit?",
                    suggested_options=["dm", "EDEKA", "Amazon"],
                )
            latency_ms = (time.perf_counter() - start) * 1000
            return AssistantResponse(
                response_type="clarification",
                intent_result=intent,
                clarification=partner_question,
                recommendations=None,
                navigation_target=None,
                latency_ms=latency_ms,
                estimated_cost_eur=estimate_cost(intent_calls=1, clarification_calls=0),
            )

        # ------------------------------------------------------------------
        # Branch B: specific + high-confidence — retrieve and rank.
        # The intent != support guard below is a safety net; A.5 catches
        # support before this branch is reached.
        # ------------------------------------------------------------------
        if (
            intent.specificity == Specificity.specific
            and intent.confidence >= settings.intent_confidence_threshold
            and intent.intent != Intent.support
        ):
            expanded_queries: list[str] | None = None
            did_expand = intent.is_basket_query and self.query_expander is not None

            if did_expand:
                sub_queries = await self.query_expander.expand(
                    intent.extracted_query, intent.language
                )
                expanded_queries = sub_queries
                all_results: dict[str, tuple] = {}
                for sub_q in sub_queries:
                    results = await self.vector_store.search(
                        sub_q, top_k=8, partner_filter=intent.target_partner
                    )
                    for product, score in results:
                        existing = all_results.get(product.product_id)
                        if existing is None or existing[1] < score:
                            all_results[product.product_id] = (product, score)
                retrieved = sorted(all_results.values(), key=lambda x: -x[1])[:15]
            else:
                retrieved = await self.vector_store.search(
                    intent.extracted_query,
                    top_k=10,
                    partner_filter=intent.target_partner,
                )

            ranked = await self.ranker.rank(retrieved, user_context)
            latency_ms = (time.perf_counter() - start) * 1000
            return AssistantResponse(
                response_type="recommendations",
                intent_result=intent,
                recommendations=ranked,
                clarification=None,
                navigation_target=None,
                latency_ms=latency_ms,
                estimated_cost_eur=estimate_cost(
                    intent_calls=1,
                    clarification_calls=0,
                    expansion_calls=1 if did_expand else 0,
                ),
                debug_expanded_queries=expanded_queries,
            )

        # ------------------------------------------------------------------
        # Branch C: vague / low-confidence — retrieve for grounding, then
        # ask a catalog-grounded clarifying question.
        # ------------------------------------------------------------------
        search_query = intent.extracted_query or query
        retrieved = await self.vector_store.search(
            search_query,
            top_k=settings.clarification_topk,
        )
        clarification = await self.clarification_agent.generate(
            query, retrieved, intent.language
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return AssistantResponse(
            response_type="clarification",
            intent_result=intent,
            clarification=clarification,
            recommendations=None,
            navigation_target=None,
            latency_ms=latency_ms,
            estimated_cost_eur=estimate_cost(intent_calls=1, clarification_calls=1),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_default_router() -> Router:
    """Wire up the default router using ClaudeClient, LocalChromaStore, and LoyaltyRanker."""
    llm = ClaudeClient()
    intent_agent = get_default_intent_agent()
    clarification_agent = ClarificationAgent(llm)
    query_expander = get_default_query_expander()
    vector_store = LocalChromaStore(persist_dir=settings.chroma_persist_dir)  # swap for BigQuery in prod
    ranker = LoyaltyRanker()
    return Router(
        intent_agent=intent_agent,
        clarification_agent=clarification_agent,
        vector_store=vector_store,
        ranker=ranker,
        query_expander=query_expander,
    )
