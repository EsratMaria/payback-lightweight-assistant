"""Router — orchestrates intent classification → retrieval → ranking → response.

This is the top-level coordinator. It holds no business logic itself: it reads
the IntentResult produced by the IntentAgent and dispatches to the right branch,
then assembles the final AssistantResponse.

Branches:
- navigational  → return navigation_target immediately (no retrieval)
- specific/high-confidence → retrieve, rank, return recommendations
- vague/low-confidence/support → retrieve for grounding, generate clarifying question
"""
from __future__ import annotations

import time
from typing import Optional

from app.agents.clarification import ClarificationAgent
from app.agents.intent_agent import IntentAgent, get_default_intent_agent
from app.config import settings
from app.llm.claude_client import ClaudeClient
from app.models.schemas import (
    AssistantResponse,
    Intent,
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


def estimate_cost(intent_calls: int, clarification_calls: int) -> float:
    """Return estimated total cost in EUR for this request's LLM calls."""
    intent_cost = intent_calls * (
        TOKENS_PER_INTENT_CALL_INPUT / 1000 * COST_PER_1K_INPUT_TOKENS_EUR
        + TOKENS_PER_INTENT_CALL_OUTPUT / 1000 * COST_PER_1K_OUTPUT_TOKENS_EUR
    )
    clarification_cost = clarification_calls * (
        TOKENS_PER_CLARIFICATION_CALL_INPUT / 1000 * COST_PER_1K_INPUT_TOKENS_EUR
        + TOKENS_PER_CLARIFICATION_CALL_OUTPUT / 1000 * COST_PER_1K_OUTPUT_TOKENS_EUR
    )
    return intent_cost + clarification_cost


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
    ) -> None:
        self.intent_agent = intent_agent
        self.clarification_agent = clarification_agent
        self.vector_store = vector_store
        self.ranker = ranker

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
        # Branch B: specific + high-confidence — retrieve and rank
        # ------------------------------------------------------------------
        if (
            intent.specificity == Specificity.specific
            and intent.confidence >= settings.intent_confidence_threshold
            and intent.intent != Intent.support
        ):
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
                estimated_cost_eur=estimate_cost(intent_calls=1, clarification_calls=0),
            )

        # ------------------------------------------------------------------
        # Branch C: vague / low-confidence / support / navigational-no-target
        # Retrieve for grounding, then ask a clarifying question.
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
    """Wire up the default router using ClaudeClient, LocalChromaStore, and LoyaltyRanker stub."""
    llm = ClaudeClient()
    intent_agent = get_default_intent_agent()
    clarification_agent = ClarificationAgent(llm)
    vector_store = LocalChromaStore(persist_dir=settings.chroma_persist_dir)
    ranker = LoyaltyRanker()
    return Router(
        intent_agent=intent_agent,
        clarification_agent=clarification_agent,
        vector_store=vector_store,
        ranker=ranker,
    )
