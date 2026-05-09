from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter

from app.models.schemas import (
    AssistantResponse,
    Intent,
    IntentResult,
    Language,
    Partner,
    Product,
    ProductRecommendation,
    Specificity,
    UserContext,
)

router = APIRouter()


class AssistRequest(dict):
    pass


from pydantic import BaseModel


class AssistRequest(BaseModel):
    query: str
    user_context: Optional[UserContext] = None


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/assist", response_model=AssistantResponse)
async def assist(request: AssistRequest) -> AssistantResponse:
    """Main assistant endpoint — stub returning mock data while implementation is pending."""
    start = time.perf_counter()

    mock_product = Product(
        product_id="mock-001",
        partner=Partner.dm,
        name="dm Bio Shampoo 250ml",
        description="Organic shampoo for all hair types",
        category="Haarpflege",
        price_eur=3.95,
        points_multiplier=2.0,
        active_promo=True,
        promo_text="Doppelte Punkte diese Woche!",
    )

    mock_intent = IntentResult(
        language=Language.de,
        intent=Intent.search,
        specificity=Specificity.specific,
        confidence=0.92,
        extracted_query=request.query,
        target_partner=None,
        reasoning="Stub intent result — LLM not called yet.",
    )

    mock_recommendation = ProductRecommendation(
        product=mock_product,
        semantic_score=0.85,
        loyalty_boost=3.0,
        final_score=0.84,
        rank=1,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return AssistantResponse(
        response_type="recommendations",
        intent_result=mock_intent,
        recommendations=[mock_recommendation],
        latency_ms=latency_ms,
        estimated_cost_eur=0.0,
    )
