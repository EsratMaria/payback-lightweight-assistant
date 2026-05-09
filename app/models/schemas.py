from __future__ import annotations

from enum import StrEnum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Partner(StrEnum):
    dm = "dm"
    edeka = "edeka"
    amazon = "amazon"


class Language(StrEnum):
    de = "de"
    en = "en"


class Intent(StrEnum):
    search = "search"
    discovery = "discovery"
    comparison = "comparison"
    support = "support"


class Specificity(StrEnum):
    specific = "specific"
    vague = "vague"
    navigational = "navigational"


class Product(BaseModel):
    product_id: str = Field(..., description="Unique product identifier")
    partner: Partner = Field(..., description="Partner this product belongs to")
    name: str = Field(..., description="Product display name")
    description: str = Field(..., description="Product description")
    category: str = Field(..., description="Product category")
    price_eur: float = Field(..., description="Price in EUR", gt=0)
    points_multiplier: float = Field(
        default=1.0, description="PAYBACK points multiplier for this product", ge=0
    )
    active_promo: bool = Field(
        default=False, description="Whether an active promotion exists"
    )
    promo_text: Optional[str] = Field(
        default=None, description="Promotional text if active_promo is True"
    )


class UserContext(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Authenticated user ID")
    partner_affinity: dict[Partner, float] = Field(
        default_factory=lambda: {
            Partner.dm: 0.33,
            Partner.edeka: 0.33,
            Partner.amazon: 0.33,
        },
        description="Relative affinity scores per partner, should sum to ~1.0",
    )
    is_new_user: bool = Field(
        default=True, description="True if the user has no purchase history"
    )


class IntentResult(BaseModel):
    language: Language = Field(..., description="Detected language of the query")
    intent: Intent = Field(..., description="Classified intent category")
    specificity: Specificity = Field(..., description="Query specificity level")
    confidence: float = Field(
        ..., description="Classification confidence score", ge=0.0, le=1.0
    )
    extracted_query: str = Field(
        ..., description="Cleaned / normalized query for retrieval"
    )
    target_partner: Optional[Partner] = Field(
        default=None, description="Explicit partner mentioned in the query, if any"
    )
    reasoning: str = Field(
        ..., description="LLM reasoning trace for the intent classification"
    )


class ProductRecommendation(BaseModel):
    product: Product = Field(..., description="The recommended product")
    semantic_score: float = Field(
        ..., description="Raw cosine similarity from vector search", ge=0.0, le=1.0
    )
    loyalty_boost: float = Field(
        ..., description="Computed loyalty/promo boost component", ge=0.0
    )
    final_score: float = Field(
        ..., description="Weighted composite score used for ranking", ge=0.0
    )
    rank: int = Field(..., description="1-based rank in the result set", ge=1)


class ClarifyingQuestion(BaseModel):
    question: str = Field(..., description="The clarifying question to surface to the user")
    suggested_options: list[str] = Field(
        ..., description="Pre-computed answer options to show as quick replies"
    )


class AssistantResponse(BaseModel):
    response_type: Literal["recommendations", "clarification", "navigation"] = Field(
        ..., description="Which action branch was taken"
    )
    intent_result: IntentResult = Field(
        ..., description="Full intent classification result"
    )
    recommendations: Optional[list[ProductRecommendation]] = Field(
        default=None, description="Ranked product recommendations (response_type=recommendations)"
    )
    clarification: Optional[ClarifyingQuestion] = Field(
        default=None, description="Clarifying question (response_type=clarification)"
    )
    navigation_target: Optional[str] = Field(
        default=None, description="Deep-link or route for navigational queries"
    )
    latency_ms: float = Field(..., description="End-to-end request latency in milliseconds")
    estimated_cost_eur: float = Field(
        ..., description="Estimated LLM inference cost for this request in EUR", ge=0.0
    )
