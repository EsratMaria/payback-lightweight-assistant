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

    # Loyalty layer - Considering PAYBACK
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
    """Mocking user profile for loyalty-aware ranking."""
    user_id: Optional[str] = Field(default=None, description="Authenticated user ID")
    partner_affinity: dict[Partner, float] = Field(
        default_factory=lambda: {
            Partner.dm: 0.33,
            Partner.edeka: 0.33,
            Partner.amazon: 0.33,
        },
        description="Relative affinity scores per partner, should sum to ~1.0 || how often a user shops there..",
    )
    is_new_user: bool = Field(
        default=True, description="True if the user has no purchase history || cold start flag"
    )


class IntentResult(BaseModel):
    """Output of the intent agent — what the LLM tells us about the query."""

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
        default=None,
        description=(
            "The recognized partner the user explicitly mentions, if any. Set when the "
            "user names one of our supported partners (dm, edeka, amazon) — whether for "
            "navigation ('open Amazon') or search constraint ('pasta from dm'). The "
            "router decides what to do with this based on the specificity field. "
            "Null if no partner is mentioned, if multiple partners are mentioned, or "
            "if the named partner is not in our system."
        ),
    )
    reasoning: str = Field(
        ..., description="LLM reasoning trace for the intent classification"
    )
    is_basket_query: bool = Field(
        default=False,
        description=(
            "True if the query implies multiple related products typically bought together "
            "(e.g. 'pasta dinner', 'Geburtstagsparty', 'ingredients for tiramisu', 'stuff for a hike'). "
            "False for single-item queries ('wireless mouse', 'Schokolade'). "
            "The router uses this signal to trigger LLM-driven query expansion before retrieval."
        ),
    )


class ExpandedQueries(BaseModel):
    """Output of the query expander — sub-queries for basket-style intents."""

    sub_queries: list[str] = Field(
        min_length=3,
        max_length=5,
        description="3-5 related search queries derived from the user's basket intent.",
    )
    reasoning: str = Field(description="One sentence explaining the decomposition.")


class ProductRecommendation(BaseModel):
    """A scored product result."""

    product: Product = Field(..., description="The recommended product")
    semantic_score: float = Field(
        ..., description="Raw cosine similarity from vector search", ge=0.0, le=1.0
    )
    loyalty_boost: float = Field(
        ..., description="Computed commercial loyalty/promo boost component", ge=0.0
    )
    diversity_bonus: float = Field(
        default=0.0, description="Partner diversity bonus applied during ranking", ge=0.0
    )
    final_score: float = Field(
        ..., description="Weighted composite score used for ranking", ge=0.0
    )
    rank: int = Field(..., description="1-based rank in the result set", ge=1)


class ClarifyingQuestion(BaseModel):
    """When the query is too vague."""

    question: str = Field(..., description="The clarifying question to surface to the user")
    suggested_options: list[str] = Field(
        ..., description="Pre-computed answer options to show as quick replies"
    )


class AssistRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    user_context: Optional[UserContext] = None


class AssistantResponse(BaseModel):
    """The thing we return from the API."""
    
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
    debug_expanded_queries: Optional[list[str]] = Field(
        default=None,
        description=(
            "Sub-queries used when basket expansion ran. Debug field only — "
            "production deployments should gate this behind a debug flag."
        ),
    )


# ---------------------------------------------------------------------------
# Eval schemas — used by evals/ runners only, not part of the production API
# ---------------------------------------------------------------------------


class JudgmentScore(BaseModel):
    """Single judge scoring run for one query response."""

    query_satisfaction: int = Field(
        ge=0, le=3,
        description="Did the response address what the user asked? 0=completely off, 3=fully satisfying",
    )
    result_quality: int = Field(
        ge=0, le=3,
        description="Are recommendations/clarification appropriate and grounded? 0=completely off, 3=fully satisfying",
    )
    language_tone_match: int = Field(
        ge=0, le=3,
        description="Did the response match the user's language? 0=wrong language, 3=perfect match",
    )
    reasoning: str = Field(
        description="One sentence per dimension explaining the score."
    )


class QueryJudgment(BaseModel):
    """Aggregated judge result for a single query across 3 runs."""

    query: str
    response_type: str
    runs: list[JudgmentScore]
    median_query_satisfaction: int
    median_result_quality: int
    median_language_tone: int
    total_median: int
    variance_flag: bool


class E2EEvalResult(BaseModel):
    """Full E2E eval result across all queries."""

    total_queries: int
    mean_total: float
    mean_query_satisfaction: float
    mean_result_quality: float
    mean_language_tone: float
    high_variance_queries: list[str]
    per_query: list[QueryJudgment]
    judge_human_agreement: dict | None = None
    duration_seconds: float
