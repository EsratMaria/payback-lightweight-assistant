from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    unified_endpoint_base_url_anthropic: str = Field(default="", alias="UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC")
    unified_endpoint_key: str = Field(default="", alias="UNIFIED_ENDPOINT_KEY")

    default_llm_provider: str = Field(
        default="claude", alias="DEFAULT_LLM_PROVIDER"
    )

    embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2", alias="EMBEDDING_MODEL"
    )

    vector_store: Literal["local", "bigquery"] = Field(
        default="local", alias="VECTOR_STORE"
    )
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")

    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")
    bigquery_dataset: str = Field(
        default="payback_assistant", alias="BIGQUERY_DATASET"
    )

    intent_confidence_threshold: float = Field(
        default=0.6, alias="INTENT_CONFIDENCE_THRESHOLD"
    )
    retrieval_weak_threshold: float = Field(
        default=0.4, alias="RETRIEVAL_WEAK_THRESHOLD"
    )
    clarification_topk: int = Field(default=20, alias="CLARIFICATION_TOPK")

    # Loyalty ranker weights (α, β, γ)
    loyalty_weight_semantic: float = Field(default=0.6, alias="LOYALTY_WEIGHT_SEMANTIC")
    loyalty_weight_commercial: float = Field(default=0.3, alias="LOYALTY_WEIGHT_COMMERCIAL")
    loyalty_weight_diversity: float = Field(default=0.1, alias="LOYALTY_WEIGHT_DIVERSITY")

    # Tuning constants for commercial score
    points_multiplier_scale: float = Field(default=0.4, alias="POINTS_MULTIPLIER_SCALE")
    active_promo_bonus: float = Field(default=0.3, alias="ACTIVE_PROMO_BONUS")

    # Tuning constant for user-affinity diversity bonus
    affinity_diversity_scale: float = Field(default=0.5, alias="AFFINITY_DIVERSITY_SCALE")

    # Query expansion: catalog-grounding settings
    expansion_preflight_topk: int = Field(default=10, alias="EXPANSION_PREFLIGHT_TOPK")
    expansion_min_relevance: float = Field(default=0.45, alias="EXPANSION_MIN_RELEVANCE")


settings = Settings()
