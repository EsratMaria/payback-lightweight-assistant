from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    unified_endpoint_base_url_anthropic: str = Field(default="", alias="UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC")
    unified_endpoint_key: str = Field(default="", alias="UNIFIED_ENDPOINT_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")

    default_llm_provider: Literal["claude", "gemini"] = Field(
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


settings = Settings()
