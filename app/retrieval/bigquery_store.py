"""BigQuery-backed vector store — production stub.

This class implements the VectorStore interface against BigQuery's VECTOR_SEARCH
function, intended as the production backend at PAYBACK scale.

WHY BIGQUERY VECTOR SEARCH:
    PAYBACK already operates a large BigQuery footprint for transactional and
    analytics data. Co-locating the product embedding index in BigQuery means:
    - No separate vector DB to operate, monitor, or pay for
    - Native joins between embeddings and other product/user signals (purchase
      history, partner data, point balances) — critical for the loyalty layer
    - Vertex AI's text-multilingual-embedding-002 produces embeddings that BigQuery
      can store and query natively via the ML.GENERATE_EMBEDDING + VECTOR_SEARCH
      functions
    - Scale: BigQuery handles 100M+ row vector indexes without operational burden

PRODUCTION SCHEMA:
    CREATE TABLE `project.payback_assistant.products` (
      product_id STRING,
      partner STRING,
      name STRING,
      description STRING,
      category STRING,
      price_eur FLOAT64,
      points_multiplier FLOAT64,
      active_promo BOOL,
      promo_text STRING,
      embedding ARRAY<FLOAT64>
    );

    CREATE VECTOR INDEX products_idx
    ON `project.payback_assistant.products` (embedding)
    OPTIONS(index_type='IVF', distance_type='COSINE');

QUERY PATTERN:
    SELECT base.*, distance
    FROM VECTOR_SEARCH(
      TABLE `project.payback_assistant.products`,
      'embedding',
      (SELECT ml_generate_embedding_result FROM ML.GENERATE_EMBEDDING(
        MODEL `project.payback_assistant.embedding_model`,
        (SELECT @query AS content)
      )),
      top_k => 10,
      distance_type => 'COSINE'
    );

MIGRATION PATH:
    1. Provision Vertex AI embedding model + BigQuery dataset via Terraform
    2. Backfill: read catalogs/*.json, write rows with ML.GENERATE_EMBEDDING
    3. Set VECTOR_STORE=bigquery in env — the rest of the app is unchanged
       (this is why the abstract VectorStore interface matters)

WHY NOT IMPLEMENTED HERE:
    Out of scope for the local development environment. The local ChromaDB store
    has identical behavior at this scale (~700 products) and avoids GCP billing
    during development. The interface is identical, so the swap is configuration-
    only at deploy time.
"""
from __future__ import annotations

from typing import Optional

from app.models.schemas import Partner, Product
from app.retrieval.vector_store import VectorStore


class BigQueryVectorStore(VectorStore):
    """Production vector store backed by BigQuery VECTOR_SEARCH and Vertex AI embeddings.

    See module docstring for full architecture, schema, and migration path.
    All methods raise NotImplementedError — set VECTOR_STORE=bigquery in .env
    and implement using google-cloud-bigquery + google-cloud-aiplatform SDKs.
    """

    async def add(self, products: list[Product]) -> None:
        # TODO: batch-embed via Vertex AI and stream-insert into BigQuery
        raise NotImplementedError

    async def search(
        self,
        query: str,
        top_k: int = 10,
        partner_filter: Optional[Partner] = None,
    ) -> list[tuple[Product, float]]:
        # TODO: run VECTOR_SEARCH parameterised query; deserialise rows into Product
        raise NotImplementedError

    async def count(self) -> int:
        # TODO: SELECT COUNT(*) FROM `{project}.{dataset}.products`
        raise NotImplementedError
