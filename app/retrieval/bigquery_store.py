from __future__ import annotations

from typing import Optional

from app.models.schemas import Partner, Product
from app.retrieval.vector_store import VectorStore


class BigQueryVectorStore(VectorStore):
    """Production vector store backed by BigQuery VECTOR_SEARCH and Vertex AI embeddings.

    Architecture overview
    ---------------------
    This backend is designed for PAYBACK's GCP production environment where latency
    requirements allow ~200 ms round-trips and the corpus can grow to millions of
    products across all partners.

    Embedding model
    ~~~~~~~~~~~~~~~
    Uses ``text-multilingual-embedding-002`` from Vertex AI, which supports both
    German and English — matching the PAYBACK user base.  Embeddings are 768-dimensional
    and stored in a FLOAT64 REPEATED column named ``embedding`` in BigQuery.

    Indexing
    ~~~~~~~~
    Products are ingested via a batch Cloud Run job that:
      1. Calls ``aiplatform.TextEmbeddingModel.get_embeddings()`` in batches of 250.
      2. Streams results to BigQuery with ``google-cloud-bigquery`` client.
      3. A scheduled ``CREATE VECTOR INDEX`` DDL refreshes the IVF index nightly.

    Query path
    ~~~~~~~~~~
    ``search()`` executes a parameterised SQL query of the form::

        SELECT
            base.*,
            distance
        FROM
            VECTOR_SEARCH(
                TABLE `{project}.{dataset}.products`,
                'embedding',
                (SELECT ml_generate_embedding_result
                 FROM ML.GENERATE_EMBEDDING(
                     MODEL `{project}.{dataset}.embedding_model`,
                     (SELECT @query AS content)
                 )),
                top_k => @top_k,
                distance_type => 'COSINE'
            )
        WHERE
            (@partner IS NULL OR base.partner = @partner)
            AND base.active = TRUE
        ORDER BY distance ASC

    Cost model
    ~~~~~~~~~~
    * Embedding inference: ~$0.0001 per 1 k characters (Vertex AI pricing).
    * VECTOR_SEARCH: billed as BigQuery on-demand per TB scanned; the IVF index
      reduces scans by ~95 % on a 1 M-row table.
    * Typical per-query cost at 300 k products: < €0.0005.

    Authentication
    ~~~~~~~~~~~~~~
    Requires Application Default Credentials with roles:
    ``roles/bigquery.dataViewer``, ``roles/bigquery.jobUser``,
    ``roles/aiplatform.user``.

    Implementation note
    ~~~~~~~~~~~~~~~~~~~
    This class intentionally raises ``NotImplementedError`` so that the local
    ChromaDB backend is used during development.  Swap ``VECTOR_STORE=bigquery``
    in ``.env`` when deploying to GCP and implement the body of each method using
    ``google-cloud-bigquery`` and ``google-cloud-aiplatform`` SDKs.
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
