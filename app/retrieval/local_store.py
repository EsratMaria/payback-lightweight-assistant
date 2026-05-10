from __future__ import annotations

from typing import Optional

import chromadb

from app.config import settings
from app.models.schemas import Partner, Product
from app.retrieval.embedder import get_embedder
from app.retrieval.vector_store import VectorStore

COLLECTION_NAME = "products"


class LocalChromaStore(VectorStore):
    """ChromaDB-backed local vector store for development and testing.

    Design decisions:
    - Only `name + description` are embedded; `category` is stored as filterable
      metadata. This keeps semantic search focused on product content while still
      allowing exact-match partner/category filtering.
    - hnsw:space=cosine must match our L2-normalised embeddings.
    - All operations are async-compatible (no blocking I/O beyond the initial
      PersistentClient construction, which happens once in __init__).
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        path = persist_dir or settings.chroma_persist_dir
        self._client = chromadb.PersistentClient(
            path=path,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_text(product: Product) -> str:
        """Text we embed: name + description only.

        Category is intentionally excluded — it lives in metadata for
        exact-match filtering, not in the embedding space.
        """
        return f"{product.name}. {product.description}"

    @staticmethod
    def _to_metadata(product: Product) -> dict:
        """Flatten Product to a ChromaDB-compatible metadata dict.

        ChromaDB does not support nested dicts or None values in metadata.
        We store promo_text as "" when None and convert back in _from_metadata.
        """
        return {
            "partner": product.partner.value,
            "name": product.name,
            "description": product.description,
            "category": product.category,
            "price_eur": product.price_eur,
            "points_multiplier": product.points_multiplier,
            "active_promo": product.active_promo,
            "promo_text": product.promo_text or "",
        }

    @staticmethod
    def _from_metadata(product_id: str, metadata: dict) -> Product:
        """Reconstruct a Product from ChromaDB metadata."""
        return Product(
            product_id=product_id,
            partner=metadata["partner"],
            name=metadata["name"],
            description=metadata["description"],
            category=metadata["category"],
            price_eur=metadata["price_eur"],
            points_multiplier=metadata["points_multiplier"],
            active_promo=bool(metadata["active_promo"]),
            promo_text=metadata["promo_text"] or None,
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    async def add(self, products: list[Product]) -> None:
        if not products:
            return

        embedder = get_embedder()
        texts = [self._embed_text(p) for p in products]
        embeddings = embedder.encode(texts)

        self._collection.upsert(
            ids=[p.product_id for p in products],
            embeddings=embeddings.tolist(),
            metadatas=[self._to_metadata(p) for p in products],
            documents=texts,
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        partner_filter: Optional[Partner] = None,
    ) -> list[tuple[Product, float]]:
        count = self._collection.count()
        if count == 0:
            return []

        embedder = get_embedder()
        query_embedding = embedder.encode([query])[0]

        # Cap n_results to actual collection size to avoid ChromaDB errors
        # when top_k exceeds the number of stored documents.
        query_kwargs: dict = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, count),
        }
        if partner_filter is not None:
            query_kwargs["where"] = {"partner": partner_filter.value}

        results = self._collection.query(**query_kwargs)

        output: list[tuple[Product, float]] = []
        for product_id, distance, metadata in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            # ChromaDB returns distances (lower = better for cosine).
            # Convert to similarity (higher = better) for downstream ranking.
            # Clamp to [0, 1]: floating-point rounding can push 1.0 - distance
            # slightly below 0 for near-orthogonal vectors.
            similarity = max(0.0, min(1.0, 1.0 - distance))
            output.append((self._from_metadata(product_id, metadata), similarity))

        return output

    async def count(self) -> int:
        return self._collection.count()

    async def reset(self) -> None:
        """Delete and recreate the collection. Used by the --rebuild ingest flag."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
