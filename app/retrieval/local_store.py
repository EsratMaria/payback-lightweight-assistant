from __future__ import annotations

from typing import Optional

from app.models.schemas import Partner, Product
from app.retrieval.vector_store import VectorStore


class LocalChromaStore(VectorStore):
    """ChromaDB-backed local vector store for development and testing."""

    async def add(self, products: list[Product]) -> None:
        # TODO: embed products with Embedder, upsert into Chroma collection
        raise NotImplementedError

    async def search(
        self,
        query: str,
        top_k: int = 10,
        partner_filter: Optional[Partner] = None,
    ) -> list[tuple[Product, float]]:
        # TODO: embed query, query Chroma with optional where filter on partner
        raise NotImplementedError

    async def count(self) -> int:
        # TODO: return collection.count()
        raise NotImplementedError
