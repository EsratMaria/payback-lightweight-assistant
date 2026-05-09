from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.models.schemas import Partner, Product


class VectorStore(ABC):
    """Abstract interface for all vector store backends."""

    @abstractmethod
    async def add(self, products: list[Product]) -> None:
        """Upsert `products` into the store, embedding their name + description.

        Implementations must be idempotent: re-adding an existing product_id
        should update, not duplicate.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        partner_filter: Optional[Partner] = None,
    ) -> list[tuple[Product, float]]:
        """Return up to `top_k` (product, cosine_similarity) pairs for `query`.

        Args:
            query: Raw natural-language query string (embedding happens inside).
            top_k: Maximum number of results to return.
            partner_filter: If provided, restrict results to this partner only.

        Returns:
            List of (Product, score) sorted descending by score.
        """
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of documents currently indexed."""
        ...
