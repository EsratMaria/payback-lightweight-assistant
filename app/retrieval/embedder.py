from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class Embedder:
    """Wraps a sentence-transformers multilingual model for local text embedding.

    The model is lazy-loaded on first access so importing this module does not
    trigger a heavyweight model download at import time.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimensionality, derived from the model at first access."""
        return self.model.get_embedding_dimension()

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim).

        Args:
            texts: Strings to encode — product text or query strings.
            normalize: L2-normalise each vector. Required for cosine similarity
                       to work correctly with ChromaDB's hnsw:space=cosine.

        Returns:
            numpy array of shape (N, dim), dtype float32.
            Returns shape (0, dim) for empty input.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Module-level singleton — loads the model once across the entire process."""
    return Embedder()
