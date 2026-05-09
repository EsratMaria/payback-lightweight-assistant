from __future__ import annotations

import numpy as np


class Embedder:
    """Wraps a sentence-transformers multilingual model for local text embedding."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), embedding_dim).

        Args:
            texts: Raw text strings to encode (product names, descriptions, or queries).

        Returns:
            numpy array of shape (N, D) where D is the model's embedding dimension.
        """
        # TODO: load model from config.EMBEDDING_MODEL and call model.encode()
        raise NotImplementedError
