from __future__ import annotations

import numpy as np


class FakeEmbedder:
    """Deterministic two-dimensional embeddings for unit tests."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self._vector(text), dtype=np.float32)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if any(word in lowered for word in ("prueba", "coverage", "cobertura")):
            return [1.0, 0.0]
        return [0.0, 1.0]

