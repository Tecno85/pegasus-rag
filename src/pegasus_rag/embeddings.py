"""Local multilingual embeddings with a small testable interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pegasus_rag.errors import ConfigurationError


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class LocalSentenceTransformer:
    """Lazily loads the model so imports and UI startup remain lightweight."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception:
                try:
                    self._model = SentenceTransformer(self.model_name)
                except Exception as offline_error:
                    raise ConfigurationError(
                        "No se pudo cargar el modelo de embeddings. Verifica la conexión "
                        "o reconstruye la caché local."
                    ) from offline_error
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        values = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(values, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        values = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(values[0], dtype=np.float32)
