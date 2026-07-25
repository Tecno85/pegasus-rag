"""A dependency-light, persistent cosine-similarity vector index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pegasus_rag.embeddings import Embedder
from pegasus_rag.models import DocumentChunk, SearchResult


class VectorIndex:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        embedder: Embedder,
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("La matriz de embeddings debe tener dos dimensiones.")
        if len(chunks) != matrix.shape[0]:
            raise ValueError("La cantidad de chunks y embeddings no coincide.")
        self.chunks = chunks
        self.embeddings = matrix
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[DocumentChunk], embedder: Embedder) -> VectorIndex:
        if not chunks:
            raise ValueError("No hay fragmentos para indexar.")
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
        return cls(chunks, embeddings, embedder)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        threshold: float = 0.16,
    ) -> list[SearchResult]:
        if not query.strip() or not self.chunks:
            return []
        query_vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        if query_vector.ndim != 1 or query_vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("La dimensión del embedding de consulta no coincide con el índice.")
        scores = self.embeddings @ query_vector
        positions = np.argsort(scores)[::-1][: max(top_k, 0)]
        return [
            SearchResult(self.chunks[int(position)], float(scores[position]))
            for position in positions
            if float(scores[position]) >= threshold
        ]

    def save(self, directory: Path, *, model_name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings, allow_pickle=False)
        (directory / "chunks.json").write_text(
            json.dumps(
                [chunk.to_dict() for chunk in self.chunks],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        metadata = {
            "format_version": 1,
            "embedding_model": model_name,
            "chunks": len(self.chunks),
            "dimensions": int(self.embeddings.shape[1]),
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        embedder: Embedder,
        *,
        expected_model: str | None = None,
    ) -> VectorIndex:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("Versión de índice no soportada; reconstruye la base.")
        if expected_model and metadata.get("embedding_model") != expected_model:
            raise ValueError("El índice usa otro modelo de embeddings; reconstruye la base.")
        chunks_data = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [DocumentChunk.from_dict(item) for item in chunks_data]
        embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
        return cls(chunks, embeddings, embedder)


def index_exists(directory: Path) -> bool:
    return all(
        (directory / name).is_file()
        for name in ("embeddings.npy", "chunks.json", "metadata.json")
    )

