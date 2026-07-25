from __future__ import annotations

import numpy as np

from pegasus_rag.embeddings import LocalSentenceTransformer


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_local_embedder_uses_normalized_numpy_arrays() -> None:
    embedder = LocalSentenceTransformer("fake")
    embedder._model = FakeModel()

    documents = embedder.embed_documents(["uno", "dos"])
    query = embedder.embed_query("uno")

    assert documents.shape == (2, 2)
    assert query.shape == (2,)
    assert documents.dtype == np.float32


def test_local_embedder_handles_empty_collection() -> None:
    embedder = LocalSentenceTransformer("fake")

    assert embedder.embed_documents([]).shape == (0, 0)

