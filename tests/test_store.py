from pathlib import Path

from conftest import FakeEmbedder

from pegasus_rag.models import DocumentChunk
from pegasus_rag.store import VectorIndex, index_exists


def make_chunk(identifier: str, text: str) -> DocumentChunk:
    return DocumentChunk(identifier, text, "guide.pdf", "Página 1", "doc")


def test_vector_search_orders_by_similarity() -> None:
    embedder = FakeEmbedder()
    index = VectorIndex.build(
        [make_chunk("tests", "La cobertura de pruebas es 80%."), make_chunk("git", "Usa GitFlow.")],
        embedder,
    )

    results = index.search("¿Cuál es la cobertura?", top_k=2, threshold=0.1)

    assert results[0].chunk.chunk_id == "tests"
    assert results[0].score == 1.0


def test_index_round_trip(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    index = VectorIndex.build([make_chunk("one", "pruebas")], embedder)
    index.save(tmp_path, model_name="fake")

    restored = VectorIndex.load(tmp_path, embedder, expected_model="fake")

    assert index_exists(tmp_path)
    assert restored.chunks == index.chunks
    assert restored.search("pruebas")[0].chunk.chunk_id == "one"

