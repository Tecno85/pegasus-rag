from pegasus_rag.chunking import chunk_sections
from pegasus_rag.models import RawSection


def test_chunking_preserves_document_metadata() -> None:
    section = RawSection(
        text="Una política importante. " * 20,
        source="manual.pdf",
        location="Página 7",
        document_id="abc",
        source_url="https://example.com/manual.pdf",
    )

    chunks = chunk_sections([section], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.source == "manual.pdf" for chunk in chunks)
    assert all(chunk.location == "Página 7" for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

