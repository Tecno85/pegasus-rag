"""Document-aware text chunking."""

from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pegasus_rag.models import DocumentChunk, RawSection


def chunk_sections(
    sections: list[RawSection], *, chunk_size: int = 1100, chunk_overlap: int = 180
) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        length_function=len,
    )
    chunks: list[DocumentChunk] = []
    for section in sections:
        for part_number, text in enumerate(splitter.split_text(section.text), start=1):
            clean_text = text.strip()
            if not clean_text:
                continue
            digest = hashlib.sha256(
                f"{section.document_id}:{section.location}:{part_number}:{clean_text}".encode()
            ).hexdigest()[:20]
            chunks.append(
                DocumentChunk(
                    chunk_id=digest,
                    text=clean_text,
                    source=section.source,
                    location=section.location,
                    document_id=section.document_id,
                    source_url=section.source_url,
                )
            )
    return chunks
