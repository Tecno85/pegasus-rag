"""Small serializable models shared across the RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RawSection:
    text: str
    source: str
    location: str
    document_id: str
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    location: str
    document_id: str
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DocumentChunk:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True, slots=True)
class SourceReference:
    number: int
    source: str
    location: str
    excerpt: str
    score: float
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: tuple[SourceReference, ...]
    grounded: bool = True

