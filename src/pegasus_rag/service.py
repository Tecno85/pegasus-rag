"""Application service that joins retrieval, grounding, and citations."""

from __future__ import annotations

from pegasus_rag.generator import AnswerGenerator
from pegasus_rag.models import Answer, SearchResult, SourceReference
from pegasus_rag.store import VectorIndex

NO_EVIDENCE = "No encontré información suficiente en la base documental para responder."


class RagService:
    def __init__(
        self,
        generator: AnswerGenerator,
        *,
        top_k: int = 5,
        threshold: float = 0.16,
    ) -> None:
        self.generator = generator
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, question: str, indexes: list[VectorIndex]) -> list[SearchResult]:
        candidates = []
        for index in indexes:
            candidates.extend(
                index.search(question, top_k=self.top_k, threshold=self.threshold)
            )
        candidates.sort(key=lambda result: result.score, reverse=True)
        unique = []
        seen = set()
        for result in candidates:
            if result.chunk.chunk_id in seen:
                continue
            seen.add(result.chunk.chunk_id)
            unique.append(result)
            if len(unique) == self.top_k:
                break
        return unique

    def ask(
        self,
        question: str,
        indexes: list[VectorIndex],
        history: list[dict[str, str]] | None = None,
    ) -> Answer:
        clean_question = question.strip()
        if not clean_question:
            return Answer(NO_EVIDENCE, (), grounded=False)
        results = self.retrieve(clean_question, indexes)
        if not results:
            return Answer(NO_EVIDENCE, (), grounded=False)
        text = self.generator.generate(clean_question, results, history or [])
        sources = tuple(
            SourceReference(
                number=number,
                source=result.chunk.source,
                location=result.chunk.location,
                excerpt=result.chunk.text[:500],
                score=result.score,
                source_url=result.chunk.source_url,
            )
            for number, result in enumerate(results, start=1)
        )
        return Answer(text, sources, grounded=True)

