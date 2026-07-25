from conftest import FakeEmbedder

from pegasus_rag.models import DocumentChunk
from pegasus_rag.service import NO_EVIDENCE, RagService
from pegasus_rag.store import VectorIndex


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question, results, history):
        self.calls += 1
        return "La cobertura mínima es 80% [Fuente 1]."


def test_service_returns_grounded_answer_and_sources() -> None:
    chunk = DocumentChunk(
        "coverage",
        "La cobertura mínima de pruebas es 80%.",
        "Guía Back-end",
        "Página 7",
        "doc",
    )
    index = VectorIndex.build([chunk], FakeEmbedder())
    generator = FakeGenerator()
    service = RagService(generator, threshold=0.2)

    answer = service.ask("¿Cuál es la cobertura de pruebas?", [index])

    assert answer.grounded
    assert answer.sources[0].location == "Página 7"
    assert "[Fuente 1]" in answer.text
    assert generator.calls == 1


def test_service_does_not_call_model_without_evidence() -> None:
    chunk = DocumentChunk("git", "Usamos GitFlow.", "Onboarding", "Página 16", "doc")
    index = VectorIndex.build([chunk], FakeEmbedder())
    generator = FakeGenerator()
    service = RagService(generator, threshold=0.9)

    answer = service.ask("¿Cuál es la cobertura de pruebas?", [index])

    assert answer.text == NO_EVIDENCE
    assert not answer.grounded
    assert generator.calls == 0

