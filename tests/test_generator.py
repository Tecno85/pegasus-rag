import pytest

from pegasus_rag.errors import MissingApiKeyError, QuotaExceededError
from pegasus_rag.generator import GeminiGenerator, build_prompt
from pegasus_rag.models import DocumentChunk, SearchResult


def test_prompt_numbers_sources_and_limits_history() -> None:
    chunk = DocumentChunk("id", "Cobertura mínima: 80%.", "Guía", "Página 7", "doc")
    history = [{"role": "user", "content": str(number)} for number in range(8)]

    prompt = build_prompt("¿Cuál es la cobertura?", [SearchResult(chunk, 0.9)], history)

    assert "[Fuente 1] Guía — Página 7" in prompt
    assert "Cobertura mínima: 80%." in prompt
    assert "user: 0" not in prompt
    assert "user: 7" in prompt


def test_generator_requires_api_key_before_client_creation() -> None:
    generator = GeminiGenerator(None, "gemini-test")

    with pytest.raises(MissingApiKeyError, match="GEMINI_API_KEY"):
        _ = generator.client


class FakeModels:
    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error

    def generate_content(self, **kwargs):
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None) -> None:
        self.models = FakeModels(response, error)


def test_generator_returns_provider_text() -> None:
    generator = GeminiGenerator("key", "gemini-test")
    response = type("Response", (), {"text": "Respuesta [Fuente 1]."})()
    generator._client = FakeClient(response=response)
    chunk = DocumentChunk("id", "Evidencia", "Guía", "Página 1", "doc")

    answer = generator.generate("Pregunta", [SearchResult(chunk, 1.0)], [])

    assert answer == "Respuesta [Fuente 1]."


def test_generator_translates_quota_error() -> None:
    generator = GeminiGenerator("key", "gemini-test")
    generator._client = FakeClient(error=RuntimeError("429 RESOURCE_EXHAUSTED quota"))
    chunk = DocumentChunk("id", "Evidencia", "Guía", "Página 1", "doc")

    with pytest.raises(QuotaExceededError, match="cuota gratuita"):
        generator.generate("Pregunta", [SearchResult(chunk, 1.0)], [])
