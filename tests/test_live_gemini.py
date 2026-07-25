import os

import pytest
from dotenv import load_dotenv

from pegasus_rag.generator import GeminiGenerator
from pegasus_rag.models import DocumentChunk, SearchResult

load_dotenv()


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") or os.getenv("RUN_LIVE_TESTS") != "1",
    reason="GEMINI_API_KEY or RUN_LIVE_TESTS not configured",
)
def test_gemini_live_grounded_smoke() -> None:
    chunk = DocumentChunk("one", "La cobertura mínima es 80%.", "Guía", "Página 7", "doc")
    generator = GeminiGenerator(
        os.environ["GEMINI_API_KEY"], os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    )

    answer = generator.generate(
        "¿Cuál es la cobertura mínima?", [SearchResult(chunk, 1.0)], []
    )

    assert "80" in answer
