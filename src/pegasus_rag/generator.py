"""Grounded answer generation through the official Gemini SDK."""

from __future__ import annotations

from typing import Protocol

from pegasus_rag.errors import (
    GenerationError,
    MissingApiKeyError,
    ProviderUnavailableError,
    QuotaExceededError,
)
from pegasus_rag.models import SearchResult

SYSTEM_INSTRUCTION = '''Eres Pegasus RAG, un asistente de conocimiento empresarial.
Responde exclusivamente con los fragmentos proporcionados. No uses conocimiento externo ni
inventes información. Escribe en el idioma de la pregunta, de forma clara y directa. Incluye
referencias inline como [Fuente 1] después de cada afirmación relevante. Si el contexto no basta,
di exactamente: "No encontré información suficiente en la base documental para responder."'''


class AnswerGenerator(Protocol):
    def generate(
        self,
        question: str,
        results: list[SearchResult],
        history: list[dict[str, str]],
    ) -> str: ...


def build_prompt(
    question: str,
    results: list[SearchResult],
    history: list[dict[str, str]],
) -> str:
    context_parts = []
    for number, result in enumerate(results, start=1):
        chunk = result.chunk
        context_parts.append(
            f"[Fuente {number}] {chunk.source} — {chunk.location}\n{chunk.text}"
        )
    recent_history = history[-6:]
    history_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent_history
    )
    context_text = "\n\n".join(context_parts)
    return f"""CONTEXTO DOCUMENTAL
{context_text}

CONVERSACIÓN RECIENTE
{history_text or '(sin mensajes anteriores)'}

PREGUNTA ACTUAL
{question}

Redacta la respuesta sustentada y cita únicamente las fuentes numeradas anteriores."""


class GeminiGenerator:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self):
        if not self.api_key:
            raise MissingApiKeyError(
                "Falta GEMINI_API_KEY. Agrégala al archivo .env o a las variables de la VM."
            )
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(
        self,
        question: str,
        results: list[SearchResult],
        history: list[dict[str, str]],
    ) -> str:
        from google.genai import types

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=build_prompt(question, results, history),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.15,
                    max_output_tokens=900,
                ),
            )
        except MissingApiKeyError:
            raise
        except Exception as exc:
            message = str(exc).lower()
            if any(token in message for token in ("quota", "resource_exhausted", "429")):
                raise QuotaExceededError(
                    "La cuota gratuita de Gemini se agotó. Intenta de nuevo cuando se renueve."
                ) from exc
            if any(token in message for token in ("api key", "api_key", "401", "403")):
                raise MissingApiKeyError(
                    "La API key de Gemini no existe, es inválida o no tiene acceso al modelo."
                ) from exc
            raise ProviderUnavailableError(
                "Gemini no está disponible temporalmente. Conservamos tus documentos en la sesión."
            ) from exc
        text = (response.text or "").strip()
        if not text:
            raise GenerationError("Gemini devolvió una respuesta vacía.")
        return text
