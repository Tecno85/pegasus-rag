"""Runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from pegasus_rag.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} debe ser un número entero.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} debe ser mayor que cero.")
    return value


def _bounded_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} debe ser un número decimal.") from exc
    if not -1 <= value <= 1:
        raise ConfigurationError(f"{name} debe estar entre -1 y 1.")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    embedding_model: str
    max_upload_mb: int
    max_upload_files: int
    top_k: int
    similarity_threshold: float
    chunk_size: int
    chunk_overlap: int
    manifest_path: Path
    raw_data_dir: Path
    index_dir: Path

    @classmethod
    def from_env(cls, project_root: Path = PROJECT_ROOT) -> Settings:
        load_dotenv(project_root / ".env")
        chunk_size = _positive_int("CHUNK_SIZE", 1100)
        chunk_overlap = _positive_int("CHUNK_OVERLAP", 180)
        if chunk_overlap >= chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE.")
        key = os.getenv("GEMINI_API_KEY", "").strip() or None
        return cls(
            gemini_api_key=key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip(),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ).strip(),
            max_upload_mb=_positive_int("MAX_UPLOAD_MB", 10),
            max_upload_files=_positive_int("MAX_UPLOAD_FILES", 5),
            top_k=_positive_int("TOP_K", 5),
            similarity_threshold=_bounded_float("SIMILARITY_THRESHOLD", 0.16),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            manifest_path=project_root / "data" / "manifest.json",
            raw_data_dir=project_root / "data" / "raw",
            index_dir=project_root / "data" / "index",
        )
