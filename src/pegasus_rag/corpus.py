"""Reproducible download and indexing of the curated Santo Pegasus corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import requests

from pegasus_rag.chunking import chunk_sections
from pegasus_rag.config import Settings
from pegasus_rag.embeddings import Embedder
from pegasus_rag.loaders import load_path
from pegasus_rag.models import RawSection
from pegasus_rag.store import VectorIndex, index_exists


def read_manifest(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("El manifiesto no contiene documentos.")
    return documents


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_documents(settings: Settings, *, force: bool = False) -> list[Path]:
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in read_manifest(settings.manifest_path):
        destination = settings.raw_data_dir / item["filename"]
        expected_hash = item["sha256"]
        if destination.exists() and sha256_file(destination) == expected_hash and not force:
            paths.append(destination)
            continue
        response = requests.get(item["url"], timeout=60)
        response.raise_for_status()
        downloaded_hash = hashlib.sha256(response.content).hexdigest()
        if downloaded_hash != expected_hash:
            raise ValueError(f"Checksum inválido al descargar {item['filename']}.")
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(destination)
        paths.append(destination)
    return paths


def build_base_index(
    settings: Settings,
    embedder: Embedder,
    *,
    force_download: bool = False,
) -> VectorIndex:
    paths = download_documents(settings, force=force_download)
    manifest = {item["filename"]: item for item in read_manifest(settings.manifest_path)}
    all_sections: list[RawSection] = []
    for path in paths:
        item = manifest[path.name]
        sections = load_path(path, source_url=item["url"])
        all_sections.extend(replace(section, source=item["title"]) for section in sections)
    chunks = chunk_sections(
        all_sections,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    index = VectorIndex.build(chunks, embedder)
    index.save(settings.index_dir, model_name=settings.embedding_model)
    return index


def load_or_build_base_index(settings: Settings, embedder: Embedder) -> VectorIndex:
    """Load the persisted corpus or build it on first start.

    Streamlit Community Cloud launches ``app.py`` directly instead of using the
    Docker entrypoint. Keeping this bootstrap in the application layer makes a
    fresh deployment reproducible while preserving the persistent local index
    used by Docker and development environments.
    """
    if index_exists(settings.index_dir):
        return VectorIndex.load(
            settings.index_dir,
            embedder,
            expected_model=settings.embedding_model,
        )
    return build_base_index(settings, embedder)
