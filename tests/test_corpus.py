from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pegasus_rag.config import Settings
from pegasus_rag.corpus import download_documents, read_manifest, sha256_file


def make_settings(tmp_path: Path, manifest_path: Path) -> Settings:
    return Settings(
        gemini_api_key=None,
        gemini_model="gemini-test",
        embedding_model="embedding-test",
        max_upload_mb=10,
        max_upload_files=5,
        top_k=5,
        similarity_threshold=0.16,
        chunk_size=100,
        chunk_overlap=20,
        manifest_path=manifest_path,
        raw_data_dir=tmp_path / "raw",
        index_dir=tmp_path / "index",
    )


def test_manifest_and_download_verify_checksum(monkeypatch, tmp_path: Path) -> None:
    content = b"valid-pdf-content"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "filename": "guide.pdf",
                        "title": "Guide",
                        "url": "https://example.com/guide.pdf",
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    response = SimpleNamespace(content=content, raise_for_status=lambda: None)
    monkeypatch.setattr("pegasus_rag.corpus.requests.get", lambda *args, **kwargs: response)
    settings = make_settings(tmp_path, manifest_path)

    paths = download_documents(settings)

    assert read_manifest(manifest_path)[0]["title"] == "Guide"
    assert paths == [tmp_path / "raw" / "guide.pdf"]
    assert sha256_file(paths[0]) == hashlib.sha256(content).hexdigest()


def test_download_rejects_wrong_checksum(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "filename": "guide.pdf",
                        "title": "Guide",
                        "url": "https://example.com/guide.pdf",
                        "sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    response = SimpleNamespace(content=b"wrong", raise_for_status=lambda: None)
    monkeypatch.setattr("pegasus_rag.corpus.requests.get", lambda *args, **kwargs: response)

    with pytest.raises(ValueError, match="Checksum"):
        download_documents(make_settings(tmp_path, manifest_path))

