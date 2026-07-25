from pathlib import Path

import pytest

from pegasus_rag.config import Settings
from pegasus_rag.errors import ConfigurationError


def test_settings_use_safe_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in (
        "GEMINI_API_KEY",
        "MAX_UPLOAD_MB",
        "MAX_UPLOAD_FILES",
        "TOP_K",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings.from_env(tmp_path)

    assert settings.gemini_api_key is None
    assert settings.max_upload_mb == 10
    assert settings.top_k == 5
    assert settings.index_dir == tmp_path / "data" / "index"


def test_overlap_must_be_smaller_than_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    with pytest.raises(ConfigurationError, match="menor"):
        Settings.from_env(tmp_path)
