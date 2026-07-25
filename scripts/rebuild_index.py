#!/usr/bin/env python3
"""Download the curated corpus and rebuild its local vector index."""

from __future__ import annotations

import argparse

from pegasus_rag.config import Settings
from pegasus_rag.corpus import build_base_index
from pegasus_rag.embeddings import LocalSentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    embedder = LocalSentenceTransformer(settings.embedding_model)
    index = build_base_index(settings, embedder, force_download=args.force_download)
    print(f"Índice listo: {len(index.chunks)} fragmentos en {settings.index_dir}")


if __name__ == "__main__":
    main()

