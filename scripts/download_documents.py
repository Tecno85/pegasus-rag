#!/usr/bin/env python3
"""Download and verify the curated source documents."""

from __future__ import annotations

import argparse

from pegasus_rag.config import Settings
from pegasus_rag.corpus import download_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Download even valid files again")
    args = parser.parse_args()
    settings = Settings.from_env()
    paths = download_documents(settings, force=args.force)
    print(f"Documentos verificados: {len(paths)}")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()

