#!/bin/sh
set -eu

if [ ! -f data/index/metadata.json ]; then
  echo "Base index not found. Building it now..."
  python scripts/rebuild_index.py
fi

exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true

