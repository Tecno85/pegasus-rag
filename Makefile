.PHONY: install data index run test lint check

install:
	python -m pip install -e ".[dev]"

data:
	python scripts/download_documents.py

index:
	python scripts/rebuild_index.py

run:
	streamlit run app.py

test:
	pytest --cov=pegasus_rag --cov-report=term-missing

lint:
	ruff check .

check: lint test

