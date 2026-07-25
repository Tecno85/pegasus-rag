FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/pegasus/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY app.py ./
COPY .streamlit ./.streamlit
COPY data/manifest.json ./data/manifest.json
COPY scripts ./scripts

RUN mkdir -p /app/data/raw /app/data/index \
    && useradd --create-home --uid 10001 pegasus \
    && chown -R pegasus:pegasus /app /home/pegasus \
    && chmod +x /app/scripts/start.sh
USER pegasus

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["./scripts/start.sh"]
