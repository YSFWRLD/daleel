FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# The optional reranker needs torch (CPU-only build; the GPU is used by Ollama on the host).
# Build with it:  docker compose build --build-arg RERANKER=bge   (and set RERANKER=bge in .env)
ARG RERANKER=none
RUN if [ "$RERANKER" = "bge" ]; then pip install --index-url https://download.pytorch.org/whl/cpu torch; fi

COPY pyproject.toml ./
COPY app ./app
COPY ingest ./ingest
COPY eval ./eval
COPY db ./db
RUN if [ "$RERANKER" = "bge" ]; then pip install ".[rerank]"; else pip install .; fi

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
