"""Embeddings via Ollama (bge-m3, multilingual, 1024 dims). Runs on the host GPU."""

from __future__ import annotations

import httpx

from app.config import get_settings


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    s = get_settings()
    out: list[list[float]] = []
    with httpx.Client(timeout=300) as client:
        for i in range(0, len(texts), batch_size):
            r = client.post(
                f"{s.ollama_url}/api/embed",
                json={"model": s.embed_model, "input": texts[i : i + batch_size], "keep_alive": s.ollama_keep_alive},
            )
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
    return out


def chunk_embedding_text(chunk: dict) -> str:
    """What gets embedded: the heading gives each chunk its document/article context."""
    return f"{chunk['heading']}\n{chunk['text']}"
