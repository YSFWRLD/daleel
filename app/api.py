import httpx
import psycopg
from fastapi import FastAPI

from app.config import get_settings

app = FastAPI(title="Saudi Compliance Assistant", version="0.1.0")


def _check_db(url: str) -> dict:
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return {"ok": ext is not None, "pgvector": ext[0] if ext else None, "chunks": n}
    except Exception as e:  # noqa: BLE001 - health endpoint reports any failure
        return {"ok": False, "error": str(e)}


def _check_ollama(url: str, needed: list[str]) -> dict:
    try:
        r = httpx.get(f"{url}/api/tags", timeout=3)
        r.raise_for_status()
        have = {m["name"] for m in r.json().get("models", [])}
        missing = [m for m in needed if m not in have and f"{m}:latest" not in have]
        return {"ok": not missing, "missing_models": missing}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@app.get("/health")
def health() -> dict:
    s = get_settings()
    db = _check_db(s.database_url)
    checks = {"db": db}
    if s.llm_provider == "ollama":
        checks["ollama"] = _check_ollama(s.ollama_url, [s.ollama_model, s.embed_model])
    else:
        checks["ollama"] = _check_ollama(s.ollama_url, [s.embed_model])
        checks["gemini"] = {"ok": bool(s.gemini_api_key)}
    return {"ok": all(c["ok"] for c in checks.values()), "llm_provider": s.llm_provider, **checks}
