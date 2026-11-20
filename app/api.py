import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm.base import LLMError, get_llm

log = logging.getLogger("compliance")
STATIC = Path(__file__).parent / "static"
_ready = threading.Event()


def _warm_up() -> None:
    """Load the BM25 index, bge-m3, the reranker and the LLM so the first question is not ~20 s slower."""
    try:
        from app.retrieval.hybrid import warm_up

        warm_up()
        llm = get_llm()
        if hasattr(llm, "warm_up"):
            llm.warm_up()
    except Exception:  # noqa: BLE001 - the API still works, just slower on the first question
        log.exception("warm-up failed")
    finally:
        _ready.set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()  # server starts answering /health at once
    yield


app = FastAPI(title="Saudi Compliance Assistant", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    provider: Literal["ollama", "gemini"] | None = None
    strategy: Literal["article", "fixed"] | None = None
    reranker: bool | None = None


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    from app.answer import answer_question

    try:
        result = answer_question(
            req.question.strip(), llm=get_llm(req.provider), strategy=req.strategy, use_reranker=req.reranker
        )
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return result.to_dict()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def _check_db(url: str) -> dict:
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return {"ok": ext is not None and n > 0, "pgvector": ext[0] if ext else None, "chunks": n}
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
    return {
        "ok": all(c["ok"] for c in checks.values()),
        "warmed_up": _ready.is_set(),
        "llm_provider": s.llm_provider,
        **checks,
    }
