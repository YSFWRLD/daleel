"""Hybrid retrieval: BM25 + vector search, fused with Reciprocal Rank Fusion, optionally reranked.

Results are grouped by (doc_id, canonical_key) so one article/control appears once, and each hit carries
its other-language counterpart: citations always show the Arabic (authoritative) text. See
docs/decisions/001-arabic-is-binding.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.retrieval.bm25 import get_index
from app.retrieval.embed import embed
from app.retrieval.vector import vector_search
from app.text import arabic

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
RRF_K = 60


@dataclass
class Hit:
    chunk: dict
    score: float  # reranker score in [0, 1] (or fused RRF score when the reranker is off)
    rrf: float
    counterpart: dict | None = None  # same article/control in the other language


@dataclass
class Retrieval:
    query: str
    lang: str
    hits: list[Hit]
    timings: dict[str, float] = field(default_factory=dict)


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


@lru_cache
def _structural_by_key() -> dict[tuple[str, str, str], dict]:
    """(doc_id, canonical_key, lang) -> first article/control chunk, for counterparts."""
    out: dict[tuple[str, str, str], dict] = {}
    for line in (PROCESSED / "chunks_article.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        out.setdefault((c["doc_id"], c["canonical_key"], c["lang"]), c)
    return out


def chunk_keys(chunk: dict) -> list[str]:
    """Canonical keys of every article/control a chunk covers (fixed-size chunks can span several).
    Rows from Postgres store them in `refs`; rows from the JSONL files in `canonical_refs`."""
    return list(chunk.get("canonical_refs") or chunk.get("refs") or [chunk.get("canonical_key")])


def counterpart(chunk: dict) -> dict | None:
    other = "ar" if chunk["lang"] == "en" else "en"
    return _structural_by_key().get((chunk["doc_id"], chunk["canonical_key"], other))


def warm_up() -> None:
    """Load everything the first question would otherwise wait for (~20 s): the BM25 indexes, the
    bge-m3 model in Ollama, and the int8 reranker. Call once at API startup."""
    s = get_settings()
    get_index(s.chunking)
    _structural_by_key()
    embed(["warm up"])
    if s.reranker == "bge":
        from app.retrieval.rerank import warm_up as warm_reranker

        warm_reranker()


def retrieve(
    query: str,
    strategy: str | None = None,
    top_k: int | None = None,
    use_reranker: bool | None = None,
    mode: str = "hybrid",
) -> Retrieval:
    """mode: "hybrid" (BM25 + vector, RRF), "vector" or "bm25" (single retriever, for ablations)."""
    if mode not in ("hybrid", "vector", "bm25"):
        raise ValueError(f"unknown retrieval mode: {mode}")
    s = get_settings()
    strategy = strategy or s.chunking
    top_k = top_k or s.top_k
    use_reranker = (s.reranker == "bge") if use_reranker is None else use_reranker
    t: dict[str, float] = {}

    dense_lists: list[list[tuple[dict, float]]] = []
    sparse: list[tuple[dict, float]] = []
    if mode != "bm25":
        t0 = time.perf_counter()
        qvec = embed([query])[0]
        t["embed"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # One list per language: in a mixed list the question's own language always ranks first, so an
        # English question never reached Arabic-only text (e.g. ECC 2-7-3, missing from the translation).
        langs = ("ar", "en") if s.vector_per_lang else (None,)
        dense_lists = [vector_search(qvec, strategy, s.candidates, lang) for lang in langs]
        t["vector"] = time.perf_counter() - t0

    if mode != "vector":
        t0 = time.perf_counter()
        sparse = get_index(strategy).search(query, s.candidates)
        t["bm25"] = time.perf_counter() - t0

    by_id = {c["id"]: c for lst in dense_lists for c, _ in lst}
    by_id.update({c["id"]: c for c, _ in sparse if c["id"] not in by_id})
    lists = [[c["id"] for c, _ in lst] for lst in (*dense_lists, sparse)]
    fused = rrf_fuse([lst for lst in lists if lst])
    # The reranker pool starts with each list's own top results: a chunk that only one list finds (Arabic-only
    # text for an English question) loses fusion to chunks found by two lists, even when it ranks first.
    pool = [cid for lst in lists for cid in lst[: s.rerank_per_list]]
    pool += sorted(fused, key=lambda cid: -fused[cid])
    ranked = list(dict.fromkeys(pool))[: s.rerank_top]

    if use_reranker:
        from app.retrieval.rerank import rerank

        t0 = time.perf_counter()
        scores = rerank(query, [f"{by_id[c]['heading']}\n{by_id[c]['text']}" for c in ranked])
        t["rerank"] = time.perf_counter() - t0
        order = sorted(zip(ranked, scores, strict=True), key=lambda x: -x[1])
    else:  # each list's top results first (as measured in eval/results.md), then the rest by fused score
        order = [(cid, fused[cid]) for cid in ranked]

    hits: list[Hit] = []
    seen: set[tuple[str, str | None]] = set()
    for cid, score in order:  # one hit per article/control (either language)
        c = by_id[cid]
        key = (c["doc_id"], c["canonical_key"])
        if key in seen:
            continue
        seen.add(key)
        hits.append(Hit(chunk=c, score=float(score), rrf=fused[cid], counterpart=counterpart(c)))
        if len(hits) == top_k:
            break
    return Retrieval(query=query, lang=arabic.detect_lang(query), hits=hits, timings=t)
