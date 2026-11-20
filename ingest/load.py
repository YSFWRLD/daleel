"""Embed the chunk files and load them into Postgres/pgvector.

python -m ingest.load                 # both strategies
python -m ingest.load --strategy article
python -m ingest.load --cache .cache/embeddings   # reuse vectors when the chunks and model are unchanged (CI)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings
from app.retrieval.embed import chunk_embedding_text, embed

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
COLUMNS = [
    "id", "strategy", "doc_id", "lang", "ref", "ref_key", "canonical_key", "refs", "authority",
    "source", "note", "heading", "title", "parent_section", "page", "text", "embedding",
]  # fmt: skip


def embed_cached(texts: list[str], cache: Path | None, name: str) -> list:
    """Embed, or reuse vectors saved for exactly these texts and this embedding model."""
    if cache is None:
        return embed(texts)
    digest = hashlib.sha256(json.dumps([get_settings().embed_model, *texts]).encode("utf-8")).hexdigest()[:16]
    path = cache / f"{name}-{digest}.npy"
    if path.exists():
        print(f"{name}: reusing cached embeddings {path.name}")
        return list(np.load(path))
    vectors = embed(texts)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(vectors, dtype=np.float32))
    return vectors


def load(strategy: str, cache: Path | None = None) -> int:
    chunks = [json.loads(line) for line in (PROCESSED / f"chunks_{strategy}.jsonl").open(encoding="utf-8")]
    t0 = time.time()
    vectors = embed_cached([chunk_embedding_text(c) for c in chunks], cache, strategy)
    print(f"{strategy}: embedded {len(chunks)} chunks in {time.time() - t0:.1f}s")
    rows = [
        (
            c["id"],
            c["strategy"],
            c["doc_id"],
            c["lang"],
            c["ref"],
            c["ref_key"],
            c["canonical_key"],
            c["canonical_refs"],
            c["authority"],
            c["source"],
            c["note"],
            c["heading"],
            c["title"],
            c["parent_section"],
            c["page"],
            c["text"],
            v,
        )  # fmt: skip
        for c, v in zip(chunks, vectors, strict=True)
    ]
    with psycopg.connect(get_settings().database_url) as conn:
        register_vector(conn)
        conn.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO chunks ({', '.join(COLUMNS)}) VALUES ({', '.join(['%s'] * len(COLUMNS))})",
                [(*r[:-1], np.array(r[-1], dtype=np.float32)) for r in rows],
            )
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["article", "fixed"], action="append")
    ap.add_argument("--cache", type=Path, help="directory for reusable embedding files")
    args = ap.parse_args()
    for s in args.strategy or ["article", "fixed"]:
        print(f"{s}: loaded {load(s, args.cache)} rows")


if __name__ == "__main__":
    main()
