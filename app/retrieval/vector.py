"""Dense retrieval with pgvector (cosine distance, HNSW index)."""

from __future__ import annotations

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import get_settings

FIELDS = (
    "id, strategy, doc_id, lang, ref, ref_key, canonical_key, refs, authority, source, note, heading, "
    "title, parent_section, page, text"
)


def vector_search(query_vec: list[float], strategy: str, k: int) -> list[tuple[dict, float]]:
    with psycopg.connect(get_settings().database_url, row_factory=dict_row) as conn:
        register_vector(conn)
        conn.execute("SET hnsw.ef_search = 100")
        q = np.array(query_vec, dtype=np.float32)
        rows = conn.execute(
            f"SELECT {FIELDS}, 1 - (embedding <=> %s) AS sim FROM chunks "
            "WHERE strategy = %s ORDER BY embedding <=> %s LIMIT %s",
            (q, strategy, q, k),
        ).fetchall()
    return [({k_: v for k_, v in r.items() if k_ != "sim"}, float(r["sim"])) for r in rows]
