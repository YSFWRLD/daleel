CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk. bge-m3 produces 1024-dim embeddings.
CREATE TABLE IF NOT EXISTS chunks (
    id             TEXT PRIMARY KEY,          -- e.g. pdpl:en:art-29:0
    strategy       TEXT NOT NULL,             -- article | fixed
    doc_id         TEXT NOT NULL,             -- pdpl | pdpl_ir | ecc | ...
    lang           TEXT NOT NULL,             -- ar | en
    ref            TEXT,                      -- "Article 29" / control "2-3-1"
    ref_key        TEXT,                      -- language-independent key: art-29 / ctl-2-3-1
    title          TEXT,
    parent_section TEXT,
    page           INT,
    text           TEXT NOT NULL,
    embedding      vector(1024)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks (strategy);
CREATE INDEX IF NOT EXISTS chunks_refkey_idx ON chunks (doc_id, ref_key);
