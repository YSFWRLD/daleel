"""End-to-end retrieval checks. Need Postgres (docker compose up db), Ollama with bge-m3, and loaded chunks
(python -m ingest.load). Run with: pytest -m integration"""

import pytest

pytestmark = pytest.mark.integration


def _retrieve(q, **kw):
    from app.retrieval.hybrid import retrieve

    try:
        return retrieve(q, **kw)
    except Exception as e:  # noqa: BLE001 - skip when the services are not running
        pytest.skip(f"retrieval services unavailable: {e}")


@pytest.mark.parametrize(
    "question,doc,key",
    [
        ("Can we store customer data on a server outside KSA?", "pdpl", "art-29"),
        ("هل يمكننا تخزين بيانات العملاء على خادم خارج المملكة؟", "pdpl", "art-29"),
        ("How long do we have to notify about a personal data breach?", "pdpl_ir", "art-24"),
        ("Is multi-factor authentication required for remote access?", "ecc", "ctl-2-2-3"),
    ],
)
@pytest.mark.parametrize("use_reranker", [False, True])
def test_expected_source_in_top5(question, doc, key, use_reranker):
    r = _retrieve(question, top_k=5, use_reranker=use_reranker)
    got = [(h.chunk["doc_id"], h.chunk["canonical_key"]) for h in r.hits]
    assert (doc, key) in got, got


def test_hits_carry_arabic_counterpart():
    r = _retrieve("Can we store customer data on a server outside KSA?", top_k=5, use_reranker=False)
    art29 = next(h for h in r.hits if h.chunk["canonical_key"] == "art-29")
    assert art29.counterpart is not None and art29.counterpart["lang"] == "ar"
    assert "خارج المملكة" in art29.counterpart["text"]


def test_one_hit_per_article():
    r = _retrieve("نقل البيانات الشخصية خارج المملكة", top_k=8, use_reranker=False)
    keys = [(h.chunk["doc_id"], h.chunk["canonical_key"]) for h in r.hits]
    assert len(keys) == len(set(keys))
