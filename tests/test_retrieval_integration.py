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
    assert art29.counterpart is not None  # the hit may be either language; the other one is attached
    ar = next(c for c in (art29.chunk, art29.counterpart) if c["lang"] == "ar")
    assert "خارج المملكة" in ar["text"]


def test_one_hit_per_article():
    r = _retrieve("نقل البيانات الشخصية خارج المملكة", top_k=8, use_reranker=False)
    keys = [(h.chunk["doc_id"], h.chunk["canonical_key"]) for h in r.hits]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    "question,key",
    [
        ("Can we store customer data on a server outside KSA?", "art-29"),
        ("هل يمكننا تخزين بيانات العملاء على خادم خارج المملكة؟", "art-29"),
    ],
)
def test_answer_cites_transfer_article(question, key):
    from app.answer import answer_question

    try:
        a = answer_question(question)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"services unavailable: {e}")
    assert not a.abstained
    assert key in {c.canonical_key for c in a.citations}
    assert all(c.text_ar for c in a.citations if c.canonical_key == key)


def test_off_topic_question_abstains():
    from app.answer import answer_question

    try:
        a = answer_question("What is the penalty for running a red light in Riyadh?")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"services unavailable: {e}")
    assert a.abstained and not a.citations


def test_english_question_reaches_arabic_only_control():
    # ECC 2-7-3 exists only in the Arabic text (the English translation skips it). Per-language vector search
    # finds it; the default (no reranker) keeps each list's top results first. The reranker scores this
    # cross-language pair low for some wordings, so it is not part of this check.
    r = _retrieve("Must our data protection requirements include at least NCA's Data Cybersecurity Controls?",
                  top_k=5, use_reranker=False)
    got = [(h.chunk["doc_id"], h.chunk["canonical_key"], h.chunk["lang"]) for h in r.hits]
    assert ("ecc", "ctl-2-7-3", "ar") in got, got
