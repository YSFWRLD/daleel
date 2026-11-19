"""Regression checks on the committed chunk files (data/processed), so CI catches a broken rebuild."""

import json
from collections import defaultdict
from pathlib import Path

import pytest

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def load(strategy):
    path = PROCESSED / f"chunks_{strategy}.jsonl"
    if not path.exists():
        pytest.skip("chunks not built yet (python -m ingest.build)")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def refs(chunks, key="canonical_key"):
    out = defaultdict(set)
    for c in chunks:
        out[(c["doc_id"], c["lang"])].add(c[key])
    return out


@pytest.mark.parametrize("doc,n", [("pdpl", 43), ("pdpl_ir", 38), ("pdpl_transfer", 9)])
@pytest.mark.parametrize("lang", ["en", "ar"])
def test_every_article_present(doc, n, lang):
    got = refs(load("article"))[(doc, lang)]
    assert got == {f"art-{i}" for i in range(1, n + 1)}


@pytest.mark.parametrize("doc", ["ecc", "ccc", "dcc"])
def test_controls_mostly_aligned_across_languages(doc):
    r = refs(load("article"))
    en, ar = r[(doc, "en")], r[(doc, "ar")]
    assert len(en) > 25
    # in canonical (Arabic) numbering the only gaps are the controls absent from the English ECC
    allowed = {("ecc", "ctl-1-7-1"), ("ecc", "ctl-2-7-3")}
    assert {(doc, k) for k in en ^ ar} <= allowed, sorted(en ^ ar)


def test_ids_unique_and_text_present():
    for strategy in ("article", "fixed"):
        chunks = load(strategy)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))
        assert all(c["text"].strip() for c in chunks)


def test_key_transfer_article_text():
    chunks = {c["id"]: c for c in load("article")}
    assert "outside the Kingdom" in chunks["pdpl:en:art-29:0"]["text"]
    assert "خارج المملكة" in chunks["pdpl:ar:art-29:0"]["text"]


def test_ocr_chunks_are_tagged():
    chunks = load("article")
    by_id = {c["id"]: c for c in chunks}
    assert by_id["dcc:en:ctl-2-6-1:0"]["source"] == "ocr"  # whole English DCC is OCR
    assert by_id["ecc:ar:ctl-2-4-3:0"]["source"] == "ocr"  # graphics page in Arabic ECC
    assert by_id["ecc:ar:ctl-2-1-1:0"]["source"] == "text"


def test_version_differences_are_noted():
    by_id = {c["id"]: c for c in load("article")}
    assert "1-7-2" in by_id["ecc:en:ctl-1-7-1:0"]["note"]
    assert by_id["ecc:ar:ctl-2-7-3:0"]["note"]
    assert not by_id["ecc:en:ctl-2-1-1:0"]["note"]


def test_no_lam_alef_extraction_bug_left():
    bad = [c["id"] for c in load("article") if c["lang"] == "ar" and "األ" in c["text"]]
    assert not bad, bad[:5]


def test_authority_and_canonical_keys():
    by_id = {c["id"]: c for c in load("article")}
    assert by_id["ecc:ar:ctl-1-7-1:0"]["authority"] == "binding"
    assert by_id["ecc:en:ctl-1-7-1:0"]["authority"] == "translation"
    assert by_id["pdpl:ar:art-29:0"]["authority"] == "official"
    # English 1-7-1 is the same control as Arabic 1-7-2
    assert by_id["ecc:en:ctl-1-7-1:0"]["canonical_key"] == "ctl-1-7-2"
    assert by_id["ecc:en:ctl-2-7-3:0"]["canonical_key"] == "ctl-2-7-4"
    assert by_id["ecc:ar:ctl-1-7-1:0"]["canonical_key"] == "ctl-1-7-1"


def test_no_control_swallows_appendices():
    long = [c["id"] for c in load("article") if len(c["text"].split()) > 400]
    assert not long, long
    assert "Appendix" not in {c["id"]: c for c in load("article")}["ecc:en:ctl-4-2-4:0"]["text"]
