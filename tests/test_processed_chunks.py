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


def refs(chunks):
    out = defaultdict(set)
    for c in chunks:
        out[(c["doc_id"], c["lang"])].add(c["ref_key"])
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
    assert len(en ^ ar) <= 3, sorted(en ^ ar)


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
