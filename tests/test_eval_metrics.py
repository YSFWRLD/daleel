import json
from pathlib import Path

from eval.metrics import citation_scores, family, first_rank, percentile, retrieval_scores

GOLD = [{"doc_id": "pdpl_ir", "canonical_key": "art-24"}, {"doc_id": "pdpl", "canonical_key": "art-20"}]


def test_first_rank_uses_any_acceptable_source_and_spanned_keys():
    ranked = [("pdpl", ["art-19"]), ("pdpl", ["art-19", "art-20"]), ("pdpl_ir", ["art-24"])]
    assert first_rank(ranked, GOLD) == 2  # a fixed-size chunk spanning art-19..20 counts
    assert first_rank([("ecc", ["art-24"])], GOLD) is None  # right key, wrong document


def test_retrieval_scores():
    assert retrieval_scores(2) == {"hit@1": 0.0, "hit@3": 1.0, "hit@5": 1.0, "rr": 0.5}
    assert retrieval_scores(None)["rr"] == 0.0


def test_citation_scores_strict_and_lenient():
    gold = [{"doc_id": "ccc", "canonical_key": "ctl-1-5-P-3"}]
    cited = [("ccc", ["ctl-1-5-P-3"]), ("ccc", ["ctl-1-5-P-1"]), ("ecc", ["ctl-1-5-3"])]
    s = citation_scores(cited, gold)
    assert s["cite_precision"] == 1 / 3 and s["cite_precision_lenient"] == 2 / 3
    assert s["cite_hit"] == 1.0 and s["cite_recall"] == 1.0
    assert citation_scores([], gold)["cite_precision"] is None


def test_family():
    assert family("ctl-2-9-3") == family("sub-2-9") == family("ctl-2-9-3-1") == "2-9"
    assert family("ctl-1-5-P-3") == "1-5" and family("art-29") == "art-29"


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 100], 50) == 3 and percentile([1, 2, 3, 4, 100], 95) == 100
    assert percentile([], 50) is None


def test_question_set_is_well_formed():
    qs = [json.loads(line) for line in Path("eval/questions.jsonl").open(encoding="utf-8")]
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["lang"] in ("en", "ar") and q["question"].strip()
        assert bool(q["gold"]) == q["answerable"], q["id"]
