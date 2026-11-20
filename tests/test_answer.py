"""Answer post-processing without a model: citation checks, quote verification and abstention."""

from app.answer import ABSTAIN, answer_question, build_sources, finalize, quote_in_source
from app.retrieval.hybrid import Hit
from app.text.arabic import fix_mirrored_brackets


def chunk(lang, text, key="art-29", doc="pdpl", source="text", note=""):
    return {
        "id": f"{doc}:{lang}:{key}:0", "doc_id": doc, "lang": lang, "canonical_key": key, "text": text,
        "heading": f"{doc} — {key}", "authority": "official" if lang == "ar" else "translation",
        "source": source, "note": note,
    }  # fmt: skip


EN = chunk(
    "en", "A Controller may Transfer Personal Data outside the Kingdom if it does not prejudice national security."
)
AR = chunk("ar", "يجوز لجهة التحكم نقل البيانات الشخصية إلى خارج المملكة إذا لم يترتب على ذلك مساس بالأمن الوطني.")
HITS = {"S1": Hit(chunk=EN, score=0.9, rrf=0.03, counterpart=AR)}


def test_finalize_keeps_valid_citation_and_both_languages():
    raw = {"answerable": True, "answer": "Yes, with conditions [S1].",
           "citations": [{"source": "S1", "quote": "may Transfer Personal Data outside the Kingdom"}]}  # fmt: skip
    text, abstained, cites = finalize(raw, HITS, "en")
    assert not abstained and text == "Yes, with conditions [S1]."
    assert cites[0].quote_verified and cites[0].text_ar == AR["text"] and cites[0].text_en == EN["text"]


def test_finalize_appends_points_as_markdown_list():
    raw = {"answerable": True, "answer": "Yes [S1].",
           "points": ["- Adequate protection [S1]", " ", "Minimum data [S1]"],
           "citations": [{"source": "S1", "quote": "may Transfer Personal Data outside the Kingdom"}]}  # fmt: skip
    text, _, _ = finalize(raw, HITS, "en")
    assert text == "Yes [S1].\n\n- Adequate protection [S1]\n- Minimum data [S1]"


def test_finalize_accepts_source_cited_by_heading():
    raw = {"answerable": True, "answer": "Yes.",
           "citations": [{"source": EN["heading"],
                          "quote": "may Transfer Personal Data outside the Kingdom"}]}  # fmt: skip
    _, abstained, cites = finalize(raw, HITS, "en")
    assert not abstained and cites[0].id == "S1"


def test_finalize_uses_inline_markers_when_citations_list_is_empty():
    raw = {"answerable": True, "answer": "Yes.", "points": ["[S1] Adequate protection.", "Not this [S9]"],
           "citations": []}  # fmt: skip
    text, abstained, cites = finalize(raw, HITS, "en")
    assert not abstained and [c.id for c in cites] == ["S1"]
    assert cites[0].quote == "" and not cites[0].quote_verified and "[S9]" not in text


def test_finalize_abstains_when_model_says_unanswerable():
    raw = {"answerable": False, "answer": "Not covered.", "citations": []}
    assert finalize(raw, HITS, "ar") == (ABSTAIN["ar"], True, [])


def test_finalize_abstains_when_only_invented_sources_are_cited():
    raw = {"answerable": True, "answer": "Yes [S7].", "citations": [{"source": "S7", "quote": "anything"}]}
    text, abstained, cites = finalize(raw, HITS, "en")
    assert abstained and cites == []


def test_finalize_flags_invented_quote_and_strips_markers():
    raw = {"answerable": True, "answer": "Yes [S1].",
           "citations": [{"source": "S1", "quote": "Transfers are always free and unrestricted [S1]"}]}  # fmt: skip
    _, abstained, cites = finalize(raw, HITS, "en")
    assert not abstained and not cites[0].quote_verified and "[S1]" not in cites[0].quote


def test_quote_in_source_tolerates_formatting_not_wording():
    assert quote_in_source("may transfer personal data  outside the Kingdom", [EN["text"]])
    assert quote_in_source("نقل البيانات الشخصية إلى خارج المملكة", [AR["text"]])
    assert not quote_in_source("data may never leave the Kingdom", [EN["text"]])


def test_sources_use_question_language_and_add_english_for_ocr():
    ocr_ar = chunk("ar", "نص عربي مقروء آليا رقم (0)", key="ctl-2-4-3", doc="ecc", source="ocr")
    en = chunk("en", "Email protection control 2-4-3", key="ctl-2-4-3", doc="ecc")
    text, ids = build_sources([Hit(chunk=ocr_ar, score=1, rrf=0, counterpart=en)], "ar")
    assert "نص عربي" in text and "Email protection control 2-4-3" in text and list(ids) == ["S1"]
    text_en, _ = build_sources([Hit(chunk=en, score=1, rrf=0, counterpart=ocr_ar)], "en")
    assert "نص عربي" not in text_en


def test_answer_question_with_fake_llm(monkeypatch):
    from app.retrieval import hybrid

    class Fake:
        name = "fake"

        def complete_json(self, system, user, schema):
            assert "Answer ONLY from the numbered sources" in system and "[S1]" in user
            return {"answerable": True, "answer": "Allowed with conditions [S1].",
                    "citations": [{"source": "S1", "quote": "outside the Kingdom"}]}  # fmt: skip

    monkeypatch.setattr("app.answer.retrieve", lambda q, **kw: hybrid.Retrieval(q, "en", [HITS["S1"]], {}))
    a = answer_question("Can data leave KSA?", llm=Fake())
    assert not a.abstained and a.citations[0].canonical_key == "art-29" and a.disclaimer


def test_fix_mirrored_brackets():
    assert (
        fix_mirrored_brackets("الفقرة )2 ( من المادة )الأولى( من النظام") == "الفقرة (2) من المادة (الأولى) من النظام"
    )
    assert fix_mirrored_brackets("(أ) نص (ب) نص") == "(أ) نص (ب) نص"
    # an earlier '(' whose number the PDF lost must not swallow the later mirrored pair
    assert fix_mirrored_brackets("الفقرة (من هذه المادة. ورد في الفقرة)2 (من") == (
        "الفقرة (من هذه المادة. ورد في الفقرة (2) من"
    )
    assert fix_mirrored_brackets("في الفقرات )1 () و2 () و4 (من المادة") == "في الفقرات (1) و(2) و(4) من المادة"
