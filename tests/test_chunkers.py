from ingest.chunkers import (
    fixed_chunks,
    norm_id,
    parse_ar_heading,
    parse_articles,
    parse_controls,
    structural_chunks,
)
from ingest.extract import Line
from ingest.grid_ocr import clean_ocr_arabic


def L(*texts, page=1):
    return [Line(t, page) for t in texts]


# --- Arabic article headings -------------------------------------------------------------------
def test_ar_heading_forward():
    assert parse_ar_heading("المادة الأولى:") == (1, "")
    assert parse_ar_heading("المادة السادسة: العدول عن الموافقة") == (6, "العدول عن الموافقة")


def test_ar_heading_word_reversed():
    assert parse_ar_heading("التعريفات :الأولى المادة") == (1, "التعريفات")
    assert parse_ar_heading("الموافقة :عشرة الحادية المادة") == (11, "الموافقة")
    # title wraps on both sides of the heading word
    assert parse_ar_heading("البيانات جمع :عشرة الخامسة المادة مباشرة صاحبها غير من") == (
        15,
        "جمع البيانات من غير صاحبها مباشرة",
    )


def test_ar_heading_rejects_cross_references():
    assert parse_ar_heading("المادة )السابعة( من النظام.") is None
    assert parse_ar_heading("المادة الخامسة من النظام") is None


# --- article parsing -------------------------------------------------------------------------------
def test_parse_articles_en_sequential_only():
    lines = L(
        "Personal Data Protection Law",
        "Article 1",
        "Definitions text.",
        "as set out in",
        "Article 5 of this Law.",  # cross-reference, out of sequence
        "Article 2",
        "Scope text.",
    )
    units = parse_articles(lines, "en")
    assert [u.ref_key for u in units] == ["art-1", "art-2"]
    assert "Article 5 of this Law." in units[0].text


def test_parse_articles_ar_fragmented_heading():
    # heading for article 2 split over two scrambled lines (as in the Implementing Regulation PDF)
    lines = L(
        "المادة الأولى: التعريفات",
        "نص المادة الأولى.",
        "الثانية",
        ":المشروعة المصلحة لأغراض المعالجة",
        "نص المادة الثانية.",
    )
    units = parse_articles(lines, "ar")
    assert [u.ref_key for u in units] == ["art-1", "art-2"]
    assert units[1].title == "المعالجة لأغراض المصلحة المشروعة"
    assert units[1].text == "نص المادة الثانية."


# --- controls --------------------------------------------------------------------------------------
def test_norm_id():
    assert norm_id("٢-٦-م-١-٤") == "2-6-P-1-4"
    assert norm_id("2.2.3.1") == "2-2-3-1"
    assert norm_id("1-1-ش-1") == "1-1-T-1"


def test_parse_controls_before_mode():
    lines = L(
        "2-1",
        "Asset Management",
        "Objective",
        "To ensure inventory.",
        "Controls",
        "2-1-1",
        "Requirements shall be identified.",
        "2-1-2 Requirements shall cover:",
        "2-1-2-1 First item.",
        "2-1-2-2 Second item.",
    )
    units = {u.ref_key: u for u in parse_controls(lines, "en", "before")}
    assert units["sub-2-1"].title == "Asset Management"
    assert "To ensure inventory." in units["sub-2-1"].text
    assert units["ctl-2-1-1"].text == "Requirements shall be identified."
    assert "2-1-2-1 First item." in units["ctl-2-1-2"].text
    assert units["ctl-2-1-2"].parent_section == "2-1 Asset Management"


def test_parse_controls_after_mode():
    # Arabic ECC stream order: text first, then its ID
    lines = L(
        "إدارة الأصول",
        "2-1",
        "للتأكد من دقة الجرد.",
        "الهدف",
        "الضوابط",
        "يجب تحديد وتوثيق واعتماد المتطلبات.",
        "2-1-1",
        "يجب تطبيق المتطلبات.",
        "2-1-2",
    )
    units = {u.ref_key: u for u in parse_controls(lines, "ar", "after")}
    assert units["sub-2-1"].title == "إدارة الأصول"
    assert units["ctl-2-1-1"].text == "يجب تحديد وتوثيق واعتماد المتطلبات."
    assert units["ctl-2-1-2"].text == "يجب تطبيق المتطلبات."


def test_parse_controls_recovers_sentence_of_missing_control_id():
    # OCR lost the "2-6-1" ID; its opening sentence must not stay in the objective
    lines = L(
        "2-6",
        "Secure Data Disposal",
        "Objective",
        "To ensure a secure data disposal.",
        "Cybersecurity requirements for secure data disposal must cover at least the following:",
        "2-6-1-1 Identification of technologies.",
        "2-6-2 Review requirements.",
    )
    units = {u.ref_key: u for u in parse_controls(lines, "en", "before")}
    assert units["sub-2-6"].text == "Objective: To ensure a secure data disposal."
    assert units["ctl-2-6-1"].text.startswith("Cybersecurity requirements for secure data disposal")


def test_parse_controls_out_of_order_subdomain():
    lines = L("2-7 A", "2-7-1 a", "2-9 C", "2-9-1 c", "2-8 B", "2-8-1 b", "2-10 D", "2-10-1 d")
    keys = {u.ref_key for u in parse_controls(lines, "en", "before")}
    assert {"ctl-2-7-1", "ctl-2-8-1", "ctl-2-9-1", "ctl-2-10-1"} <= keys


def test_parse_controls_stops_at_appendix():
    lines = L("1-1 Gov", "1-1-1 real control", "2-1 Def", "2-1-1 real", "1-1-1", "appendix row")
    units = {u.ref_key: u for u in parse_controls(lines, "en", "before")}
    assert "appendix row" not in units["ctl-2-1-1"].text


# --- chunk builders -------------------------------------------------------------------------------
def test_structural_chunks_split_long_units():
    lines = L("Article 1", " ".join(["word"] * 900))
    chunks = structural_chunks(parse_articles(lines, "en"), "pdpl", "PDPL", "en")
    assert len(chunks) > 1
    assert {c.ref_key for c in chunks} == {"art-1"}
    assert all(c.heading.startswith("PDPL — Article 1") for c in chunks)


def test_fixed_chunks_track_refs():
    lines = L("Article 1", " ".join(["a"] * 280), "Article 2", " ".join(["b"] * 280))
    chunks = fixed_chunks(parse_articles(lines, "en"), "pdpl", "PDPL", "en")
    assert chunks[0].refs == ["art-1", "art-2"]
    assert all(c.strategy == "fixed" for c in chunks)


def test_clean_ocr_arabic():
    assert clean_ocr_arabic("يجب أن تغطي المتطلبات بحد Gol ما tah") == "يجب أن تغطي المتطلبات بحد أدنى ما يلي:"
    assert "(Domain Keys Identified Mail)" in clean_ocr_arabic("إطار (Domain Keys Identified Mail) آمن")


def test_parse_controls_stops_at_appendices_heading():
    lines = L("4-2 Cloud", "4-2-1 real control text.", "Appendices", "Appendix (A): Terms and Definitions", "Term text")
    units = {u.ref_key: u for u in parse_controls(lines, "en", "before")}
    assert units["ctl-4-2-1"].text == "real control text."


def test_parse_controls_joins_id_split_by_stray_space():
    # CCC Arabic writes 2-3-P-1-10 as "2-٣-م-1-1 0"
    lines = L("2-3 Sub", "2-3-P-1 parent:", "2-3-P-1-1 first", "2-3-P-1-9 ninth", "2-٣-م-1-1 0 tenth item")
    units = {u.ref_key: u for u in parse_controls(lines, "ar", "before")}
    assert "2-3-P-1-10 tenth item" in units["ctl-2-3-P-1"].text
