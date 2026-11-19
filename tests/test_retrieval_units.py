from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.hybrid import rrf_fuse


def test_tokenize_normalizes_arabic_variants():
    # alef/ta-marbuta/ya variants and diacritics must not split the vocabulary
    assert tokenize("المادة الأُولى") == tokenize("الماده الاولي")


def test_tokenize_keeps_control_ids_whole():
    assert "2-3-1" in tokenize("See control 2-3-1 of ECC")
    assert "29" in tokenize("المادة ٢٩")


def test_tokenize_drops_stop_words_and_stems_english():
    toks = tokenize("Can we store the data outside the Kingdom?")
    assert "the" not in toks and "we" not in toks
    assert "store" in toks and "kingdom" in toks


def test_bm25_finds_matching_chunk_in_either_language():
    chunks = [
        {"id": "a", "lang": "en", "heading": "PDPL Article 29", "text": "transfer personal data outside the Kingdom"},
        {"id": "b", "lang": "en", "heading": "ECC 2-9-1", "text": "backup and recovery management"},
        {"id": "c", "lang": "ar", "heading": "المادة 29", "text": "نقل البيانات الشخصية إلى خارج المملكة"},
    ]
    idx = BM25Index(chunks)
    assert idx.search("data transfer outside the Kingdom", 1)[0][0]["id"] == "a"
    assert idx.search("نقل البيانات خارج المملكة", 1)[0][0]["id"] == "c"
    assert idx.search("backup", 3, lang="en")[0][0]["id"] == "b"


def test_rrf_rewards_agreement_between_retrievers():
    scores = rrf_fuse([["x", "y", "z"], ["y", "x", "w"]])
    assert scores["x"] == scores["y"] > scores["z"] > 0
    assert scores["w"] < scores["x"]
