from collections import Counter

import pytest

from app.text import arabic


@pytest.mark.parametrize(
    "words,expected",
    [
        (["الأولى"], 1),
        (["العاشرة"], 10),
        (["الحادية", "عشرة"], 11),
        (["السادسة", "عشرة:"], 16),
        (["العشرون"], 20),
        (["الحادية", "والعشرون"], 21),
        (["التاسعة", "والعشرون"], 29),
        (["الثالثة", "والأربعون"], 43),
        (["(السابعة)"], 7),
    ],
)
def test_parse_ordinal(words, expected):
    assert arabic.parse_ordinal(words)[0] == expected


def test_parse_ordinal_rejects_non_ordinals():
    assert arabic.parse_ordinal(["من", "النظام"]) is None


@pytest.mark.parametrize("n", [1, 9, 10, 11, 16, 20, 21, 29, 30, 38, 43])
def test_ordinal_words_roundtrip(n):
    assert arabic.parse_ordinal(arabic.ordinal_words(n))[0] == n


def test_fix_lam_alef():
    assert arabic.fix_lam_alef("األمن السيبراني") == "الأمن السيبراني"
    assert arabic.fix_lam_alef("الضوابط األساسية لألمن") == "الضوابط الأساسية للأمن"
    assert arabic.fix_lam_alef("وال يجوز إال بعد") == "ولا يجوز إلا بعد"
    # documents without the bug only get the always-safe repair; correct words stay intact
    assert arabic.fix_lam_alef("بالألفاظ والعبارات", aggressive=False) == "بالألفاظ والعبارات"
    assert arabic.fix_lam_alef("األمن بالألفاظ", aggressive=False) == "الأمن بالألفاظ"


def test_lam_alef_bug_count():
    assert arabic.lam_alef_bug_count("األمن األساسية") == 2
    assert arabic.lam_alef_bug_count("الأمن الأساسية") == 0


def test_fix_reversed_line():
    vocab = Counter({w: 5 for w in ["نظام", "حماية", "البيانات", "الشخصية", "الصادر", "بالمرسوم", "الملكي"]})
    reversed_line = "نظام حماية البيانات الشخصية"[::-1]
    assert arabic.fix_reversed_line(reversed_line, vocab) == "نظام حماية البيانات الشخصية"
    assert arabic.fix_reversed_line("نظام حماية البيانات", vocab) == "نظام حماية البيانات"


def test_clean_unicode_folds_presentation_forms():
    # U+FEE3 U+FED8 U+FEF4 U+FEAA are presentation forms of م ق ي د
    assert arabic.clean_unicode("ﻣﻘﻴﺪ") == "مقيد"
    assert arabic.clean_unicode("ماــــظن") == "ماظن"


def test_normalize_for_search():
    assert arabic.normalize_for_search("الأُولى") == "الاولي"
    assert arabic.normalize_for_search("المادة ٢٩") == "الماده 29"


def test_detect_lang():
    assert arabic.detect_lang("هل يمكن تخزين بيانات العملاء خارج المملكة؟") == "ar"
    assert arabic.detect_lang("Can we store customer data outside KSA?") == "en"


def test_join_split_suffix():
    assert arabic.join_split_suffix("من وقت علم ها بالحادثة") == "من وقت علمها بالحادثة"
    assert arabic.join_split_suffix("أغراض معالجت ها.") == "أغراض معالجتها."
    assert arabic.join_split_suffix("ها هو النص") == "ها هو النص"  # demonstrative at the start
    assert arabic.join_split_suffix("قال ها هو النص") == "قال ها هو النص"  # "ها هو" is kept
    assert arabic.join_split_suffix("جهاتها المختصة") == "جهاتها المختصة"
