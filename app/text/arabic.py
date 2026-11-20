"""Arabic text utilities: extraction repair, search normalization, ordinal parsing."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

AR_RANGE = re.compile("[" + chr(0x0600) + "-" + chr(0x06FF) + "]")
AR_WORD = re.compile("[" + chr(0x0621) + "-" + chr(0x064A) + "]+")
DIACRITICS = re.compile(
    "["
    + chr(0x0610)
    + "-"
    + chr(0x061A)
    + chr(0x064B)
    + "-"
    + chr(0x065F)
    + chr(0x0670)
    + chr(0x06D6)
    + "-"
    + chr(0x06ED)
    + "]"
)
TATWEEL = chr(0x0640)
# zero-width and bidi control characters
ZERO_WIDTH = re.compile(
    "["
    + chr(0x200B)
    + "-"
    + chr(0x200F)
    + chr(0x202A)
    + "-"
    + chr(0x202E)
    + chr(0x2066)
    + "-"
    + chr(0x2069)
    + chr(0xFEFF)
    + "]"
)
# Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digits -> ASCII
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Junk glyphs some SDAIA PDFs emit for list numbers (font without a ToUnicode map)
JUNK_GLYPHS = re.compile(r"[!\"#$%&']+")
BRACKET_SWAP = str.maketrans("()[]{}<>", ")(][}{><")


def has_arabic(text: str) -> bool:
    return bool(AR_RANGE.search(text))


def detect_lang(text: str) -> str:
    """'ar' if the text has more Arabic letters than Latin letters, else 'en'."""
    ar = len(AR_RANGE.findall(text))
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    return "ar" if ar > lat else "en"


def to_ascii_digits(text: str) -> str:
    return text.translate(DIGITS)


def clean_unicode(text: str) -> str:
    """NFKC folds Arabic presentation forms (e.g. U+FEE3) into base letters; drop tatweel and bidi marks."""
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH.sub("", text)
    return text.replace(TATWEEL, "")


# --- lam-alef ligature repair -------------------------------------------------------------------
# Several NCA PDFs map the lam-alef ligature to the two letters in the wrong order:
#   "الأمن" -> "األمن"  (alef, hamza-alef, lam)     "للأمن" -> "لألمن"     "إلا" -> "إال"
_LA_AFTER_ALEF = re.compile(r"ا([أإآ])ل")
_LA_AFTER_LAM = re.compile(r"ل([أإآ])ل")
_LA_STANDALONE = {"ال": "لا", "وال": "ولا", "فال": "فلا", "إال": "إلا", "أال": "ألا", "بال": "بلا"}


def lam_alef_bug_count(text: str) -> int:
    """Occurrences of a pattern that never appears in correct Arabic ("األ")."""
    return len(re.findall(r"ا[أإآ]ل", text))


def fix_lam_alef(text: str, aggressive: bool = True) -> str:
    """'األ' never occurs in correct Arabic, so that repair is always safe. The other two rules
    ('لأل' -> 'للأ', standalone 'إال' -> 'إلا') would damage correct words such as 'بالألفاظ',
    so they only run on documents that show the bug (aggressive=True)."""
    text = _LA_AFTER_ALEF.sub(r"ال\1", text)
    if not aggressive:
        return text
    text = _LA_AFTER_LAM.sub(r"لل\1", text)
    return re.sub(
        "(?<!" + AR_WORD.pattern[:-1] + ")(" + "|".join(_LA_STANDALONE) + ")(?!" + AR_WORD.pattern[:-1] + ")",
        lambda m: _LA_STANDALONE[m.group(1)],
        text,
    )


def fix_mirrored_brackets(text: str) -> str:
    """Some Arabic PDFs store brackets mirrored: "الفقرة )2 (" instead of "الفقرة (2)". A ')' with no open
    '(' before it, followed within 40 characters by a '(' (no other bracket between), is a mirrored pair:
    swap it. Correct text such as "(أ) نص (ب)" never has an unopened ')' so it is left alone.
    A number in reversed brackets between Arabic words (")2 (") is fixed first: the scanner alone misses it
    when an earlier '(' was left unclosed because the PDF lost its number."""
    ar = "؀-ۿ"
    text = re.sub(rf"(?<=[{ar}])\s*\)\s*([0-9٠-٩]{{1,3}})\s*\(\s*(?=[{ar}])", r" (\1) ", text)
    # scrambled lists of paragraph numbers: "الفقرات )1 () و2 () و4 (من" -> "الفقرات (1) و(2) و(4) من"
    text = re.sub(
        r"(الفقرات|الفقرتين)\s*([()\s0-9و]*\d[()\s0-9و]*?)\s*(?=من)",
        lambda m: m[1] + " " + " و".join(f"({n})" for n in re.findall(r"\d+", m[2])) + " "
        if m[2].count("(") + m[2].count(")") else m[0],
        text,
    )
    chars = list(text)
    depth, i = 0, 0
    while i < len(chars):
        ch = chars[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth > 0:
                depth -= 1
            else:
                j = next((k for k in range(i + 1, min(len(chars), i + 41)) if chars[k] in "()"), None)
                if j is not None and chars[j] == "(":
                    chars[i], chars[j] = "(", ")"
                    i = j
        i += 1
    out = "".join(chars)
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", out)) if out != text else text


# --- reversed-line repair ------------------------------------------------------------------------
def build_vocab(texts: list[str]) -> Counter:
    vocab: Counter = Counter()
    for t in texts:
        vocab.update(AR_WORD.findall(t))
    return vocab


def _vocab_score(line: str, vocab: Counter) -> float:
    words = AR_WORD.findall(line)
    if not words:
        return 0.0
    return sum(1 for w in words if vocab.get(w, 0) >= 3) / len(words)


def fix_reversed_line(line: str, vocab: Counter) -> str:
    """Some lines come out in visual order (characters reversed). Keep whichever reading is real Arabic."""
    if len(AR_WORD.findall(line)) < 2:
        return line
    rev = line[::-1].translate(BRACKET_SWAP)
    s_orig, s_rev = _vocab_score(line, vocab), _vocab_score(rev, vocab)
    return rev if s_rev >= 0.6 and s_rev > s_orig + 0.3 else line


# --- search normalization ------------------------------------------------------------------------
def normalize_for_search(text: str) -> str:
    """Aggressive normalization used for BM25 and matching (never for display)."""
    text = clean_unicode(text)
    text = DIACRITICS.sub("", text)
    text = re.sub("[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return to_ascii_digits(text).lower()


# --- ordinal numbers ("المادة الحادية والعشرون" -> 21) ----------------------------------------
_UNITS = {
    "الأولى": 1,
    "الأول": 1,
    "الحادية": 1,
    "الحادي": 1,
    "الثانية": 2,
    "الثاني": 2,
    "الثالثة": 3,
    "الثالث": 3,
    "الرابعة": 4,
    "الرابع": 4,
    "الخامسة": 5,
    "الخامس": 5,
    "السادسة": 6,
    "السادس": 6,
    "السابعة": 7,
    "السابع": 7,
    "الثامنة": 8,
    "الثامن": 8,
    "التاسعة": 9,
    "التاسع": 9,
    "العاشرة": 10,
    "العاشر": 10,
}
_TENS = {
    "العشرون": 20,
    "العشرين": 20,
    "الثلاثون": 30,
    "الثلاثين": 30,
    "الأربعون": 40,
    "الأربعين": 40,
    "الخمسون": 50,
    "الخمسين": 50,
}
_UNITS_N = {normalize_for_search(k): v for k, v in _UNITS.items()}
_TENS_N = {normalize_for_search(k): v for k, v in _TENS.items()}
_TEEN = {"عشره", "عشر"}


_FEM_UNITS = ["الأولى", "الثانية", "الثالثة", "الرابعة", "الخامسة", "السادسة", "السابعة", "الثامنة", "التاسعة"]
_FEM_TENS = {2: "العشرون", 3: "الثلاثون", 4: "الأربعون", 5: "الخمسون"}


def ordinal_words(n: int) -> list[str]:
    """Feminine ordinal as used in article headings: 16 -> ['السادسة', 'عشرة'] (normalized)."""
    if n < 10:
        words = [_FEM_UNITS[n - 1]]
    elif n == 10:
        words = ["العاشرة"]
    elif n < 20:
        words = ["الحادية" if n == 11 else _FEM_UNITS[n - 11], "عشرة"]
    elif n % 10 == 0:
        words = [_FEM_TENS[n // 10]]
    else:
        unit = "الحادية" if n % 10 == 1 else _FEM_UNITS[n % 10 - 1]
        words = [unit, "و" + _FEM_TENS[n // 10]]
    return [normalize_for_search(w) for w in words]


def parse_ordinal(words: list[str]) -> tuple[int, int] | None:
    """Parse an Arabic ordinal at the start of `words`. Returns (value, words_consumed)."""
    w = [normalize_for_search(x).strip("()[]:،.") for x in words[:3]]
    if not w or not w[0]:
        return None
    if w[0] in _TENS_N:
        return _TENS_N[w[0]], 1
    if w[0] not in _UNITS_N:
        return None
    unit = _UNITS_N[w[0]]
    if len(w) > 1 and w[1] in _TEEN and unit < 10:
        return 10 + unit, 2
    if len(w) > 1 and w[1].startswith("و") and w[1][1:] in _TENS_N and unit < 10:
        return _TENS_N[w[1][1:]] + unit, 2
    return unit, 1
