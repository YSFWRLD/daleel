"""Bilingual BM25 over the chunk files (in memory: ~900 chunks, builds in well under a second)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import snowballstemmer
from rank_bm25 import BM25Okapi

from app.text import arabic

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

_AR_STEM = snowballstemmer.stemmer("arabic")
_EN_STEM = snowballstemmer.stemmer("english")
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[ء-ي]+")
_STOP = {
    # English
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "on", "by", "with", "be", "is", "are", "shall",
    "as", "at", "that", "this", "it", "its", "from", "any", "all", "can", "we", "our", "do", "does", "which",
    # Arabic (normalized forms)
    "في", "من", "علي", "الي", "عن", "ان", "او", "و", "ما", "التي", "الذي", "هذا", "هذه", "ذلك", "مع", "كل",
    "هل", "يجب", "لا", "اي",
}  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Normalize (alef/ya/ta-marbuta, diacritics, digits), drop stop words, light stemming.
    Control IDs like 2-3-1 stay whole so exact references score."""
    out = []
    for tok in _TOKEN.findall(arabic.normalize_for_search(text)):
        if tok in _STOP or len(tok) < 2 and not tok.isdigit():
            continue
        if tok[0].isascii():
            out.append(tok if any(ch.isdigit() for ch in tok) else _EN_STEM.stemWord(tok))
        else:
            out.append(_AR_STEM.stemWord(tok))
    return out


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(f"{c['heading']} {c['text']}") for c in chunks])

    def search(self, query: str, k: int, lang: str | None = None) -> list[tuple[dict, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if lang and self.chunks[i]["lang"] != lang:
                continue
            out.append((self.chunks[i], float(scores[i])))
            if len(out) == k:
                break
        return out


@lru_cache
def get_index(strategy: str) -> BM25Index:
    path = PROCESSED / f"chunks_{strategy}.jsonl"
    return BM25Index([json.loads(line) for line in path.open(encoding="utf-8")])
