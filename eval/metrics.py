"""Evaluation metrics. Pure functions so they can be unit-tested without services.

Gold sources are a list of acceptable (doc_id, canonical_key) pairs: a question about breach notification
is correctly answered from PDPL Art. 20 or from Implementing Regulation Art. 24. A retrieved chunk or a
citation "matches" when its document is the same and the gold key is among the keys the chunk covers
(fixed-size chunks can span several articles).
"""

from __future__ import annotations

import math

Key = tuple[str, str]


def matches(doc_id: str, keys: list[str], gold: list[dict]) -> bool:
    return any(g["doc_id"] == doc_id and g["canonical_key"] in keys for g in gold)


def first_rank(ranked: list[tuple[str, list[str]]], gold: list[dict]) -> int | None:
    """1-based rank of the first retrieved item matching any gold source, or None."""
    for i, (doc_id, keys) in enumerate(ranked, start=1):
        if matches(doc_id, keys, gold):
            return i
    return None


def retrieval_scores(rank: int | None, ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    out = {f"hit@{k}": float(rank is not None and rank <= k) for k in ks}
    out["rr"] = 1.0 / rank if rank else 0.0  # averaged over questions -> MRR
    return out


def family(key: str) -> str:
    """Controls belong to a sub-domain: ctl-2-9-3, ctl-2-9-1 and sub-2-9 -> "2-9"; CCC ctl-1-5-P-3 -> "1-5".
    Articles stand alone."""
    parts = key.split("-")
    return "-".join(parts[1:3]) if parts[0] in ("ctl", "sub") and len(parts) >= 3 else key


def matches_family(doc_id: str, keys: list[str], gold: list[dict]) -> bool:
    fams = {family(k) for k in keys}
    return any(g["doc_id"] == doc_id and family(g["canonical_key"]) in fams for g in gold)


def citation_scores(cited: list[tuple[str, list[str]]], gold: list[dict]) -> dict[str, float | None]:
    """precision: share of citations that point at a gold source.
    precision_lenient: same, but a sibling control of the same sub-domain also counts (for "what are the
    requirements for X" questions the model often cites the whole sub-domain, which is reasonable).
    hit: at least one citation points at a gold source.
    recall: share of gold sources that were cited (gold lists alternatives, so this is strict)."""
    if not cited:
        return {"cite_precision": None, "cite_precision_lenient": None, "cite_hit": 0.0, "cite_recall": 0.0}
    good = [matches(d, k, gold) for d, k in cited]
    near = [matches_family(d, k, gold) for d, k in cited]
    covered = [g for g in gold if any(d == g["doc_id"] and g["canonical_key"] in k for d, k in cited)]
    return {
        "cite_precision": sum(good) / len(good),
        "cite_precision_lenient": sum(near) / len(near),
        "cite_hit": float(any(good)),
        "cite_recall": len(covered) / len(gold) if gold else 0.0,
    }


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (no interpolation): p50/p95 of small samples stay real observations."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    rank = max(1, math.ceil(p / 100 * len(vals)))
    return vals[rank - 1]


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
