"""Grounded answers: retrieve, ask the LLM to answer only from numbered sources, then verify citations.

Rules (docs/decisions/001-arabic-is-binding.md):
- every answer must cite at least one retrieved source, or the assistant says it doesn't know
- citations always carry the Arabic text (authoritative) and, when available, the English translation
- quotes are checked against the source text; unverifiable quotes are flagged, not shown as fact
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import asdict, dataclass, field

from app.llm.base import LLM, get_llm
from app.policy import DISCLAIMER
from app.retrieval.hybrid import Hit, chunk_keys, retrieve
from app.text import arabic

PROMPT_VERSION = "v2"  # v1: answerable first, no scenario/out-of-scope rules

# Size limits stop a known failure of small models under constrained decoding: repeating the same list item
# until the request times out (seen on 8 of 276 eval calls before the limits were added).
# "answerable" comes last: the model decides after writing the answer and finding its quotes. Deciding
# first made it refuse questions that describe a situation ("we do X, is that compliant?") - see
# eval/results.md, prompt v1 vs v2.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "maxLength": 600},
        "points": {"type": "array", "items": {"type": "string", "maxLength": 300}, "maxItems": 6},
        "citations": {
            "type": "array",
            "minItems": 1,  # the model sometimes cited [S1] inline but returned no citations
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "quote": {"type": "string", "maxLength": 300}},
                "required": ["source", "quote"],
            },
        },
        "answerable": {"type": "boolean"},
    },
    "required": ["answer", "points", "citations", "answerable"],
}

SYSTEM = """You are a compliance assistant for Saudi regulations (PDPL and its regulations, NCA cybersecurity controls).
Answer ONLY from the numbered sources in the user message. Never use outside knowledge.

Rules:
1. Every claim in "answer" must cite its source ids in square brackets, e.g. [S1] or [S1][S3].
2. If no source contains a rule that addresses the question, set "answerable" to false, explain in one
   sentence that the provided regulations do not cover it, and return an empty "citations" list. Do not guess.
   If the question asks about a regulator, framework or law that is not one of the sources (for example
   SAMA frameworks, labour law or tax law), it is not answerable: never present these sources as if they
   were that other regulation.
3. Many questions describe the asker's own situation or practice and ask whether it is allowed or compliant.
   These ARE answerable when a source states the relevant rule, even though the sources do not mention the
   asker: apply the rule to the situation. Start "answer" with a clear verdict (for example "No, this does
   not comply" or "Yes, provided that ..."), then list in "points" what the rule requires.
4. "citations": one entry per source you relied on, with its id and a SHORT quote (under 30 words)
   copied character-for-character from that source's text.
5. Write in {language}. Be concise and practical for a compliance officer.
   Include the conditions and exceptions the sources state.
   "answer": a direct answer in 1-2 sentences with its source ids.
   "points": the conditions, exceptions or steps, one short item each with its source ids ([] if none).
6. If a source has a NOTE about differences between language versions, the Arabic text prevails; say so."""

ABSTAIN = {
    "en": (
        "I don't know. The regulations I have (PDPL, its Implementing Regulation, the Transfer Regulation, "
        "and NCA's ECC, CCC and DCC) don't clearly answer this question."
    ),
    "ar": (
        "لا أعلم. الأنظمة المتوفرة لدي (نظام حماية البيانات الشخصية ولائحته التنفيذية ولائحة نقل البيانات، "
        "وضوابط الهيئة الوطنية للأمن السيبراني ECC وCCC وDCC) لا تجيب عن هذا السؤال بوضوح."
    ),
}
LANG_NAME = {"en": "English", "ar": "Arabic (Modern Standard Arabic)"}
AUTHORITY_LABEL = {
    "binding": "binding Arabic text",
    "official": "official Arabic text",
    "translation": "English translation",
}


@dataclass
class Citation:
    id: str  # S1, S2, ...
    doc_id: str
    title: str  # e.g. "PDPL — Article 29"
    canonical_key: str | None
    quote: str
    quote_verified: bool
    text_ar: str | None  # authoritative text, always shown when it exists
    text_en: str | None
    authority_ar: str | None
    source_ar: str | None  # "ocr" -> numbers inside may be misread
    note: str
    keys: list[str] = field(default_factory=list)  # every article/control the cited chunk covers

@dataclass
class Answer:
    question: str
    lang: str
    answer: str
    abstained: bool
    citations: list[Citation]
    related: list[str]  # headings of retrieved sources (shown when abstaining)
    disclaimer: str
    model: str
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _versions(hit: Hit) -> tuple[dict | None, dict | None]:
    """(arabic chunk, english chunk) for a hit."""
    pair = [hit.chunk, hit.counterpart] if hit.counterpart else [hit.chunk]
    ar = next((c for c in pair if c and c["lang"] == "ar"), None)
    en = next((c for c in pair if c and c["lang"] == "en"), None)
    return ar, en


def build_sources(hits: list[Hit], lang: str) -> tuple[str, dict[str, Hit]]:
    """Numbered source block for the prompt: the version in the question's language when it exists
    (keeps the prompt within the context window); OCR'd Arabic also gets the English for exact numbers."""
    blocks, ids = [], {}
    for n, hit in enumerate(hits, start=1):
        sid = f"S{n}"
        ids[sid] = hit
        ar, en = _versions(hit)
        primary = (ar if lang == "ar" else en) or hit.chunk
        lines = [
            f"[{sid}] {primary['heading']}",
            f"({AUTHORITY_LABEL.get(primary['authority'], primary['authority'])})",
        ]
        if primary.get("note"):
            lines.append(f"NOTE: {primary['note']}")
        lines.append(primary["text"])
        if primary["lang"] == "ar" and primary.get("source") == "ocr" and en:
            lines.append(f"English translation (use for exact numbers; the Arabic above was OCR'd): {en['text']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), ids


def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", arabic.normalize_for_search(s)).strip()


def quote_in_source(quote: str, sources: list[str]) -> bool:
    """Exact (normalized) substring, or at least 85% of the quote's words in order-insensitive overlap with
    one source. Tolerates the model fixing whitespace or punctuation, not invented wording."""
    q = _norm(quote)
    if not q:
        return False
    for src in sources:
        s = _norm(src)
        if q in s:
            return True
        qw, sw = q.split(), set(s.split())
        if len(qw) >= 4 and sum(w in sw for w in qw) / len(qw) >= 0.85:
            return True
    return False


def finalize(raw: dict, ids: dict[str, Hit], lang: str) -> tuple[str, bool, list[Citation]]:
    """Validate the model output: keep citations that point at real sources, check quotes, abstain otherwise."""
    answer = str(raw.get("answer", "")).strip()
    points = [str(p).strip().lstrip("-•* ").strip() for p in raw.get("points") or [] if str(p).strip()]
    points = list(dict.fromkeys(points))  # drop repeated items
    if points:  # rendered as a Markdown list by the chat page
        answer = answer + "\n\n" + "\n".join(f"- {p}" for p in points)
    cites: list[Citation] = []
    seen = set()
    by_heading = {}  # models sometimes cite the source's heading instead of its id
    for k, h in ids.items():
        for v in _versions(h):
            if v:
                by_heading.setdefault(v["heading"].strip().lower(), k)
    raw_cites = list(raw.get("citations") or [])
    # sources cited inline ([S2]) but missing from "citations" are still cited: add them without a quote
    listed = {str(c.get("source", "")).strip().strip("[]").strip() for c in raw_cites}
    for sid in dict.fromkeys(re.findall(r"\[(S\d+)\]", answer)):
        if sid in ids and sid not in listed:
            raw_cites.append({"source": sid, "quote": ""})
    for c in raw_cites:
        sid = str(c.get("source", "")).strip().strip("[]").strip()
        if sid not in ids:
            name = sid.lower()
            sid = by_heading.get(name) or next((k for hd, k in by_heading.items() if name and hd.startswith(name)), "")
        if sid not in ids or sid in seen:
            continue  # cited a source that was never provided
        seen.add(sid)
        hit = ids[sid]
        ar, en = _versions(hit)
        texts = [x["text"] for x in (ar, en) if x]
        quote = re.sub(r"\s*\[S\d+\]", "", str(c.get("quote", ""))).strip()  # models echo markers
        cites.append(
            Citation(
                id=sid,
                doc_id=hit.chunk["doc_id"],
                title=(ar if lang == "ar" and ar else en or ar)["heading"],
                canonical_key=hit.chunk["canonical_key"],
                quote=quote,
                quote_verified=quote_in_source(quote, texts),
                text_ar=ar["text"] if ar else None,
                text_en=en["text"] if en else None,
                authority_ar=ar["authority"] if ar else None,
                source_ar=ar["source"] if ar else None,
                note=" ".join(dict.fromkeys(x["note"] for x in (ar, en) if x and x.get("note"))),
                keys=chunk_keys(hit.chunk),
            )
        )
    # inline markers must point at kept citations
    kept = {c.id for c in cites}
    answer = re.sub(r"\[(S\d+)\]", lambda m: m.group(0) if m.group(1) in kept else "", answer).strip()
    if not raw.get("answerable") or not cites or not answer:
        return ABSTAIN[lang], True, []
    if not re.search(r"\[S\d+\]", answer):
        answer += " " + "".join(f"[{c.id}]" for c in cites)
    return answer, False, cites


def answer_question(
    question: str,
    llm: LLM | None = None,
    strategy: str | None = None,
    use_reranker: bool | None = None,
    mode: str = "hybrid",
) -> Answer:
    lang = arabic.detect_lang(question)
    t0 = time.perf_counter()
    r = retrieve(question, strategy=strategy, use_reranker=use_reranker, mode=mode)
    timings = {**r.timings, "retrieval_total": time.perf_counter() - t0}
    llm = llm or get_llm()
    related = [h.chunk["heading"] for h in r.hits]

    if not r.hits:
        return Answer(question, lang, ABSTAIN[lang], True, [], related, DISCLAIMER[lang], llm.name, timings)

    context, ids = build_sources(r.hits, lang)
    user = f"Question: {question}\n\nSources:\n\n{context}"
    t0 = time.perf_counter()
    schema = copy.deepcopy(ANSWER_SCHEMA)  # constrain citation sources to the ids actually provided
    schema["properties"]["citations"]["items"]["properties"]["source"]["enum"] = list(ids)
    raw = llm.complete_json(SYSTEM.replace("{language}", LANG_NAME[lang]), user, schema)
    timings["llm"] = time.perf_counter() - t0
    text, abstained, cites = finalize(raw, ids, lang)
    timings["total"] = sum(v for k, v in timings.items() if k in ("retrieval_total", "llm"))
    return Answer(question, lang, text, abstained, cites, related, DISCLAIMER[lang], llm.name, timings)
