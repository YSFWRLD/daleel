"""Structure-aware chunking (articles / controls) plus a fixed-size baseline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.text import arabic
from ingest.extract import ID_TOKEN, STRUCTURAL_LABELS, Line

MAX_WORDS = 380  # split structural units longer than this
SPLIT_WORDS, SPLIT_OVERLAP = 300, 40
FIXED_WORDS, FIXED_OVERLAP = 300, 50


@dataclass
class Unit:
    ref_key: str  # language-independent: art-29 / ctl-2-3-1 / ctl-2-6-P-1 / sub-2-3
    ref: str  # display: "Article 29" / "المادة 29" / "Control 2-3-1"
    title: str
    parent_section: str
    page: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.lines)).strip()


@dataclass
class Chunk:
    id: str
    strategy: str
    doc_id: str
    lang: str
    ref: str | None
    ref_key: str | None
    refs: list[str]
    title: str
    parent_section: str
    page: int
    heading: str
    text: str
    source: str = "text"  # "ocr" when the text came from OCR (numbers inside may be misread)
    note: str = ""  # e.g. known differences between the Arabic and English versions

    def to_dict(self) -> dict:
        return asdict(self)


# --- articles ------------------------------------------------------------------------------------
EN_HEADING = re.compile(r"^Article\s*\(?(\d+)\)?\s*(?:[:.\-–]\s*)?(.*)$", re.IGNORECASE)


def parse_ar_heading(text: str) -> tuple[int, str] | None:
    """Recognise 'المادة الأولى: التعريفات' and its word-reversed variants ('التعريفات :الأولى المادة')."""
    toks = text.replace(":", " : ").split()
    if "المادة" not in toks:
        return None
    if toks[0] == "المادة":
        rest = toks[1:]
        reversed_line = False
    else:
        # Word-reversed line ("التعريفات :الأولى المادة"), possibly with the title wrapping on
        # both sides of the heading word. Reading order = reverse tokens, rotate to start at المادة.
        k = toks.index("المادة")
        prev = [t for t in toks[:k] if t != ":"]
        if not prev or not arabic.parse_ordinal(prev[-1:]):
            return None
        rev = list(reversed(toks))
        j = rev.index("المادة")
        rest = rev[j + 1 :] + rev[:j]
        reversed_line = True
    while rest and rest[0] == ":":
        rest = rest[1:]
    if not rest or rest[0][:1] in "()":
        return None  # "المادة (السابعة) من النظام" is a cross-reference, not a heading
    parsed = arabic.parse_ordinal(rest)
    if not parsed:
        return None
    num, used = parsed
    after = rest[used:]
    # a real heading has ':' or nothing right after the ordinal (not "من النظام ...")
    if not reversed_line and after and after[0] != ":":
        return None
    return num, " ".join(t for t in after if t != ":").strip()


def _fragmented_ar_heading(lines: list[Line], i: int, expected: int) -> tuple[str, int] | None:
    """Heading split over up to 3 short lines in scrambled order, e.g. 'السادسة' / ':عشرة المصلحة ...'.

    Accept only if the window holds every word of the *expected* ordinal plus a ':' and no brackets
    (cross-references are bracketed). Returns (title, lines consumed).
    """
    want = arabic.ordinal_words(expected)
    window = [ln.text for ln in lines[i : i + 3]]
    first = [arabic.normalize_for_search(t) for t in window[0].replace(":", " ").split()]
    if not (set(first) & (set(want) | {"الماده"})):
        return None
    for n in range(1, len(window) + 1):
        part = window[:n]
        if any(len(t) > 90 or re.search(r"[()]", t) for t in part):
            return None
        toks = [arabic.normalize_for_search(t) for t in " ".join(part).replace(":", " : ").split()]
        if all(w in toks for w in want) and ":" in toks:
            colon_line = next(t for t in part if ":" in t)
            rest = [
                t
                for t in colon_line.replace(":", " ").split()
                if arabic.normalize_for_search(t) not in set(want) | {"الماده"}
            ]
            return " ".join(reversed(rest)), n
    return None


def parse_articles(lines: list[Line], lang: str) -> list[Unit]:
    units: list[Unit] = []
    expected = 1
    i = 0
    while i < len(lines):
        ln = lines[i]
        text = ln.text
        # heading word alone on its line: look at it together with the next line
        if text in ("المادة", "Article") and i + 1 < len(lines):
            text = f"{text} {lines[i + 1].text}"
        head = None
        if lang == "en":
            m = EN_HEADING.match(text)
            if m:
                head = (int(m.group(1)), m.group(2).strip())
        else:
            head = parse_ar_heading(text)
            if not (head and head[0] == expected):
                frag = _fragmented_ar_heading(lines, i, expected)
                if frag:
                    title, used = frag
                    units.append(Unit(f"art-{expected}", f"المادة {expected}", title, "", ln.page))
                    expected += 1
                    i += used
                    continue
        if head and head[0] == expected:
            num, rest = head
            title = rest if (lang == "ar" or len(rest) < 80) else ""
            ref = f"Article {num}" if lang == "en" else f"المادة {num}"
            u = Unit(f"art-{num}", ref, title, "", ln.page)
            if lang == "en" and rest and not title:
                u.lines.append(rest)
            units.append(u)
            expected += 1
            if text != ln.text:
                i += 1  # consumed the next line too
        elif units:
            if not re.match(r"^(Chapter|Part|الفصل|الباب)\b", ln.text):
                units[-1].lines.append(ln.text)
        i += 1
    return units


# --- controls ------------------------------------------------------------------------------------
ID_PREFIX = re.compile(rf"^({ID_TOKEN})(?:\s+(.*))?$")


def norm_id(raw: str) -> str:
    s = arabic.to_ascii_digits(raw).replace(".", "-").replace("م", "P").replace("ش", "T")
    return s.strip("-")


def id_level(nid: str) -> int:
    return sum(1 for p in nid.split("-") if p.isdigit())


def id_sort_key(nid: str) -> tuple:
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in nid.split("-"))


def parse_controls(lines: list[Line], lang: str, id_position: str, collect: list[str] | None = None) -> list[Unit]:
    """Group NCA controls: one unit per subdomain objective and per main control (with its subcontrols).

    id_position: 'before' when an ID precedes its text (English, CCC Arabic),
                 'after' when the ID line follows its text (ECC/DCC Arabic stream order).
    """
    lab_obj = {"objective", "الهدف"}
    lab_ctl = {"controls", "الضوابط"}

    # skip the table of contents: start at the subdomain heading before the first real control
    def match_id(t: str):
        m = ID_PREFIX.match(t)
        return (norm_id(m.group(1)), (m.group(2) or "").strip()) if m else None

    first3 = next((k for k, ln in enumerate(lines) if (m := match_id(ln.text)) and id_level(m[0]) >= 3), None)
    if first3 is None:
        return []
    start = max((k for k in range(first3) if (m := match_id(lines[k].text)) and id_level(m[0]) == 2), default=first3)
    if id_position == "after":
        start = max(0, start - 2)  # the subdomain title precedes its ID

    subs: dict[str, Unit] = {}
    ctrls: dict[str, Unit] = {}
    order: list[str] = []
    last_key: tuple = ()
    seen: set[str] = set()
    cur_sub: str | None = None
    target: list[str] | None = None  # 'before' mode: where text goes
    buffer: list[str] = []  # 'after' mode: text waiting for its ID

    def sub_title(key: str) -> str:
        return subs[key].title if key in subs else ""

    def open_id(nid: str, page: int) -> list[str]:
        nonlocal cur_sub
        lvl = id_level(nid)
        parts = nid.split("-")
        if lvl == 2:
            cur_sub = nid
            subs[nid] = Unit(f"sub-{nid}", nid, "", "", page)
            order.append(f"sub-{nid}")
            return subs[nid].lines  # title lines until "Objective"
        # level >= 3: main control is the first 3 numeric parts (+ P/T letter in CCC)
        main_parts, n = [], 0
        for p in parts:
            main_parts.append(p)
            n += p.isdigit()
            if n == 3:
                break
        main = "-".join(main_parts)
        if main not in ctrls:
            sub = "-".join(parts[:2])
            ctrls[main] = Unit(f"ctl-{main}", main, "", sub, page)
            order.append(f"ctl-{main}")
            # OCR can miss a control's ID, leaving its opening sentence ("In addition to ...",
            # "... shall cover the following:") at the end of the subdomain objective. Move it back.
            if id_position == "before" and sub in subs and subs[sub].lines:
                last = subs[sub].lines[-1]
                if last.rstrip().endswith(":") or re.match(r"^(In addition to|بالإضافة)", last):
                    ctrls[main].lines.append(subs[sub].lines.pop())
        u = ctrls[main]
        if nid != main:
            u.lines.append(nid)  # keep subcontrol numbers inline in the text
        return u.lines

    for ln in lines[start:]:
        t = ln.text
        low = t.lower()
        m = match_id(t)
        if m and id_level(m[0]) >= 2:
            nid, inline = m
            key = id_sort_key(nid)
            # Accept any ID not seen yet: some PDFs store a whole subdomain out of order in the
            # text stream (CCC Arabic puts 2-8 after 2-10).
            if nid not in seen and not (last_key and key[0] < last_key[0]):
                seen.add(nid)
                last_key = max(last_key, key)
                if collect is not None:
                    collect.append(nid)
                dest = open_id(nid, ln.page)
                if id_position == "after":
                    if id_level(nid) == 2:
                        buffer = buffer[-2:]  # title is the last line or two before the ID
                    dest.extend(buffer)
                    buffer = []
                    if inline:
                        dest.append(inline)
                    target = None
                else:
                    if inline:
                        dest.append(inline)
                    target = dest
                continue
            if int(nid.split("-")[0]) < int(last_key[0][1]):
                break  # IDs jumped back to an earlier domain: appendix / mapping tables
        if low in lab_obj:
            if cur_sub:
                subs[cur_sub].title = subs[cur_sub].title or " ".join(subs[cur_sub].lines)
                subs[cur_sub].lines = []
                if id_position == "after":
                    subs[cur_sub].lines.extend(buffer)
                    buffer = []
                else:
                    target = subs[cur_sub].lines
            continue
        if low in lab_ctl:
            buffer, target = [], None
            continue
        if low in STRUCTURAL_LABELS:
            continue
        if id_position == "after":
            buffer.append(t)
        elif target is not None:
            target.append(t)

    units: list[Unit] = []
    for k in order:
        if k.startswith("sub-"):
            u = subs[k[4:]]
            u.parent_section = ""
            label = "Objective" if lang == "en" else "الهدف"
            if u.lines:
                u.lines.insert(0, f"{label}:")
            u.ref = f"{'Subdomain' if lang == 'en' else 'المكون الفرعي'} {u.ref}"
        else:
            u = ctrls[k[4:]]
            u.title = sub_title(u.parent_section)
            u.parent_section = f"{u.parent_section} {u.title}".strip()
            u.ref = f"{'Control' if lang == 'en' else 'الضابط'} {u.ref}"
        if u.text:
            units.append(u)
    return units


# --- chunk builders ------------------------------------------------------------------------------
def _split_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    out, step = [], size - overlap
    for s in range(0, len(words), step):
        out.append(words[s : s + size])
        if s + size >= len(words):
            break
    return out


def structural_chunks(units: list[Unit], doc_id: str, doc_title: str, lang: str) -> list[Chunk]:
    chunks = []
    for u in units:
        words = u.text.split()
        pieces = _split_words(words, SPLIT_WORDS, SPLIT_OVERLAP) if len(words) > MAX_WORDS else [words]
        heading = f"{doc_title} — {u.ref}" + (f": {u.title}" if u.title else "")
        for n, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{lang}:{u.ref_key}:{n}",
                    strategy="article",
                    doc_id=doc_id,
                    lang=lang,
                    ref=u.ref,
                    ref_key=u.ref_key,
                    refs=[u.ref_key],
                    title=u.title,
                    parent_section=u.parent_section,
                    page=u.page,
                    heading=heading,
                    text=" ".join(piece),
                )
            )
    return chunks


def fixed_chunks(units: list[Unit], doc_id: str, doc_title: str, lang: str) -> list[Chunk]:
    """Baseline: ignore structure, slide a word window over the document; record which units it covers."""
    words: list[tuple[str, str, int]] = [(w, u.ref_key, u.page) for u in units for w in u.text.split()]
    chunks = []
    step = FIXED_WORDS - FIXED_OVERLAP
    for n, s in enumerate(range(0, max(len(words), 1), step)):
        win = words[s : s + FIXED_WORDS]
        if not win:
            break
        refs = list(dict.fromkeys(r for _, r, _ in win))
        chunks.append(
            Chunk(
                id=f"{doc_id}:{lang}:fixed:{n}",
                strategy="fixed",
                doc_id=doc_id,
                lang=lang,
                ref=None,
                ref_key=refs[0],
                refs=refs,
                title="",
                parent_section="",
                page=win[0][2],
                heading=doc_title,
                text=" ".join(w for w, _, _ in win),
            )
        )
        if s + FIXED_WORDS >= len(words):
            break
    return chunks
