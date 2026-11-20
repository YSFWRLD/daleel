"""PDF -> cleaned lines (with page numbers). Handles Arabic repair, running headers/footers and OCR."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

from app.text import arabic

ID_TOKEN = r"[0-9٠-٩]+(?:[-.](?:[0-9٠-٩]+|[PTمش]))+"
ID_ONLY = re.compile(rf"^\s*({ID_TOKEN})\s*$")
# Table-of-contents leaders, check marks, lone page numbers, glyph junk
NOISE = re.compile(r"^(?:[.…·\s]{6,}.*|.*\.{6,}.*|[✔✓\s]+|\d{1,3}|[!\"#$%&'()*+,\-–•.:;/\s]+)$")
STRUCTURAL_LABELS = {
    "objective",
    "controls",
    "الهدف",
    "الضوابط",
}
CLASSIFICATION_LABELS = {
    "public",
    "confidential",
    "secret",
    "top",
    "top secret",
    "data classification level",
    "عام",
    "مقيد",
    "سري",
    "سري للغاية",
    "مستوى تصنيف البيانات",
}


@dataclass
class Line:
    text: str
    page: int  # 1-based


def _ocr_page(page: pymupdf.Page, ocr) -> list[str]:
    """OCR a page whose text is drawn as vector graphics, then rebuild table reading order.

    NCA control tables put the ID in a left column, vertically centred on its cell, so IDs are
    matched to paragraphs by position rather than by OCR output order.
    """
    import numpy as np

    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    result, _ = ocr(img)
    items = []
    for box, txt, conf in result or []:
        txt = txt.strip()
        if not txt or float(conf) < 0.5:
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        if min(ys) < 0.07 * pix.h or max(ys) > 0.93 * pix.h:
            continue  # running header / footer zone
        items.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys), "t": txt})

    ids = [i for i in items if ID_ONLY.match(i["t"])]
    labels = [i for i in items if i["t"].lower() in STRUCTURAL_LABELS]
    texts = [i for i in items if i not in ids and i not in labels and i["t"].lower() not in CLASSIFICATION_LABELS]
    # DCC tables have per-classification-level columns on the right ("At least annually", ...).
    # Keep them out of the paragraph flow and attach them to the row they belong to.
    side = [i for i in texts if i["x0"] > 0.55 * pix.w]
    texts = [i for i in texts if i["x0"] <= 0.55 * pix.w]

    # group text lines into paragraphs (same column, small vertical gap)
    texts.sort(key=lambda i: i["y0"])
    paras: list[dict] = []
    for it in texts:
        p = paras[-1] if paras else None
        if p and it["y0"] - p["y1"] < 18 and abs(it["x0"] - p["x0"]) < 60:
            p["items"].append(it)
            p["y1"] = max(p["y1"], it["y1"])
        else:
            paras.append({"x0": it["x0"], "y0": it["y0"], "y1": it["y1"], "items": [it], "tag": None})

    def centre(i):
        return (i["y0"] + i["y1"]) / 2

    def split_by_centres(items: list[dict], cents: list[float]) -> list[list[dict]]:
        """Split lines into len(cents) contiguous groups whose mid-heights best match the ID centres
        (IDs are vertically centred on cells of different heights)."""
        n, k = len(items), len(cents)
        best: dict[tuple[int, int], tuple[float, list[int]]] = {(0, 0): (0.0, [])}
        for j in range(1, k + 1):
            for e in range(j, n - (k - j) + 1):
                opts = []
                for s in range(j - 1, e):
                    if (s, j - 1) in best:
                        grp = items[s:e]
                        mid = (grp[0]["y0"] + grp[-1]["y1"]) / 2
                        cost = best[(s, j - 1)][0] + abs(mid - cents[j - 1])
                        # a cell should start after a finished sentence, not mid-sentence
                        # (strong: some IDs are top-aligned rather than centred, which skews the centres)
                        if s > 0 and not items[s - 1]["t"].rstrip().endswith((".", ":", ";")):
                            cost += 100
                        opts.append((cost, best[(s, j - 1)][1] + [e]))
                if opts:
                    best[(e, j)] = min(opts, key=lambda o: o[0])
        cuts = best[(n, k)][1] if (n, k) in best else [n]
        groups, s = [], 0
        for e in cuts:
            groups.append(items[s:e])
            s = e
        return groups

    tags = ids + labels
    used: set[int] = set()
    split_paras: list[dict] = []
    for p in paras:  # IDs/labels left of the paragraph and inside its vertical span
        cands = [
            (k, m)
            for k, m in enumerate(tags)
            if k not in used and m["x1"] < p["x0"] + 5 and p["y0"] - 15 <= centre(m) <= p["y1"] + 15
        ]
        if cands:
            col = max(m["x1"] for _, m in cands)  # nearest column to the left = deepest level
            cands = sorted([c for c in cands if c[1]["x1"] > col - 40], key=lambda c: centre(c[1]))
            cands = cands[: len(p["items"])]
            groups = split_by_centres(p["items"], [centre(m) for _, m in cands])
            for (k, m), grp in zip(cands, groups):
                used.add(k)
                split_paras.append({**p, "y0": grp[0]["y0"], "y1": grp[-1]["y1"], "items": grp, "tag": m["t"]})
        else:
            split_paras.append(p)
    paras = split_paras
    for p in paras:
        p["lines"] = [i["t"] for i in p["items"]]
    leftover = [(k, m) for k, m in enumerate(ids) if k not in used]
    for p in paras:  # merged cells: parent-control text sits above its centred ID
        if p["tag"] is None and leftover:
            below = [(k, m) for k, m in leftover if p["y0"] <= centre(m) <= p["y1"] + 300 and m["x1"] < p["x0"] + 5]
            if below:
                k, m = min(below, key=lambda c: centre(c[1]))
                leftover.remove((k, m))
                p["tag"] = m["t"]

    for it in side:  # attach right-hand column values to the overlapping tagged row
        rows = [p for p in paras if p["tag"] and p["y0"] - 25 <= centre(it) <= p["y1"] + 25]
        if rows:
            min(rows, key=lambda p: abs((p["y0"] + p["y1"]) / 2 - centre(it))).setdefault("side", []).append(it)

    out = []
    for p in sorted(paras, key=lambda p: p["y0"]):
        body = " ".join(p["lines"])
        if p.get("side"):
            cols: list[list[dict]] = []
            for it in sorted(p["side"], key=lambda i: i["x0"]):
                if cols and abs(it["x0"] - cols[-1][0]["x0"]) < 60:
                    cols[-1].append(it)
                else:
                    cols.append([it])
            vals = [" ".join(i["t"] for i in sorted(c, key=lambda i: i["y0"])) for c in cols]
            body += " (by data classification level: " + " / ".join(vals) + ")"
        if p["tag"] and p["tag"].lower() in STRUCTURAL_LABELS:
            out += [p["tag"], body]  # "Objective" / "Controls" stay on their own line
        elif p["tag"]:
            out.append(f"{p['tag']} {body}")
        else:
            out.append(body)
    return out


# "1-", "1." and "أ-"; not "أ." or "4)", which also occur as broken words and page furniture
MARKER_PIECE = re.compile(r"^(?:[0-9٠-٩]{1,2}\s*[-.]?|[أبجدهوزحطي]\s*-?|[-.])$")
MARKER_DONE = re.compile(r"^(?:[0-9٠-٩]{1,2}\s*[-.]|[أبجدهوزحطي]\s*-)$")


def merge_markers(lines: list[str]) -> list[str]:
    """Arabic PDFs put paragraph markers on their own lines ("1", "-", then the text; or "أ-"). Alone they look
    like page numbers or repeated headers and get filtered out, so glue them onto the text that follows.
    A bare number or letter only counts when a "-" completes it, so page numbers stay apart."""
    out: list[str] = []
    pending = ""
    for t in lines:
        s = t.strip()
        if not s:
            continue
        if MARKER_PIECE.match(s) and (pending or s[0] not in "-."):  # a marker starts with its number/letter
            cand = pending + s
            if len(cand) <= 4 and MARKER_PIECE.match(cand):
                pending = cand
                continue
            if pending:  # e.g. "-" followed by "3": drop the stray piece, start a new marker
                out.append(pending)
            pending = s
            continue
        if pending:
            if MARKER_DONE.match(pending):
                s = f"{pending} {s}"
            else:
                out.append(pending)  # not a marker after all (e.g. a page number); leave it to the filters
            pending = ""
        out.append(s)
    if pending:
        out.append(pending)
    return out


def extract_lines(
    path: str,
    lang: str,
    page_range: tuple[int, int] | None = None,
    ocr: bool = False,
) -> list[Line]:
    doc = pymupdf.open(path)
    first, last = page_range or (1, len(doc))
    ocr_engine = None
    if ocr:
        from rapidocr_onnxruntime import RapidOCR

        ocr_engine = RapidOCR()

    pages: list[list[str]] = []
    for pno in range(first - 1, last):
        page = doc[pno]
        if ocr_engine is not None:
            raw = _ocr_page(page, ocr_engine)
        else:
            raw = page.get_text().splitlines()
        pages.append([arabic.clean_unicode(t).strip() for t in raw])

    if lang == "ar":
        joined = "\n".join("\n".join(p) for p in pages)
        aggressive = arabic.lam_alef_bug_count(joined) > 5
        pages = [[arabic.fix_lam_alef(t, aggressive) for t in p] for p in pages]
        pages = [[arabic.fix_mirrored_brackets(arabic.JUNK_GLYPHS.sub(" ", t).strip()) for t in p] for p in pages]
        pages = [merge_markers(p) for p in pages]

    # running headers / footers: short lines repeated on many pages
    def key(t: str) -> str:  # ignore a page number glued to the start/end of a header line
        t = re.sub(r"\s+", " ", t).strip()
        if re.match(r"^(Article|المادة)\b", t, re.IGNORECASE):
            return t  # "Article 12" is a heading, not "Article" + page number
        return re.sub(r"^[\d٠-٩]+\s+|\s+[\d٠-٩]+$", "", t).strip()

    freq = Counter(k for p in pages for k in {key(t) for t in p if t})
    threshold = max(3, int(0.3 * len(pages)))
    repeated = {k for k, n in freq.items() if n >= threshold and len(k) < 80 and k.lower() not in STRUCTURAL_LABELS}

    lines: list[Line] = []
    for i, p in enumerate(pages):
        for t in p:
            t = re.sub(r"\s+", " ", t).strip()
            if not t or key(t) in repeated or NOISE.match(t):
                continue
            if t.lower() in CLASSIFICATION_LABELS:
                continue
            lines.append(Line(t, first + i))
    return lines


def repair_reversed(lines_by_doc: dict[str, list[Line]]) -> None:
    """Fix visually-reversed Arabic lines using a vocabulary built from all Arabic documents."""
    vocab = arabic.build_vocab([ln.text for lines in lines_by_doc.values() for ln in lines])
    for lines in lines_by_doc.values():
        for ln in lines:
            # reversing a visual-order line can re-create the 'األ' artifact; repair it again
            ln.text = arabic.fix_lam_alef(arabic.fix_reversed_line(ln.text, vocab), aggressive=False)
