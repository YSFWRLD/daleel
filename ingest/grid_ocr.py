"""Recover control tables from PDF pages whose text is drawn as graphics (Arabic ECC).

Tesseract reads the Arabic text well but garbles the Arabic-Indic control numbers. The table grid,
however, is real vector lines. So rows come from the grid, IDs come from the known control order
(taken from the English document between the IDs on the neighbouring text pages), and OCR is only
used for the cell text. A page is accepted only if its rows match the expected IDs exactly.

Output lines follow the 'after' convention used by the Arabic ECC text pages (text, then its ID).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

import pymupdf

from app.text import arabic

TESSDATA = os.environ.get("TESSDATA_DIR", "D:/Tesseract/tessdata")
TESSERACT = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@dataclass
class PageResult:
    page: int
    ok: bool
    lines: list[str]
    note: str


def _h_segments(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    segs = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < 1 and abs(p1.x - p2.x) > 15:
                    segs.append((min(p1.x, p2.x), max(p1.x, p2.x), p1.y))
            elif it[0] == "re":
                r = it[1]
                if r.height < 2 and r.width > 15:
                    segs.append((r.x0, r.x1, r.y0))
    return segs


def _ys(segs, x0, x1) -> list[float]:
    """Distinct y positions of segments covering the column [x0, x1]."""
    ys = sorted(s[2] for s in segs if s[0] <= x0 + 3 and s[1] >= x1 - 3)
    out: list[float] = []
    for y in ys:
        if not out or y - out[-1] > 3:
            out.append(y)
    return out


def _level(nid: str) -> int:
    return sum(1 for p in nid.split("-") if p.isdigit())


def ocr_grid_page(page: pymupdf.Page, expected: list[str], dpi: int = 300) -> PageResult:
    pno = page.number + 1
    segs = _h_segments(page)
    if len(segs) < 6 or not expected:
        return PageResult(pno, False, [], "no grid or no expected IDs")
    right = max(s[1] for s in segs)
    x_main = max(s[0] for s in segs if abs(s[1] - right) < 3)  # control-ID column start
    x_sub = max((s[0] for s in segs if abs(s[1] - x_main) < 3), default=x_main)  # subcontrol-ID column
    # shaded rows (subdomain titles, objective/controls labels) are filled rectangles, sometimes
    # without a border line, so their edges are row boundaries too
    filled = [
        it[1]
        for d in page.get_drawings()
        if d.get("fill") is not None
        for it in d["items"]
        if it[0] == "re" and it[1].height > 8 and it[1].x0 <= x_main + 3 and it[1].x1 >= right - 3
    ]
    main = _ys(segs + [(r.x0, r.x1, y) for r in filled for y in (r.y0, r.y1)], x_main, right)
    bottom = max(s[2] for s in segs)
    if bottom - main[-1] > 3:
        main.append(bottom)
    rows = list(zip(main, main[1:]))
    sub = _ys(segs, x_sub, x_main)

    x_left = min(s[0] for s in segs)

    def text_in(y0: float, y1: float, x_limit: float) -> str:
        """OCR one table cell (text column only, so the garbled ID glyphs are excluded)."""
        clip = pymupdf.Rect(x_left + 1, y0 + 0.5, x_limit - 1, y1 - 0.5)
        if clip.height < 4:
            return ""
        # Tesseract sometimes drops a lone short line; a wider crop (with the cell borders) or
        # raw-line mode (psm 13) recovers it. Garbled ID digits caught by the wide crop are stripped.
        wide = pymupdf.Rect(x_left - 25, y0 + 1, x_limit + 15, y1 - 1)  # keep row borders out of the crop
        attempts = ((clip, dpi, "6"), (clip, 400, "7"), (wide, dpi, "6"), (clip, dpi, "13"))
        for c, d, psm in attempts:
            png = _padded_png(page, c, d)
            res = subprocess.run(
                [TESSERACT, "stdin", "stdout", "--tessdata-dir", TESSDATA, "-l", "ara+eng", "--psm", psm],
                input=png,
                capture_output=True,
            )
            t = " ".join(arabic.clean_unicode(res.stdout.decode("utf-8", "replace")).split()).strip("| ")
            if len(t) > 3:
                return clean_ocr_arabic(t)
        return ""

    out: list[str] = []
    q = list(expected)
    i = 0
    while i < len(rows) and q:
        y0, y1 = rows[i]
        e = q[0]
        bands = [y0] + [y for y in sub if y0 + 1 < y < y1 - 1] + [y1]
        bands = list(zip(bands, bands[1:]))
        if _level(e) == 2:  # subdomain: title row, objective row, "controls" label row
            if i + 2 >= len(rows):
                return PageResult(pno, False, [], f"subdomain {e} split across pages")
            out += [text_in(y0, y1, x_main), e]
            oy0, oy1 = rows[i + 1]
            out += [text_in(oy0, oy1, x_main), "الهدف", "الضوابط"]
            q.pop(0)
            i += 3
            continue
        if _level(e) >= 4:  # subcontrols continuing from the previous page
            for by0, by1 in bands:
                if not q or _level(q[0]) < 4:
                    return PageResult(pno, False, [], f"row {i}: more subcontrol bands than expected")
                out += [text_in(by0, by1, x_sub), q.pop(0)]
            i += 1
            continue
        kids = 0
        while kids + 1 < len(q) and _level(q[kids + 1]) >= 4 and q[kids + 1].startswith(e + "-"):
            kids += 1
        want = 1 if kids == 0 else kids + 1
        if len(bands) != want:
            return PageResult(pno, False, [], f"{e}: {len(bands)} bands, expected {want}")
        if kids == 0:
            out += [text_in(y0, y1, x_main), q.pop(0)]
        else:
            by0, by1 = bands[0]
            out += [text_in(by0, by1, x_main), q.pop(0)]
            for by0, by1 in bands[1:]:
                out += [text_in(by0, by1, x_sub), q.pop(0)]
        i += 1
    if q:
        return PageResult(pno, False, [], f"{len(q)} expected IDs left over (first {q[0]})")
    return PageResult(pno, True, [t for t in out if t], f"{len(expected)} IDs")


def _padded_png(page: pymupdf.Page, clip: pymupdf.Rect, dpi: int) -> bytes:
    """Render a cell with a white margin: Tesseract often returns nothing when text touches the edge."""
    import io

    from PIL import Image, ImageOps

    im = Image.open(io.BytesIO(page.get_pixmap(dpi=dpi, clip=clip).tobytes("png"))).convert("L")
    buf = io.BytesIO()
    ImageOps.expand(im, border=30, fill=255).save(buf, "PNG")
    return buf.getvalue()


_LATIN_JUNK = re.compile(r"(?<![(\w])[A-Za-z][a-z]{0,5}[!.)]?(?![\w)])")
# "بحد أدنى ما يلي" is the phrase Tesseract garbles most ("بحد Gol ما tah", "Gol doy ما يلي")
_MIN_PHRASE = re.compile(
    r"(?:بحد\s+)?(?:[A-Za-z]+[!.)]?\s+){1,2}ما\s+(?:يلى|يلي|[A-Za-z]+)\s*:?|بحد\s+ما(?:\s+(?:يلى|يلي))?\s*:?"
)


def clean_ocr_arabic(t: str) -> str:
    """Drop short Latin fragments Tesseract hallucinates inside Arabic text ("بحد Gol ما tah").
    Real English terms in these documents are in parentheses or upper case, so they are kept."""
    t = _MIN_PHRASE.sub(" بحد أدنى ما يلي: ", t)
    parts = re.split(r"(\([^)]*\))", t)  # leave parenthesised English terms untouched
    t = "".join(p if p.startswith("(") else _LATIN_JUNK.sub(" ", p) for p in parts)
    t = t.replace("|", " ").replace("يلى", "يلي")
    # symbols Tesseract substitutes for Arabic-Indic digits it cannot read; better absent than wrong
    t = re.sub(r"[©®€£¥§™]", "", t)
    t = re.sub(r"\(\s*\)|\)\s*\(", " ", t)
    t = re.sub(r"(?:^|\s)[0-9٠-٩\-]{1,9}\s*$", "", t)  # stray ID glyphs at the end of a wide crop
    return " ".join(t.split()).strip(" -")
