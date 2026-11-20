"""Build chunk files from data/sources.yaml.

python -m ingest.build            # writes data/processed/chunks_{article,fixed}.jsonl + report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from app.text import arabic
from ingest.chunkers import fixed_chunks, parse_articles, parse_controls, structural_chunks
from ingest.extract import extract_lines, repair_reversed

DATA = Path(__file__).resolve().parent.parent / "data"


def recover_graphics_pages(cfg: dict, lines_by: dict, ocr_pages: dict) -> list[str]:
    """For documents with `grid_ocr: {lang: true}`, OCR pages that have no text layer, using the
    other language's control order to label the table rows (see ingest/grid_ocr.py)."""
    import pymupdf

    from ingest.chunkers import ID_PREFIX, norm_id
    from ingest.extract import Line
    from ingest.grid_ocr import ocr_grid_page

    notes = []
    for d in cfg["documents"]:
        for lang, on in (d.get("grid_ocr") or {}).items():
            other = "en" if lang == "ar" else "ar"
            if not on or (d["doc_id"], other) not in lines_by:
                continue
            seq: list[str] = []
            parse_controls(
                lines_by[(d["doc_id"], other)], other, (d.get("id_position") or {}).get(other, "before"), collect=seq
            )
            pos = {nid: k for k, nid in enumerate(seq)}
            lines = lines_by[(d["doc_id"], lang)]
            ids_on: dict[int, list[int]] = {}
            for ln in lines:
                m = ID_PREFIX.match(ln.text)
                if m and norm_id(m.group(1)) in pos:
                    ids_on.setdefault(ln.page, []).append(pos[norm_id(m.group(1))])
            doc = pymupdf.open(str(DATA / d["files"][lang]))
            added: list[Line] = []
            for p in range(2, len(doc)):
                if len(doc[p - 1].get_text().strip()) > 400 or p - 1 not in ids_on:
                    continue
                lo = max(ids_on[p - 1])
                # last control page: nothing after it, so it must hold the rest of the sequence
                hi = min(ids_on[p + 1]) if p + 1 in ids_on else len(seq)
                if not (lo < hi and hi - lo < 80):
                    continue
                res = ocr_grid_page(doc[p - 1], seq[lo + 1 : hi])
                notes.append(f"- {d['doc_id']}/{lang} page {p}: {'OK' if res.ok else 'SKIPPED'} — {res.note}")
                if res.ok:
                    added += [Line(t, p) for t in res.lines]
                    ocr_pages.setdefault((d["doc_id"], lang), set()).add(p)
            if added:
                merged = sorted(lines + added, key=lambda ln: ln.page)  # stable: keeps order within a page
                lines_by[(d["doc_id"], lang)] = merged
    return notes


def main() -> int:
    cfg = yaml.safe_load((DATA / "sources.yaml").read_text(encoding="utf-8"))
    lines_by: dict[tuple[str, str], list] = {}
    ocr_pages: dict[tuple[str, str], set[int]] = {}
    for d in cfg["documents"]:
        for lang, rel in d["files"].items():
            if not rel:
                continue
            pages = (d.get("pages") or {}).get(lang)
            ocr = bool((d.get("ocr") or {}).get(lang))
            print(f"extracting {d['doc_id']}/{lang}{' (OCR)' if ocr else ''} ...", flush=True)
            lines_by[(d["doc_id"], lang)] = extract_lines(str(DATA / rel), lang, tuple(pages) if pages else None, ocr)
            if ocr:
                ocr_pages[(d["doc_id"], lang)] = {ln.page for ln in lines_by[(d["doc_id"], lang)]}
    grid_notes = recover_graphics_pages(cfg, lines_by, ocr_pages)
    repair_reversed({f"{k[0]}/{k[1]}": v for k, v in lines_by.items() if k[1] == "ar"})

    out = {"article": [], "fixed": []}
    report = [
        "# Chunking report",
        "",
        "| doc | lang | units | chunks (article) | chunks (fixed) | notes |",
        "|---|---|---|---|---|---|",
    ]
    keys_by_doc: dict[str, dict[str, set]] = {}
    for d in cfg["documents"]:
        for lang in d["files"]:
            if (d["doc_id"], lang) not in lines_by:
                continue
            lines = lines_by[(d["doc_id"], lang)]
            if d["unit"] == "article":
                units = parse_articles(lines, lang)
            else:
                units = parse_controls(lines, lang, (d.get("id_position") or {}).get(lang, "before"))
            title = d["short"][lang]
            a, f = structural_chunks(units, d["doc_id"], title, lang), fixed_chunks(units, d["doc_id"], title, lang)
            notes_for = {k: v.get(lang, "") for k, v in (d.get("version_notes") or {}).items()}
            scanned = ocr_pages.get((d["doc_id"], lang), set())
            alias = ((d.get("aliases") or {}).get(lang)) or {}
            for c in a + f:
                if lang == "ar":  # brackets split across lines are only fixable on the joined text
                    c.text = arabic.join_split_suffix(arabic.fix_mirrored_brackets(c.text))
                c.source = "ocr" if c.page in scanned else "text"
                c.note = " ".join(n for r in c.refs if (n := notes_for.get(r)))
                c.authority = (d.get("authority") or {}).get(lang, "")
                c.canonical_refs = [alias.get(r, r) for r in c.refs]
                c.canonical_key = alias.get(c.ref_key, c.ref_key) if c.ref_key else None
            out["article"] += a
            out["fixed"] += f
            alias = ((d.get("aliases") or {}).get(lang)) or {}
            keys = {alias.get(u.ref_key, u.ref_key) for u in units}  # compare in canonical numbering
            keys_by_doc.setdefault(d["doc_id"], {})[lang] = keys
            notes = ""
            if d["unit"] == "article":
                nums = sorted(int(k.split("-")[1]) for k in keys)
                notes = f"articles 1..{nums[-1] if nums else 0}"
            report.append(f"| {d['doc_id']} | {lang} | {len(units)} | {len(a)} | {len(f)} | {notes} |")

    if grid_notes:
        report += ["", "## Graphics-only pages recovered with grid OCR", ""] + grid_notes
    report += ["", "## English vs Arabic structure (canonical keys present in one language only)", ""]
    for doc, by_lang in keys_by_doc.items():
        if len(by_lang) == 2:
            en, ar = by_lang.get("en", set()), by_lang.get("ar", set())
            report.append(f"- **{doc}**: only EN {sorted(en - ar) or '—'}; only AR {sorted(ar - en) or '—'}")

    (DATA / "processed").mkdir(exist_ok=True)
    for strategy, chunks in out.items():
        with open(DATA / "processed" / f"chunks_{strategy}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in chunks)
    (DATA / "processed" / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
