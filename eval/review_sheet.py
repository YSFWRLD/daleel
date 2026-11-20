"""Write eval/questions_review.md: every question with its gold sources (title + opening text), for the
domain-expert review of the gold set.  python -m eval.review_sheet"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT.parent / "data" / "processed" / "chunks_article.jsonl"


def main() -> None:
    first: dict[tuple[str, str, str], dict] = {}
    for c in map(json.loads, CHUNKS.open(encoding="utf-8")):
        first.setdefault((c["doc_id"], c["canonical_key"], c["lang"]), c)
    qs = [json.loads(line) for line in (ROOT / "questions.jsonl").open(encoding="utf-8")]
    out = [
        "# Evaluation questions: review sheet",
        "",
        "For each question check: (1) the question is realistic, (2) the gold sources are the ones a compliance "
        "officer would cite, (3) nothing important is missing from the gold list, (4) unanswerable questions "
        "really are not covered by these regulations. Gold lists every acceptable source; citing any one counts.",
        "Edit `eval/questions.jsonl` (set `status` to `reviewed`), or note corrections here.",
        "",
    ]
    for q in qs:
        out.append(f"### {q['id']} ({q['lang'].upper()}{', unanswerable' if not q['answerable'] else ''})")
        out.append("")
        out.append(f"**Q:** {q['question']}")
        out.append("")
        for g in q["gold"]:
            c = first.get((g["doc_id"], g["canonical_key"], "en")) or first.get((g["doc_id"], g["canonical_key"], "ar"))
            head = c["heading"] if c else f"{g['doc_id']} {g['canonical_key']}"
            text = (c["text"][:220] + "…") if c else ""
            out.append(f"- Gold: **{head}** — {text}")
        if q.get("expected_notes"):
            out.append(f"- Note: {q['expected_notes']}")
        out.append("")
    (ROOT / "questions_review.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {ROOT / 'questions_review.md'} ({len(qs)} questions)")


if __name__ == "__main__":
    main()
