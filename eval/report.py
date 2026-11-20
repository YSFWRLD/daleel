"""Build eval/results.md from the run files in eval/runs/.  python -m eval.report"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def num(v, digits=2) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def sec(v) -> str:
    return "–" if v is None else f"{v:.1f} s"


def load() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*.json")) if p.stem != "smoke"]


def retrieval_table(runs: list[dict]) -> list[str]:
    rows = [r for r in runs if not r["config"]["answers"]]
    if not rows:
        return []
    order = {"hybrid": 0, "vector": 1, "bm25": 2}
    rows.sort(key=lambda r: (r["config"]["strategy"], order[r["config"]["mode"]], r["config"]["reranker"] == "off",
                             r["config"].get("per_lang", "off") == "off", r["config"].get("rerank_top", 10)))
    out = [
        "| Chunking | Search | Vector per language | Reranker (pool) "
        "| Hit@1 | Hit@3 | Hit@5 | MRR | Hit@5 EN | Hit@5 AR | p50 | p95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c, s = r["config"], r["summary"]
        a = s["all"]
        pool = f" ({c.get('rerank_top', 10)}, top {c.get('per_list', 0)}/list)" if c["reranker"] == "on" else ""
        per_lang = c.get("per_lang", "off") if c["mode"] != "bm25" else "–"
        out.append(
            f"| {c['strategy']} | {c['mode']} | {per_lang} | {c['reranker']}{pool} | "
            f"{pct(a['hit@1'])} | {pct(a['hit@3'])} | "
            f"{pct(a['hit@5'])} | {num(a['mrr'])} | {pct(s['en']['hit@5'])} | {pct(s['ar']['hit@5'])} | "
            f"{sec(a['latency_p50'])} | {sec(a['latency_p95'])} |"
        )
    return out


def answer_table(runs: list[dict]) -> list[str]:
    rows = [r for r in runs if r["config"]["answers"]]
    if not rows:
        return []
    out = [
        "| Run | Model | Prompt | Chunking | Reranker | Answered (answerable) | Abstained (unanswerable) "
        "| Citation hit | Citation precision (strict / lenient) | Quotes verified | Faithfulness | p50 | p95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c, a = r["config"], r["summary"]["all"]
        out.append(
            f"| {r['name']} | {c['model'].replace('ollama:', '')} | {c.get('prompt', '–')} | "
            f"{c['strategy']} | {c['reranker']} | "
            f"{pct(a['answered_when_answerable'])} | {pct(a['abstained_when_unanswerable'])} | "
            f"{pct(a['cite_hit_all'])} | {pct(a['cite_precision'])} / {pct(a['cite_precision_lenient'])} | "
            f"{pct(a['quote_verified'])} | {pct(a['faithfulness'])} | "
            f"{sec(a['latency_p50'])} | {sec(a['latency_p95'])} |"
        )
    return out


def breakdown(runs: list[dict]) -> list[str]:
    """Default configuration split by question type and language."""
    by = {r["name"]: r["summary"] for r in runs}
    ret, ans = by.get("ret-article-hybrid-rerank-off"), by.get("ans-qwen7b-article")
    if not ret and not ans:
        return []
    out = [
        "| Subset | Retrieval Hit@5 | MRR | Answered (answerable) | Abstained (unanswerable) "
        "| Citation hit | Faithfulness |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in (("direct", "Direct questions"), ("scenario", "Scenario questions"),
                       ("en", "English"), ("ar", "Arabic"), ("all", "All")):
        r, a = (ret or {}).get(key, {}), (ans or {}).get(key, {})
        out.append(
            f"| {label} | {pct(r.get('hit@5'))} | {num(r.get('mrr'))} | {pct(a.get('answered_when_answerable'))} | "
            f"{pct(a.get('abstained_when_unanswerable'))} | {pct(a.get('cite_hit_all'))} | "
            f"{pct(a.get('faithfulness'))} |"
        )
    return out


def misses(runs: list[dict]) -> list[str]:
    """Questions the default configuration gets wrong, for error analysis."""
    out = []
    failed = [(r["name"], x["id"]) for r in runs for x in r["rows"] if x.get("error")]
    out.append(f"Failed calls (counted as abstentions) across all runs: {len(failed)}"
               + (" — " + ", ".join(f"{n}:{i}" for n, i in failed) if failed else ""))
    out.append("")
    for r in runs:
        if r["name"] == "ret-article-hybrid-rerank-off":
            bad = [x for x in r["rows"] if x["answerable"] and not x["hit@5"]]
            out.append("")
            out.append(f"Retrieval misses (gold not in top 5), default configuration: {len(bad)}")
            out += [f"- `{x['id']}` ({x['lang']}) {x['question']} — gold "
                    f"{', '.join(g['doc_id'] + ':' + g['canonical_key'] for g in x['gold'])}; "
                    f"got {', '.join(x['retrieved'][:3])}" for x in bad]
        if r["name"] == "ans-qwen7b-article":
            bad = [x for x in r["rows"] if not x["abstain_correct"]]
            out.append("")
            out.append(f"Wrong abstention decisions, default configuration: {len(bad)}")
            out += [f"- `{x['id']}` ({x['lang']}) {x['question']} — "
                    f"{'abstained but answerable' if x['abstained'] else 'answered but unanswerable'}" for x in bad]
    return out


def main() -> None:
    runs = load()
    qs = [json.loads(line) for line in (ROOT / "questions.jsonl").open(encoding="utf-8")]
    n_en = sum(q["lang"] == "en" for q in qs)
    n_un = sum(not q["answerable"] for q in qs)
    lines = [
        "# Evaluation results",
        "",
        f"Question set: {len(qs)} questions ({n_en} English, {len(qs) - n_en} Arabic); "
        f"{len(qs) - n_un} answerable, {n_un} unanswerable. Gold sources list every acceptable article/control. "
        "Status: draft gold set, pending expert review.",
        "",
        "Generated by `python -m eval.report` from `eval/runs/*.json`; reproduce with `bash eval/run_all.sh`.",
        "Hardware: RTX 2070 Super 8 GB (LLM + embeddings via Ollama), optional reranker on CPU (int8).",
        "",
        "## Retrieval",
        "",
        "Hit@k: a gold source is among the first k distinct articles/controls. MRR over the first 10. "
        "Latency: retrieval only, warm.",
        "",
        *retrieval_table(runs),
        "",
        "## Answers",
        "",
        "- Answered / Abstained: correct abstention behaviour on answerable / unanswerable questions.",
        "- Citation hit: at least one citation is a gold source "
        "(abstaining on an answerable question counts as a miss).",
        "- Citation precision: share of citations that are gold sources; lenient also accepts sibling controls "
        "of the same sub-domain.",
        "- Quotes verified: the model's quote was found in the cited source text.",
        "- Faithfulness: share of answer claims supported by the cited sources, judged by Qwen 2.5 7B "
        "(a weak judge, and the same model family as the system under test).",
        "- Latency: end to end (retrieval + generation), warm.",
        "",
        *answer_table(runs),
        "",
        "## Breakdown (default configuration: article chunks, hybrid search, no reranker, Qwen 2.5 7B)",
        "",
        "Scenario questions describe a company practice and ask whether it complies.",
        "",
        *breakdown(runs),
        "",
        "## Error analysis",
        "",
        *misses(runs),
        "",
        "## Caveats",
        "",
        "- The gold set is a draft written from the source texts, not yet reviewed by a compliance expert "
        "(`eval/questions_review.md`).",
        "- Prompt v2 and the scenario questions were developed while looking at these same questions, so the v2 "
        "numbers are optimistic; a held-out set is needed for an unbiased estimate.",
        "- 69 questions: one question moves a percentage by 1.4-1.7 points overall and by 6-11 points in the "
        "unanswerable and scenario subsets. Small differences between rows are not significant.",
        "- Faithfulness is judged by Qwen 2.5 7B, the same model family as the system under test.",
        "- The `-baseline` rows (and rows with \"Vector per language: off\") ran before per-language vector "
        "search, the per-list reranker pool and the inline-citation fallback were added. All other rows use "
        "the current code.",
        "- Generation runs at temperature 0, but GPU inference is not bit-for-bit deterministic: rerunning the "
        "same configuration moved citation hit by about 2 points.",
        "- Known failures the current default still has: it answers two out-of-scope questions (SAMA's framework, "
        "unpaid overtime) by applying unrelated controls and articles, and search misses the gold source for "
        "two scenarios whose wording shares nothing with the regulation (a cloud host in Germany vs "
        "\"transfer outside the Kingdom\"; cybersecurity reporting lines). An LLM scope check and query "
        "rewrite were tried with Qwen 7B and dropped: they rejected valid ECC questions and invented article "
        "numbers.",
        "- An English question now reaches Arabic-only text (ECC 2-7-3 has no English counterpart). Without the "
        "reranker it ranks first for `vd-03` and for a rephrased version; with the reranker on, the rephrased "
        "version ranks it 9th, outside the 5 sources the model sees.",
        "- Prompt v1 ran before the output size limits (`maxItems`/`maxLength`, `num_predict`) were added.",
        "",
    ]
    (ROOT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
