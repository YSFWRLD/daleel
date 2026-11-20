"""Run the evaluation set and write one result file per configuration to eval/runs/<name>.json.

Retrieval-only (fast, no LLM):
    python -m eval.run_eval --name hybrid-rerank --strategy article --mode hybrid --reranker off
Full answers (+ optional LLM-judge faithfulness):
    python -m eval.run_eval --name qwen7b --answers --judge
    python -m eval.run_eval --name qwen1.5b --answers --model qwen2.5:1.5b --judge
Then build the tables:  python -m eval.report

Needs the same services as the API: Postgres with loaded chunks and Ollama (bge-m3 + the LLM).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from eval.metrics import citation_scores, first_rank, matches, mean, percentile, retrieval_scores

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "questions.jsonl"
RUNS = ROOT / "runs"

JUDGE_SYSTEM = """You are a strict fact checker for answers about regulations.
Split the ANSWER into its individual factual claims (ignore citation markers like [S1] and ignore
statements that only say the information comes from a source). For each claim decide whether the SOURCES
explicitly support it. A claim that adds details, numbers, conditions or conclusions not stated in the
sources is NOT supported. Return JSON only."""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"],
            },
        }
    },
    "required": ["claims"],
}


def load_questions(ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    qs = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8") if line.strip()]
    if ids:
        qs = [q for q in qs if q["id"] in ids]
    return qs[:limit] if limit else qs


def run_retrieval(q: dict, args) -> dict:
    from app.retrieval.hybrid import chunk_keys, retrieve

    t0 = time.perf_counter()
    r = retrieve(q["question"], strategy=args.strategy, top_k=10, use_reranker=args.reranker == "on", mode=args.mode)
    latency = time.perf_counter() - t0
    ranked = [(h.chunk["doc_id"], chunk_keys(h.chunk)) for h in r.hits]
    rank = first_rank(ranked, q["gold"]) if q["answerable"] else None
    return {
        "retrieved": [f"{d}:{k[0]}" for d, k in ranked[:5]],
        "rank": rank,
        **(retrieval_scores(rank) if q["answerable"] else {}),
        "latency": latency,
        "timings": r.timings,
    }


def judge(llm, answer: dict, lang: str) -> dict:
    """Faithfulness: share of the answer's claims that the cited sources support."""
    blocks = []
    for c in answer["citations"]:
        text = (c["text_en"] if lang == "en" else c["text_ar"]) or c["text_en"] or c["text_ar"] or ""
        blocks.append(f"[{c['id']}] {c['title']}\n{text}")
    text = re.sub(r"\[S\d+\]", "", answer["answer"])
    sources = "\n\n".join(blocks)
    user = f"SOURCES:\n\n{sources}\n\nANSWER:\n{text}"
    try:
        out = llm.complete_json(JUDGE_SYSTEM, user, JUDGE_SCHEMA)
    except Exception as e:  # noqa: BLE001 - a judge failure should not lose the run
        return {"faithfulness": None, "judge_error": str(e)}
    claims = [c for c in out.get("claims") or [] if str(c.get("claim", "")).strip()]
    if not claims:
        return {"faithfulness": None, "claims": []}
    return {"faithfulness": sum(bool(c.get("supported")) for c in claims) / len(claims), "claims": claims}


def run_answer(q: dict, args, llm) -> dict:
    from app.answer import answer_question

    t0 = time.perf_counter()
    a = answer_question(q["question"], llm=llm, strategy=args.strategy, use_reranker=args.reranker == "on",
                        mode=args.mode).to_dict()
    latency = time.perf_counter() - t0
    cited = [(c["doc_id"], c["keys"] or [c["canonical_key"]]) for c in a["citations"]]
    out = {
        "answer": a["answer"],
        "abstained": a["abstained"],
        "abstain_correct": a["abstained"] != q["answerable"],
        "citations": [f"{d}:{k[0]}" for d, k in cited],
        "citation_ok": [matches(d, k, q["gold"]) for d, k in cited],
        "quote_verified": [c["quote_verified"] for c in a["citations"] if c["quote"]],
        "latency": latency,
        "timings": a["timings"],
        "_full": a,  # for the judge; dropped before saving
    }
    if q["answerable"] and not a["abstained"]:
        out.update(citation_scores(cited, q["gold"]))
    return out


def summarize(rows: list[dict], answers: bool) -> dict:
    ans = [r for r in rows if r["answerable"]]
    s: dict = {"n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(rows) - len(ans)}

    def block(sub: list[dict]) -> dict:
        b: dict = {}
        a_sub = [r for r in sub if r["answerable"]]
        if not answers:
            for m in ("hit@1", "hit@3", "hit@5"):
                b[m] = mean([r[m] for r in a_sub])
            b["mrr"] = mean([r["rr"] for r in a_sub])
        else:
            u_sub = [r for r in sub if not r["answerable"]]
            b["answered_when_answerable"] = mean([float(not r["abstained"]) for r in a_sub])
            b["abstained_when_unanswerable"] = mean([float(r["abstained"]) for r in u_sub])
            b["abstention_accuracy"] = mean([float(r["abstain_correct"]) for r in sub])
            b["cite_precision"] = mean([r.get("cite_precision") for r in a_sub])
            b["cite_precision_lenient"] = mean([r.get("cite_precision_lenient") for r in a_sub])
            b["cite_hit"] = mean([r.get("cite_hit") for r in a_sub if not r["abstained"]])
            b["cite_hit_all"] = mean([r.get("cite_hit", 0.0) for r in a_sub])  # abstaining counts as a miss
            b["cite_recall"] = mean([r.get("cite_recall") for r in a_sub if not r["abstained"]])
            b["quote_verified"] = mean([float(v) for r in sub for v in r["quote_verified"]])
            b["faithfulness"] = mean([r.get("faithfulness") for r in sub])
        lat = [r["latency"] for r in sub]
        b["latency_p50"], b["latency_p95"] = percentile(lat, 50), percentile(lat, 95)
        return b

    s["all"] = block(rows)
    for lang in ("en", "ar"):
        s[lang] = block([r for r in rows if r["lang"] == lang])
    s["scenario"] = block([r for r in rows if "scenario" in r["tags"]])
    s["direct"] = block([r for r in rows if "scenario" not in r["tags"]])
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--strategy", choices=["article", "fixed"], default="article")
    p.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    p.add_argument("--reranker", choices=["on", "off"], default="off")
    p.add_argument("--per-lang", choices=["on", "off"], default="on", help="vector search per language")
    p.add_argument("--rerank-top", type=int, default=None, help="fused candidates sent to the reranker")
    p.add_argument("--per-list", type=int, default=None, help="each retriever's top results kept in the pool")
    p.add_argument("--answers", action="store_true", help="generate answers (otherwise retrieval only)")
    p.add_argument("--provider", choices=["ollama", "gemini"], default="ollama")
    p.add_argument("--model", default=None, help="LLM model override, e.g. qwen2.5:1.5b")
    p.add_argument("--judge", action="store_true", help="score faithfulness with an LLM judge")
    p.add_argument("--judge-model", default=None, help="judge model (default: the configured Ollama model)")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--sleep", type=float, default=0.0, help="pause between questions (API rate limits)")
    args = p.parse_args()

    from app.config import get_settings
    from app.llm.base import get_llm

    settings = get_settings()  # cached: overrides apply to retrieve() below
    settings.vector_per_lang = args.per_lang == "on"
    if args.rerank_top:
        settings.rerank_top = args.rerank_top
    if args.per_list is not None:
        settings.rerank_per_list = args.per_list
    qs = load_questions(args.ids, args.limit)
    llm = get_llm(args.provider, args.model) if args.answers else None
    # load bge-m3, the BM25 index, the reranker and the LLM first so early latencies are not inflated
    from app.retrieval.hybrid import get_index, warm_up

    warm_up()
    get_index(args.strategy)
    if args.answers:
        llm.complete_json("Reply with JSON.", "ping", {"type": "object", "properties": {}})
    rows = []
    for i, q in enumerate(qs, 1):
        base = {k: q[k] for k in ("id", "lang", "question", "answerable", "gold", "tags")}
        try:
            res = run_answer(q, args, llm) if args.answers else run_retrieval(q, args)
        except Exception as e:  # noqa: BLE001 - one failed call is recorded, not allowed to lose the run
            print(f"[{i}/{len(qs)}] {q['id']} ERROR {type(e).__name__}: {e}", flush=True)
            res = {"error": f"{type(e).__name__}: {e}", "latency": None, "timings": {}, "rank": None,
                   "retrieved": [], "abstained": True, "abstain_correct": not q["answerable"], "citations": [],
                   "citation_ok": [], "quote_verified": [], "_full": None}
            if not args.answers and q["answerable"]:
                res.update(retrieval_scores(None))
        rows.append({**base, **res})
        if res.get("error"):
            continue
        mark = ("abstain" if res["abstained"] else ",".join(res["citations"])) if args.answers else res["rank"]
        print(f"[{i}/{len(qs)}] {q['id']} {res['latency']:.1f}s -> {mark}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    if args.answers and args.judge:
        judge_llm = get_llm("ollama", args.judge_model)
        for r in rows:
            if not r["abstained"]:
                r.update(judge(judge_llm, r["_full"], r["lang"]))
                print(f"judge {r['id']}: {r.get('faithfulness')}", flush=True)
    for r in rows:
        r.pop("_full", None)

    config = {k: getattr(args, k) for k in ("strategy", "mode", "reranker", "answers", "provider", "model", "judge")}
    config.update(per_lang=args.per_lang, rerank_top=settings.rerank_top, per_list=settings.rerank_per_list)
    if args.answers:
        from app.answer import PROMPT_VERSION

        config["model"] = llm.name
        config["prompt"] = PROMPT_VERSION
        config["judge_model"] = get_llm("ollama", args.judge_model).name if args.judge else None
    summary = summarize(rows, args.answers)
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{args.name}.json"
    out.write_text(json.dumps({"name": args.name, "config": config, "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["all"], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
