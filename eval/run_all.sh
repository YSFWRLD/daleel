#!/usr/bin/env bash
# Run every experiment in eval/results.md. Needs Postgres + loaded chunks and Ollama with bge-m3,
# qwen2.5:7b-instruct-q4_K_M and qwen2.5:1.5b. Retrieval runs take ~1-3 min each, answer runs ~10 min.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/Scripts/python.exe}"
run() { "$PY" -X utf8 -m eval.run_eval "$@" 2>&1 | grep -E "ERROR|Traceback|Error:|^wrote"; }

# Retrieval: chunking x search mode x reranker
for strategy in article fixed; do
  for mode in hybrid vector bm25; do
    for rr in on off; do
      run --name "ret-$strategy-$mode-rerank-$rr" --strategy "$strategy" --mode "$mode" --reranker "$rr"
    done
  done
done

# Answers (faithfulness judged by Qwen 2.5 7B)
# Default: no reranker. The chunking and model ablations ran with the reranker on.
run --name ans-qwen7b-article --answers --judge
run --name ans-qwen7b-article-rerank --answers --judge --reranker on
run --name ans-qwen7b-fixed --answers --judge --strategy fixed --reranker on
run --name ans-qwen1.5b-article --answers --judge --model qwen2.5:1.5b --reranker on

"$PY" -X utf8 -m eval.report
