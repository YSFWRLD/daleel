# Roadmap

## Current plan
- [x] 1. Scaffold, Docker Compose (Postgres + pgvector), `/health`
- [x] 2. PDF extraction (Arabic repair, OCR) and article/control-aware chunking
- [x] 3. Embeddings (bge-m3), hybrid BM25 + vector retrieval, RRF, reranker
- [x] 4. Grounded answers with verified citations, `/ask`, chat page
- [ ] 5. Evaluation: ~50 bilingual questions, hit@5, MRR, citation accuracy, faithfulness, abstention, latency
- [ ] 6. CI (GitHub Actions), API Docker image, README

## Later phase (after evaluation)

### Chat history sidebar
A list of past conversations that can be reopened (questions, answers and citations), plus "New chat".
- Store conversations in the browser (localStorage): no server or database change, private to the user's
  machine. Lost if the browser's site data is cleared.
- Alternative if multi-device or multi-user history is ever needed: a `conversations` table in Postgres and
  `GET/POST /conversations` endpoints (needs user accounts to keep histories apart).
- Mostly `app/static/index.html`.

### Follow-up questions with conversation memory
Let questions like "what about sensitive data?" use the earlier turns.
- Rewrite the follow-up into a standalone question from the previous turns (one extra LLM call, ~1-2 s),
  then retrieve and answer as today. Retrieval must not run on the vague follow-up as written.
- Do it after evaluation, and add follow-up questions to the eval set to check it doesn't lower
  retrieval hit rate or citation accuracy.
- Touches `app/answer.py` (rewrite step), `/ask` (accept previous turns) and the chat page.
