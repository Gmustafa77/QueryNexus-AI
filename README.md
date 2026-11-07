# Aetna Insurance Virtual Assistant — Production RAG

A grounded question-answering assistant for insurance plan PDFs (EOC, SBC, manuals).
Every answer is generated **only** from the uploaded documents and cites the
**document, page and section** it came from. If the evidence is not good enough,
the assistant says so and does not guess.

## Architecture

```
PDF upload ─► PDFLoader ─► DocumentChunker ─► EmbeddingService ─► ChromaManager (persistent)
             validate,      section + token     batched, cached,    deterministic IDs,
             headings,      aware, metadata     retried, normalised  dedup, version replace
             header/footer
             removal

Question ─► validate + injection check
         ─► Retriever: vector top-N  +  BM25 top-N  ─► weighted RRF fusion
                     ─► optional cross-encoder rerank ─► relevance gate (no-answer if weak)
         ─► build_context: dedupe, token budget, labels [S1]..[Sn]
         ─► LLM (strict grounded prompt, timeout + retry)
         ─► verify_answer: drop fabricated labels, check figures vs evidence,
                           withhold uncited answers (STRICT_CITATIONS)
         ─► UI: answer + source cards (file · page · section · scores) + warnings
```

| Package | Module | Responsibility |
|---|---|---|
| `ingestion/` | `pdf_loader.py` | Validation, page text, font-based heading hints, header/footer removal, file hash |
| `chunking/` | `text_splitter.py` | Section-aware, token-aware chunks; section carried across pages; deterministic IDs |
| `embeddings/` | `embedding_service.py` | OpenAI / Azure / Gemini; batching, SQLite cache, retries, L2 normalisation |
| `vectorstore/` | `chroma_manager.py` | Chroma persistence (cosine), upsert, dedup, update/delete by document, filters |
| `retrieval/` | `retriever.py`, `bm25.py`, `reranker.py` | Hybrid search, RRF fusion, thresholds, optional reranking |
| `prompts/` | `prompt_template.py` | Grounded system prompt, escaped evidence blocks with citation labels |
| `rag/` | `rag_pipeline.py`, `grounding.py` | Orchestration; context building; citation and figure verification |
| `utils/` | `config.py`, `logger.py`, `retry.py`, `text.py` | Env config, structured logs with request IDs, backoff, tokens |
| `evaluation/` | `evaluate.py` | Retrieval and answer-quality metrics |
| root | `app.py` | Streamlit UI (`frontend/streamlit_ui.py` delegates to it) |

## Setup (Windows, uv)

```powershell
cd C:\Users\<you>\Downloads\rag_app
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
copy .env.example .env      # then fill in keys for your provider
uv run streamlit run app.py
```

With pip: `python -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements.txt`.

### Upgrading from the previous version

* Set `CHROMA_COLLECTION_NAME` to a **new** name (e.g. `aetna_insurance_docs_v2`) and
  re-upload your PDFs. New collections use cosine distance and the richer metadata.
  An old collection still works (L2 distances are converted), but it has no
  `document_id`/section metadata, so dedup and section citations are weaker.
* `CHUNK_SIZE`/`CHUNK_OVERLAP` (characters) still work; prefer `CHUNK_SIZE_TOKENS`/`CHUNK_OVERLAP_TOKENS`.
* `RetrievedChunk.similarity_score` is now **cosine similarity (higher = better)**.
* `RAGPipeline.ask(question)` and `ingest_documents(paths) -> int` are unchanged.
  New: `ask(question, filters={"source_file": [...]}, top_k=8)`, `ingest_with_report(paths)`,
  `list_documents()`, `delete_document(document_id)`.

## Behaviour

* **Re-upload unchanged file** → skipped (content hash). **Changed file, same name** →
  the new version is indexed first, then the old version's chunks are deleted.
* **No/weak evidence** → fixed no-answer message, **no LLM call**.
* **Model answer without valid citations** → withheld when `STRICT_CITATIONS=true`.
* **Figures in the answer that are not in the cited sources** → warning shown to the user.
* **Prompt injection** → evidence is XML-escaped and treated as data. Suspicious questions
  are flagged and logged by hash only. Questions are never logged in plain text at INFO level.
* **tiktoken offline** (corporate proxy) → falls back to approximate token counts automatically.

## Tests

```powershell
uv run pytest
```

39 offline tests (no API keys needed) cover the config, PDF loader, chunker, embedding
cache/batching, BM25, filters, prompt escaping, grounding verification, retry,
ingestion dedup/update, hybrid retrieval, no-answer gating, metadata filters, the
evaluation harness, and a headless Streamlit UI run.

## Evaluation

Create a JSONL file of real questions from your documents (see `evaluation/sample_eval.jsonl`):

```json
{"id": "deductible", "question": "What is the in-network deductible?", "answerable": true,
 "expected_sources": [{"source_file": "EOC_2026.pdf", "page_number": 12}],
 "expected_answer_contains": ["$1,500"]}
```

```powershell
uv run python -m evaluation.evaluate --dataset evaluation\my_eval.jsonl --docs "data\pdfs\*.pdf" --output evaluation\report.json --min-hit-rate 0.85
```

| Metric | Meaning |
|---|---|
| `retrieval_hit_at_k`, `retrieval_mrr`, `retrieval_recall_at_k` | Was the expected page retrieved, and how high was it ranked |
| `answer_accuracy` | Expected key facts present in the answer |
| `citation_precision` | Share of citations that point to an expected page |
| `faithfulness` | Share of answer words supported by the cited evidence (lexical proxy) |
| `groundedness_rate` | Answers with valid citations and no unsupported figures |
| `no_answer_accuracy` | Refuses unanswerable questions, answers answerable ones |

**Tuning loop:** run the evaluation → adjust `MIN_SIMILARITY` (raise it if out-of-scope
questions get answered, lower it if valid questions are refused), `RETRIEVAL_TOP_K`,
`CHUNK_SIZE_TOKENS`, and try `RERANKER=cross_encoder` → re-run and compare.

## Known limits

* Scanned/image-only PDFs need OCR before upload (they are reported as "no extractable text").
* Tables are extracted as reading-order text; complex multi-column tables may lose structure.
* Each question is answered independently (no follow-up memory), which avoids context
  carried over from earlier answers into new ones.

## Answer cache (exact + semantic)

`RAGPipeline.ask()` checks a persistent cache (`data/answer_cache.sqlite3`) before retrieval and the LLM.
No UI changes. A hit skips retrieval, the LLM call, and (for exact repeats) the embedding call.

* **Exact match**: same question after lower-casing / whitespace / trailing-punctuation normalisation.
* **Semantic match**: cosine similarity >= `ANSWER_CACHE_SIMILARITY_THRESHOLD` (default 0.93) **and** a guard
  passes: both questions must contain the same numbers and the same meaning-flipping words (in-network vs
  out-of-network, not, family vs individual, copay vs coinsurance vs deductible...). Edit `QUALIFIERS` in
  `caching/answer_cache.py` for your documents. Near-matches rejected by the guard are counted and logged.
* **Scope**: entries only match when model, prompt/grounding code, retrieval and chunking settings, source mode,
  document filter, Top-K and a *content fingerprint of the indexed documents* are all identical. Ingesting,
  replacing or deleting a document therefore can never serve a stale answer, even after a restart or when
  another process changed the store (within `DOC_LIST_CACHE_TTL_SECONDS`).
* **Invalidation**: scope change (above), explicit clear on ingest/delete/clear in this process, fixed TTL from
  creation (`ANSWER_CACHE_TTL_SECONDS`, default 24h), LRU eviction (`ANSWER_CACHE_MAX_ENTRIES`, default 1000).
* **Only cached**: grounded answers with verified citations and no warnings. Never cached: no-answers, errors,
  partial-source (hybrid fallback) answers, prompt-injection-flagged questions.
* **Monitoring**: log lines `Answer cache HIT|MISS|STORE|BYPASS|INVALIDATED` (question text is never logged, only a
  hash prefix; misses log the best similarity so you can tune the threshold), `pipeline.answer_cache_stats()`,
  `RAGResponse.cache_status`, per-stage `timings_ms` in the "Answered question" log line, and
  `python -m scripts.cache_stats`.
* **Disable**: `ANSWER_CACHE_ENABLED=false`, or `ANSWER_CACHE_SEMANTIC_ENABLED=false` for exact-only.
* The cache file stores question text in plain text. Treat `data/` as sensitive. Evaluation runs with the cache off.
