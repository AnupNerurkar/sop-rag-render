# EduMind

**A role-aware agentic retrieval-augmented assistant for institutional SOPs, policies, and circulars in engineering colleges.**

EduMind answers questions only from an institution's own documents and shows where each answer came from. It runs dense retrieval over BGE embeddings next to BM25 lexical retrieval, merges the two ranked lists with Reciprocal Rank Fusion (RRF), and can reorder the result with a cross-encoder. Access control is a metadata filter inside retrieval, so a chunk a user may not see never reaches the language model. A LangGraph workflow classifies the query, plans sub-queries, validates context, generates a cited answer, scores confidence from retrieval evidence, and retries with a broader query when confidence falls below 0.40.

This repository is the source code accompanying the paper *EduMind: A Role-Aware Agentic Retrieval-Augmented Assistant for Institutional SOPs, Policies, and Circulars in Engineering Colleges* (Department of Computer Engineering, Sardar Patel Institute of Technology, Mumbai).

**Authors:** Anup Nerurkar, Jaynish Thakar, Aditya Malekar, Nipun Alwala, Prithvi Panchal. **Guides:** Dr. Kiran Talele, Dr. Saurabh Mehta.

## Why

Colleges run on SOPs for admissions, examinations, fee refunds, procurement, placements, and recruitment, but the files sit in departmental folders, the website, email threads, and the ERP, often in several versions. Keyword search misses everyday phrasing ("can I sit for the exam with 70% attendance?" against a regulation about "eligibility for term-end examination"), and a general chatbot answers from the internet, not the college.

A two-round survey of 95 respondents (76 students, 13 faculty, 6 student office bearers) confirmed the need:

- mean ease of finding accurate procedural information today: **2.57 / 5**
- **48.4%** need a day or more to get a clear procedural answer; only 20.0% get one within minutes
- **42.1%** start by asking a friend or senior or searching email/WhatsApp groups
- **66.3%** said they were likely to use such a tool (faculty: 76.9%)
- most valued features: a direct link to the official document and deadline information (41.1% each)

Those results drove four design choices: hybrid retrieval, a citation on every answer that opens the source document, version supersession so only the current version of a policy can be cited, and an approval workflow so committees can keep SOPs current.

## Features

- **Grounded, cited answers.** The model sees numbered `[SOURCE N]` blocks and must cite each claim. Citations are resolved deterministically from the context the model saw, never from the model, and show document title, department, section, and version.
- **Hybrid retrieval.** BGE dense search and BM25 lexical search fused with RRF (k = 60), with optional cross-encoder reranking. Each stage is switchable per query.
- **RBAC inside retrieval.** Four roles (Public, Student, Faculty, Admin). The role from a verified JWT becomes a `where` filter on the vector query, so restricted chunks are never candidates. The document viewer re-checks access on every request.
- **Retrieval-based confidence.** Confidence is computed from the top retrieved chunk's score before generation, so the model never grades itself. It is shown as a percentage with a hallucination-risk label.
- **LangGraph workflow** with a confidence-gated reflection loop (τ = 0.40, one retry), exposed at `/api/agent/chat`.
- **Incremental ingestion** of `.doc`, `.docx`, and `.pdf`: SHA-256 duplicate detection, table-aware extraction, version supersession within a department, structure-aware chunking.
- **Committee workflow.** Committee leads submit SOPs; nothing is chunked, embedded, or indexed until an administrator approves it.
- **Account approval.** New Student and Faculty accounts start pending and cannot sign in until an administrator approves them.
- **Pluggable generation backend** via `LLM_BACKEND`: a local Ollama server (Qwen2.5-7B) or the Hugging Face Inference API (Qwen2.5-32B-Instruct).
- **Streaming chat** over SSE, with citations and confidence delivered when generation finishes.

## Architecture

The system has four planes. Ingestion runs only when an administrator uploads or approves a document, so nothing on the query path re-chunks or re-embeds.

```
Serving plane      Browser SPA (HTML/CSS/JS) ── FastAPI + JWT auth (REST, SSE) ── pluggable LLM backend
                                  │ question + role                 ▲ cited answer + confidence
Retrieval &        role → permitted access levels (metadata filter)
reasoning plane      ├─ dense search (BGE, cosine) ─┐
                     └─ BM25 search (lexical) ──────┴─ RRF (k = 60) ─ cross-encoder rerank ─ LangGraph agents
                                                                                              analyse → plan → retrieve
                                                                                              → validate → generate
                                                                                              → cite → score
                                                                                              (reflect & retry if low)
Ingestion plane    upload / approved committee SOP → SHA-256 check → extraction → cleaning + metadata
                   → structure-aware chunking (1,200 / 150) → BGE embedding (768-d)
Storage plane      vector store (vectors + chunk metadata incl. access level)
                   SQLite ledger (hashes, versions, audit log, users, chat history)
```

### LangGraph workflow

| Node | Behaviour |
|---|---|
| Query Analyzer | Answers greetings and declines clearly out-of-scope questions without an LLM call; resolves short follow-ups from conversation memory; extracts department, category, and version hints. |
| Planner | Splits comparative or conjunctive questions into up to three retrieval steps. |
| Retrieval Agent | Runs role-filtered hybrid retrieval for each plan step and merges the results. |
| Context Validator | Removes duplicates by chunk id and text prefix, drops unnamed sources, applies a 0.15 relevance floor without emptying the set, keeps at most six chunks. |
| Response Generator | Builds the grounded prompt and calls the LLM; returns a fixed fallback without calling the model when context is empty. |
| Citation Formatter | Maps `[SOURCE n]` markers to citations, preferring the newest version per (title, department) and dropping orphan markers. |
| Confidence Evaluator | Scores confidence; finalises if ≥ 0.40 or the retry budget is spent, otherwise routes to Reflection. |
| Reflection Agent | Strips question phrasing, clears metadata filters, issues one broadened single-step plan. |
| Final Response | Stores the turn in bounded memory (20 turns) and adds a note when support is weak. |

## Algorithms and parameters

The paper gives each stage as pseudocode (Algorithms 1–13). The values used in deployment:

| Stage | Method | Parameters |
|---|---|---|
| Ingestion | SHA-256 of raw bytes; version supersession within a department | unchanged file is a no-op; superseded versions lose their vectors |
| Chunking | `[SUB PROCESS]` / `[PROCESS: …]` markers as hard boundaries, recursive split only for long blocks | S_max = 1,500, target 1,200, overlap 150 characters |
| Dense retrieval | `BAAI/bge-base-en-v1.5`, L2-normalised, cosine | 768-d; query prefix `Represent this sentence for searching relevant passages: ` |
| Lexical retrieval | BM25 Okapi | k1 = 1.5, b = 0.75, negative IDF floored at 0.25 × mean IDF |
| Fusion | Reciprocal Rank Fusion, `RRF(d) = Σ 1 / (k + r_i(d))` | k = 60 |
| Reranking | `BAAI/bge-reranker-base` cross-encoder, sigmoid of the logit | switchable per query |
| Confidence | `round(100 · s_top)`, reranker score if reranked, else cosine | reflection threshold τ = 0.40, retry budget β = 1 |
| Prompt | numbered `[SOURCE N]` blocks, cut at a sentence boundary | 8,000-character budget |

| Path | k_dense | k_bm25 | k_fused | k_final | prompt chunks |
|---|---|---|---|---|---|
| Chat (`/api/chat/*`) | 10 | 10 | 10 | 3 | 3 |
| Agent (`/api/agent/chat`) and retriever defaults | 25 | 25 | 25 | 5 | 6 |

Role-to-access mapping: Admin reads all levels; Faculty reads Public, Student, Faculty; Student reads Public, Student; Public reads Public only. An unauthenticated request is treated as Public.

## Results

Measured on 21 SOP documents (634 chunks) from one engineering college, CPU-only (Intel Core i9-13900HX, 32 GB RAM), with the configuration described above.

### Retrieval (20 queries, K = 5)

Relevance is a department-match proxy, since the corpus has no human relevance labels.

| Configuration | P@5 | R@5 | MRR | Hit@1 | Hit@3 | Mean (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|
| Dense only | 0.680 | 0.149 | 0.833 | 0.700 | 1.000 | 514* | 9,481* |
| Hybrid (dense + BM25 + RRF) | **0.710** | **0.158** | 0.827 | 0.700 | 0.950 | **41** | **115** |
| Hybrid + cross-encoder | 0.620 | 0.143 | 0.733 | 0.550 | 0.950 | 2,710* | 6,910* |

\* includes a one-time model load on the first query.

Hybrid retrieval raised Precision@5 at a steady-state cost of about 41 ms per query. Reranking lowered the proxy metrics and added about 2.5 s per query; the department proxy penalises exactly the cross-department promotions a reranker makes, so this shows no evidence of benefit on this corpus rather than proof that reranking is useless.

### End-to-end answers (46 questions)

| Metric | Result |
|---|---|
| Average correctness | 1.63 / 2.00 (95% CI 1.43–1.83) |
| Fully correct | 34 (73.9%) |
| At least acceptably correct | 41 (89.1%) |
| Average groundedness | 1.46 / 2.00 (95% CI 1.23–1.68) |
| Fully grounded | 28 (60.9%) |
| Citation accuracy | 32 (69.6%) |

Every out-of-scope question was declined without invented content. The most common failure was a correct claim cited to a chunk that did not specifically support it (14 of 46): the retriever found the right document but not always the exact passage. The agentic layer has not yet been evaluated on its own, so no claim is made that the reflection loop improves answers.

## This repository: the hosted build

The code here is the hosted (Render) build of EduMind. To fit a small CPU-only web service it substitutes some components of the evaluated configuration:

| Evaluated configuration | This repository |
|---|---|
| ChromaDB collection (approximate nearest neighbour) | SQLite-backed vector store with exact cosine search (`vector_store/sqlite_store.py`) |
| In-memory `rank_bm25` index | SQLite FTS5 with its built-in `bm25()` ranking (`retrieval/fts5.py`) |
| Local `bge-reranker-base` cross-encoder | API-based listwise reranker (`retrieval/rerank_client.py`) |
| Confidence from the top result | Confidence averages the top result with the top three |

The algorithms, RRF fusion, role filter, LangGraph workflow, and citation logic are the same in both builds. All results above were measured on the evaluated configuration, not on this build.

## Repository layout

| Module | Files |
|---|---|
| Ingestion, ledger, chunking | `ingestion_pipeline.py`, `ledger.py`, `chunker.py` |
| Embedding (BGE) | `embeddings/embedder.py`, `embeddings/embed_pipeline.py` |
| Vector store and indexing | `vector_store/sqlite_store.py`, `vector_store/index_pipeline.py` |
| Keyword index, hybrid retrieval, RRF | `retrieval/fts5.py`, `retrieval/hybrid_search.py`, `retrieval/retriever.py`, `retrieval/fusion.py` |
| Role filter | `retrieval/filters.py` |
| Reranking | `retrieval/reranker.py`, `retrieval/rerank_client.py` |
| Prompting, generation, citations | `rag/prompt_builder.py`, `rag/rag_engine.py`, `rag/citation_engine.py` |
| LangGraph workflow | `agents/multi_agent_graph.py`, `agents/agent_state.py` |
| JWT and RBAC; committee workflow | `backend/auth.py`, `backend/signup_policy.py`, `backend/committee_manager.py` |
| API and SPA | `backend/app.py`, `frontend/app.js` |
| Evaluation scripts | `scripts/evaluate_retrieval.py`, `scripts/evaluate_answers.py` |
| Tests | `tests/` |

## Running it

### Local

```bash
pip install -r requirements.txt

# .env
# LLM_BACKEND=hf                       # or ollama for a local Qwen2.5-7B
# HF_API_TOKEN=...
# HF_MODEL=Qwen/Qwen2.5-32B-Instruct
# JWT_SECRET_KEY=any-long-random-string

python seed_users.py                   # creates demo accounts, one per role
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1
# open http://localhost:8000
```

### On premises with Docker Compose

The same image runs fully on premises next to a local Ollama container serving Qwen2.5-7B, so no document text leaves the institution:

```bash
docker compose -f docker/docker-compose.yml up --build
# app → http://localhost:8000, Ollama → http://localhost:11434
```

### Hosted (Render)

`Dockerfile` and `render.yaml` package EduMind as one image. The vector index is committed to the repository and copied into the image at build time, so a new container answers immediately without an external database. A Render web service has no GPU and no room for an Ollama server, so the hosted deployment sets `LLM_BACKEND=hf` and calls the Hugging Face Inference API; set `HF_API_TOKEN` as a secret. This means retrieved SOP text leaves the institution; a college that cannot accept that should use the Docker Compose setup instead. On the free plan, documents uploaded at runtime do not survive a spin-down or redeploy.

## Tests

```bash
pytest                   # full suite
pytest -m "not slow"     # skip tests that need the embedding model or a live vector store
python scripts/evaluate_retrieval.py
python scripts/evaluate_answers.py
```

## Limitations

- The retrieval benchmark is small (20 queries) and uses a department-match proxy; 31 of the 46 end-to-end questions were scored by a single annotator.
- The four-role model does not express per-committee or per-department permissions, and the evaluated corpus has no Faculty or Admin chunks, so RBAC is tested only on the levels present.
- When versions conflict the system flags the conflict but does not decide which is correct; an uploaded SOP is assumed authoritative.
- Corpus, embedding model, and prompts are English-only.
- Legacy `.doc` files were opened through Microsoft Word COM automation on a Windows ingestion host, since python-docx cannot read them.

## Future work

Human relevance labels to settle the reranking question; a larger double-annotated answer set with chance-corrected agreement; calibrating confidence against human judgements; evaluating the reflection loop in isolation; step-by-step checklists with required approvals, explicit version dates, and college-wide vs department-level labels (requested by faculty); a mobile-friendly interface, deadline and contact extraction, voice input, and Indian-language support (requested by students); and connecting ingestion to the college ERP.

## Data availability

The SOP corpus (Vidyalankar Institute of Technology, Mumbai) consists of confidential internal records of the institution and is not released as a dataset with the paper. Any documents bundled with the hosted demo remain the institution's property and are not licensed for reuse. The pipeline does not depend on this college: any institution can reproduce the setup by ingesting its own SOPs. Survey responses were anonymous and are reported only in aggregate.
