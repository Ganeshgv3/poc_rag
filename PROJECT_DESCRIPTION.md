# Project Description - RAG PDF Chat App

## Overview

This repository is a Retrieval-Augmented Generation (RAG) app for chatting with uploaded PDFs.
It provides a modern API + web client flow and a legacy Streamlit flow.

- Current app: `api.py` (FastAPI) + `frontend/` (React + Vite)
- Legacy app: `app.py` (Streamlit; chat path uses the same `rag_pipeline` / LangGraph flow as the API)

**Orchestration modules:** `rag_pipeline.py` (LangGraph graph + LangChain Ollama), `rag_routing.py` (small-talk / binary / not-found / `clean_answer_text`), `rag_agentic.py` (optional agentic RAG: post-retrieval grading + follow-up hybrid search), `prompts.py` (system and user prompt assembly, question normalization, retrieval query variants).

At a high level, the app:
1. uploads and stores PDFs,
2. extracts and chunks text,
3. embeds chunks with Sentence Transformers,
4. stores vectors in a configured vector backend,
5. retrieves relevant chunks for each question (hybrid dense + BM25 + RRF when enabled),
6. optionally refines retrieval (**agentic RAG**, when enabled) using a short Ollama “grader” pass and merged follow-up queries,
7. routes each turn through a **LangGraph** state machine and, when an LLM answer is needed, calls **Ollama** via **LangChain** (`ChatOllama` + LCEL) for grounded generation.

**Small-talk** is answered on a dedicated graph branch **before** retrieval, so those turns skip vector search and reduce latency.

## Current architecture

### 1) Ingestion

- PDF files are uploaded from UI and stored under `data/uploads`.
- Text is extracted using PyMuPDF (`fitz`) via `text_chunking.extract_text`.
- Text is normalized (line merge, whitespace cleanup, de-hyphenation) and chunked with overlap.
- Chunking is tunable with environment variables:
  - `PDF_CHUNK_SIZE` (default around 1100)
  - `PDF_CHUNK_OVERLAP` (default around 180)

### 2) Embedding and vector store

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Vector backend is selected through env:
  - `VECTOR_BACKEND=chroma` (default, local persistent store under `data/chroma`)
  - `VECTOR_BACKEND=qdrant` (Qdrant via `QDRANT_URL`, optional `QDRANT_API_KEY`)
- Vector helper logic is centralized in `vector_store.py` (Chroma client or Qdrant adapter; adapter exposes Chroma-like `query` / `get` for hybrid retrieval).
- `chroma_helpers.py` is a compatibility shim re-exporting vector helpers.

### 3) Retrieval and answer generation

- **Hybrid retrieval** (default): dense vector similarity plus **BM25** lexical scoring, merged with **reciprocal rank fusion (RRF)**. Implemented in `hybrid_retrieval.py` (`retrieve_with_hybrid`); invoked from the LangGraph **retrieve** node in `rag_pipeline.py` (also used by the legacy Streamlit path through the same pipeline). Requires the `rank-bm25` package; if hybrid is disabled or the dependency is missing, the app falls back to dense-only search.
- Dense search uses the same embedding model and vector backend as before, with a larger candidate pool before fusion; final context size remains the same `top_k` passed into retrieval (e.g. 3 for API chat; Streamlit exposes a slider).
- Tunable via env: `HYBRID_SEARCH_ENABLED`, `HYBRID_RRF_K`, `HYBRID_DENSE_POOL`, `HYBRID_SPARSE_POOL`, `HYBRID_MAX_TOTAL_CHUNKS` (see `.env.example`).
- **Question normalization (`prompts.expand_question_shorthand`)** runs in the graph’s **expand** step before retrieval. It handles `&` → “and”, spaced `/` → “or”, and maps common **procedural openers** (case-insensitive) to a canonical **“how to ”** prefix—for example “guide me to …”, “show me how to …”, “walk me through …”, “help me to …”, “steps to …”, “how do i …”—so retrieval and generation stay aligned across paraphrases.
- **`retrieval_query_variants`** (`prompts.py`) supplies up to a few distinct query strings per turn (full normalized question plus optional prefix-stripped tails such as after “how to ”) so hybrid search can recall table-like or imperative chunks; the retrieve node uses the first few variants (capped) and merges with deduplication.
- Prompt assembly and message shaping live in `prompts.py` (`SYSTEM_PROMPT`, memory-aware user blob via `build_chat_messages_for_ollama`). System and task text stress **intent-stable** answers across procedural wording and discourage empty or generic “rephrase” replies when context supports the topic.
- **LangGraph** (`rag_pipeline.py`): compiled `StateGraph` with nodes **expand** → (**small_talk** *or* **retrieve**) → **agentic_refine** → **not_found** / **binary** / **llm**. The **agentic_refine** node is a no-op when agentic RAG is off or when retrieval returned no chunks; otherwise it may expand the context set before routing. A second compile uses `interrupt_before=["llm"]` so the streaming API can run expand, retrieval, agentic refinement, and routing inside the graph, then stream **LangChain** `ChatOllama` tokens for the LLM path only.
- **LangChain**: `langchain-community` `ChatOllama` with `OLLAMA_API_URL` and `OLLAMA_MODEL`; sync answers use a small LCEL chain (`RunnableLambda` → `llm` → `StrOutputParser`). Python deps include `langchain`, `langchain-core`, `langchain-community`, and `langgraph` (see `requirements.txt`).
- Optional **LangSmith** tracing for LangChain/LangGraph: `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY` in `.env.example`.
- Both normal and streaming chat responses are supported in API mode.

### 4) Agentic RAG (optional)

- Implemented in **`rag_agentic.py`**, wired as the **`agentic_refine`** node after the first hybrid retrieval.
- When **`AGENTIC_RAG_ENABLED`** is true (see `.env.example`), a compact **grader** call (`ChatOllama`, temperature 0) reads the user question plus short previews of retrieved chunks and returns JSON: whether context is **sufficient**, and optional **follow_up_queries** (bounded count, short strings).
- If the grader marks coverage as weak and proposes queries, the app runs additional **`retrieve_with_hybrid`** passes (with the same variant helper where applicable), **deduplicates** chunks by text, and caps the merged list (**`AGENTIC_RAG_MERGED_CAP`**, etc.). Invalid JSON from the grader is treated as sufficient so the pipeline does not stall.
- **`run_pdf_rag_sync`** / **`stream_pdf_rag_llm_tokens`** accept optional **`agentic_enabled`**; when omitted, the flag is resolved from the environment (`resolve_agentic_enabled`).

### 5) Conversation memory

- Optional conversational memory is supported for follow-up questions.
- Controlled by:
  - `CONTEXT_MEMORY_ENABLED`
  - `CONTEXT_MEMORY_MAX_MESSAGES`
- Prior user/assistant turns can be included before the current RAG context message.
- **Elliptical follow-ups** (for example “how to do” right after a topic-setting user question) are resolved in the graph **expand** step: when the current line is short or vague, `question_for_rag` is built from the **latest prior user message** plus the follow-up before shorthand expansion, so retrieval stays on-topic. **Binary yes/no routing** uses the **literal current user line** only, so a merged retrieval string that still begins with “is there …” does not force a second “Yes.” instead of a procedural answer.

## Data and persistence

- **MySQL**: users, documents metadata, chats, and messages (API flow). Chat rows support **soft delete** (`deleted_at`), **pin** (`pinned_at`), and **archive** (`archived_at`); the chat list returns only non-deleted, non-archived chats, ordered with pinned items first. Assistant message rows store optional **RAG metrics** (`retrieval_seconds`, `latency_seconds`, `accuracy_label`, `accuracy_score`) so the UI can show retrieval time and accuracy for every answer after reload.
- **Vector store**: Chroma on disk or Qdrant (based on `VECTOR_BACKEND`).
- **Uploads**: original files under `data/uploads`.
- **Legacy Streamlit registry**: `data/files.json`.

## API responsibilities (`api.py`)

- Initializes required MySQL tables at startup.
- Handles authentication with JWT:
  - register, login, current-user endpoints.
- Handles file lifecycle:
  - upload/index, list, PDF fetch, delete.
- Handles chat lifecycle:
  - non-stream chat endpoint,
  - stream chat endpoint,
  - chat/message listing and assistant message persistence endpoints,
  - `PATCH /api/chats/{id}` for title rename, pin/unpin, and archive,
  - `DELETE /api/chats/{id}` for soft-delete (sets `deleted_at`).
- Orchestrates chat through `rag_pipeline.py` (LangGraph + LangChain to Ollama), including vector error mapping to HTTP 503 when the backend is unreachable.

## Frontend responsibilities (`frontend/`)

- Provides login/register experience.
- Enforces protected chat route with token-based access.
- Manages file list, chat history, and message composer UI.
- **Recents sidebar**: each conversation has an overflow menu (⋯) for Rename (inline edit), Pin / Unpin, Archive, and Delete (soft delete with confirmation). PDF viewer, export, and document delete remain as before.
- Calls FastAPI endpoints via `frontend/src/api.js` (`VITE_API_URL` configurable).

## Run modes

- Legacy mode:
  - `streamlit run app.py`
- Current mode:
  - Backend: `uvicorn api:app --host 0.0.0.0 --port 8000 --reload`
  - Frontend: `cd frontend && yarn dev`

## Key environment variables

- MySQL: `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
- Auth: `JWT_SECRET`, `JWT_EXPIRES_MINUTES`
- LLM: `OLLAMA_API_URL`, `OLLAMA_MODEL`
- Retrieval memory: `CONTEXT_MEMORY_ENABLED`, `CONTEXT_MEMORY_MAX_MESSAGES`
- Chunking: `PDF_CHUNK_SIZE`, `PDF_CHUNK_OVERLAP`
- Vector backend: `VECTOR_BACKEND`, `CHROMA_URL`, `QDRANT_URL`, `QDRANT_API_KEY`
- Hybrid retrieval: `HYBRID_SEARCH_ENABLED`, `HYBRID_RRF_K`, `HYBRID_DENSE_POOL`, `HYBRID_SPARSE_POOL`, `HYBRID_MAX_TOTAL_CHUNKS`
- Agentic RAG: `AGENTIC_RAG_ENABLED`, `AGENTIC_RAG_MERGED_CAP`, `AGENTIC_RAG_MAX_FOLLOWUP_QUERIES`, `AGENTIC_RAG_GRADER_NUM_PREDICT`
- Optional LangSmith (LangChain/LangGraph): `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`
