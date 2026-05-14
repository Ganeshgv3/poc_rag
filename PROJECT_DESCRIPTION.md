# Project Description - RAG PDF Chat App

## Overview

This repository is a Retrieval-Augmented Generation (RAG) app for chatting with uploaded PDFs.
It provides a modern API + web client flow and a legacy Streamlit flow.

- Current app: `api.py` (FastAPI) + `frontend/` (React + Vite)
- Legacy app: `app.py` (Streamlit; chat path uses the same `rag_pipeline` / LangGraph flow as the API)

**Orchestration modules:** `rag_pipeline.py` (LangGraph graph + LangChain Ollama), `rag_routing.py` (small-talk / binary / not-found / `clean_answer_text`).

At a high level, the app:
1. uploads and stores PDFs,
2. extracts and chunks text,
3. embeds chunks with Sentence Transformers,
4. stores vectors in a configured vector backend,
5. retrieves relevant chunks for each question (hybrid dense + BM25 + RRF when enabled),
6. routes each turn through a **LangGraph** state machine and, when an LLM answer is needed, calls **Ollama** via **LangChain** (`ChatOllama` + LCEL) for grounded generation.

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
- Prompt assembly and message shaping live in `prompts.py`.
- Question shorthand normalization is supported (for example `&` and spaced `/` handling), applied in the graph’s **expand** step before retrieval.
- **LangGraph** (`rag_pipeline.py`): compiled `StateGraph` with nodes for expand → (small_talk **or** retrieve) → not_found / binary / LLM. A second compile uses `interrupt_before=["llm"]` so the streaming API can run retrieval + routing in the graph, then stream **LangChain** `ChatOllama` tokens for the LLM path only.
- **LangChain**: `langchain-community` `ChatOllama` with `OLLAMA_API_URL` and `OLLAMA_MODEL`; sync answers use a small LCEL chain (`RunnableLambda` → `llm` → `StrOutputParser`). Python deps include `langchain`, `langchain-core`, `langchain-community`, and `langgraph` (see `requirements.txt`).
- Optional **LangSmith** tracing for LangChain/LangGraph: `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY` in `.env.example`.
- Both normal and streaming chat responses are supported in API mode.

### 4) Conversation memory

- Optional conversational memory is supported for follow-up questions.
- Controlled by:
  - `CONTEXT_MEMORY_ENABLED`
  - `CONTEXT_MEMORY_MAX_MESSAGES`
- Prior user/assistant turns can be included before the current RAG context message.

## Data and persistence

- **MySQL**: users, documents metadata, chats, and messages (API flow). Chat rows support **soft delete** (`deleted_at`), **pin** (`pinned_at`), and **archive** (`archived_at`); the chat list returns only non-deleted, non-archived chats, ordered with pinned items first.
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
- Optional LangSmith (LangChain/LangGraph): `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`
