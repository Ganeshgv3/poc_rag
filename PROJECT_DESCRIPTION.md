# Project Description - RAG PDF Chat App

## What this project does

This project is a Retrieval-Augmented Generation (RAG) application for chatting with uploaded PDF files.
Users upload a PDF, the app splits the text into chunks, creates embeddings, stores those vectors in ChromaDB, and answers user questions by retrieving relevant chunks before calling an LLM (Ollama).

The repository contains two app flows:

- `app.py`: legacy Streamlit single-app experience.
- `api.py` + `frontend/`: current FastAPI backend with React frontend and JWT auth.

## Main architecture

1. PDF upload
- PDF file is uploaded from UI.
- Raw PDF is saved to `data/uploads`.

2. Extraction and chunking
- Text is extracted with PyMuPDF (`fitz`).
- Text is chunked with overlap for better retrieval quality.

3. Embedding and vector storage
- Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.
- Chunk text + vectors are stored in local ChromaDB at `data/chroma`.

4. Retrieval and answer generation
- User question is embedded.
- Top-k relevant chunks are retrieved from Chroma.
- Retrieved context is injected into the prompt (`prompts.py`).
- Ollama generates the final answer.

## Data storage used

- **ChromaDB (local, filesystem)**: chunk documents, embeddings, vector index.
- **MySQL**: users, documents metadata, chats, and messages (FastAPI flow).
- **JSON file (`data/files.json`)**: file registry for Streamlit legacy flow.
- **Uploads folder (`data/uploads`)**: original uploaded PDFs.

## Backend responsibilities (`api.py`)

- Authentication (register/login/me) with JWT.
- File upload and indexing endpoint.
- Chat endpoints (normal + streaming SSE).
- Database table initialization on startup.
- Retrieval pipeline + Ollama call orchestration.

## Frontend responsibilities (`frontend/`)

- Login/register UX.
- Protected chat route.
- File list and chat history UI.
- Sends chat requests to FastAPI and renders responses.

## Key decisions in current implementation

- Uses local ChromaDB persistent storage (`data/chroma`) for vector data.
- Uses local Ollama endpoint for generation (`OLLAMA_API_URL`).
- Uses one Chroma collection per document hash (`pdf_<sha-prefix>`).
- Uses chunk overlap to improve retrieval continuity.

## Run modes

- Legacy mode: `streamlit run app.py`
- Current mode:
  - Backend: `uvicorn api:app --host 0.0.0.0 --port 8000 --reload`
  - Frontend: `cd frontend && yarn dev`
