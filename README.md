# RAG Chat App (React + FastAPI + MySQL)

This project now includes:

- ChatGPT-style frontend (React + Vite + Yarn)
- Auth system (Register/Login with JWT)
- MySQL persistence for users, docs, chats, messages
- ChromaDB for chunk/vector storage
- Ollama for answer generation

## Prerequisites

- Python 3.10+
- Node.js 18+ and Yarn
- MySQL 8+
- Ollama installed: [https://ollama.com](https://ollama.com)

Start Ollama:

```bash
ollama serve
ollama pull llama3.1:8b
```

## Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` with your MySQL credentials and JWT secret:

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=rag_app
JWT_SECRET=your_strong_secret
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

Run backend API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API runs on `http://localhost:8000`.

## Frontend setup

```bash
cd frontend
yarn
yarn dev
```

Frontend runs on `http://localhost:5173`.

To override API URL, set `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000/api
```

## Features implemented

- Register and login pages
- Protected chat route
- JWT-based API authentication
- PDF upload and indexing
- Chat interface with left sidebar and message composer (ChatGPT-like layout)
- Per-user file listing and chat/message persistence
- Centralized system prompt in `prompts.py` (used by both `api.py` and `app.py`)

## Legacy Streamlit app

Your original Streamlit app is still available at `app.py` and can still be run:

```bash
streamlit run app.py
```

