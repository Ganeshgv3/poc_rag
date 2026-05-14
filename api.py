import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import jwt
import numpy as np
import pymysql
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from werkzeug.security import check_password_hash, generate_password_hash

from chroma_helpers import create_chroma_client, delete_named_collection, get_or_create_vector_collection
from env_load import format_env_search_list, load_dotenv_for_project
from prompts import expand_question_shorthand
from rag_metrics import accuracy_from_signals
from rag_pipeline import retry_rag_llm_if_weak, run_pdf_rag_sync, stream_pdf_rag_llm_tokens
from rag_routing import clean_answer_text
from text_chunking import chunk_text, extract_text


BASE_DIR = Path(__file__).parent
load_dotenv_for_project(BASE_DIR)

DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-secret")
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES", "120"))


def rag_chat_llm_temperature() -> float:
    try:
        return float(os.getenv("RAG_CHAT_TEMPERATURE", "0.0"))
    except ValueError:
        return 0.0


def rag_chat_llm_num_predict() -> int:
    try:
        return max(160, int(os.getenv("RAG_CHAT_NUM_PREDICT", "320")))
    except ValueError:
        return 320


def context_memory_enabled() -> bool:
    return (os.getenv("CONTEXT_MEMORY_ENABLED") or "true").strip().lower() not in ("0", "false", "no", "off")


def context_memory_max_messages() -> int:
    try:
        return max(0, int(os.getenv("CONTEXT_MEMORY_MAX_MESSAGES", "40")))
    except ValueError:
        return 24


def fetch_prior_messages(cur, chat_id: int, *, exclude_last_user: bool) -> List[Dict[str, str]]:
    """Load recent DB messages for Ollama. Chronological order (oldest first within the window)."""
    limit = context_memory_max_messages()
    if limit <= 0 or not chat_id:
        return []
    cur.execute(
        "SELECT role, content FROM messages WHERE chat_id=%s ORDER BY id DESC LIMIT %s",
        (chat_id, limit),
    )
    rows = list(reversed(cur.fetchall() or []))
    # Stream path inserts the current user row before this fetch; strip it so the model
    # does not see the same question twice (it is also inside the final RAG user blob).
    if exclude_last_user and rows and rows[-1].get("role") == "user":
        rows = rows[:-1]
    out: List[Dict[str, str]] = []
    for r in rows:
        role = r.get("role")
        content = (r.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": str(role), "content": content})
    return out


app = FastAPI(title="RAG Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer(auto_error=False)


def db_conn():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "rag_app"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def init_db():
    try:
        conn = db_conn()
    except pymysql.err.OperationalError as exc:
        errno = exc.args[0] if exc.args else None
        if errno == 1045:
            user = os.getenv("MYSQL_USER", "root")
            password_set = bool(os.getenv("MYSQL_PASSWORD", "").strip())
            raise RuntimeError(
                "MySQL login failed (1045 Access denied). Set MYSQL_USER and MYSQL_PASSWORD "
                f"(variable name must be exactly MYSQL_PASSWORD). "
                f"Env files checked (later overrides earlier): {format_env_search_list(BASE_DIR)}. "
                f"You can point to a specific file with DOTENV_PATH=/path/to/.env. "
                f"Currently user={user!r}, MYSQL_PASSWORD={'set' if password_set else 'missing or empty'}."
            ) from exc
        if errno == 1049:
            dbname = os.getenv("MYSQL_DATABASE", "rag_app")
            raise RuntimeError(
                f"MySQL database {dbname!r} does not exist (1049). Create it first, e.g. "
                f"`CREATE DATABASE {dbname}` then restart the API."
            ) from exc
        raise RuntimeError(f"MySQL connection failed: {exc}") from exc
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(120) NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    user_id BIGINT NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    stored_name VARCHAR(255) NOT NULL,
                    path TEXT NOT NULL,
                    sha256 VARCHAR(64) NOT NULL,
                    collection_name VARCHAR(120) NOT NULL,
                    chunks INT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uniq_user_sha (user_id, sha256),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    user_id BIGINT NOT NULL,
                    document_id BIGINT NOT NULL,
                    title VARCHAR(255) NOT NULL DEFAULT 'New Chat',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP NULL DEFAULT NULL,
                    pinned_at TIMESTAMP NULL DEFAULT NULL,
                    archived_at TIMESTAMP NULL DEFAULT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    chat_id BIGINT NOT NULL,
                    role ENUM('user', 'assistant') NOT NULL,
                    content MEDIUMTEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                )
                """
            )
            try:
                cur.execute("ALTER TABLE chats ADD COLUMN deleted_at TIMESTAMP NULL DEFAULT NULL")
            except pymysql.err.OperationalError as exc:
                if exc.args and exc.args[0] != 1060:
                    raise
            try:
                cur.execute("ALTER TABLE chats ADD COLUMN pinned_at TIMESTAMP NULL DEFAULT NULL")
            except pymysql.err.OperationalError as exc:
                if exc.args and exc.args[0] != 1060:
                    raise
            try:
                cur.execute("ALTER TABLE chats ADD COLUMN archived_at TIMESTAMP NULL DEFAULT NULL")
            except pymysql.err.OperationalError as exc:
                if exc.args and exc.args[0] != 1060:
                    raise
            for msg_col_sql in (
                "ALTER TABLE messages ADD COLUMN retrieval_seconds DOUBLE NULL DEFAULT NULL",
                "ALTER TABLE messages ADD COLUMN latency_seconds DOUBLE NULL DEFAULT NULL",
                "ALTER TABLE messages ADD COLUMN accuracy_label VARCHAR(24) NULL DEFAULT NULL",
                "ALTER TABLE messages ADD COLUMN accuracy_score INT NULL DEFAULT NULL",
            ):
                try:
                    cur.execute(msg_col_sql)
                except pymysql.err.OperationalError as exc:
                    if exc.args and exc.args[0] != 1060:
                        raise
    finally:
        conn.close()


_embedding_model = None
_chroma_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = create_chroma_client(CHROMA_DIR)
    return _chroma_client


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def create_token(user: Dict) -> str:
    payload = {
        # JWT "sub" must be a string per RFC 7519.
        "sub": str(user["id"]),
        "email": user["email"],
        "exp": datetime.now(tz=timezone.utc) + timedelta(minutes=JWT_EXPIRES_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_user_id(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return int(payload["sub"])
    except Exception:
        return None


def auth_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = decode_user_id(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user_id


class RegisterPayload(BaseModel):
    name: str
    email: str
    password: str


class LoginPayload(BaseModel):
    email: str
    password: str


class ChatPayload(BaseModel):
    document_id: int
    question: str
    chat_id: Optional[int] = None
    replace_user_message_id: Optional[int] = None


class AssistantMessagePayload(BaseModel):
    content: str
    retrieval_seconds: Optional[float] = None
    latency_seconds: Optional[float] = None
    accuracy_label: Optional[str] = None
    accuracy_score: Optional[int] = None


class ChatPatchPayload(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None
    archived: Optional[bool] = None


def sse_event(payload: Dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def chunk_text_for_stream(text: str, chunk_size: int = 12) -> List[str]:
    cleaned = text or ""
    if not cleaned:
        return []
    return [cleaned[i : i + chunk_size] for i in range(0, len(cleaned), chunk_size)]


def build_chat_title(questions: List[str]) -> str:
    corpus = " ".join(" ".join((q or "").strip().split()) for q in questions if (q or "").strip())
    if not corpus:
        return "New Chat"

    lowered = re.sub(r"[^a-z0-9\s]", " ", corpus.lower())
    tokens = [token for token in lowered.split() if token]
    if not tokens:
        return "New Chat"

    theme_map = {
        "Education Background": {"education", "qualification", "college", "degree", "bsc", "msc", "university"},
        "Career Experience": {"experience", "career", "work", "years", "company", "role", "designation"},
        "Technical Skills": {"skills", "skill", "technology", "technologies", "stack", "python", "java", "html"},
        "Project Discussion": {"project", "projects", "built", "developed", "implementation", "architecture"},
        "Personal Profile": {"name", "age", "dob", "birthday", "about", "intro", "introduction"},
        "Contact Details": {"email", "phone", "mobile", "address", "linkedin", "github"},
    }

    token_set = set(tokens)
    scored_themes = []
    for title, words in theme_map.items():
        score = len(token_set & words)
        if score:
            scored_themes.append((score, title))

    if scored_themes:
        scored_themes.sort(reverse=True)
        if len(scored_themes) > 1 and scored_themes[0][0] == scored_themes[1][0]:
            return f"{scored_themes[0][1]} & {scored_themes[1][1]}"
        return scored_themes[0][1]

    ignored = {
        "what",
        "which",
        "who",
        "whom",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "has",
        "have",
        "had",
        "will",
        "would",
        "should",
        "please",
        "tell",
        "me",
        "about",
        "the",
        "a",
        "an",
    }
    meaningful = [token for token in tokens if token not in ignored and len(token) > 2]
    if not meaningful:
        return "Conversation"

    freq: Dict[str, int] = {}
    for token in meaningful:
        freq[token] = freq.get(token, 0) + 1
    top_tokens = [token for token, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:3]]
    return " ".join(top_tokens).title()


def is_generic_chat_title(title: str) -> bool:
    if not title:
        return True
    cleaned = title.strip().lower()
    if cleaned in {"new chat", "new conversation", "conversation"}:
        return True
    return cleaned.endswith("?") or len(cleaned.split()) <= 2


def _vector_store_unreachable_detail(exc: BaseException) -> Optional[str]:
    """If exc is a connectivity failure to the vector backend, return an API detail string."""
    seen: set[int] = set()
    stack: List[BaseException] = [exc]
    while stack:
        e = stack.pop()
        eid = id(e)
        if eid in seen:
            continue
        seen.add(eid)
        if isinstance(e, (httpx.ConnectError, httpx.TimeoutException)):
            return (
                "Cannot reach the vector database. If you use Qdrant (VECTOR_BACKEND=qdrant), "
                "start Qdrant and confirm QDRANT_URL (default http://localhost:6333). "
                "For local on-disk vectors without Qdrant, set VECTOR_BACKEND=chroma or unset it."
            )
        if e.__cause__ is not None:
            stack.append(e.__cause__)
        if e.__context__ is not None and e.__context__ is not e.__cause__:
            stack.append(e.__context__)
        args0 = e.args[0] if e.args else None
        if isinstance(args0, BaseException) and args0 is not e.__cause__:
            stack.append(args0)
    return None


@app.post("/api/auth/register")
def register(payload: RegisterPayload):
    name = payload.name.strip()
    email = payload.email.strip().lower()
    password = payload.password
    if not name or not email or len(password) < 6:
        raise HTTPException(status_code=400, detail="Name, valid email, and password >= 6 required.")

    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email=%s", (email,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Email already exists.")
            cur.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
                (name, email, generate_password_hash(password)),
            )
            user_id = cur.lastrowid
        token = create_token({"id": user_id, "email": email})
        return {"token": token, "user": {"id": user_id, "name": name, "email": email}}
    finally:
        conn.close()


@app.post("/api/auth/login")
def login(payload: LoginPayload):
    email = payload.email.strip().lower()
    password = payload.password
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, email, password_hash FROM users WHERE email=%s", (email,))
            user = cur.fetchone()
        if not user or not check_password_hash(user["password_hash"], password):
            raise HTTPException(status_code=401, detail="Invalid credentials.")
        token = create_token(user)
        return {"token": token, "user": {"id": user["id"], "name": user["name"], "email": user["email"]}}
    finally:
        conn.close()


@app.get("/api/auth/me")
def me(user_id: int = Depends(auth_user)):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, email FROM users WHERE id=%s", (user_id,))
            user = cur.fetchone()
        return {"user": user}
    finally:
        conn.close()


@app.get("/api/files")
def list_files(user_id: int = Depends(auth_user)):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, filename, sha256, chunks, created_at FROM documents WHERE user_id=%s ORDER BY created_at DESC",
                (user_id,),
            )
            docs = cur.fetchall()
        return {"files": docs}
    finally:
        conn.close()


@app.get("/api/files/{document_id}/pdf")
def get_document_pdf(document_id: int, user_id: int = Depends(auth_user)):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT path, filename FROM documents WHERE id=%s AND user_id=%s",
                (document_id, user_id),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found.")
        file_path = Path(row["path"])
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="PDF file is missing on the server.")
        download_name = row.get("filename") or file_path.name
        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            filename=download_name,
        )
    finally:
        conn.close()


@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...), user_id: int = Depends(auth_user)):
    if file is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")
    pdf_bytes = await file.read()
    digest = sha256_bytes(pdf_bytes)

    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE user_id=%s AND sha256=%s", (user_id, digest))
            existing = cur.fetchone()
            if existing:
                raise HTTPException(status_code=409, detail="This PDF is already indexed.")

            text = extract_text(pdf_bytes)
            chunks = chunk_text(text)
            if not chunks:
                raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
            model = get_embedding_model()
            embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
            embeddings = np.asarray(embeddings, dtype=np.float32).tolist()

            original_name = Path(file.filename).name
            stored_name = f"{digest[:12]}_{original_name}"
            file_path = UPLOADS_DIR / stored_name
            file_path.write_bytes(pdf_bytes)

            collection_name = f"pdf_{digest[:16]}"
            try:
                collection = get_or_create_vector_collection(get_chroma_client(), collection_name)
                ids = [f"{digest}_{idx}" for idx in range(len(chunks))]
                metadatas = [
                    {"chunk_index": idx, "filename": original_name, "sha256": digest}
                    for idx in range(len(chunks))
                ]
                collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
            except Exception as exc:
                detail = _vector_store_unreachable_detail(exc)
                if detail:
                    raise HTTPException(status_code=503, detail=detail) from exc
                raise

            cur.execute(
                """
                INSERT INTO documents (user_id, filename, stored_name, path, sha256, collection_name, chunks)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (user_id, original_name, stored_name, str(file_path), digest, collection_name, len(chunks)),
            )
            document_id = cur.lastrowid
            mysql_db = os.getenv("MYSQL_DATABASE", "rag_app")
            vector_backend = (os.getenv("VECTOR_BACKEND") or "chroma").strip()
            print(
                "\n=== PDF uploaded & indexed ===",
                f"File: {original_name}",
                f"Chunked into {len(chunks)} chunks; embeddings stored in vector store",
                f"MySQL database: {mysql_db}  (documents.id={document_id})",
                f"Vector collection name: {collection_name}  (VECTOR_BACKEND={vector_backend})",
                "================================\n",
                sep="\n",
                flush=True,
            )
        return {"message": "Indexed successfully.", "document_id": document_id}
    finally:
        conn.close()


@app.delete("/api/files/{document_id}")
def delete_file(document_id: int, user_id: int = Depends(auth_user)):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, path, collection_name
                FROM documents
                WHERE id=%s AND user_id=%s
                """,
                (document_id, user_id),
            )
            doc = cur.fetchone()
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found.")

            collection_name = doc.get("collection_name")
            if collection_name:
                try:
                    delete_named_collection(get_chroma_client(), collection_name)
                except Exception:
                    # Keep API resilient if vector collection is already missing/corrupt.
                    pass

            path_value = doc.get("path")
            if path_value:
                file_path = Path(path_value)
                if file_path.exists():
                    file_path.unlink()

            cur.execute("DELETE FROM documents WHERE id=%s AND user_id=%s", (document_id, user_id))
        return {"message": "Document deleted successfully."}
    finally:
        conn.close()


@app.post("/api/chat")
def chat(payload: ChatPayload, user_id: int = Depends(auth_user)):
    question = payload.question.strip()
    document_id = payload.document_id
    chat_id = payload.chat_id
    if not question or not document_id:
        raise HTTPException(status_code=400, detail="document_id and question are required.")

    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, collection_name FROM documents WHERE id=%s AND user_id=%s",
                (document_id, user_id),
            )
            doc = cur.fetchone()
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found.")

            existing_chat = None
            if chat_id:
                cur.execute(
                    "SELECT id, title FROM chats WHERE id=%s AND user_id=%s AND document_id=%s AND deleted_at IS NULL AND archived_at IS NULL",
                    (chat_id, user_id, document_id),
                )
                existing_chat = cur.fetchone()
                if not existing_chat:
                    raise HTTPException(status_code=404, detail="Chat not found for this document.")
            else:
                cur.execute(
                    "INSERT INTO chats (user_id, document_id, title) VALUES (%s, %s, %s)",
                    (user_id, document_id, build_chat_title([question])),
                )
                chat_id = cur.lastrowid

            prior_messages: List[Dict[str, str]] = []
            if context_memory_enabled() and chat_id:
                prior_messages = fetch_prior_messages(cur, chat_id, exclude_last_user=False)

            started_at = time.perf_counter()
            try:
                answer, contexts, distances, retrieval_seconds = run_pdf_rag_sync(
                    question=question,
                    collection_name=doc["collection_name"],
                    embedding_model=get_embedding_model(),
                    chroma_client=get_chroma_client(),
                    ollama_base_url=OLLAMA_API_URL,
                    ollama_model=OLLAMA_MODEL,
                    prior_messages=prior_messages or None,
                    top_k=3,
                    temperature=rag_chat_llm_temperature(),
                    num_predict=rag_chat_llm_num_predict(),
                    allow_inference=True,
                )
            except Exception as exc:
                detail = _vector_store_unreachable_detail(exc)
                if detail:
                    raise HTTPException(status_code=503, detail=detail) from exc
                raise
            answer = clean_answer_text(answer, question)
            elapsed = round(time.perf_counter() - started_at, 2)
            acc_label, acc_score = accuracy_from_signals(answer, contexts, distances)
            mysql_db = os.getenv("MYSQL_DATABASE", "rag_app")
            print(
                "\n--- Question & answer (POST /api/chat) ---",
                f"MySQL database: {mysql_db}",
                f"Vector collection: {doc['collection_name']}",
                f"document_id={document_id}  chat_id={chat_id}",
                f"Question: {question}",
                f"Answer: {answer}",
                "-----------------------------------------\n",
                sep="\n",
                flush=True,
            )

            cur.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (%s, 'user', %s)",
                (chat_id, question),
            )
            cur.execute(
                """
                INSERT INTO messages (chat_id, role, content, retrieval_seconds, latency_seconds, accuracy_label, accuracy_score)
                VALUES (%s, 'assistant', %s, %s, %s, %s, %s)
                """,
                (chat_id, answer, retrieval_seconds, elapsed, acc_label, acc_score),
            )
            if chat_id:
                cur.execute(
                    "SELECT content FROM messages WHERE chat_id=%s AND role='user' ORDER BY id ASC LIMIT 8",
                    (chat_id,),
                )
                user_questions = [row["content"] for row in cur.fetchall()]
                improved_title = build_chat_title(user_questions)

            if chat_id and existing_chat and (is_generic_chat_title(existing_chat.get("title", "")) or len(user_questions) <= 3):
                cur.execute(
                    "UPDATE chats SET title=%s WHERE id=%s",
                    (improved_title, chat_id),
                )

        return {
            "answer": answer,
            "contexts": contexts,
            "distances": distances,
            "retrieval_seconds": retrieval_seconds,
            "accuracy_label": acc_label,
            "accuracy_score": acc_score,
            "latency_seconds": elapsed,
            "chat_id": chat_id,
        }
    finally:
        conn.close()


@app.post("/api/chat/stream")
def chat_stream(payload: ChatPayload, user_id: int = Depends(auth_user)):
    question = payload.question.strip()
    document_id = payload.document_id
    chat_id = payload.chat_id
    if not question or not document_id:
        raise HTTPException(status_code=400, detail="document_id and question are required.")

    conn = db_conn()
    replace_assistant_message_id: Optional[int] = None
    replace_id: Optional[int] = payload.replace_user_message_id
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, collection_name FROM documents WHERE id=%s AND user_id=%s",
            (document_id, user_id),
        )
        doc = cur.fetchone()
        if not doc:
            conn.close()
            raise HTTPException(status_code=404, detail="Document not found.")

        existing_chat = None
        if chat_id:
            cur.execute(
                "SELECT id, title FROM chats WHERE id=%s AND user_id=%s AND document_id=%s AND deleted_at IS NULL AND archived_at IS NULL",
                (chat_id, user_id, document_id),
            )
            existing_chat = cur.fetchone()
            if not existing_chat:
                conn.close()
                raise HTTPException(status_code=404, detail="Chat not found for this document.")
        else:
            cur.execute(
                "INSERT INTO chats (user_id, document_id, title) VALUES (%s, %s, %s)",
                (user_id, document_id, build_chat_title([question])),
            )
            chat_id = cur.lastrowid

        if replace_id:
            if not chat_id:
                raise HTTPException(status_code=400, detail="chat_id is required when editing a message.")
            cur.execute(
                """
                SELECT m.id, m.role
                FROM messages m
                INNER JOIN chats c ON c.id = m.chat_id
                WHERE m.id = %s AND m.chat_id = %s AND c.user_id = %s AND c.document_id = %s
                    AND c.deleted_at IS NULL
                    AND c.archived_at IS NULL
                """,
                (replace_id, chat_id, user_id, document_id),
            )
            target = cur.fetchone()
            if not target or target.get("role") != "user":
                raise HTTPException(status_code=404, detail="User message not found for this chat.")
            cur.execute(
                "UPDATE messages SET content=%s WHERE id=%s AND chat_id=%s",
                (question, replace_id, chat_id),
            )
            cur.execute(
                """
                SELECT id FROM messages
                WHERE chat_id=%s AND id > %s AND role='assistant'
                ORDER BY id ASC
                LIMIT 1
                """,
                (chat_id, replace_id),
            )
            assistant_row = cur.fetchone()
            if assistant_row:
                replace_assistant_message_id = int(assistant_row["id"])
        else:
            cur.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (%s, 'user', %s)",
                (chat_id, question),
            )

        prior_messages: List[Dict[str, str]] = []
        if context_memory_enabled() and chat_id:
            prior_messages = fetch_prior_messages(cur, chat_id, exclude_last_user=True)

    try:
        contexts, distances, retrieval_seconds, prefilled, token_iter, question_for_rag = stream_pdf_rag_llm_tokens(
            question=question,
            collection_name=doc["collection_name"],
            embedding_model=get_embedding_model(),
            chroma_client=get_chroma_client(),
            ollama_base_url=OLLAMA_API_URL,
            ollama_model=OLLAMA_MODEL,
            prior_messages=prior_messages or None,
            top_k=3,
            temperature=rag_chat_llm_temperature(),
            num_predict=rag_chat_llm_num_predict(),
            allow_inference=True,
        )
    except Exception as exc:
        conn.close()
        detail = _vector_store_unreachable_detail(exc)
        if detail:
            raise HTTPException(status_code=503, detail=detail) from exc
        raise

    def event_stream():
        answer_parts: List[str] = []
        stream_wall0 = time.perf_counter()
        try:
            yield sse_event({"type": "meta", "chat_id": chat_id, "retrieval_seconds": retrieval_seconds})
            if prefilled:
                answer = clean_answer_text(prefilled, question)
                for part in chunk_text_for_stream(answer):
                    answer_parts.append(part)
                    yield sse_event({"type": "token", "delta": part})
                final_answer = answer
            else:
                for delta in token_iter:
                    if not delta:
                        continue
                    answer_parts.append(delta)
                    yield sse_event({"type": "token", "delta": delta})
                raw_joined = "".join(answer_parts)
                cleaned = clean_answer_text(raw_joined, question)
                qfr = (question_for_rag or "").strip() or expand_question_shorthand(question)
                final_answer = retry_rag_llm_if_weak(
                    streamed_raw=raw_joined,
                    cleaned_answer=cleaned,
                    question=question,
                    question_for_rag=qfr,
                    contexts=contexts,
                    prior_messages=prior_messages,
                    ollama_base_url=OLLAMA_API_URL,
                    ollama_model=OLLAMA_MODEL,
                    allow_inference=True,
                    base_num_predict=rag_chat_llm_num_predict(),
                )

            acc_label, acc_score = accuracy_from_signals(final_answer, contexts, distances)
            total_latency = round(time.perf_counter() - stream_wall0, 2)
            mysql_db = os.getenv("MYSQL_DATABASE", "rag_app")
            print(
                "\n--- Question & answer (POST /api/chat/stream) ---",
                f"MySQL database: {mysql_db}",
                f"Vector collection: {doc['collection_name']}",
                f"document_id={document_id}  chat_id={chat_id}",
                f"Question: {question}",
                f"Answer: {final_answer}",
                "--------------------------------------------------\n",
                sep="\n",
                flush=True,
            )
            with conn.cursor() as cur:
                if replace_id and replace_assistant_message_id:
                    cur.execute(
                        """
                        UPDATE messages SET content=%s, retrieval_seconds=%s, latency_seconds=%s,
                            accuracy_label=%s, accuracy_score=%s
                        WHERE id=%s AND chat_id=%s
                        """,
                        (
                            final_answer,
                            retrieval_seconds,
                            total_latency,
                            acc_label,
                            acc_score,
                            replace_assistant_message_id,
                            chat_id,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO messages (chat_id, role, content, retrieval_seconds, latency_seconds, accuracy_label, accuracy_score)
                        VALUES (%s, 'assistant', %s, %s, %s, %s, %s)
                        """,
                        (chat_id, final_answer, retrieval_seconds, total_latency, acc_label, acc_score),
                    )
                cur.execute(
                    "SELECT content FROM messages WHERE chat_id=%s AND role='user' ORDER BY id ASC LIMIT 8",
                    (chat_id,),
                )
                user_questions = [row["content"] for row in cur.fetchall()]
                improved_title = build_chat_title(user_questions)
                if existing_chat and (is_generic_chat_title(existing_chat.get("title", "")) or len(user_questions) <= 3):
                    cur.execute(
                        "UPDATE chats SET title=%s WHERE id=%s",
                        (improved_title, chat_id),
                    )
            yield sse_event(
                {
                    "type": "done",
                    "chat_id": chat_id,
                    "answer": final_answer,
                    "contexts": contexts,
                    "distances": distances,
                    "retrieval_seconds": retrieval_seconds,
                    "latency_seconds": total_latency,
                    "accuracy_label": acc_label,
                    "accuracy_score": acc_score,
                }
            )
        except Exception as exc:
            yield sse_event({"type": "error", "detail": str(exc)})
        finally:
            conn.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/chats")
def list_chats(document_id: int, user_id: int = Depends(auth_user)):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, created_at, pinned_at, archived_at
                FROM chats
                WHERE user_id=%s AND document_id=%s AND deleted_at IS NULL AND archived_at IS NULL
                ORDER BY (pinned_at IS NULL) ASC, pinned_at DESC, created_at DESC
                """,
                (user_id, document_id),
            )
            chats = cur.fetchall()
        return {"chats": chats}
    finally:
        conn.close()


@app.get("/api/chats/{chat_id}/messages")
def chat_messages(chat_id: int, user_id: int = Depends(auth_user)):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM chats WHERE id=%s AND user_id=%s AND deleted_at IS NULL AND archived_at IS NULL",
                (chat_id, user_id),
            )
            chat = cur.fetchone()
            if not chat:
                raise HTTPException(status_code=404, detail="Chat not found.")

            cur.execute(
                """
                SELECT id, role, content, created_at, retrieval_seconds, latency_seconds, accuracy_label, accuracy_score
                FROM messages WHERE chat_id=%s ORDER BY id ASC
                """,
                (chat_id,),
            )
            messages = cur.fetchall()
        return {"messages": messages}
    finally:
        conn.close()


@app.patch("/api/chats/{chat_id}")
def patch_chat(chat_id: int, payload: ChatPatchPayload, user_id: int = Depends(auth_user)):
    if payload.title is None and payload.pinned is None and payload.archived is None:
        raise HTTPException(status_code=400, detail="No changes provided.")

    sets: List[str] = []
    args: List = []

    if payload.title is not None:
        title = (payload.title or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="title cannot be empty.")
        sets.append("title=%s")
        args.append(title[:255])

    if payload.pinned is not None:
        if payload.pinned:
            sets.append("pinned_at=CURRENT_TIMESTAMP")
        else:
            sets.append("pinned_at=NULL")

    if payload.archived is not None:
        if payload.archived:
            sets.append("archived_at=CURRENT_TIMESTAMP")
            sets.append("pinned_at=NULL")
        else:
            sets.append("archived_at=NULL")

    if not sets:
        raise HTTPException(status_code=400, detail="No changes provided.")

    sql = f"UPDATE chats SET {', '.join(sets)} WHERE id=%s AND user_id=%s AND deleted_at IS NULL"
    args.extend([chat_id, user_id])

    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(args))
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Chat not found.")
            cur.execute(
                """
                SELECT id, title, created_at, pinned_at, archived_at
                FROM chats WHERE id=%s AND user_id=%s
                """,
                (chat_id, user_id),
            )
            row = cur.fetchone()
        return {"chat": row}
    finally:
        conn.close()


@app.delete("/api/chats/{chat_id}")
def soft_delete_chat(chat_id: int, user_id: int = Depends(auth_user)):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chats SET deleted_at = CURRENT_TIMESTAMP
                WHERE id=%s AND user_id=%s AND deleted_at IS NULL
                """,
                (chat_id, user_id),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Chat not found.")
        return {"message": "Chat removed."}
    finally:
        conn.close()


@app.post("/api/chats/{chat_id}/messages/assistant")
def save_assistant_message(chat_id: int, payload: AssistantMessagePayload, user_id: int = Depends(auth_user)):
    content = (payload.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required.")

    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM chats WHERE id=%s AND user_id=%s AND deleted_at IS NULL AND archived_at IS NULL",
                (chat_id, user_id),
            )
            chat = cur.fetchone()
            if not chat:
                raise HTTPException(status_code=404, detail="Chat not found.")

            cur.execute(
                """
                INSERT INTO messages (chat_id, role, content, retrieval_seconds, latency_seconds, accuracy_label, accuracy_score)
                VALUES (%s, 'assistant', %s, %s, %s, %s, %s)
                """,
                (
                    chat_id,
                    content,
                    payload.retrieval_seconds,
                    payload.latency_seconds,
                    payload.accuracy_label,
                    payload.accuracy_score,
                ),
            )
        return {"message": "Assistant message saved."}
    finally:
        conn.close()


@app.on_event("startup")
def on_startup():
    init_db()


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
