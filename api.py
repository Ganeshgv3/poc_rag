import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import fitz
import jwt
import numpy as np
import pymysql
import requests
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from werkzeug.security import check_password_hash, generate_password_hash

from prompts import build_prompt


BASE_DIR = Path(__file__).parent


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


load_env_file(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-secret")
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES", "120"))

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_app")

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
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def init_db():
    conn = db_conn()
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
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def get_or_create_collection(client: chromadb.PersistentClient, collection_name: str):
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    return "\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 1100, overlap: int = 180) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


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


class AssistantMessagePayload(BaseModel):
    content: str


def ask_ollama(question: str, contexts: List[str], model_name: str) -> str:
    prompt = build_prompt(question=question, contexts=contexts, allow_inference=True)
    response = requests.post(
        f"{OLLAMA_API_URL}/api/chat",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 220},
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    message = payload.get("message") or {}
    return str(message.get("content") or payload.get("response") or "").strip()


def stream_ollama_answer(question: str, contexts: List[str], model_name: str):
    prompt = build_prompt(question=question, contexts=contexts, allow_inference=True)
    response = requests.post(
        f"{OLLAMA_API_URL}/api/chat",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0.2, "num_predict": 220},
        },
        stream=True,
        timeout=180,
    )
    response.raise_for_status()
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        chunk = json.loads(raw_line)
        message = chunk.get("message") or {}
        delta = str(message.get("content") or "")
        if delta:
            yield delta


def sse_event(payload: Dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def chunk_text_for_stream(text: str, chunk_size: int = 12) -> List[str]:
    cleaned = text or ""
    if not cleaned:
        return []
    return [cleaned[i : i + chunk_size] for i in range(0, len(cleaned), chunk_size)]


def is_small_talk(question: str) -> bool:
    lowered = re.sub(r"[^a-z0-9\s]", " ", question.lower()).strip()
    if not lowered:
        return False
    compact = " ".join(lowered.split())
    exact_phrases = {
        "hi",
        "hello",
        "hey",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "thanks",
        "thank you",
        "thx",
        "ok",
        "okay",
        "cool",
        "nice",
        "great",
    }
    if compact in exact_phrases:
        return True
    starts_with_phrases = (
        "hi ",
        "hello ",
        "hey ",
        "thanks ",
        "thank you ",
        "how are you ",
    )
    return compact.startswith(starts_with_phrases)


def small_talk_reply(question: str) -> str:
    compact = " ".join(re.sub(r"[^a-z0-9\s]", " ", question.lower()).split())
    if compact.startswith(("thanks", "thank you", "thx")):
        return "You are welcome. I am here whenever you want to ask something about your PDF."
    if compact.startswith("how are you"):
        return "I am doing great. Ready to help you explore your PDF."
    if compact in {"ok", "okay", "cool", "nice", "great"}:
        return "Great. Ask me any detail you want from the selected PDF."
    return "Hi there. I am ready to help. Ask me anything from your selected PDF."


NOT_FOUND_REPLIES = [
    "I could not spot that detail in the selected PDF yet. Please try a more specific question.",
    "I could not find that in this PDF. If you want, ask with a related keyword and I will check again.",
    "That exact detail is not visible in the selected PDF right now. Try rephrasing and I can re-check.",
]


def friendly_not_found_reply(question: str) -> str:
    seed = abs(hash((question or "").strip().lower()))
    return NOT_FOUND_REPLIES[seed % len(NOT_FOUND_REPLIES)]


def clean_answer_text(answer: str) -> str:
    if not answer:
        return "I am here to help. Could you please rephrase your question?"
    text = answer
    # Remove common markdown emphasis artifacts for plain readable UI output.
    text = text.replace("**", "").replace("__", "").replace("`", "")
    # Remove orphan/standalone bullet symbols that sometimes appear from LLM formatting.
    text = re.sub(r"(?m)^\s*[•◦○●▪▫]\s*$", "", text)
    # Normalize bullet-prefixed lines into plain sentences.
    text = re.sub(r"(?m)^\s*[-*•◦○●▪▫]\s+", "", text)
    # Normalize excessive spacing/newlines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove stray spaces around punctuation.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    cleaned = text.strip()
    if not cleaned:
        return "I am here to help. Could you please rephrase your question?"
    return cleaned


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


def retrieve_context(query: str, collection_name: str, top_k: int = 3) -> Tuple[List[str], List[float]]:
    model = get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_vector = np.asarray(query_embedding, dtype=np.float32).tolist()[0]
    collection = get_or_create_collection(get_chroma_client(), collection_name)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "distances"],
    )
    docs = result.get("documents", [[]])[0] or []
    distances = result.get("distances", [[]])[0] or []
    return docs, distances


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
            collection = get_or_create_collection(get_chroma_client(), collection_name)
            ids = [f"{digest}_{idx}" for idx in range(len(chunks))]
            metadatas = [
                {"chunk_index": idx, "filename": original_name, "sha256": digest}
                for idx in range(len(chunks))
            ]
            collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

            cur.execute(
                """
                INSERT INTO documents (user_id, filename, stored_name, path, sha256, collection_name, chunks)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (user_id, original_name, stored_name, str(file_path), digest, collection_name, len(chunks)),
            )
            document_id = cur.lastrowid
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
                    get_chroma_client().delete_collection(name=collection_name)
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
                    "SELECT id, title FROM chats WHERE id=%s AND user_id=%s AND document_id=%s",
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

            started_at = time.perf_counter()
            contexts, distances = retrieve_context(question, doc["collection_name"], top_k=3)
            if re.match(
                r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b",
                question.lower(),
            ):
                text = " ".join(contexts).lower()
                keywords = [t for t in re.findall(r"[a-z0-9\+\#\.]+", question.lower()) if len(t) > 2]
                answer = "Yes." if any(k in text for k in keywords) else "No."
            elif is_small_talk(question):
                answer = small_talk_reply(question)
            elif not contexts:
                answer = friendly_not_found_reply(question)
            else:
                answer = ask_ollama(question, contexts, OLLAMA_MODEL)
            answer = clean_answer_text(answer)

            cur.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (%s, 'user', %s)",
                (chat_id, question),
            )
            cur.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (%s, 'assistant', %s)",
                (chat_id, answer),
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
            elapsed = round(time.perf_counter() - started_at, 2)

        return {
            "answer": answer,
            "contexts": contexts,
            "distances": distances,
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
                "SELECT id, title FROM chats WHERE id=%s AND user_id=%s AND document_id=%s",
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

        cur.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (%s, 'user', %s)",
            (chat_id, question),
        )

    try:
        contexts, distances = retrieve_context(question, doc["collection_name"], top_k=3)
    except Exception:
        conn.close()
        raise

    def event_stream():
        answer_parts: List[str] = []
        try:
            yield sse_event({"type": "meta", "chat_id": chat_id})
            if re.match(
                r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b",
                question.lower(),
            ):
                text = " ".join(contexts).lower()
                keywords = [t for t in re.findall(r"[a-z0-9\+\#\.]+", question.lower()) if len(t) > 2]
                answer = "Yes." if any(k in text for k in keywords) else "No."
                answer = clean_answer_text(answer)
                for part in chunk_text_for_stream(answer):
                    answer_parts.append(part)
                    yield sse_event({"type": "token", "delta": part})
            elif is_small_talk(question):
                answer = clean_answer_text(small_talk_reply(question))
                for part in chunk_text_for_stream(answer):
                    answer_parts.append(part)
                    yield sse_event({"type": "token", "delta": part})
            elif not contexts:
                answer = clean_answer_text(friendly_not_found_reply(question))
                for part in chunk_text_for_stream(answer):
                    answer_parts.append(part)
                    yield sse_event({"type": "token", "delta": part})
            else:
                for delta in stream_ollama_answer(question, contexts, OLLAMA_MODEL):
                    if not delta:
                        continue
                    answer_parts.append(delta)
                    yield sse_event({"type": "token", "delta": delta})

            final_answer = clean_answer_text("".join(answer_parts))
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (chat_id, role, content) VALUES (%s, 'assistant', %s)",
                    (chat_id, final_answer),
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
            yield sse_event({"type": "done", "chat_id": chat_id, "answer": final_answer, "contexts": contexts, "distances": distances})
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
                "SELECT id, title, created_at FROM chats WHERE user_id=%s AND document_id=%s ORDER BY created_at DESC",
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
                "SELECT id FROM chats WHERE id=%s AND user_id=%s",
                (chat_id, user_id),
            )
            chat = cur.fetchone()
            if not chat:
                raise HTTPException(status_code=404, detail="Chat not found.")

            cur.execute(
                "SELECT role, content, created_at FROM messages WHERE chat_id=%s ORDER BY id ASC",
                (chat_id,),
            )
            messages = cur.fetchall()
        return {"messages": messages}
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
            cur.execute("SELECT id FROM chats WHERE id=%s AND user_id=%s", (chat_id, user_id))
            chat = cur.fetchone()
            if not chat:
                raise HTTPException(status_code=404, detail="Chat not found.")

            cur.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (%s, 'assistant', %s)",
                (chat_id, content),
            )
        return {"message": "Assistant message saved."}
    finally:
        conn.close()


@app.on_event("startup")
def on_startup():
    init_db()


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
