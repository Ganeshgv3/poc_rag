import hashlib
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from chroma_helpers import (
    create_chroma_client,
    get_or_create_vector_collection,
    load_streamlit_file_registry,
    save_streamlit_file_registry,
)
from env_load import load_dotenv_for_project
from rag_pipeline import run_pdf_rag_sync
from rag_routing import NOT_FOUND_REPLIES, is_small_talk
from text_chunking import chunk_text, extract_text


BASE_DIR = Path(__file__).parent
load_dotenv_for_project(BASE_DIR)

# Streamlit's source watcher can introspect optional transformer modules
# and raise noisy import errors when extras (e.g., torchvision) are absent.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
import streamlit as st
from sentence_transformers import SentenceTransformer

DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
FILES_LEGACY = DATA_DIR / "files.json"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_TOP_K = int(os.getenv("TOP_K_DEFAULT", "3"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "220"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
FALLBACK_MODELS = [
    "gpt-oss:20b-cloud",
    "llama3.1:8b",
    "qwen3.5:latest",
    "gemma4:e4b",
    "llama3.2:latest",
    "kimi-k2.6:cloud",
    "glm-5:cloud",
]


def context_memory_enabled() -> bool:
    return (os.getenv("CONTEXT_MEMORY_ENABLED") or "true").strip().lower() not in ("0", "false", "no", "off")


def context_memory_max_messages() -> int:
    try:
        return max(0, int(os.getenv("CONTEXT_MEMORY_MAX_MESSAGES", "40")))
    except ValueError:
        return 40


def session_messages_to_memory(chat_messages: List[Dict]) -> List[Dict[str, str]]:
    """Map Streamlit chat history to Ollama roles (excludes the current user message when caller passes a slice)."""
    lim = context_memory_max_messages()
    out: List[Dict[str, str]] = []
    for m in chat_messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": str(role), "content": content})
    if lim and len(out) > lim:
        out = out[-lim:]
    return out


DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="PDF Chat RAG", page_icon="📄", layout="wide")
st.title("PDF Chat Assistant")
st.caption("Upload PDFs, chat like ChatGPT, and get answers grounded in your files.")


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma_client():
    return create_chroma_client(CHROMA_DIR)


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def ingest_pdf(
    pdf_name: str,
    pdf_bytes: bytes,
    model: SentenceTransformer,
    client,
    progress_callback=None,
) -> Tuple[bool, str]:
    def emit_progress(value: int, text: str) -> None:
        if progress_callback is not None:
            progress_callback(value, text)

    emit_progress(5, "Checking file...")
    digest = sha256_bytes(pdf_bytes)
    records = load_streamlit_file_registry(client, FILES_LEGACY)
    already_exists = next((x for x in records if x["sha256"] == digest), None)
    if already_exists:
        emit_progress(100, "Already indexed.")
        return False, f"Already indexed: {already_exists['filename']}"

    emit_progress(20, "Extracting text...")
    text = extract_text(pdf_bytes)
    if not text:
        emit_progress(100, "Extraction failed.")
        return False, "Could not extract text from this PDF."

    emit_progress(40, "Creating chunks...")
    chunks = chunk_text(text)
    if not chunks:
        emit_progress(100, "Chunking failed.")
        return False, "No valid text chunks found."

    emit_progress(65, "Generating embeddings...")
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32).tolist()

    emit_progress(80, "Saving PDF file...")
    stored_name = f"{digest[:12]}_{pdf_name}"
    file_path = UPLOADS_DIR / stored_name
    file_path.write_bytes(pdf_bytes)

    collection_name = f"pdf_{digest[:16]}"
    collection = get_or_create_vector_collection(client, collection_name)

    ids = [f"{digest}_{idx}" for idx in range(len(chunks))]
    metadatas = [{"chunk_index": idx, "filename": pdf_name, "sha256": digest} for idx in range(len(chunks))]
    emit_progress(90, "Writing vectors to database...")
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    records.append(
        {
            "filename": pdf_name,
            "stored_name": stored_name,
            "path": str(file_path),
            "sha256": digest,
            "collection_name": collection_name,
            "chunks": len(chunks),
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
        }
    )
    save_streamlit_file_registry(client, records)
    emit_progress(100, "Upload complete.")
    vector_backend = (os.getenv("VECTOR_BACKEND") or "chroma").strip()
    print(
        "\n=== PDF uploaded & indexed (Streamlit) ===",
        f"File: {pdf_name}",
        f"Chunked into {len(chunks)} chunks; embeddings stored in vector store",
        f"Vector collection name: {collection_name}  (VECTOR_BACKEND={vector_backend})",
        "(MySQL not used in Streamlit mode; registry saved to vector store / files.json)",
        "==========================================\n",
        sep="\n",
        flush=True,
    )
    return True, f"Indexed {pdf_name} with {len(chunks)} chunks."


def accuracy_from_distances(distances: List[float]) -> Tuple[str, int]:
    if not distances:
        return "Low", 0
    avg_distance = float(sum(distances) / len(distances))
    score = max(0, min(100, int((1.2 - avg_distance) * 100)))
    if score >= 80:
        return "High", score
    if score >= 55:
        return "Medium", score
    return "Low", score


def support_score_from_context(answer: str, contexts: List[str]) -> int:
    if not answer.strip() or not contexts:
        return 0
    answer_tokens = {
        tok
        for tok in re.findall(r"[a-z0-9\+\#\.]+", answer.lower())
        if len(tok) > 2 and tok not in {"the", "and", "for", "with", "from", "that", "this"}
    }
    if not answer_tokens:
        return 0
    context_text = " ".join(contexts).lower()
    matched = sum(1 for tok in answer_tokens if tok in context_text)
    ratio = matched / max(1, len(answer_tokens))
    return int(max(0, min(100, ratio * 100)))


def is_not_found_answer(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    return any(reply.lower() == normalized for reply in NOT_FOUND_REPLIES)


def accuracy_from_signals(answer: str, contexts: List[str], distances: List[float]) -> Tuple[str, int]:
    if is_not_found_answer(answer):
        return "Low", 0
    dist_label, dist_score = accuracy_from_distances(distances)
    _ = dist_label
    support_score = support_score_from_context(answer, contexts)
    final_score = int(round((dist_score * 0.5) + (support_score * 0.5)))
    if final_score >= 80:
        return "High", final_score
    if final_score >= 55:
        return "Medium", final_score
    return "Low", final_score


def get_available_models() -> List[str]:
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=4)
        response.raise_for_status()
        models_data = response.json().get("models", [])
        names = [str(m.get("name", "")).strip() for m in models_data if m.get("name")]
        if names:
            return names
    except Exception:  # pylint: disable=broad-except
        pass
    return FALLBACK_MODELS.copy()


if "messages_by_file" not in st.session_state:
    st.session_state.messages_by_file = {}
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = DEFAULT_OLLAMA_MODEL
if "pending_model_intro" not in st.session_state:
    st.session_state.pending_model_intro = None

embed_model = get_embedding_model()
chroma_client = get_chroma_client()

selected_sha = None
records = []

with st.sidebar:
    st.subheader("Files")
    upload = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
    available_models = get_available_models()
    if DEFAULT_OLLAMA_MODEL not in available_models:
        available_models.insert(0, DEFAULT_OLLAMA_MODEL)
    current_model = st.session_state.selected_model_name
    if current_model not in available_models:
        current_model = DEFAULT_OLLAMA_MODEL
    selected_model = st.selectbox(
        "Ollama model",
        options=available_models,
        index=available_models.index(current_model),
        help="Select a local model. Cloud-tag models may return 403 on local-only setups.",
    )
    previous_model = st.session_state.selected_model_name
    if selected_model != previous_model:
        st.session_state.selected_model_name = selected_model
        if previous_model:
            st.session_state.pending_model_intro = f"Hi, I'm `{selected_model}`. Ready to help with your PDF."
    fast_mode = st.checkbox("Fast mode", value=True)
    reasoning_mode = st.checkbox("Reasoning mode (infer when needed)", value=True)
    top_k_default = DEFAULT_TOP_K if fast_mode else max(DEFAULT_TOP_K + 2, 5)
    top_k = st.slider("Top-k context", min_value=2, max_value=8, value=top_k_default, step=1)
    max_output_tokens = st.slider(
        "Max output tokens", min_value=64, max_value=512, value=DEFAULT_MAX_OUTPUT_TOKENS, step=32
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.1)
    if st.button("🛑 Stop process", use_container_width=True):
        st.session_state.stop_requested = True

    if upload is not None:
        progress_bar = st.progress(0, text="Preparing upload...")

        def update_upload_progress(value: int, text: str) -> None:
            progress_bar.progress(value, text=text)

        progress_bar.progress(10, text="Reading PDF bytes...")
        file_bytes = upload.read()
        ok, message = ingest_pdf(
            upload.name,
            file_bytes,
            embed_model,
            chroma_client,
            progress_callback=update_upload_progress,
        )
        progress_bar.empty()
        if ok:
            st.success(message)
        else:
            st.info(message)

    records = load_streamlit_file_registry(chroma_client, FILES_LEGACY)
    if not records:
        st.info("No PDF indexed yet.")
    else:
        label_map = {f"{r['filename']} ({r['chunks']} chunks)": r["sha256"] for r in records}
        selected_label = st.radio("Select PDF", list(label_map.keys()))
        selected_sha = label_map[selected_label]

        st.markdown("### Stored files")
        for rec in records:
            st.write(f"- {rec['filename']}")
            st.caption(f"URL: file://{rec['path']}")
            with open(rec["path"], "rb") as f_handle:
                st.download_button(
                    label=f"Download {rec['filename']}",
                    data=f_handle.read(),
                    file_name=rec["filename"],
                    mime="application/pdf",
                    key=f"download_{rec['sha256']}",
                )

if not selected_sha:
    st.info("Upload and select a PDF to start chatting.")
    st.stop()

current_record = next((r for r in records if r["sha256"] == selected_sha), None)
if current_record is None:
    st.error("Selected file metadata is missing.")
    st.stop()

chat_key = current_record["sha256"]
if chat_key not in st.session_state.messages_by_file:
    st.session_state.messages_by_file[chat_key] = []
if st.session_state.pending_model_intro:
    st.session_state.messages_by_file[chat_key].append(
        {
            "role": "assistant",
            "content": st.session_state.pending_model_intro,
            "is_friendly": True,
            "sources": [],
        }
    )
    st.session_state.pending_model_intro = None

st.subheader(f"Chat: {current_record['filename']}")

for message in st.session_state.messages_by_file[chat_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if not message.get("is_friendly"):
                st.caption(f"Accuracy: {message['accuracy_label']} ({message['accuracy_score']}%)")
            if "latency_seconds" in message:
                st.caption(f"Response time: {message['latency_seconds']:.2f}s")
            if message.get("sources"):
                with st.expander("Retrieved context"):
                    for idx, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**Chunk {idx}**")
                        st.write(source)

user_question = st.chat_input("Ask a question about selected PDF...")
if user_question:
    st.session_state.messages_by_file[chat_key].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            started_at = time.perf_counter()
            contexts: List[str] = []
            distances: List[float] = []
            accuracy_label, accuracy_score = "N/A", 0
            if st.session_state.stop_requested:
                answer = "Response cancelled."
                is_friendly = True
                st.session_state.stop_requested = False
            else:
                with st.spinner("Generating answer..."):
                    prior_messages = None
                    if context_memory_enabled():
                        prior_messages = session_messages_to_memory(
                            st.session_state.messages_by_file[chat_key][:-1]
                        )
                        if not prior_messages:
                            prior_messages = None
                    answer, contexts, distances = run_pdf_rag_sync(
                        question=user_question.strip(),
                        collection_name=current_record["collection_name"],
                        embedding_model=embed_model,
                        chroma_client=chroma_client,
                        ollama_base_url=OLLAMA_API_URL,
                        ollama_model=st.session_state.selected_model_name.strip(),
                        prior_messages=prior_messages,
                        top_k=top_k,
                        temperature=temperature,
                        num_predict=max_output_tokens,
                        allow_inference=reasoning_mode,
                    )
                is_friendly = is_small_talk(user_question.strip())
                if not is_friendly:
                    accuracy_label, accuracy_score = accuracy_from_signals(answer, contexts, distances)
            elapsed_seconds = time.perf_counter() - started_at

            vector_backend = (os.getenv("VECTOR_BACKEND") or "chroma").strip()
            print(
                "\n--- Question & answer (Streamlit) ---",
                f"Vector collection: {current_record['collection_name']}  (VECTOR_BACKEND={vector_backend})",
                f"PDF: {current_record.get('filename', '')}",
                f"Question: {user_question.strip()}",
                f"Answer: {answer}",
                "---------------------------------------\n",
                sep="\n",
                flush=True,
            )

            st.markdown(answer)
            if not is_friendly:
                st.caption(f"Accuracy: {accuracy_label} ({accuracy_score}%)")
            st.caption(f"Response time: {elapsed_seconds:.2f}s")
            if contexts:
                with st.expander("Retrieved context"):
                    for idx, source in enumerate(contexts, start=1):
                        st.markdown(f"**Chunk {idx}**")
                        st.write(source)

            st.session_state.messages_by_file[chat_key].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "accuracy_label": accuracy_label,
                    "accuracy_score": accuracy_score,
                    "is_friendly": is_friendly,
                    "latency_seconds": elapsed_seconds,
                    "sources": contexts,
                }
            )
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Ollama. Run `ollama serve` first.")
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 403:
                st.error(
                    "Ollama returned 403 (Forbidden). "
                    "If you selected a `*:cloud` model, switch to a local model like "
                    "`llama3.1:8b`."
                )
            else:
                st.error(f"Ollama HTTP error: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unexpected error: {exc}")
