"""
Agentic RAG: after the first hybrid retrieval, optionally ask a small planner model
whether the chunks are sufficient; if not, run one refinement pass with new queries
and merge unique chunks (bounded) before generation.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Tuple

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from chroma_helpers import get_or_create_vector_collection
from hybrid_retrieval import retrieve_with_hybrid
from prompts import retrieval_query_variants


def agentic_rag_enabled_from_env() -> bool:
    return (os.getenv("AGENTIC_RAG_ENABLED") or "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def resolve_agentic_enabled(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return bool(explicit)
    return agentic_rag_enabled_from_env()


def agentic_merged_chunk_cap(top_k: int) -> int:
    try:
        cap = int(os.getenv("AGENTIC_RAG_MERGED_CAP", "12"))
    except ValueError:
        cap = 12
    return max(int(top_k), min(cap, 24))


def agentic_grader_num_predict() -> int:
    try:
        return max(80, int(os.getenv("AGENTIC_RAG_GRADER_NUM_PREDICT", "220")))
    except ValueError:
        return 220


def agentic_max_followup_queries() -> int:
    try:
        return max(0, min(3, int(os.getenv("AGENTIC_RAG_MAX_FOLLOWUP_QUERIES", "2"))))
    except ValueError:
        return 2


_GRADER_SYSTEM = (
    "You are a retrieval judge for PDF question answering. "
    "Given the user's question and short excerpts from retrieved chunks, decide if the excerpts "
    "likely contain enough evidence to answer the question factually. "
    "Treat 'guide me to run', 'how to run', 'how do I run', and 'I want to run' as the same procedural intent when judging coverage. "
    "If excerpts are empty, off-topic, or missing key entities/numbers the question asks for, "
    "propose 1–2 different search queries (keywords or short phrases) that might surface the right passage. "
    "Output ONLY valid JSON, no markdown, no explanation. Schema:\n"
    '{"sufficient": <true|false>, "follow_up_queries": [<string>, ...]}\n'
    "follow_up_queries: at most 2 strings, each under 120 characters, use English or the same language as the question."
)


def _strip_json_fence(text: str) -> str:
    raw = (text or "").strip()
    if "```" not in raw:
        return raw
    parts = raw.split("```")
    for block in parts:
        block = block.strip()
        if block.lower().startswith("json"):
            block = block[4:].lstrip()
        if block.startswith("{") and "sufficient" in block:
            return block
    return raw


def parse_grader_json(text: str) -> Tuple[bool, List[str]]:
    """Default to sufficient=True on parse failure (avoid extra latency / bad loops)."""
    raw = _strip_json_fence(text)
    data: dict
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return True, []
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return True, []

    sufficient = bool(data.get("sufficient", True))
    queries = data.get("follow_up_queries") or data.get("queries") or []
    if not isinstance(queries, list):
        return sufficient, []
    out: List[str] = []
    for q in queries:
        s = str(q).strip()
        if not s or len(s) > 200:
            continue
        out.append(s[:120])
        if len(out) >= agentic_max_followup_queries():
            break
    return sufficient, out


def _merge_retrieval_batches(
    batches: List[Tuple[List[str], List[float]]],
    cap: int,
) -> Tuple[List[str], List[float]]:
    seen: set[str] = set()
    docs_out: List[str] = []
    dists_out: List[float] = []
    for batch_docs, batch_dists in batches:
        dist_list = batch_dists or []
        for i, doc in enumerate(batch_docs or []):
            text = str(doc or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            d = float(dist_list[i]) if i < len(dist_list) else 0.45
            docs_out.append(str(doc))
            dists_out.append(d)
            if len(docs_out) >= cap:
                return docs_out, dists_out
    return docs_out, dists_out


def _context_previews(contexts: List[str], *, max_chunks: int, preview_len: int) -> str:
    lines: List[str] = []
    for i, ctx in enumerate((contexts or [])[:max_chunks], start=1):
        c = (ctx or "").strip().replace("\n", " ")
        if len(c) > preview_len:
            c = c[: preview_len - 1] + "…"
        lines.append(f"[{i}] {c}")
    return "\n".join(lines) if lines else "(no excerpts)"


def grade_retrieval_sufficiency(
    *,
    question: str,
    question_for_rag: str,
    contexts: List[str],
    ollama_base_url: str,
    ollama_model: str,
) -> Tuple[bool, List[str]]:
    previews = _context_previews(contexts, max_chunks=6, preview_len=480)
    user_block = (
        f"Original question:\n{question.strip()}\n\n"
        f"Normalized question (for retrieval):\n{(question_for_rag or question).strip()}\n\n"
        f"Retrieved excerpts:\n{previews}\n\n"
        "Return JSON only with keys sufficient (boolean) and follow_up_queries (array of strings)."
    )
    llm = ChatOllama(
        base_url=ollama_base_url.rstrip("/"),
        model=ollama_model,
        temperature=0.0,
        num_predict=agentic_grader_num_predict(),
    )
    msg = HumanMessage(content=f"{_GRADER_SYSTEM}\n\n---\n\n{user_block}")
    out = llm.invoke([msg])
    text = getattr(out, "content", None) or str(out)
    return parse_grader_json(str(text))


def run_followup_retrievals(
    *,
    queries: List[str],
    collection_name: str,
    embedding_model: Any,
    chroma_client: Any,
    top_k: int,
) -> List[Tuple[List[str], List[float]]]:
    if not queries or not collection_name:
        return []
    collection = get_or_create_vector_collection(chroma_client, collection_name)
    batches: List[Tuple[List[str], List[float]]] = []
    per_q_k = max(2, min(top_k, 6))
    for fq in queries:
        q = fq.strip()
        if not q:
            continue
        for variant in retrieval_query_variants(q)[:3]:
            docs, dists = retrieve_with_hybrid(variant, collection, embedding_model, per_q_k)
            if docs:
                batches.append((list(docs or []), list(dists or [])))
    return batches


def refine_contexts_agentic(
    *,
    question: str,
    question_for_rag: str,
    contexts: List[str],
    distances: List[float],
    collection_name: str,
    top_k: int,
    embedding_model: Any,
    chroma_client: Any,
    ollama_base_url: str,
    ollama_model: str,
) -> Tuple[List[str], List[float]]:
    """
    Single refinement round: grade → optional follow-up hybrid retrieval → merge (capped).
    """
    if not contexts:
        return contexts, distances

    sufficient, follow_ups = grade_retrieval_sufficiency(
        question=question,
        question_for_rag=question_for_rag,
        contexts=contexts,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
    )
    if sufficient or not follow_ups:
        return contexts, distances

    q_lower = (question_for_rag or question).strip().casefold()
    filtered: List[str] = []
    for fq in follow_ups:
        fl = fq.strip().casefold()
        if not fl or fl == q_lower:
            continue
        filtered.append(fq.strip())
    if not filtered:
        return contexts, distances

    cap = agentic_merged_chunk_cap(top_k)
    seed_batch = [(list(contexts or []), list(distances or []))]
    extra = run_followup_retrievals(
        queries=filtered,
        collection_name=collection_name,
        embedding_model=embedding_model,
        chroma_client=chroma_client,
        top_k=top_k,
    )
    merged_docs, merged_dists = _merge_retrieval_batches(seed_batch + extra, cap)
    if len(merged_docs) <= len(contexts):
        return contexts, distances
    return merged_docs, merged_dists
