"""
LangGraph-orchestrated PDF RAG with LangChain ChatOllama.

- LangGraph: explicit routing (small-talk short-circuit, retrieve, binary / not-found / LLM).
- LangChain: ChatOllama + LCEL (messages -> model -> string) for the grounded answer path.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langgraph.graph import END, START, StateGraph

from chroma_helpers import get_or_create_vector_collection
from hybrid_retrieval import retrieve_with_hybrid
from prompts import (
    build_chat_messages_for_ollama,
    expand_question_shorthand,
    retrieval_query_variants,
)
from rag_routing import (
    binary_yes_no_from_context,
    clean_answer_text,
    friendly_not_found_reply,
    is_binary_question_prefix,
    is_small_talk,
    small_talk_reply,
)


class RagGraphState(TypedDict, total=False):
    question: str
    question_for_rag: str
    collection_name: str
    top_k: int
    contexts: List[str]
    distances: List[float]
    prior_messages: Optional[List[Dict[str, str]]]
    ollama_base_url: str
    ollama_model: str
    temperature: float
    num_predict: int
    allow_inference: bool
    answer: str


def _dict_messages_to_lc(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def _merge_hybrid_retrieval_batches(
    batches: List[Tuple[List[str], List[float]]],
    top_k: int,
) -> Tuple[List[str], List[float]]:
    """Deduplicate chunk text (case-folded), preserve first-seen order, cap at top_k."""
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
            if len(docs_out) >= top_k:
                return docs_out, dists_out
    return docs_out, dists_out


@lru_cache(maxsize=32)
def _chat_ollama(base_url: str, model: str, temperature: float, num_predict: int) -> ChatOllama:
    return ChatOllama(
        base_url=base_url.rstrip("/"),
        model=model,
        temperature=temperature,
        num_predict=num_predict,
    )


def _lc_rag_answer_chain(base_url: str, model: str, temperature: float, num_predict: int) -> RunnableSerializable:
    llm = _chat_ollama(base_url, model, temperature, num_predict)

    def to_messages(payload: Dict[str, Any]) -> List[BaseMessage]:
        return _dict_messages_to_lc(
            build_chat_messages_for_ollama(
                payload["question_for_rag"],
                payload["contexts"],
                allow_inference=bool(payload.get("allow_inference", True)),
                prior_messages=payload.get("prior_messages"),
            )
        )

    return RunnableLambda(to_messages) | llm | StrOutputParser()


def _expand_node(state: RagGraphState) -> Dict[str, Any]:
    q = (state.get("question") or "").strip()
    return {"question_for_rag": expand_question_shorthand(q)}


def _small_talk_node(state: RagGraphState) -> Dict[str, Any]:
    raw = (state.get("question") or "").strip()
    return {"answer": clean_answer_text(small_talk_reply(raw))}


def _make_retrieve_node(embedding_model: Any, chroma_client: Any):
    def retrieve_node(state: RagGraphState) -> Dict[str, Any]:
        name = state.get("collection_name") or ""
        if not name:
            return {"contexts": [], "distances": []}
        top_k = int(state.get("top_k") or 3)
        collection = get_or_create_vector_collection(chroma_client, name)
        q = state.get("question_for_rag") or ""
        batches: List[Tuple[List[str], List[float]]] = []
        for variant in retrieval_query_variants(q)[:4]:
            docs, dists = retrieve_with_hybrid(variant, collection, embedding_model, top_k)
            if docs:
                batches.append((list(docs or []), list(dists or [])))
        if not batches:
            return {"contexts": [], "distances": []}
        merged_docs, merged_dists = _merge_hybrid_retrieval_batches(batches, top_k)
        return {"contexts": merged_docs, "distances": merged_dists}

    return retrieve_node


def _binary_node(state: RagGraphState) -> Dict[str, Any]:
    q = state.get("question_for_rag") or ""
    ctx = state.get("contexts") or []
    return {"answer": clean_answer_text(binary_yes_no_from_context(q, ctx))}


def _not_found_node(state: RagGraphState) -> Dict[str, Any]:
    raw = (state.get("question") or "").strip()
    return {"answer": clean_answer_text(friendly_not_found_reply(raw))}


def _llm_node(state: RagGraphState) -> Dict[str, Any]:
    chain = _lc_rag_answer_chain(
        str(state.get("ollama_base_url") or "http://localhost:11434"),
        str(state.get("ollama_model") or "llama3.1:8b"),
        float(state.get("temperature") if state.get("temperature") is not None else 0.2),
        int(state.get("num_predict") or 220),
    )
    payload = {
        "question_for_rag": state.get("question_for_rag") or "",
        "contexts": state.get("contexts") or [],
        "allow_inference": state.get("allow_inference", True),
        "prior_messages": state.get("prior_messages"),
    }
    text = chain.invoke(payload)
    return {"answer": clean_answer_text(str(text or "").strip())}


def _after_expand(state: RagGraphState) -> Literal["small_talk", "retrieve"]:
    if is_small_talk((state.get("question") or "").strip()):
        return "small_talk"
    return "retrieve"


def _after_retrieve(state: RagGraphState) -> Literal["not_found", "binary", "llm"]:
    contexts = state.get("contexts") or []
    if not contexts:
        return "not_found"
    if is_binary_question_prefix(state.get("question_for_rag") or ""):
        return "binary"
    return "llm"


def _build_pdf_rag_graph(embedding_model: Any, chroma_client: Any) -> StateGraph:
    g: StateGraph = StateGraph(RagGraphState)
    g.add_node("expand", _expand_node)
    g.add_node("small_talk", _small_talk_node)
    g.add_node("retrieve", _make_retrieve_node(embedding_model, chroma_client))
    g.add_node("binary", _binary_node)
    g.add_node("not_found", _not_found_node)
    g.add_node("llm", _llm_node)

    g.add_edge(START, "expand")
    g.add_conditional_edges(
        "expand",
        _after_expand,
        {"small_talk": "small_talk", "retrieve": "retrieve"},
    )
    g.add_edge("small_talk", END)
    g.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {"not_found": "not_found", "binary": "binary", "llm": "llm"},
    )
    g.add_edge("not_found", END)
    g.add_edge("binary", END)
    g.add_edge("llm", END)
    return g


_compiled_full = None
_compiled_until_llm = None
_graph_cache_key: Optional[Tuple[int, int]] = None


def get_compiled_rag_graphs(embedding_model: Any, chroma_client: Any):
    """Return (full_graph, graph_that_interrupts_before_llm) for sync vs streaming."""
    global _compiled_full, _compiled_until_llm, _graph_cache_key
    key = (id(embedding_model), id(chroma_client))
    if _graph_cache_key != key or _compiled_full is None or _compiled_until_llm is None:
        builder = _build_pdf_rag_graph(embedding_model, chroma_client)
        _compiled_full = builder.compile()
        _compiled_until_llm = builder.compile(interrupt_before=["llm"])
        _graph_cache_key = key
    return _compiled_full, _compiled_until_llm


def _fresh_thread_config() -> Dict[str, Any]:
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


def run_pdf_rag_sync(
    *,
    question: str,
    collection_name: str,
    embedding_model: Any,
    chroma_client: Any,
    ollama_base_url: str,
    ollama_model: str,
    prior_messages: Optional[List[Dict[str, str]]] = None,
    top_k: int = 3,
    temperature: float = 0.2,
    num_predict: int = 220,
    allow_inference: bool = True,
) -> Tuple[str, List[str], List[float]]:
    """Run full LangGraph + LangChain path. Returns (answer, contexts, distances)."""
    full, _ = get_compiled_rag_graphs(embedding_model, chroma_client)
    initial: RagGraphState = {
        "question": question.strip(),
        "collection_name": collection_name,
        "top_k": top_k,
        "prior_messages": prior_messages,
        "ollama_base_url": ollama_base_url,
        "ollama_model": ollama_model,
        "temperature": temperature,
        "num_predict": num_predict,
        "allow_inference": allow_inference,
    }
    out = full.invoke(initial, _fresh_thread_config())
    return (
        str(out.get("answer") or "").strip(),
        list(out.get("contexts") or []),
        list(out.get("distances") or []),
    )


def stream_pdf_rag_llm_tokens(
    *,
    question: str,
    collection_name: str,
    embedding_model: Any,
    chroma_client: Any,
    ollama_base_url: str,
    ollama_model: str,
    prior_messages: Optional[List[Dict[str, str]]] = None,
    top_k: int = 3,
    temperature: float = 0.2,
    num_predict: int = 220,
    allow_inference: bool = True,
) -> Tuple[List[str], List[float], Optional[str], Iterator[str]]:
    """
    Run graph until the LLM node when streaming is required.

    Returns (contexts, distances, prefilled_answer_or_none, token_iterator).
    When prefilled_answer_or_none is set, the iterator is empty and the caller should use it as the full answer.
    """
    _, partial = get_compiled_rag_graphs(embedding_model, chroma_client)
    initial: RagGraphState = {
        "question": question.strip(),
        "collection_name": collection_name,
        "top_k": top_k,
        "prior_messages": prior_messages,
        "ollama_base_url": ollama_base_url,
        "ollama_model": ollama_model,
        "temperature": temperature,
        "num_predict": num_predict,
        "allow_inference": allow_inference,
    }
    out = partial.invoke(initial, _fresh_thread_config())
    contexts = list(out.get("contexts") or [])
    distances = list(out.get("distances") or [])
    if out.get("answer"):
        return contexts, distances, str(out["answer"]), iter(())

    def _gen() -> Iterator[str]:
        llm = _chat_ollama(ollama_base_url, ollama_model, temperature, num_predict)
        messages = _dict_messages_to_lc(
            build_chat_messages_for_ollama(
                out.get("question_for_rag") or "",
                contexts,
                allow_inference=allow_inference,
                prior_messages=prior_messages,
            )
        )
        for chunk in llm.stream(messages):
            piece = getattr(chunk, "content", None) or ""
            if piece:
                yield str(piece)

    return contexts, distances, None, _gen()
