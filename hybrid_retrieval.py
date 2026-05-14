"""Hybrid RAG retrieval: dense (embedding) + sparse (BM25) with reciprocal rank fusion (RRF)."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

_HYBRID_ENABLED = (os.getenv("HYBRID_SEARCH_ENABLED") or "true").strip().lower() not in ("0", "false", "no", "off")
_RRF_K = max(1, int(os.getenv("HYBRID_RRF_K", "60")))
_DENSE_POOL = max(int(os.getenv("HYBRID_DENSE_POOL", "24")), 1)
_SPARSE_POOL = max(int(os.getenv("HYBRID_SPARSE_POOL", "40")), 1)
_MAX_CHUNKS = max(100, int(os.getenv("HYBRID_MAX_TOTAL_CHUNKS", "12000")))


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1]


def _normalize_id(raw: Any) -> str:
    return str(raw)


def _dense_query(
    collection: Any,
    embedding_model: Any,
    query: str,
    n_results: int,
) -> Dict[str, Any]:
    query_embedding = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    query_vector = np.asarray(query_embedding, dtype=np.float32).tolist()[0]
    n_results = max(1, min(n_results, _MAX_CHUNKS))
    try:
        return collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "distances", "ids"],
        )
    except Exception:
        return collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "distances"],
        )


def _dense_only(
    collection: Any,
    embedding_model: Any,
    query: str,
    top_k: int,
) -> Tuple[List[str], List[float]]:
    result = _dense_query(collection, embedding_model, query, top_k)
    docs = result.get("documents", [[]])[0] or []
    distances = result.get("distances", [[]])[0] or []
    return docs, distances


def retrieve_with_hybrid(
    query: str,
    collection: Any,
    embedding_model: Any,
    top_k: int = 3,
) -> Tuple[List[str], List[float]]:
    """
    Return (context_chunks, distance_like_scores) for downstream RAG.
    Distances are dense cosine distance when known; otherwise a neutral default for UI heuristics.
    """
    top_k = max(1, top_k)
    if not _HYBRID_ENABLED:
        return _dense_only(collection, embedding_model, query, top_k)

    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return _dense_only(collection, embedding_model, query, top_k)

    try:
        got = collection.get(include=["documents", "ids"])
        all_docs: List[str] = list(got.get("documents") or [])
        all_ids: List[str] = [_normalize_id(i) for i in (got.get("ids") or [])]
        if not all_docs or not all_ids or len(all_docs) != len(all_ids):
            return _dense_only(collection, embedding_model, query, top_k)

        if len(all_docs) > _MAX_CHUNKS:
            return _dense_only(collection, embedding_model, query, top_k)

        dense_n = min(_DENSE_POOL, len(all_docs))
        dense_result = _dense_query(collection, embedding_model, query, dense_n)
        dense_docs = dense_result.get("documents", [[]])[0] or []
        dense_dist = dense_result.get("distances", [[]])[0] or []
        dense_ids_raw = dense_result.get("ids", [[]])[0] or []
        dense_ids = [_normalize_id(i) for i in dense_ids_raw]
        if len(dense_ids) != len(dense_docs):
            dense_ids = []
            for d in dense_docs:
                match_id = ""
                for j, doc in enumerate(all_docs):
                    if doc == d:
                        match_id = all_ids[j]
                        break
                dense_ids.append(match_id)
        if len(dense_ids) != len(dense_docs) or any(not cid for cid in dense_ids):
            return _dense_only(collection, embedding_model, query, top_k)

        id_to_doc = dict(zip(all_ids, all_docs))
        id_to_dense_dist: Dict[str, float] = {}
        for i, cid in enumerate(dense_ids):
            if i < len(dense_dist):
                id_to_dense_dist[cid] = float(dense_dist[i])
            elif i < len(dense_docs) and cid in id_to_doc:
                id_to_dense_dist[cid] = 0.5

        corpus_tokens = [_tokenize(d) or ["_"] for d in all_docs]
        q_tokens = _tokenize(query)
        if not q_tokens:
            out_ids = dense_ids[:top_k]
            out_docs = [id_to_doc.get(i, "") for i in out_ids]
            out_dist = [id_to_dense_dist.get(i, 0.45) for i in out_ids]
            return out_docs, out_dist

        bm25 = BM25Okapi(corpus_tokens)
        sparse_scores = bm25.get_scores(q_tokens)
        scores_arr = np.asarray(sparse_scores, dtype=np.float64)
        order = np.argsort(scores_arr)[::-1]

        sparse_ids: List[str] = []
        for idx in order:
            if len(sparse_ids) >= _SPARSE_POOL:
                break
            if float(sparse_scores[int(idx)]) <= 0:
                continue
            sparse_ids.append(all_ids[int(idx)])

        rrf: Dict[str, float] = defaultdict(float)
        for rank, cid in enumerate(dense_ids):
            rrf[cid] += 1.0 / (_RRF_K + rank + 1)
        for rank, cid in enumerate(sparse_ids):
            rrf[cid] += 1.0 / (_RRF_K + rank + 1)

        ranked = sorted(rrf.keys(), key=lambda c: rrf[c], reverse=True)[:top_k]
        out_docs = [id_to_doc[c] for c in ranked]
        out_dist = [id_to_dense_dist.get(c, 0.28) for c in ranked]
        return out_docs, out_dist
    except Exception:
        return _dense_only(collection, embedding_model, query, top_k)
