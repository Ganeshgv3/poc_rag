"""Hybrid RAG retrieval: dense (embedding) + sparse (BM25) with configurable ranking."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from retrieval_filter import (
    MetaFilter,
    chroma_where_clause,
    filter_indexed_rows,
    metadata_matches,
)
from retrieval_ranking import RankingMethod, rank_candidate_ids, ranking_method_from_env

_HYBRID_ENABLED = (os.getenv("HYBRID_SEARCH_ENABLED") or "true").strip().lower() not in ("0", "false", "no", "off")
_DENSE_POOL = max(int(os.getenv("HYBRID_DENSE_POOL", "24")), 1)
_SPARSE_POOL = max(int(os.getenv("HYBRID_SPARSE_POOL", "40")), 1)
_MAX_CHUNKS = max(100, int(os.getenv("HYBRID_MAX_TOTAL_CHUNKS", "12000")))


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1]


def _normalize_id(raw: Any) -> str:
    return str(raw)


def _collection_get(
    collection: Any,
    *,
    include: List[str],
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"include": include}
    if where:
        kwargs["where"] = where
    try:
        return collection.get(**kwargs)
    except TypeError:
        if where:
            got = collection.get(include=include)
            return _apply_where_to_get_result(got, where, include)
        return collection.get(include=include)


def _apply_where_to_get_result(
    got: Dict[str, Any],
    where: Dict[str, Any],
    include: List[str],
) -> Dict[str, Any]:
    ids = list(got.get("ids") or [])
    docs = list(got.get("documents") or [])
    metas = list(got.get("metadatas") or [])
    f_ids, f_docs, f_metas = filter_indexed_rows(ids, docs, metas, where)
    out: Dict[str, Any] = {"ids": f_ids}
    if "documents" in include:
        out["documents"] = f_docs
    if "metadatas" in include:
        out["metadatas"] = f_metas
    return out


def _dense_query(
    collection: Any,
    embedding_model: Any,
    query: str,
    n_results: int,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    query_embedding = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    query_vector = np.asarray(query_embedding, dtype=np.float32).tolist()[0]
    n_results = max(1, min(n_results, _MAX_CHUNKS))
    include = ["documents", "distances", "ids", "metadatas"]
    try:
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": n_results,
            "include": include,
        }
        if where:
            kwargs["where"] = where
        return collection.query(**kwargs)
    except TypeError:
        kwargs = {
            "query_embeddings": [query_vector],
            "n_results": n_results,
            "include": ["documents", "distances", "ids"],
        }
        if where:
            kwargs["where"] = where
        try:
            return collection.query(**kwargs)
        except TypeError:
            pass
    except Exception:
        pass
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
    metadata_filter: Optional[MetaFilter] = None,
) -> Tuple[List[str], List[float]]:
    where = chroma_where_clause(metadata_filter)
    result = _dense_query(collection, embedding_model, query, top_k, where=where)
    docs = result.get("documents", [[]])[0] or []
    distances = result.get("distances", [[]])[0] or []
    metas = (result.get("metadatas") or [[]])[0] or []
    if metadata_filter and metas:
        filtered_docs: List[str] = []
        filtered_dists: List[float] = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            if metadata_matches(meta, metadata_filter):
                filtered_docs.append(doc)
                filtered_dists.append(float(distances[i]) if i < len(distances) else 0.45)
        if filtered_docs:
            return filtered_docs[:top_k], filtered_dists[:top_k]
    return docs, distances


def _load_corpus(
    collection: Any,
    metadata_filter: Optional[MetaFilter],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    where = chroma_where_clause(metadata_filter)
    include = ["documents", "ids", "metadatas"]
    got = _collection_get(collection, include=include, where=where)
    all_docs: List[str] = list(got.get("documents") or [])
    all_ids: List[str] = [_normalize_id(i) for i in (got.get("ids") or [])]
    all_metas: List[Dict[str, Any]] = [dict(m or {}) for m in (got.get("metadatas") or [])]
    if metadata_filter and (not where or len(all_ids) != len(all_docs)):
        all_ids, all_docs, all_metas = filter_indexed_rows(all_ids, all_docs, all_metas, metadata_filter)
    elif not all_metas and all_ids:
        all_metas = [{} for _ in all_ids]
    return all_ids, all_docs, all_metas


def retrieve_with_hybrid(
    query: str,
    collection: Any,
    embedding_model: Any,
    top_k: int = 3,
    metadata_filter: Optional[MetaFilter] = None,
    ranking_method: Optional[RankingMethod] = None,
) -> Tuple[List[str], List[float]]:
    """
    Return (context_chunks, distance_like_scores) for downstream RAG.
    Distances are dense cosine distance when known; otherwise a neutral default for UI heuristics.

    metadata_filter: restrict candidates (filename, sha256, chunk_index, Chroma operators).
    ranking_method: rrf | dense | sparse | weighted (defaults from RETRIEVAL_RANKING_METHOD).
    """
    top_k = max(1, top_k)
    method: RankingMethod = ranking_method or ranking_method_from_env()

    if method == "dense" or not _HYBRID_ENABLED:
        return _dense_only(collection, embedding_model, query, top_k, metadata_filter)

    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return _dense_only(collection, embedding_model, query, top_k, metadata_filter)

    try:
        all_ids, all_docs, _all_metas = _load_corpus(collection, metadata_filter)
        if not all_docs or not all_ids or len(all_docs) != len(all_ids):
            return _dense_only(collection, embedding_model, query, top_k, metadata_filter)

        if len(all_docs) > _MAX_CHUNKS:
            return _dense_only(collection, embedding_model, query, top_k, metadata_filter)

        where = chroma_where_clause(metadata_filter)
        dense_n = min(_DENSE_POOL, len(all_docs))
        dense_result = _dense_query(collection, embedding_model, query, dense_n, where=where)
        dense_docs = dense_result.get("documents", [[]])[0] or []
        dense_dist = dense_result.get("distances", [[]])[0] or []
        dense_ids_raw = dense_result.get("ids", [[]])[0] or []
        dense_metas = (dense_result.get("metadatas") or [[]])[0] or []
        dense_ids = [_normalize_id(i) for i in dense_ids_raw]

        if metadata_filter:
            kept_docs: List[str] = []
            kept_ids: List[str] = []
            kept_dists: List[float] = []
            for i, doc in enumerate(dense_docs):
                meta = dense_metas[i] if i < len(dense_metas) else {}
                if not metadata_matches(meta, metadata_filter):
                    continue
                kept_docs.append(doc)
                kept_ids.append(dense_ids[i] if i < len(dense_ids) else "")
                kept_dists.append(float(dense_dist[i]) if i < len(dense_dist) else 0.5)
            dense_docs, dense_ids, dense_dist = kept_docs, kept_ids, kept_dists

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
            return _dense_only(collection, embedding_model, query, top_k, metadata_filter)

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
        sparse_scores_arr = bm25.get_scores(q_tokens)
        scores_arr = np.asarray(sparse_scores_arr, dtype=np.float64)
        order = np.argsort(scores_arr)[::-1]

        sparse_ids: List[str] = []
        id_to_sparse_score: Dict[str, float] = {}
        for idx in order:
            if len(sparse_ids) >= _SPARSE_POOL:
                break
            score = float(sparse_scores_arr[int(idx)])
            if score <= 0:
                continue
            cid = all_ids[int(idx)]
            sparse_ids.append(cid)
            id_to_sparse_score[cid] = score

        ranked_ids = rank_candidate_ids(
            method,
            dense_ids=dense_ids,
            sparse_ids=sparse_ids,
            id_to_dense_dist=id_to_dense_dist,
            id_to_sparse_score=id_to_sparse_score,
            top_k=top_k,
        )
        out_docs = [id_to_doc[c] for c in ranked_ids if c in id_to_doc]
        out_dist = [id_to_dense_dist.get(c, 0.28) for c in ranked_ids]
        return out_docs, out_dist
    except Exception:
        return _dense_only(collection, embedding_model, query, top_k, metadata_filter)
