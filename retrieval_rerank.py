"""Optional cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Sequence, Tuple


def rerank_enabled_from_env() -> bool:
    return (os.getenv("RERANK_ENABLED") or "false").strip().lower() in ("1", "true", "yes", "on")


def rerank_model_name() -> str:
    return (os.getenv("RERANK_MODEL") or "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()


def rerank_max_candidates() -> int:
    try:
        return max(4, min(80, int(os.getenv("RERANK_MAX_CANDIDATES", "24"))))
    except ValueError:
        return 24


def rerank_doc_char_limit() -> int:
    try:
        return max(300, min(4000, int(os.getenv("RERANK_DOC_CHAR_LIMIT", "1600"))))
    except ValueError:
        return 1600


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder  # pylint: disable=import-error

    return CrossEncoder(model_name)


def rerank_contexts(
    query: str,
    docs: Sequence[str],
    dists: Sequence[float],
    *,
    top_k: int,
) -> Tuple[List[str], List[float]]:
    """Rerank docs with cross-encoder; fallback to input order on failure."""
    if not docs:
        return [], []
    if not rerank_enabled_from_env():
        return list(docs)[:top_k], list(dists)[:top_k]

    max_candidates = min(len(docs), rerank_max_candidates())
    pairs = []
    trimmed_docs: List[str] = []
    for doc in list(docs)[:max_candidates]:
        text = str(doc or "").strip()
        if not text:
            continue
        text = text[: rerank_doc_char_limit()]
        pairs.append((query, text))
        trimmed_docs.append(str(doc))
    if not pairs:
        return list(docs)[:top_k], list(dists)[:top_k]

    try:
        model = _cross_encoder(rerank_model_name())
        scores = model.predict(pairs)
    except (ImportError, RuntimeError, ValueError, OSError):
        return list(docs)[:top_k], list(dists)[:top_k]

    indexed = []
    for i, score in enumerate(scores):
        dense_dist = float(dists[i]) if i < len(dists) else 0.45
        indexed.append((float(score), i, trimmed_docs[i], dense_dist))
    indexed.sort(key=lambda row: row[0], reverse=True)
    chosen = indexed[: max(1, top_k)]
    return [row[2] for row in chosen], [row[3] for row in chosen]
