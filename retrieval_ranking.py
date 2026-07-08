"""Configurable fusion / ranking for hybrid retrieval."""

from __future__ import annotations

import os
from typing import Dict, List, Literal, Sequence

RankingMethod = Literal["rrf", "dense", "sparse", "weighted"]

_VALID_METHODS = frozenset({"rrf", "dense", "sparse", "weighted"})


def ranking_method_from_env() -> RankingMethod:
    raw = (os.getenv("RETRIEVAL_RANKING_METHOD") or "rrf").strip().lower()
    if raw in _VALID_METHODS:
        return raw  # type: ignore[return-value]
    return "rrf"


def ranking_rrf_k() -> int:
    try:
        return max(1, int(os.getenv("HYBRID_RRF_K", "60")))
    except ValueError:
        return 60


def ranking_dense_weight() -> float:
    try:
        return max(0.0, float(os.getenv("RETRIEVAL_RANKING_DENSE_WEIGHT", "0.55")))
    except ValueError:
        return 0.55


def ranking_sparse_weight() -> float:
    try:
        return max(0.0, float(os.getenv("RETRIEVAL_RANKING_SPARSE_WEIGHT", "0.45")))
    except ValueError:
        return 0.45


def _minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def rank_candidate_ids(
    method: RankingMethod,
    *,
    dense_ids: Sequence[str],
    sparse_ids: Sequence[str],
    id_to_dense_dist: Dict[str, float],
    id_to_sparse_score: Dict[str, float],
    top_k: int,
    rrf_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
) -> List[str]:
    """
    Order candidate chunk ids and return the top_k.
    dense_ids / sparse_ids are best-first lists from each retriever.
    """
    top_k = max(1, top_k)
    k = rrf_k if rrf_k is not None else ranking_rrf_k()
    dw = dense_weight if dense_weight is not None else ranking_dense_weight()
    sw = sparse_weight if sparse_weight is not None else ranking_sparse_weight()

    if method == "dense":
        seen: set[str] = set()
        out: List[str] = []
        for cid in dense_ids:
            if cid in seen:
                continue
            seen.add(cid)
            out.append(cid)
            if len(out) >= top_k:
                return out
        return out

    if method == "sparse":
        seen = set()
        out = []
        for cid in sparse_ids:
            if cid in seen:
                continue
            seen.add(cid)
            out.append(cid)
            if len(out) >= top_k:
                return out
        return out

    if method == "weighted":
        dense_sim: Dict[str, float] = {}
        for cid in dense_ids:
            dist = id_to_dense_dist.get(cid)
            if dist is None:
                dense_sim[cid] = 0.5
            else:
                dense_sim[cid] = max(0.0, 1.0 - float(dist))
        sparse_norm = _minmax_norm({cid: id_to_sparse_score.get(cid, 0.0) for cid in sparse_ids})
        dense_norm = _minmax_norm(dense_sim)
        universe = set(dense_ids) | set(sparse_ids)
        combined: Dict[str, float] = {}
        for cid in universe:
            combined[cid] = dw * dense_norm.get(cid, 0.0) + sw * sparse_norm.get(cid, 0.0)
        ranked = sorted(combined.keys(), key=lambda c: combined[c], reverse=True)
        return ranked[:top_k]

    # rrf (default)
    from collections import defaultdict

    rrf: Dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(dense_ids):
        rrf[cid] += 1.0 / (k + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] += 1.0 / (k + rank + 1)
    if not rrf:
        return []
    ranked = sorted(rrf.keys(), key=lambda c: rrf[c], reverse=True)
    return ranked[:top_k]
