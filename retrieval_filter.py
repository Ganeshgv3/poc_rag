"""Metadata filters for vector retrieval (Chroma `where` / in-memory / Qdrant)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MetaFilter = Dict[str, Any]


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or ("true" if default else "false")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def meta_filter_enabled_from_env() -> bool:
    return _truthy_env("RETRIEVAL_META_FILTER_ENABLED", default=True)


def parse_meta_filter_json(raw: str) -> MetaFilter:
    text = (raw or "").strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("metadata filter must be a JSON object")
    return dict(data)


def meta_filter_from_env() -> MetaFilter:
    raw = (os.getenv("RETRIEVAL_META_FILTER") or "").strip()
    if not raw:
        return {}
    try:
        return parse_meta_filter_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {}


def sha256_meta_filter(sha256: str) -> MetaFilter:
    digest = (sha256 or "").strip()
    if not digest:
        return {}
    return {"sha256": digest}


def merge_meta_filters(*filters: Optional[MetaFilter]) -> MetaFilter:
    merged: MetaFilter = {}
    for f in filters:
        if not f:
            continue
        for key, value in f.items():
            if value is None:
                continue
            merged[str(key)] = value
    return merged


def resolve_metadata_filter(
    *,
    explicit: Optional[MetaFilter] = None,
    document_sha256: Optional[str] = None,
    use_env: bool = True,
) -> Optional[MetaFilter]:
    """
    Build the effective metadata filter for a retrieval call.
    Returns None when filtering is disabled or no constraints are set.
    """
    if not meta_filter_enabled_from_env() and explicit is None and not document_sha256:
        return None

    parts: List[MetaFilter] = []
    if use_env:
        env_f = meta_filter_from_env()
        if env_f:
            parts.append(env_f)
    if document_sha256:
        parts.append(sha256_meta_filter(document_sha256))
    if explicit:
        parts.append(explicit)

    merged = merge_meta_filters(*parts)
    if not merged:
        return None
    return merged


def chroma_where_clause(meta_filter: Optional[MetaFilter]) -> Optional[Dict[str, Any]]:
    """Normalize filter dict for ChromaDB `where` (equality or operator objects)."""
    if not meta_filter:
        return None
    where: Dict[str, Any] = {}
    for key, value in meta_filter.items():
        if value is None:
            continue
        if isinstance(value, dict):
            where[str(key)] = value
        else:
            where[str(key)] = value
    return where or None


def _match_scalar(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        for op, bound in expected.items():
            op_l = str(op).lower()
            try:
                av = actual
                bv = bound
                if op_l in ("$eq", "eq"):
                    if av != bv:
                        return False
                elif op_l in ("$ne", "ne"):
                    if av == bv:
                        return False
                elif op_l in ("$gte", "gte"):
                    if av is None or av < bv:
                        return False
                elif op_l in ("$gt", "gt"):
                    if av is None or av <= bv:
                        return False
                elif op_l in ("$lte", "lte"):
                    if av is None or av > bv:
                        return False
                elif op_l in ("$lt", "lt"):
                    if av is None or av >= bv:
                        return False
                elif op_l in ("$in", "in"):
                    if av not in (bound if isinstance(bound, (list, tuple, set)) else [bound]):
                        return False
                else:
                    return False
            except TypeError:
                return False
        return True
    return actual == expected


def metadata_matches(meta: Optional[Mapping[str, Any]], meta_filter: Optional[MetaFilter]) -> bool:
    if not meta_filter:
        return True
    row = dict(meta or {})
    for key, expected in meta_filter.items():
        if not _match_scalar(row.get(key), expected):
            return False
    return True


def filter_indexed_rows(
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Optional[Sequence[Optional[Mapping[str, Any]]]],
    meta_filter: Optional[MetaFilter],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Keep rows whose metadata satisfies meta_filter."""
    if not meta_filter:
        meta_list = [dict(m or {}) for m in (metadatas or [])]
        return list(ids), list(documents), meta_list

    out_ids: List[str] = []
    out_docs: List[str] = []
    out_meta: List[Dict[str, Any]] = []
    meta_seq = list(metadatas or [])
    for i, doc_id in enumerate(ids):
        meta = dict(meta_seq[i]) if i < len(meta_seq) else {}
        if not metadata_matches(meta, meta_filter):
            continue
        out_ids.append(str(doc_id))
        out_docs.append(str(documents[i]) if i < len(documents) else "")
        out_meta.append(meta)
    return out_ids, out_docs, out_meta


def build_qdrant_filter(meta_filter: Optional[MetaFilter]) -> Any:
    """Map a simple metadata filter to a Qdrant Filter (equality and numeric ranges)."""
    if not meta_filter:
        return None
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range
    except ModuleNotFoundError:
        return None

    must: List[Any] = []
    for key, value in meta_filter.items():
        field = str(key)
        if isinstance(value, dict):
            gte = value.get("$gte", value.get("gte"))
            lte = value.get("$lte", value.get("lte"))
            gt = value.get("$gt", value.get("gt"))
            lt = value.get("$lt", value.get("lt"))
            eq = value.get("$eq", value.get("eq"))
            if eq is not None:
                must.append(FieldCondition(key=field, match=MatchValue(value=eq)))
            elif any(v is not None for v in (gte, lte, gt, lt)):
                must.append(
                    FieldCondition(
                        key=field,
                        range=Range(gte=gte, lte=lte, gt=gt, lt=lt),
                    )
                )
            else:
                continue
        else:
            must.append(FieldCondition(key=field, match=MatchValue(value=value)))
    if not must:
        return None
    return Filter(must=must)
