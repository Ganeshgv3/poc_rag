"""Source citation helpers for retrieved contexts."""

from __future__ import annotations

import re
from typing import Any, Dict, List

_HEADER_RE = re.compile(r"^\[Page\s+(?P<page>\d+)\s+\|\s+(?P<kind>[^\]]+)\]\s*$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^Section:\s*(?P<section>.+)$", re.IGNORECASE)


def _strip_chunk_header(text: str) -> str:
    lines = (text or "").splitlines()
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if _HEADER_RE.match(s):
            continue
        if _SECTION_RE.match(s):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _chunk_meta(text: str) -> Dict[str, Any]:
    page = None
    kind = "text"
    section = ""
    for ln in (text or "").splitlines()[:4]:
        s = ln.strip()
        hm = _HEADER_RE.match(s)
        if hm:
            page = int(hm.group("page"))
            kind = hm.group("kind").strip().lower()
            continue
        sm = _SECTION_RE.match(s)
        if sm:
            section = sm.group("section").strip()
    return {"page": page, "content_type": kind, "section": section}


def build_source_citations(contexts: List[str], *, max_sources: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, ctx in enumerate(contexts or [], start=1):
        meta = _chunk_meta(ctx)
        body = _strip_chunk_header(ctx)
        snippet = re.sub(r"\s+", " ", body).strip()
        if len(snippet) > 220:
            snippet = snippet[:219] + "..."
        key = f"{meta.get('page')}|{meta.get('section')}|{snippet[:80]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "rank": idx,
                "page": meta.get("page"),
                "content_type": meta.get("content_type"),
                "section": meta.get("section"),
                "snippet": snippet,
            }
        )
        if len(out) >= max(1, max_sources):
            break
    return out
