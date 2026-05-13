"""
PDF text normalization and RAG chunking.

Splits on paragraphs / sentences / lines when possible instead of fixed character
cuts, and cleans common PDF extraction artefacts.
"""

from __future__ import annotations

import os
import re
from typing import List


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(100, int(raw))
    except ValueError:
        return default


def default_chunk_size() -> int:
    return _env_int("PDF_CHUNK_SIZE", 1100)


def default_chunk_overlap() -> int:
    return _env_int("PDF_CHUNK_OVERLAP", 180)


def normalize_pdf_text(text: str) -> str:
    """Clean PyMuPDF-style text: junk control chars, hyphenated line breaks, whitespace."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\x00", "")
    t = t.replace("\ufeff", "")
    # Hyphenation at line wrap: "exam-\nple" -> "example"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # Merge lines that are obviously one sentence (single newlines inside a block)
    t = re.sub(r"([a-z,;])\n([a-z])", r"\1 \2", t)
    # Collapse horizontal whitespace; keep paragraph breaks
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_text(pdf_bytes: bytes) -> str:
    """Extract and normalize plain text from a PDF."""
    import fitz  # PyMuPDF; lazy import so chunking helpers stay importable without it

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        parts: List[str] = []
        for page in doc:
            parts.append(page.get_text("text") or "")
    finally:
        doc.close()
    return normalize_pdf_text("\n\n".join(parts))


def _find_chunk_end(text: str, start: int, chunk_size: int) -> int:
    """Exclusive index to cut at; prefers paragraph / sentence / line / word before hard limit."""
    text_len = len(text)
    hard_end = min(start + chunk_size, text_len)
    if hard_end >= text_len:
        return text_len

    window = text[start:hard_end]
    min_take = max(80, chunk_size // 3)
    min_rel = min_take if len(window) >= min_take else 0

    def rel_ok(rel_end: int) -> bool:
        return rel_end >= min_rel or start + rel_end >= text_len - 1

    # Prefer largest break at or before window end (search backward)
    for sep in ("\n\n", ". ", ".\n", "? ", "?\n", "! ", "!\n", "\n"):
        rel = window.rfind(sep)
        if rel >= 0 and rel_ok(rel + len(sep)):
            return min(start + rel + len(sep), text_len)

    sp = window.rfind(" ")
    if sp >= 0 and rel_ok(sp + 1):
        return min(start + sp + 1, text_len)

    return hard_end


def _next_chunk_start(text: str, start: int, end: int, overlap: int) -> int:
    """
    First index of the next chunk. Reuses up to `overlap` characters from the tail
    of [start:end), without moving backward before `start`.
    """
    text_len = len(text)
    if end >= text_len:
        return text_len
    span = end - start
    if span <= 0:
        return min(start + 1, text_len)
    # Share min(overlap, span-1) chars so we always advance when more text remains
    shared = min(overlap, max(1, span - 1))
    nxt = end - shared
    if nxt <= start:
        nxt = end
    # Nudge forward to a line or word boundary without skipping the overlap region entirely
    probe = text[nxt:min(end, nxt + 64)]
    cut_nl = probe.find("\n")
    cut_sp = probe.find(" ")
    if cut_nl >= 0 and cut_sp >= 0:
        cut = min(cut_nl, cut_sp)
    else:
        cut = cut_nl if cut_nl >= 0 else cut_sp
    if cut > 0:
        nxt = min(nxt + cut + 1, end)
    return nxt


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    min_chunk_chars: int = 50,
) -> List[str]:
    """
    Build overlapping chunks with natural boundaries when possible.

    min_chunk_chars: very short tail pieces are merged into the previous chunk.
    """
    chunk_size = chunk_size if chunk_size is not None else default_chunk_size()
    overlap = overlap if overlap is not None else default_chunk_overlap()
    overlap = min(overlap, chunk_size // 2)

    text = normalize_pdf_text(text)
    if not text:
        return []

    raw_chunks: List[str] = []
    start = 0
    text_len = len(text)
    guard = 0
    max_iters = max(16, (text_len // max(1, chunk_size - overlap)) + 32)
    _it = 0

    while start < text_len:
        _it += 1
        if _it > max_iters:
            tail = text[start:].strip()
            if tail:
                raw_chunks.append(tail)
            break
        end = _find_chunk_end(text, start, chunk_size)
        if end <= start:
            end = min(start + min(chunk_size, text_len - start), text_len)
        piece = text[start:end].strip()
        if piece:
            raw_chunks.append(piece)
        if end >= text_len:
            break
        next_start = _next_chunk_start(text, start, end, overlap)
        if next_start <= start:
            guard += 1
            next_start = min(end + 1, text_len) if end < text_len else text_len
            if guard > text_len:
                break
        else:
            guard = 0
        start = next_start

    merged: List[str] = []
    for c in raw_chunks:
        if not c:
            continue
        if merged and len(c) < min_chunk_chars:
            merged[-1] = f"{merged[-1]}\n\n{c}".strip()
        else:
            merged.append(c)

    return merged
