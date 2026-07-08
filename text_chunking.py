"""
PDF text normalization and RAG chunking.

- Table-aware extraction via PyMuPDF ``find_tables`` (markdown) plus prose blocks in
  reading order, with table regions deduplicated from plain text.
- Prose: paragraph / sentence boundaries with overlap.
- Tables: row-batched chunks with repeated header rows and page context for retrieval.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple

ContentKind = Literal["prose", "table"]


@dataclass(frozen=True)
class ContentBlock:
    kind: ContentKind
    page: int
    text: str
    section: str = ""


@dataclass(frozen=True)
class TextChunk:
    text: str
    content_type: ContentKind
    page: int
    section: str = ""


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = (os.getenv(name) or ("true" if default else "false")).strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return raw in ("1", "true", "yes", "on", "")


def default_chunk_size() -> int:
    return _env_int("PDF_CHUNK_SIZE", 1100, minimum=100)


def default_chunk_overlap() -> int:
    return _env_int("PDF_CHUNK_OVERLAP", 180, minimum=0)


def default_table_max_rows_per_chunk() -> int:
    return _env_int("PDF_TABLE_MAX_ROWS_PER_CHUNK", 18, minimum=4)


def default_table_row_overlap() -> int:
    return _env_int("PDF_TABLE_ROW_OVERLAP", 2, minimum=0)


def table_extraction_enabled() -> bool:
    return _env_bool("PDF_TABLE_EXTRACTION_ENABLED", default=True)


def normalize_pdf_text(text: str, *, preserve_table_layout: bool = False) -> str:
    """Clean PyMuPDF-style text: junk control chars, hyphenated line breaks, whitespace."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\x00", "")
    t = t.replace("\ufeff", "")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    if not preserve_table_layout:
        t = re.sub(r"([a-z,;])\n([a-z])", r"\1 \2", t)
    if preserve_table_layout:
        lines = [ln.rstrip() for ln in t.split("\n")]
    else:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _rect_overlap_ratio(inner: Any, outer: Any) -> float:
    """Share of ``inner`` area covered by intersection with ``outer`` (0..1)."""
    ix0 = max(inner.x0, outer.x0)
    iy0 = max(inner.y0, outer.y0)
    ix1 = min(inner.x1, outer.x1)
    iy1 = min(inner.y1, outer.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    inner_area = max(1e-6, (inner.x1 - inner.x0) * (inner.y1 - inner.y0))
    return inter / inner_area


def _block_text(block: dict) -> str:
    parts: List[str] = []
    for line in block.get("lines") or []:
        spans = line.get("spans") or []
        line_text = "".join(str(s.get("text", "")) for s in spans)
        if line_text.strip():
            parts.append(line_text)
    return "\n".join(parts).strip()


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if s.endswith(":") and len(s.split()) <= 12:
        return True
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4:
        upper = sum(1 for c in letters if c.isupper())
        if upper / len(letters) >= 0.75 and len(s.split()) <= 14:
            return True
    return False


def _detect_section_heading(text: str) -> str:
    for line in text.split("\n")[:4]:
        if _looks_like_heading(line):
            return line.strip()[:200]
    return ""


def _prose_looks_like_table(text: str) -> bool:
    """Heuristic when ``find_tables`` misses a grid but text is columnar."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return False
    pipe_lines = sum(1 for ln in lines if ln.count("|") >= 2)
    if pipe_lines >= max(2, len(lines) // 2):
        return True
    col_lines = sum(1 for ln in lines if re.search(r"\S {2,}\S {2,}\S", ln))
    if col_lines >= max(3, (len(lines) * 2) // 3):
        return True
    # After PDF normalization, columns may be single-space separated
    rows_3 = sum(1 for ln in lines if len(ln.split()) >= 3)
    return rows_3 >= max(3, (len(lines) * 2) // 3)


def _columnar_text_to_markdown(text: str) -> str:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    rows: List[List[str]] = []
    for ln in lines:
        if "|" in ln:
            cells = [c.strip() for c in ln.split("|") if c.strip()]
        elif re.search(r" {2,}", ln):
            cells = [c.strip() for c in re.split(r" {2,}", ln) if c.strip()]
        else:
            cells = ln.split()
        if cells:
            rows.append(cells)
    if not rows:
        return text
    ncol = max(len(r) for r in rows)
    norm = [r + [""] * (ncol - len(r)) for r in rows]
    md_lines = ["| " + " | ".join(r) + " |" for r in norm]
    if len(md_lines) >= 2:
        sep = "| " + " | ".join(["---"] * ncol) + " |"
        return "\n".join([md_lines[0], sep] + md_lines[1:])
    return "\n".join(md_lines)


def _table_to_markdown(table: Any) -> str:
    try:
        md = table.to_markdown()
        if md and md.strip():
            return md.strip()
    except Exception:
        pass
    try:
        rows = table.extract()
        if not rows:
            return ""
        lines: List[str] = []
        for row in rows:
            cells = [str(c or "").replace("|", "/").strip() for c in row]
            lines.append("| " + " | ".join(cells) + " |")
        if len(lines) >= 2:
            ncol = len(rows[0])
            sep = "| " + " | ".join(["---"] * ncol) + " |"
            return "\n".join([lines[0], sep] + lines[1:])
        return "\n".join(lines)
    except Exception:
        return ""


def _page_ordered_blocks(page: Any, *, use_tables: bool) -> List[ContentBlock]:
    import fitz

    page_no = int(page.number) + 1
    items: List[Tuple[float, float, ContentBlock]] = []
    table_rects: List[Any] = []

    if use_tables:
        try:
            finder = page.find_tables()
            tables = getattr(finder, "tables", None) or []
            for table in tables:
                md = _table_to_markdown(table)
                if not md:
                    continue
                bbox = fitz.Rect(table.bbox)
                table_rects.append(bbox)
                section = _detect_section_heading(md)
                items.append(
                    (
                        bbox.y0,
                        bbox.x0,
                        ContentBlock(kind="table", page=page_no, text=md, section=section),
                    )
                )
        except Exception:
            table_rects = []

    try:
        page_dict = page.get_text("dict") or {}
    except Exception:
        page_dict = {}

    for block in page_dict.get("blocks") or []:
        if block.get("type") != 0:
            continue
        raw = _block_text(block)
        if not raw:
            continue
        bbox = fitz.Rect(block["bbox"])
        if table_rects:
            covered = sum(1 for tr in table_rects if _rect_overlap_ratio(bbox, tr) > 0.55)
            if covered:
                continue
        if _prose_looks_like_table(raw):
            text = normalize_pdf_text(raw, preserve_table_layout=True)
            if not text:
                continue
            section = _detect_section_heading(text)
            kind = "table"
            body = _columnar_text_to_markdown(text) or text
        else:
            text = normalize_pdf_text(raw)
            if not text:
                continue
            section = _detect_section_heading(text)
            kind = "prose"
            body = text
        items.append(
            (
                bbox.y0,
                bbox.x0,
                ContentBlock(kind=kind, page=page_no, text=body, section=section),
            )
        )

    if not items:
        fallback = normalize_pdf_text(page.get_text("text") or "")
        if fallback:
            items.append((0.0, 0.0, ContentBlock(kind="prose", page=page_no, text=fallback)))

    items.sort(key=lambda t: (t[0], t[1]))
    return [b for _, _, b in items]


def _merge_page_columnar_rows(page_blocks: List[ContentBlock]) -> List[ContentBlock]:
    """
    PDFs often emit one text block per table row. Merge consecutive short prose
    blocks on the same page when they look like columnar data.
    """
    if not page_blocks:
        return page_blocks
    out: List[ContentBlock] = []
    run: List[ContentBlock] = []

    def flush_run() -> None:
        nonlocal run
        if not run:
            return
        if len(run) >= 3 and all(b.kind == "prose" and b.text.count("\n") < 2 for b in run):
            combined = "\n".join(b.text for b in run)
            data_lines = [ln.strip() for ln in combined.split("\n") if ln.strip()]
            table_lines = [ln for ln in data_lines if len(ln.split()) >= 3]
            prefix_lines = [ln for ln in data_lines if len(ln.split()) < 3]
            if len(table_lines) >= 3 and _prose_looks_like_table("\n".join(table_lines)):
                section = next((b.section for b in run if b.section), "")
                page = run[0].page
                for line in prefix_lines:
                    out.append(
                        ContentBlock(kind="prose", page=page, text=line, section=section)
                    )
                md = _columnar_text_to_markdown("\n".join(table_lines))
                if md:
                    out.append(
                        ContentBlock(kind="table", page=page, text=md, section=section)
                    )
                    run = []
                    return
        out.extend(run)
        run = []

    for block in page_blocks:
        if block.kind == "table":
            flush_run()
            out.append(block)
            continue
        if run and block.page != run[-1].page:
            flush_run()
        if block.kind == "prose" and len(block.text) < 220:
            run.append(block)
            if len(run) > 40:
                flush_run()
        else:
            flush_run()
            out.append(block)
    flush_run()
    return out


def extract_content_blocks(pdf_bytes: bytes) -> List[ContentBlock]:
    """Extract prose and table blocks in page reading order."""
    import fitz

    use_tables = table_extraction_enabled()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    blocks: List[ContentBlock] = []
    try:
        current_section = ""
        for page in doc:
            page_blocks: List[ContentBlock] = []
            for block in _page_ordered_blocks(page, use_tables=use_tables):
                section = block.section or current_section
                if block.section:
                    current_section = block.section
                page_blocks.append(
                    ContentBlock(
                        kind=block.kind,
                        page=block.page,
                        text=block.text,
                        section=section,
                    )
                )
            page_blocks = _merge_page_columnar_rows(page_blocks)
            blocks.extend(page_blocks)
    finally:
        doc.close()
    return blocks


def extract_text(pdf_bytes: bytes) -> str:
    """Extract normalized plain text (prose + tables as markdown) for legacy callers."""
    parts: List[str] = []
    for block in extract_content_blocks(pdf_bytes):
        if block.kind == "table":
            parts.append(block.text)
        else:
            parts.append(block.text)
    return normalize_pdf_text("\n\n".join(parts))


def _chunk_header(page: int, section: str, kind: ContentKind) -> str:
    label = "Table" if kind == "table" else "Text"
    bits = [f"[Page {page} | {label}]"]
    if section:
        bits.append(f"Section: {section.strip()}")
    return "\n".join(bits)


def _split_markdown_table(md: str, max_data_rows: int) -> List[str]:
    lines = [ln for ln in md.strip().splitlines() if ln.strip()]
    if len(lines) <= 2:
        return [md.strip()] if md.strip() else []
    header = lines[:2]
    body = lines[2:]
    if not body:
        return ["\n".join(header)]
    out: List[str] = []
    for i in range(0, len(body), max_data_rows):
        out.append("\n".join(header + body[i : i + max_data_rows]))
    return out


def _split_markdown_table_with_overlap(md: str, max_data_rows: int, row_overlap: int) -> List[str]:
    lines = [ln for ln in md.strip().splitlines() if ln.strip()]
    if len(lines) <= 2:
        return [md.strip()] if md.strip() else []
    header = lines[:2]
    body = lines[2:]
    if not body:
        return ["\n".join(header)]

    max_rows = max(1, int(max_data_rows))
    overlap = max(0, min(int(row_overlap), max_rows - 1))
    step = max(1, max_rows - overlap)

    out: List[str] = []
    i = 0
    while i < len(body):
        part = body[i : i + max_rows]
        if not part:
            break
        out.append("\n".join(header + part))
        if i + max_rows >= len(body):
            break
        i += step
    return out


def _is_data_dense_text(text: str) -> bool:
    """
    Heuristic for chunks likely containing many entities/values where smaller windows
    generally retrieve better than broad prose windows.
    """
    s = (text or "").strip()
    if not s:
        return False
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 4:
        short_lines = sum(1 for ln in lines if len(ln) <= 90)
        if short_lines / max(1, len(lines)) >= 0.6:
            return True
    tokens = re.findall(r"\w+", s)
    if not tokens:
        return False
    digit_tokens = sum(1 for t in tokens if any(ch.isdigit() for ch in t))
    punct_separators = s.count(":") + s.count(";") + s.count("|")
    return (digit_tokens / len(tokens) >= 0.2) or (punct_separators >= max(8, len(s) // 160))


def _adaptive_chunk_params(text: str, chunk_size: int, overlap: int) -> Tuple[int, int]:
    if not _is_data_dense_text(text):
        return chunk_size, overlap
    tuned_size = max(480, int(chunk_size * 0.68))
    tuned_overlap = max(90, min(int(overlap * 0.75), tuned_size // 3))
    return tuned_size, tuned_overlap


def _find_chunk_end(text: str, start: int, chunk_size: int) -> int:
    text_len = len(text)
    hard_end = min(start + chunk_size, text_len)
    if hard_end >= text_len:
        return text_len

    window = text[start:hard_end]
    min_take = max(80, chunk_size // 3)
    min_rel = min_take if len(window) >= min_take else 0

    def rel_ok(rel_end: int) -> bool:
        return rel_end >= min_rel or start + rel_end >= text_len - 1

    for sep in ("\n\n", ". ", ".\n", "? ", "?\n", "! ", "!\n", "\n"):
        rel = window.rfind(sep)
        if rel >= 0 and rel_ok(rel + len(sep)):
            return min(start + rel + len(sep), text_len)

    sp = window.rfind(" ")
    if sp >= 0 and rel_ok(sp + 1):
        return min(start + sp + 1, text_len)

    return hard_end


def _next_chunk_start(text: str, start: int, end: int, overlap: int) -> int:
    text_len = len(text)
    if end >= text_len:
        return text_len
    span = end - start
    if span <= 0:
        return min(start + 1, text_len)
    shared = min(overlap, max(1, span - 1))
    nxt = end - shared
    if nxt <= start:
        nxt = end
    probe = text[nxt : min(end, nxt + 64)]
    cut_nl = probe.find("\n")
    cut_sp = probe.find(" ")
    if cut_nl >= 0 and cut_sp >= 0:
        cut = min(cut_nl, cut_sp)
    else:
        cut = cut_nl if cut_nl >= 0 else cut_sp
    if cut > 0:
        nxt = min(nxt + cut + 1, end)
    return nxt


def chunk_plain_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    min_chunk_chars: int = 50,
) -> List[str]:
    """Overlapping prose chunks with natural boundaries."""
    chunk_size = chunk_size if chunk_size is not None else default_chunk_size()
    overlap = overlap if overlap is not None else default_chunk_overlap()

    text = normalize_pdf_text(text)
    if not text:
        return []
    chunk_size, overlap = _adaptive_chunk_params(text, chunk_size, overlap)
    overlap = min(overlap, chunk_size // 2)

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


def chunk_content_blocks(
    blocks: Sequence[ContentBlock],
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    table_max_rows: int | None = None,
) -> List[TextChunk]:
    """Turn ordered content blocks into index-ready chunks with page/section context."""
    chunk_size = chunk_size if chunk_size is not None else default_chunk_size()
    overlap = overlap if overlap is not None else default_chunk_overlap()
    table_max_rows = table_max_rows if table_max_rows is not None else default_table_max_rows_per_chunk()
    table_row_overlap = default_table_row_overlap()

    out: List[TextChunk] = []
    prose_buf: List[str] = []
    prose_page = 1
    prose_section = ""

    def flush_prose() -> None:
        nonlocal prose_buf, prose_page, prose_section
        if not prose_buf:
            return
        joined = "\n\n".join(prose_buf).strip()
        prose_buf = []
        if not joined:
            return
        for piece in chunk_plain_text(joined, chunk_size=chunk_size, overlap=overlap):
            header = _chunk_header(prose_page, prose_section, "prose")
            out.append(
                TextChunk(
                    text=f"{header}\n\n{piece}",
                    content_type="prose",
                    page=prose_page,
                    section=prose_section,
                )
            )

    for block in blocks:
        if block.kind == "prose":
            if prose_buf and block.page != prose_page and prose_buf:
                flush_prose()
            if block.section:
                prose_section = block.section
            prose_page = block.page
            prose_buf.append(block.text)
            combined = "\n\n".join(prose_buf)
            effective_chunk_size, _ = _adaptive_chunk_params(combined, chunk_size, overlap)
            flush_threshold = int(effective_chunk_size * 1.2)
            if len(combined) >= max(420, flush_threshold):
                flush_prose()
            continue

        flush_prose()
        section = block.section or prose_section
        for part in _split_markdown_table_with_overlap(block.text, table_max_rows, table_row_overlap):
            header = _chunk_header(block.page, section, "table")
            out.append(
                TextChunk(
                    text=f"{header}\n\n{part}",
                    content_type="table",
                    page=block.page,
                    section=section,
                )
            )
        prose_page = block.page
        prose_section = section

    flush_prose()
    return out


def extract_and_chunk(pdf_bytes: bytes) -> List[TextChunk]:
    """Full pipeline: PDF bytes → chunks with metadata for vector indexing."""
    blocks = extract_content_blocks(pdf_bytes)
    if not blocks:
        return []
    return chunk_content_blocks(blocks)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    min_chunk_chars: int = 50,
) -> List[str]:
    """Build overlapping chunks from plain text (no table structure)."""
    return chunk_plain_text(text, chunk_size, overlap, min_chunk_chars)


def chunk_metadata(tc: TextChunk) -> Dict[str, Any]:
    """Vector-store metadata fields for a :class:`TextChunk`."""
    return {
        "content_type": tc.content_type,
        "page": int(tc.page),
        "section": (tc.section or "")[:200],
    }
