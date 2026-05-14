"""Heuristic accuracy signals for RAG answers (shared by API, Streamlit, and tests)."""

from __future__ import annotations

import re
from typing import List, Tuple

from rag_routing import NOT_FOUND_REPLIES


def accuracy_from_distances(distances: List[float]) -> Tuple[str, int]:
    if not distances:
        return "Low", 0
    avg_distance = float(sum(distances) / len(distances))
    score = max(0, min(100, int((1.2 - avg_distance) * 100)))
    if score >= 80:
        return "High", score
    if score >= 55:
        return "Medium", score
    return "Low", score


def support_score_from_context(answer: str, contexts: List[str]) -> int:
    if not answer.strip() or not contexts:
        return 0
    answer_tokens = {
        tok
        for tok in re.findall(r"[a-z0-9\+\#\.]+", answer.lower())
        if len(tok) > 2 and tok not in {"the", "and", "for", "with", "from", "that", "this"}
    }
    if not answer_tokens:
        return 0
    context_text = " ".join(contexts).lower()
    matched = sum(1 for tok in answer_tokens if tok in context_text)
    ratio = matched / max(1, len(answer_tokens))
    return int(max(0, min(100, ratio * 100)))


def is_not_found_answer(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    return any(reply.lower() == normalized for reply in NOT_FOUND_REPLIES)


def accuracy_from_signals(answer: str, contexts: List[str], distances: List[float]) -> Tuple[str, int]:
    if not contexts:
        return "N/A", 0
    if is_not_found_answer(answer):
        return "Low", 0
    _dist_label, dist_score = accuracy_from_distances(distances)
    support_score = support_score_from_context(answer, contexts)
    final_score = int(round((dist_score * 0.5) + (support_score * 0.5)))
    if final_score >= 80:
        return "High", final_score
    if final_score >= 55:
        return "Medium", final_score
    return "Low", final_score
