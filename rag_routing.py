"""Shared RAG routing helpers (small-talk, binary QA, not-found, answer cleanup)."""

from __future__ import annotations

import re
from typing import List

_BINARY_KEYWORD_SKIP = frozenset({"and", "or"})

NOT_FOUND_REPLIES = [
    "I could not spot that detail in the selected PDF yet. Please try a more specific question.",
    "I could not find that in this PDF. If you want, ask with a related keyword and I will check again.",
    "That exact detail is not visible in the selected PDF right now. Try rephrasing and I can re-check.",
]


def is_small_talk(question: str) -> bool:
    lowered = re.sub(r"[^a-z0-9\s]", " ", question.lower()).strip()
    if not lowered:
        return False
    compact = " ".join(lowered.split())
    exact_phrases = {
        "hi",
        "hello",
        "hey",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "thanks",
        "thank you",
        "thx",
        "ok",
        "okay",
        "cool",
        "nice",
        "great",
    }
    if compact in exact_phrases:
        return True
    starts_with_phrases = (
        "hi ",
        "hello ",
        "hey ",
        "thanks ",
        "thank you ",
        "how are you ",
    )
    return compact.startswith(starts_with_phrases)


def small_talk_reply(question: str) -> str:
    compact = " ".join(re.sub(r"[^a-z0-9\s]", " ", question.lower()).split())
    if compact.startswith(("thanks", "thank you", "thx")):
        return "You are welcome. I am here whenever you want to ask something about your PDF."
    if compact.startswith("how are you"):
        return "I am doing great. Ready to help you explore your PDF."
    if compact in {"ok", "okay", "cool", "nice", "great"}:
        return "Great. Ask me any detail you want from the selected PDF."
    return "Hi there. I am ready to help. Ask me anything from your selected PDF."


def friendly_not_found_reply(question: str) -> str:
    seed = abs(hash((question or "").strip().lower()))
    return NOT_FOUND_REPLIES[seed % len(NOT_FOUND_REPLIES)]


def clean_answer_text(answer: str) -> str:
    if not answer:
        return "I am here to help. Could you please rephrase your question?"
    text = answer
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?m)^\s*[•◦○●▪▫]\s*$", "", text)
    text = re.sub(r"(?m)^\s*[-*•◦○●▪▫]\s+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    cleaned = text.strip()
    if not cleaned:
        return "I am here to help. Could you please rephrase your question?"
    return cleaned


def is_binary_question_prefix(question_for_rag: str) -> bool:
    return bool(
        re.match(
            r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b",
            (question_for_rag or "").lower(),
        )
    )


def binary_yes_no_from_context(question_for_rag: str, contexts: List[str]) -> str:
    text = " ".join(contexts).lower()
    keywords = [
        t
        for t in re.findall(r"[a-z0-9\+\#\.]+", question_for_rag.lower())
        if len(t) > 2 and t not in _BINARY_KEYWORD_SKIP
    ]
    return "Yes." if any(k in text for k in keywords) else "No."
