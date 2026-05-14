"""Shared RAG routing helpers (small-talk, binary QA, not-found, answer cleanup)."""

from __future__ import annotations

import re
from typing import List, Optional

_BINARY_KEYWORD_SKIP = frozenset({"and", "or"})

_STOP_FOR_HINT = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "must",
        "shall",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "me",
        "my",
        "your",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "much",
        "many",
        "some",
        "any",
        "no",
        "not",
        "only",
        "just",
        "also",
        "too",
        "very",
        "about",
        "into",
        "over",
        "after",
        "before",
        "between",
        "through",
        "during",
        "please",
        "tell",
        "give",
        "show",
        "get",
        "pdf",
        "document",
        "file",
        "page",
        "here",
        "there",
        "again",
        "still",
        "yet",
        "like",
        "want",
        "need",
        "know",
    }
)


def _meaningful_tokens_for_hint(question: str, *, max_tokens: int = 6) -> List[str]:
    raw = (question or "").strip().lower()
    if not raw:
        return []
    tokens = [t for t in re.findall(r"[a-z0-9][a-z0-9\+\#\.]*", raw) if len(t) > 1]
    out: List[str] = []
    for t in tokens:
        if t in _STOP_FOR_HINT:
            continue
        out.append(t)
        if len(out) >= max_tokens:
            break
    return out


def friendly_empty_answer_reply(source_question: Optional[str] = None) -> str:
    """
    When the model returns nothing useful after cleanup, show a warm, varied message
    that references the user's wording when possible (never the old generic rephrase line).
    """
    q = (source_question or "").strip()
    tokens = _meaningful_tokens_for_hint(q)
    seed = abs(hash(q.casefold())) if q else 0

    if tokens:
        hint = " ".join(tokens).title()
        if len(hint) > 52:
            hint = hint[:49].rstrip() + "…"
        templates = (
            "I could not line that up with the selected PDF yet for {hint}. Try the same idea using a name, year, or label exactly as it appears in the document.",
            "The PDF did not surface a clear match for {hint} this time. Ask with a short phrase you can see on the page—an amount, heading, or date works well.",
            "I did not find enough in this PDF to answer about {hint} just yet. Narrow it to a few keywords from the file and I will search again.",
            "Nothing solid showed up for {hint} in the selected document. Point me at a concrete word or number from the PDF and I will take another look.",
        )
        return templates[seed % len(templates)].format(hint=hint)

    templates = (
        "I could not connect that to the selected PDF yet. Try a question built from words you see in the file—a heading, value, name, or date.",
        "The document did not give me a confident match for that. Ask using a short phrase copied from the PDF if you can.",
        "I was not able to pull an answer from this PDF for that request. Try one or two specific terms that appear on the page.",
        "That did not match the selected PDF closely enough yet. Use a concrete detail from the document and I will try again.",
    )
    return templates[seed % len(templates)]


def clean_answer_text(answer: str, source_question: Optional[str] = None) -> str:
    if not answer:
        return friendly_empty_answer_reply(source_question)
    text = answer
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?m)^\s*[•◦○●▪▫]\s*$", "", text)
    text = re.sub(r"(?m)^\s*[-*•◦○●▪▫]\s+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    cleaned = text.strip()
    if not cleaned:
        return friendly_empty_answer_reply(source_question)
    return cleaned

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
