from __future__ import annotations

import re
from typing import Dict, List, Optional

MEMORY_SYSTEM_PROMPT = (
    "You are in one continuous conversation about one PDF. "
    "Use prior turns to resolve follow-ups, pronouns, shorthand, and elliptical questions. "
    "Treat the latest retrieved Context as primary ground truth for document facts. "
    "If memory conflicts with Context, trust Context. "
    "Respect the user's original intent even when shorthand is normalized (for example '&' to 'and', '/' to 'or'). "
    "Treat different polite or procedural wordings ('how to', 'guide me to', 'show me how to') as the same task when they ask for the same outcome."
)


SYSTEM_PROMPT = (
    "You are an expert PDF RAG assistant for enterprise document QA.\n"
    "Answer using only the provided PDF Context and valid conversation memory.\n"
    "Never fabricate values, names, dates, IDs, totals, legal, or financial facts.\n"
    "If evidence is insufficient, say so clearly and briefly.\n"
    "If inference is needed and strongly supported, prefix with 'Inferred:' and keep assumptions explicit.\n"
    "Shorthand handling: '&' means AND (cover all requested parts), spaced '/' means OR (compare supported alternatives).\n"
    "If slash appears inside a single field/code token, treat it as one combined label when context supports that reading.\n"
    "Procedural questions: phrasings such as 'how to …', 'guide me to …', 'show me how to …', 'walk me through …', "
    "'steps to …', 'how do I …', and 'help me to …' (when asking for a procedure) are the same user intent. "
    "If Context describes UI steps, menus, or buttons that accomplish the goal, give those steps—do not treat a different opener as a different question.\n"
    "Examples of same intent: 'guide me to run a portfolio valuation', 'how to run a portfolio valuation', and 'how do I value a portfolio' "
    "all target the same workflow—map them to the same headings or controls in Context (e.g. Value a Portfolio, Run, Start, Dashboard).\n"
    "Paraphrase and grammar: questions that use different surface forms but the same intent must be answered the same way. "
    "For example 'how much did X complete in 2009', 'how much X completed in 2009', and 'X completed in 2009' (same X and year) "
    "all ask for the same underlying fact in the document—extract and return that value or phrase from Context.\n"
    "Do not ask the user to rephrase when Context clearly supports an answer. "
    "Do not reply with only a generic 'rephrase your question' unless Context truly has no usable overlap with the request.\n"
)


_PROC_OPENERS = re.compile(
    r"^(?:"
    r"guide\s+me\s+to\s+|"
    r"guide\s+me\s+on\s+|"
    r"guide\s+me\s+through\s+|"
    r"show\s+me\s+how\s+to\s+|"
    r"walk\s+me\s+through\s+|"
    r"help\s+me\s+to\s+|"
    r"i\s+want\s+to\s+|"
    r"i\s+need\s+to\s+|"
    r"steps\s+to\s+|"
    r"how\s+do\s+i\s+"
    r")",
    re.IGNORECASE,
)

# Light typo fixes so retrieval matches doc spelling (esp. product terms).
_COMMON_TYPO_PATTERNS = (
    (re.compile(r"\bportflio\b", re.IGNORECASE), "portfolio"),
    (re.compile(r"\bportfolo\b", re.IGNORECASE), "portfolio"),
    (re.compile(r"\bvalaution\b", re.IGNORECASE), "valuation"),
)


def expand_question_shorthand(text: str) -> str:
    """
    Normalize common shorthand for retrieval + model reasoning.

    - Leading procedural phrases ('guide me to', 'show me how to', …) -> 'how to ' so
      retrieval aligns with doc-style imperatives and paraphrases match.
    - '&' (with spaces or between word characters) -> ' and '
    - '/' used as a list separator (spaces around slash) -> ' or '
    Does not rewrite tight tokens like dates or paths (e.g. 2024/25, https://...).
    """
    s = (text or "").strip()
    if not s:
        return s
    s = _PROC_OPENERS.sub("how to ", s)
    s = re.sub(r"(?i)^how\s+to\s+how\s+to\s+", "how to ", s)
    for pattern, replacement in _COMMON_TYPO_PATTERNS:
        s = pattern.sub(replacement, s)
    # Slash as "or" only when used like "A / B" (avoids 2024/25, http://, N/A).
    s = re.sub(r"\s+/\s+", " or ", s)
    # Ampersand: spaced form first, then compact "X&Y" between word-like chars.
    s = re.sub(r"\s*&\s*", " and ", s)
    s = re.sub(r"(?<=[\w\)\]])&(?=[\w\(])", " and ", s)
    return re.sub(r"\s+", " ", s).strip()


# Leading phrases that often hurt dense retrieval but can be stripped for a second recall pass
# (original question is still used for the LLM and as the first retrieval query).
_RETRIEVAL_STRIP_PREFIXES: tuple[str, ...] = (
    "how much did ",
    "how many did ",
    "how long did ",
    "how much ",
    "how many ",
    "how long ",
    "how to ",  # after procedural normalization; yields e.g. 'create file format' tail for embedding
    "what did ",
    "what was ",
    "what is ",
    "what are ",
    "what were ",
    "tell me ",
    "please tell me ",
    "can you tell me ",
    "could you tell me ",
    "can you ",
    "could you ",
    "please ",
    "what about ",
    "how about ",
    "i want to know ",
    "i need to know ",
)


def _procedural_retrieval_extras(q: str, lowered: str) -> List[str]:
    """Short extra queries for procedural 'how to run …' style questions (improves recall vs UI copy)."""
    extras: List[str] = []
    if lowered.startswith("how to "):
        tail = q[7:].strip()
        tl = tail.lower()
        for verb in ("run ", "running ", "perform ", "execute "):
            if tl.startswith(verb):
                inner = tail[len(verb) :].strip()
                if len(inner) >= 6:
                    extras.append(inner)
                break
    if "portfolio" in lowered and "valuation" in lowered:
        extras.extend(("portfolio valuation", "value a portfolio"))
    return extras


def retrieval_query_variants(question_for_rag: str) -> List[str]:
    """
    Return ordered unique query strings for hybrid retrieval.

    The first variant is always the full normalized question. Additional variants
    strip common WH / polite prefixes so embeddings align better with table-like
    chunks (e.g. 'Bob completed in 2009' vs 'how much Bob completed in 2009').
    """
    q = (question_for_rag or "").strip()
    if not q:
        return []
    lowered = q.lower()
    out: List[str] = [q]
    for prefix in _RETRIEVAL_STRIP_PREFIXES:
        if lowered.startswith(prefix):
            tail = q[len(prefix) :].strip()
            if len(tail) >= 6 and tail.lower() != lowered:
                out.append(tail)
            break
    out.extend(_procedural_retrieval_extras(q, lowered))
    seen: set[str] = set()
    uniq: List[str] = []
    for item in out:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def build_prompt(question: str, contexts: List[str], allow_inference: bool) -> str:
    """
    Builds the single user message we send to Ollama.

    Notes:
    - Keep context bounded for latency; 1400 chars/chunk reduces dropped UI labels vs 900.
    - Even when inference is allowed, grounded extraction is preferred.
    """

    trimmed_contexts = [ctx[:1400] for ctx in (contexts or [])]
    context_block = "\n\n---\n\n".join(trimmed_contexts)

    if allow_inference:
        behavior = (
            "Inference mode is enabled. "
            "Use direct extraction first. "
            "If extraction is incomplete but evidence strongly implies an answer, provide a short inferred answer prefixed with 'Inferred:'. "
            "If evidence is weak or missing, respond with a short friendly not-found message and gently ask a related follow-up."
        )
    else:
        behavior = (
            "Only answer from the provided context. "
            "If the answer is missing from context, use a short friendly not-found reply."
        )

    return f"""
{SYSTEM_PROMPT}
{behavior}

Context:
{context_block}

Question:
{question}

Formatting rules:
- Return plain text only.
- Do not use markdown, bullets, tables, bold markers, or decorative symbols.
- Do not add headings like "Answer:" or "Response:".
- Keep answers concise and directly useful.
Memory rules:
- Use prior turns for continuity only (coreference, follow-ups, constraints, user intent).
- Do not treat prior assistant claims as facts when they conflict with retrieved Context.
- If the current question depends on prior turns, resolve it explicitly before answering.
Reasoning rules:
- Resolve user typos and grammar issues internally before answering.
- Treat typo-correction as hidden reasoning; do not describe correction unless asked.
- If multiple typo interpretations are possible, choose the most context-supported one.
- Interpret '&' as AND and spaced '/' as OR as described in the system instructions; if the question lists multiple AND parts, answer all that the context supports; if OR alternatives, compare each to the document.
- For how-to and navigation questions, match the user's goal to headings, tabs, buttons, and numbered steps in Context even when their wording used 'guide me', 'show me', or 'walk me through' instead of 'how to'.
- Treat 'guide me to run …' like 'how to run …': the normalized question in this prompt already aligns those forms—search Context for the same workflow labels (menus, buttons, feature names) and answer with concrete steps, not an empty reply.
- Map verbose WH-questions to the same factual target as shorter phrasing when they share the same entities (names, IDs, years, amounts): answer with the value or fact from Context, not a request to rephrase.
- Prefer exact field values (name, PAN, UAN, CTC, net pay, dates, IDs) when available.
- For numeric answers, preserve units/currency exactly as seen in context.
- Mention uncertainty in one short line when details are implicit.
- If question is unrelated to selected PDF, respond with a short friendly not-found reply.
Output quality rules:
- Avoid repetition and filler.
- Avoid contradictory statements in the same answer.
- If answer is empty after reasoning, give a brief not-found reply that names what was missing; do not use the exact phrase 'rephrase your question' as the whole answer when Context partially matches the topic.
Conversation rules:
- If user sends small-talk (greeting, thanks, short casual message), reply briefly and politely in 1 sentence.
- For small-talk, do not force PDF citations or refusal messages.
- After small-talk reply, gently guide user back to PDF queries.
- Keep tone friendly, professional, and concise.
""".strip()


def _build_memory_chain(prior_messages: List[Dict[str, str]]) -> str:
    if not prior_messages:
        return ""
    lines: List[str] = []
    for idx, message in enumerate(prior_messages, start=1):
        role = "USER" if message["role"] == "user" else "ASSISTANT"
        lines.append(f"{idx}. {role}: {message['content']}")
    return "\n".join(lines)


def build_chat_messages_for_ollama(
    question: str,
    contexts: List[str],
    allow_inference: bool,
    prior_messages: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """One-chain memory prompt for Ollama with sharp task constraints."""
    prior = prior_messages or []
    cleaned: List[Dict[str, str]] = []
    for m in prior:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        max_len = 1600 if role == "user" else 2800
        if len(content) > max_len:
            content = content[: max_len - 3] + "..."
        cleaned.append({"role": str(role), "content": content})

    current_prompt = build_prompt(question, contexts, allow_inference)
    if not cleaned:
        return [{"role": "user", "content": current_prompt}]

    history_block = _build_memory_chain(cleaned)
    one_chain_prompt = (
        f"{MEMORY_SYSTEM_PROMPT}\n\n"
        "Conversation Memory Chain (oldest to latest):\n"
        f"{history_block}\n\n"
        "Current Task:\n"
        "Answer only the latest question below using retrieved Context as primary evidence.\n\n"
        f"{current_prompt}"
    )
    return [{"role": "user", "content": one_chain_prompt}]

