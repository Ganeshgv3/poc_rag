from __future__ import annotations

from typing import Dict, List, Optional

MEMORY_SYSTEM_PROMPT = (
    "Earlier messages are a short recap of previous turns in this chat about the same PDF. "
    "Use them only for continuity (pronouns, implicit references, what the user already asked). "
    "The final user message contains freshly retrieved excerpts under Context; "
    "treat that Context as the primary ground truth for factual claims in your reply."
)


SYSTEM_PROMPT = (
    "You are an expert PDF RAG assistant for enterprise document QA.\n"
    "Your first priority is to answer from the provided PDF context only.\n"
    "If exact wording is missing but intent is clearly related to the selected PDF, infer cautiously from nearby evidence.\n"
    "When inference is used, begin with 'Inferred:' and keep assumptions minimal and explicit.\n"
    "Never fabricate values, names, dates, IDs, totals, or legal/financial facts.\n"
)


def build_prompt(question: str, contexts: List[str], allow_inference: bool) -> str:
    """
    Builds the single user message we send to Ollama.

    Notes:
    - Keep context compact for latency.
    - Even when inference is allowed, grounded extraction is preferred.
    """

    trimmed_contexts = [ctx[:900] for ctx in (contexts or [])]
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
Reasoning rules:
- Resolve user typos and grammar issues internally before answering.
- Treat typo-correction as hidden reasoning; do not describe correction unless asked.
- If multiple typo interpretations are possible, choose the most context-supported one.
- Prefer exact field values (name, PAN, UAN, CTC, net pay, dates, IDs) when available.
- For numeric answers, preserve units/currency exactly as seen in context.
- Mention uncertainty in one short line when details are implicit.
- If question is unrelated to selected PDF, respond with a short friendly not-found reply.
Output quality rules:
- Avoid repetition and filler.
- Avoid contradictory statements in the same answer.
- If answer is empty after reasoning, respond with a short friendly not-found reply.
Conversation rules:
- If user sends small-talk (greeting, thanks, short casual message), reply briefly and politely in 1 sentence.
- For small-talk, do not force PDF citations or refusal messages.
- After small-talk reply, gently guide user back to PDF queries.
- Keep tone friendly, professional, and concise.
""".strip()


def build_chat_messages_for_ollama(
    question: str,
    contexts: List[str],
    allow_inference: bool,
    prior_messages: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Multi-turn messages for Ollama: optional recap + current RAG user prompt."""
    prior = prior_messages or []
    cleaned: List[Dict[str, str]] = []
    for m in prior:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        max_len = 1200 if role == "user" else 2000
        if len(content) > max_len:
            content = content[: max_len - 3] + "..."
        cleaned.append({"role": str(role), "content": content})

    messages: List[Dict[str, str]] = []
    if cleaned:
        messages.append({"role": "system", "content": MEMORY_SYSTEM_PROMPT})
    messages.extend(cleaned)
    messages.append({"role": "user", "content": build_prompt(question, contexts, allow_inference)})
    return messages

