"""Lightweight continuous-eval metrics for stored QA turns."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


def _tok(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def retrieval_recall_at_k(question: str, contexts: List[str]) -> float:
    q = set(_tok(question))
    if not q or not contexts:
        return 0.0
    covered: set[str] = set()
    for ctx in contexts:
        c = set(_tok(ctx))
        covered |= (q & c)
    return round(len(covered) / max(1, len(q)), 4)


def faithfulness_proxy(answer: str, contexts: List[str]) -> float:
    at = _tok(answer)
    if not at or not contexts:
        return 0.0
    ctext = " ".join(contexts)
    ct = set(_tok(ctext))
    supported = sum(1 for t in at if t in ct)
    return round(supported / max(1, len(at)), 4)


def answer_relevancy_proxy(question: str, answer: str) -> float:
    return round(_jaccard(_tok(question), _tok(answer)), 4)


def evaluate_pair(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    rec = retrieval_recall_at_k(question, contexts)
    faith = faithfulness_proxy(answer, contexts)
    rel = answer_relevancy_proxy(question, answer)
    overall = round((0.4 * rec) + (0.4 * faith) + (0.2 * rel), 4)
    return {
        "recall_at_k": rec,
        "faithfulness": faith,
        "answer_relevancy": rel,
        "overall": overall,
    }


def evaluate_chat_pairs(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for p in pairs:
        rows.append(
            evaluate_pair(
                question=str(p.get("question") or ""),
                answer=str(p.get("answer") or ""),
                contexts=list(p.get("contexts") or []),
            )
        )
    if not rows:
        return {"count": 0, "averages": {"recall_at_k": 0.0, "faithfulness": 0.0, "answer_relevancy": 0.0, "overall": 0.0}}
    n = len(rows)
    avg = {
        "recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 4),
        "faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 4),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in rows) / n, 4),
        "overall": round(sum(r["overall"] for r in rows) / n, 4),
    }
    return {"count": n, "averages": avg, "rows": rows}
