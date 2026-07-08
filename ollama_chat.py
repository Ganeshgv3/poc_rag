"""ChatOllama via langchain-ollama (community package no longer exports it in 0.4+)."""

from __future__ import annotations

try:
    from langchain_ollama import ChatOllama
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ChatOllama requires the langchain-ollama package. "
        "Install project dependencies: pip install -r requirements.txt"
    ) from exc

__all__ = ["ChatOllama"]
