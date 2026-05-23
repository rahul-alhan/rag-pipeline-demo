"""Pure-Python formatting helpers — no langchain / openai deps so the
contract is unit-testable in isolation."""
from __future__ import annotations

from typing import Protocol


class _DocLike(Protocol):
    page_content: str

    @property
    def metadata(self) -> dict: ...


def format_context(docs) -> str:
    """Render retrieved docs into the bracketed [doc i | source: X] block
    that the RAG prompt expects."""
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", f"doc_{i}")
        parts.append(f"[doc {i} | source: {src}]\n{d.page_content}")
    return "\n\n".join(parts)
