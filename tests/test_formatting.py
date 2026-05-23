"""Context formatter contract — what the prompt actually receives."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.formatting import format_context


@dataclass
class _Doc:
    page_content: str
    metadata: dict = field(default_factory=dict)


def test_empty_docs_renders_empty_string():
    assert format_context([]) == ""


def test_single_doc_includes_source_and_content():
    out = format_context([_Doc("the body", {"source": "policy.md"})])
    assert "[doc 1 | source: policy.md]" in out
    assert "the body" in out


def test_missing_source_falls_back_to_position_label():
    out = format_context([_Doc("body", {})])
    assert "[doc 1 | source: doc_1]" in out


def test_multiple_docs_are_separated_by_blank_line():
    out = format_context([
        _Doc("first", {"source": "a"}),
        _Doc("second", {"source": "b"}),
    ])
    # blank line between blocks
    assert "first\n\n[doc 2" in out
    assert "second" in out


def test_doc_numbering_is_1_indexed():
    out = format_context([_Doc("x"), _Doc("y"), _Doc("z")])
    assert "[doc 1" in out and "[doc 2" in out and "[doc 3" in out
    assert "[doc 0" not in out
