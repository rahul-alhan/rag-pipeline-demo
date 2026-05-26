"""End-to-end answer() test with injected fake retriever + generation.

Verifies the production contract: retriever output is formatted into the
prompt context, the generation runnable is invoked, and the returned
RAGAnswer carries through the question, answer, contexts, and sources.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline import RAGAnswer, answer


@dataclass
class _Doc:
    page_content: str
    metadata: dict = field(default_factory=dict)


class _FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return self.docs


class _FakeGeneration:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.response


def test_answer_threads_retriever_output_through_generation():
    docs = [
        _Doc("policy body", {"source": "policy.md"}),
        _Doc("handbook body", {"source": "handbook.md"}),
    ]
    retriever = _FakeRetriever(docs)
    gen = _FakeGeneration("Final answer [source: policy.md]")

    out = answer("what is the SLA?", retriever=retriever, generation=gen)

    assert isinstance(out, RAGAnswer)
    assert out.question == "what is the SLA?"
    assert out.answer == "Final answer [source: policy.md]"
    assert out.contexts == ["policy body", "handbook body"]
    assert out.sources == ["policy.md", "handbook.md"]
    # Retriever called once with the user's query
    assert retriever.calls == ["what is the SLA?"]
    # Generation got formatted context and the question
    payload = gen.calls[0]
    assert payload["question"] == "what is the SLA?"
    assert "policy.md" in payload["context"]
    assert "handbook.md" in payload["context"]


def test_answer_handles_missing_source_metadata():
    """A doc without a `source` key should not crash; sources list gets empty string."""
    retriever = _FakeRetriever([_Doc("anonymous", {})])
    gen = _FakeGeneration("ok")
    out = answer("q", retriever=retriever, generation=gen)
    assert out.sources == [""]


def test_answer_handles_empty_retrieval():
    retriever = _FakeRetriever([])
    gen = _FakeGeneration("I don't have enough information.")
    out = answer("q", retriever=retriever, generation=gen)
    assert out.contexts == []
    assert out.sources == []
    assert out.answer == "I don't have enough information."


def test_rag_answer_to_dict_shape():
    a = RAGAnswer(question="q", answer="a", contexts=["c1"], sources=["s1"])
    assert a.to_dict() == {"question": "q", "answer": "a", "contexts": ["c1"], "sources": ["s1"]}
