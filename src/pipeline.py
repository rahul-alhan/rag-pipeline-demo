"""RAG chain with citation enforcement and structured output."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import CONFIG
from .formatting import format_context
from .retriever import get_retriever


SYSTEM_PROMPT = """You are a precise assistant. Answer ONLY using the provided context.
Rules:
1. If the answer is not in the context, say: "I don't have enough information."
2. Cite the source for every claim using [source: <doc_name>] inline.
3. Do not invent facts, names, dates, or figures not present in the context.
4. Prefer concise, structured answers."""

USER_PROMPT = """Context:
{context}

Question: {question}

Answer (with inline [source: ...] citations):"""


@dataclass
class RAGAnswer:
    question: str
    answer: str
    contexts: list[str]
    sources: list[str]

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "sources": self.sources,
        }


def build_chain():
    retriever = get_retriever()
    llm = ChatOpenAI(model=CONFIG.llm_model, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
    )
    return retriever, prompt | llm | StrOutputParser()


def answer(question: str) -> RAGAnswer:
    retriever, generation = build_chain()
    docs = retriever.invoke(question)
    ctx = format_context(docs)
    text = generation.invoke({"context": ctx, "question": question})
    return RAGAnswer(
        question=question,
        answer=text,
        contexts=[d.page_content for d in docs],
        sources=[d.metadata.get("source", "") for d in docs],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    args = p.parse_args()
    result = answer(args.query)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
