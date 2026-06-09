"""Ingest documents → chunk → embed → persist to the configured backend
(Chroma for local dev, OpenSearch for production)."""
from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CONFIG


def load_documents(docs_dir: str):
    docs = []
    for pattern in ("**/*.md", "**/*.txt"):
        text_loader = DirectoryLoader(
            docs_dir,
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        docs.extend(text_loader.load())

    for pdf in Path(docs_dir).rglob("*.pdf"):
        docs.extend(PyPDFLoader(str(pdf)).load())

    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(docs_dir: str, persist_dir: str | None = None):
    """Build the index for the configured backend (CONFIG.backend).

    For Chroma: persists to a local directory.
    For OpenSearch: pushes to the configured cluster/index.
    """
    print(f"Loading documents from {docs_dir}...")
    raw = load_documents(docs_dir)
    print(f"  → {len(raw)} documents")

    chunks = chunk_documents(raw)
    print(f"  → {len(chunks)} chunks")

    if CONFIG.backend == "opensearch":
        from .retriever_opensearch import build_opensearch_index
        print(f"Pushing to OpenSearch index '{CONFIG.opensearch_index}' at {CONFIG.opensearch_host}...")
        build_opensearch_index(chunks, recreate=True)
        print("Done.")
        return None

    # Default: Chroma local
    from langchain_chroma import Chroma

    persist_dir = persist_dir or CONFIG.persist_dir
    embeddings = OpenAIEmbeddings(model=CONFIG.embedding_model)
    vec = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=CONFIG.collection_name,
    )
    print(f"Persisted to {persist_dir}")
    return vec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="./docs")
    p.add_argument("--persist", default=CONFIG.persist_dir)
    args = p.parse_args()
    build_index(args.docs, args.persist)


if __name__ == "__main__":
    main()
