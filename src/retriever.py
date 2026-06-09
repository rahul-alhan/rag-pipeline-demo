"""Backend-agnostic retriever — dispatches to Chroma (local) or OpenSearch (prod)
based on CONFIG.backend. Both paths return a LangChain Runnable with the same
`.invoke(query)` contract."""
from __future__ import annotations

from .config import CONFIG


def get_retriever(persist_dir: str | None = None):
    if CONFIG.backend == "opensearch":
        # Deferred import — Chroma users shouldn't need opensearch-py installed.
        from .retriever_opensearch import get_opensearch_retriever
        return get_opensearch_retriever()

    # Default: Chroma with MMR
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    persist_dir = persist_dir or CONFIG.persist_dir
    vec = Chroma(
        persist_directory=persist_dir,
        collection_name=CONFIG.collection_name,
        embedding_function=OpenAIEmbeddings(model=CONFIG.embedding_model),
    )
    return vec.as_retriever(
        search_type="mmr",
        search_kwargs={"k": CONFIG.top_k, "lambda_mult": CONFIG.mmr_lambda},
    )
