"""Dense retrieval with MMR over the persisted Chroma index."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .config import CONFIG


def get_retriever(persist_dir: str | None = None):
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
