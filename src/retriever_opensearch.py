"""OpenSearch backend for the RAG retriever.

Uses langchain-community's OpenSearchVectorSearch so the returned retriever
exposes the same `.invoke(query)` contract as the Chroma path. Hybrid search
(BM25 + kNN) is enabled when the index template supports it; otherwise we
fall back to pure vector kNN.
"""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from .config import CONFIG


def get_opensearch_retriever():
    """Return a LangChain retriever backed by OpenSearch.

    Drop-in replacement for `retriever.get_retriever()` when CONFIG.backend
    is set to 'opensearch'. Expects an existing index populated by
    `ingest_opensearch.build_index`.
    """
    # Deferred import — opensearch-py + langchain-community shouldn't be
    # required for users running the Chroma path.
    from langchain_community.vectorstores import OpenSearchVectorSearch

    embeddings = OpenAIEmbeddings(model=CONFIG.embedding_model)
    store = OpenSearchVectorSearch(
        opensearch_url=CONFIG.opensearch_host,
        index_name=CONFIG.opensearch_index,
        embedding_function=embeddings,
        http_auth=(CONFIG.opensearch_user, CONFIG.opensearch_password),
        use_ssl=CONFIG.opensearch_host.startswith("https"),
        verify_certs=CONFIG.opensearch_verify_certs,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        engine="lucene",
    )
    return store.as_retriever(search_kwargs={"k": CONFIG.top_k})


def build_opensearch_index(documents, recreate: bool = False) -> None:
    """Build (or rebuild) the OpenSearch index from a list of LangChain Documents.

    Called by `ingest.build_index` when CONFIG.backend == 'opensearch'. The
    `recreate=True` path deletes the index first; default appends.
    """
    from langchain_community.vectorstores import OpenSearchVectorSearch
    from opensearchpy import OpenSearch

    embeddings = OpenAIEmbeddings(model=CONFIG.embedding_model)

    if recreate:
        client = OpenSearch(
            hosts=[CONFIG.opensearch_host],
            http_auth=(CONFIG.opensearch_user, CONFIG.opensearch_password),
            use_ssl=CONFIG.opensearch_host.startswith("https"),
            verify_certs=CONFIG.opensearch_verify_certs,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        if client.indices.exists(index=CONFIG.opensearch_index):
            client.indices.delete(index=CONFIG.opensearch_index)

    OpenSearchVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        opensearch_url=CONFIG.opensearch_host,
        index_name=CONFIG.opensearch_index,
        http_auth=(CONFIG.opensearch_user, CONFIG.opensearch_password),
        use_ssl=CONFIG.opensearch_host.startswith("https"),
        verify_certs=CONFIG.opensearch_verify_certs,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        engine="lucene",
        bulk_size=200,
    )
