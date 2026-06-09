import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 4
    mmr_lambda: float = 0.5

    # Backend selection. "chroma" (default, local file) or "opensearch" (production).
    # Override via RAG_BACKEND env var.
    backend: str = os.getenv("RAG_BACKEND", "chroma")

    # Chroma settings (used when backend == "chroma")
    persist_dir: str = "./chroma_db"
    collection_name: str = "rag_demo"

    # OpenSearch settings (used when backend == "opensearch")
    opensearch_host: str = os.getenv("OPENSEARCH_HOST", "https://localhost:9200")
    opensearch_index: str = os.getenv("OPENSEARCH_INDEX", "rag-demo")
    opensearch_user: str = os.getenv("OPENSEARCH_USER", "admin")
    opensearch_password: str = os.getenv("OPENSEARCH_PASSWORD", "admin")
    opensearch_verify_certs: bool = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"

    faithfulness_gate: float = 0.85
    precision_gate: float = 0.80


CONFIG = RAGConfig()
