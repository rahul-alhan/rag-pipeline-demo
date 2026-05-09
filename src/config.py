from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 4
    mmr_lambda: float = 0.5
    persist_dir: str = "./chroma_db"
    collection_name: str = "rag_demo"

    faithfulness_gate: float = 0.85
    precision_gate: float = 0.80


CONFIG = RAGConfig()
