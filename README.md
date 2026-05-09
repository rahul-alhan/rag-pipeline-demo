# RAG Pipeline Demo

Production-style Retrieval-Augmented Generation pipeline with chunking strategy, dense retrieval, hallucination control, and automated quality gates via **RAGAS**.

> Mirrors the architecture used in production at Algoworks for content moderation and copyright risk assessment over a large unstructured corpus.

---

## Architecture

```
                ┌──────────────┐
   docs/   ──▶  │   ingest.py  │  ──▶  Chunking (recursive, 512 tok, 50 overlap)
                └──────────────┘                │
                                                ▼
                                         OpenAI embeddings
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │   Chroma Vec DB │
                                       └─────────────────┘
                                                │
   query  ──▶  retriever.py  ──▶  top-k dense retrieval (k=4, MMR)
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │   pipeline.py   │  ──▶  GPT-4 + cited answer
                                       └─────────────────┘
                                                │
                                                ▼
                                       evaluate.py
                                       (RAGAS: faithfulness,
                                        answer relevance, precision@k)
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set your OpenAI key
export OPENAI_API_KEY=sk-...

# 3. Drop documents into ./docs (txt/md/pdf)
#    A small sample corpus is included.

# 4. Build the index
python -m src.ingest --docs ./docs --persist ./chroma_db

# 5. Ask a question
python -m src.pipeline --query "What are the copyright escalation rules?"

# 6. Run RAGAS evaluation on the eval set
python -m src.evaluate --eval-set eval/eval_set.json
```

Or open `notebooks/demo.ipynb` for an end-to-end walkthrough.

---

## Design Choices

| Decision | Rationale |
|---|---|
| **RecursiveCharacterTextSplitter** | Preserves semantic boundaries (paragraphs → sentences → words) |
| **chunk_size=512, overlap=50** | Empirical sweet spot — large enough for context, small enough to keep precision@k tight |
| **Chroma (local)** | Zero-infra for demo; production version uses OpenSearch with kNN |
| **MMR retrieval** | Reduces redundancy in top-k; meaningfully boosts answer relevance |
| **Source citation enforced** | Output schema requires `[source: <doc>]` tags — primary hallucination mitigation |
| **RAGAS quality gates** | Faithfulness ≥ 0.85, precision@k ≥ 0.80 enforced before promoting prompt changes |

---

## Evaluation Results (sample corpus)

| Metric | Score |
|---|---|
| Faithfulness | 0.91 |
| Answer Relevance | 0.88 |
| Context Precision @ k=4 | 0.83 |
| Context Recall | 0.86 |

> Numbers above are illustrative; actual values depend on your corpus and eval set.

---

## Repository Layout

```
rag-pipeline-demo/
├── README.md
├── requirements.txt
├── docs/                       # sample corpus
│   ├── sample_policy.md
│   └── sample_handbook.md
├── eval/
│   └── eval_set.json           # ground-truth Q/A for RAGAS
├── src/
│   ├── __init__.py
│   ├── config.py               # central config (chunking, retrieval, model)
│   ├── ingest.py               # chunking + embedding + persist
│   ├── retriever.py            # dense retrieval with MMR
│   ├── pipeline.py             # RAG chain with citation enforcement
│   └── evaluate.py             # RAGAS harness
└── notebooks/
    └── demo.ipynb
```

---

## Production Notes

In production this same pattern is deployed with:
- **OpenSearch kNN** instead of Chroma (managed, hybrid sparse+dense)
- **AWS Lambda** for the retrieval API, **API Gateway** for ingress
- **MLflow** prompt versioning + experiment registry
- **CloudWatch** structured logs for every retrieval (query, doc IDs, scores, answer)
- **EventBridge** triggers for re-embedding when corpus drifts

---

## License

MIT
