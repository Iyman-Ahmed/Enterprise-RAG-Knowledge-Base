---
title: Enterprise RAG Knowledge Base
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.37.0
app_file: app.py
pinned: false
license: mit
---

# Enterprise RAG Knowledge Base with Evaluation Pipeline

> Production-grade Retrieval-Augmented Generation system with hybrid search, vector database, automated evaluation, and hallucination detection — built for enterprise document Q&A.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)](https://www.trychroma.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)](https://fastapi.tiangolo.com)

---

## Why This Project

RAG is the #1 most in-demand AI engineering skill in 2026. This project demonstrates production-grade RAG pipeline design — not a tutorial chatbot, but a system with evaluation metrics, hybrid search, reranking, and hallucination guardrails.

---

## What It Does

Upload company documents (PDFs, DOCX, TXT, Markdown) and ask natural language questions. The system:

1. **Ingests documents** — section-aware chunking with metadata extraction
2. **Hybrid search** — combines dense embeddings (sentence-transformers) + sparse BM25 for optimal retrieval
3. **Cross-encoder reranking** — reranks top-k results for precision
4. **LLM generation** — answers grounded in retrieved context with inline citations
5. **Hallucination detection** — flags answers not supported by source documents
6. **Evaluation dashboard** — automated RAGAS metrics (faithfulness, relevance, context precision)

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────┐
│   Query Processing    │
│   - Query expansion   │
│   - Intent detection  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     ┌────────────────────────┐
│   Hybrid Retrieval    │     │   Vector Store          │
│   Dense: all-MiniLM   │◄───│   ChromaDB + BM25 index │
│   Sparse: BM25        │     │   Metadata filtering    │
│   Fusion: RRF scoring │     └────────────────────────┘
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Cross-Encoder       │
│   Reranking           │
│   ms-marco-MiniLM     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     ┌────────────────────────┐
│   LLM Generation      │     │   Guardrails            │
│   Groq / OpenAI       │────►│   - Citation validation │
│   Structured output   │     │   - Hallucination check │
│   with citations      │     │   - Confidence scoring  │
└──────────┬───────────┘     └────────────────────────┘
           │
           ▼
┌──────────────────────┐
│   Evaluation Pipeline │
│   RAGAS metrics       │
│   - Faithfulness      │
│   - Answer relevancy  │
│   - Context precision │
│   - Context recall    │
└──────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector DB | ChromaDB with persistent storage |
| Sparse Search | BM25 (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| LLM | Groq `llama-3.3-70b` / OpenAI GPT-4o-mini |
| Framework | LangChain 0.2 + LangSmith tracing |
| Backend | FastAPI + SQLAlchemy |
| Frontend | Streamlit / Gradio |
| Evaluation | RAGAS + custom metrics |
| Deployment | Docker → Hugging Face Spaces |

---

## Key Features

- **Hybrid search fusion** — Reciprocal Rank Fusion combining dense + sparse retrieval
- **Metadata-aware chunking** — preserves document structure, headers, page numbers
- **Citation grounding** — every answer includes source document + page references
- **Hallucination scoring** — NLI-based check comparing answer against retrieved context
- **Evaluation pipeline** — automated RAGAS metrics on test datasets
- **Token optimization** — context window management for cost efficiency
- **Multi-format ingestion** — PDF, DOCX, TXT, Markdown with OCR fallback

---

## Project Structure

```
Enterprise-RAG-Knowledge-Base/
├── app.py                  # Streamlit/Gradio UI
├── api/
│   ├── main.py            # FastAPI endpoints
│   ├── routes/
│   └── models/
├── pipeline/
│   ├── ingestion.py       # Document loading + chunking
│   ├── embeddings.py      # Embedding generation
│   ├── retrieval.py       # Hybrid search + reranking
│   ├── generation.py      # LLM response generation
│   └── guardrails.py      # Hallucination detection
├── evaluation/
│   ├── ragas_eval.py      # RAGAS evaluation pipeline
│   ├── test_dataset.json  # Golden Q&A test set
│   └── metrics.py         # Custom evaluation metrics
├── vectorstore/
│   └── chroma_manager.py  # ChromaDB operations
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Metrics & Results

| Metric | Score |
|---|---|
| Retrieval Hit Rate (@10) | TBD |
| Context Precision | TBD |
| Faithfulness (RAGAS) | TBD |
| Answer Relevancy | TBD |
| Hallucination Detection F1 | TBD |
| Avg Response Latency | TBD |

---

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/Enterprise-RAG-Knowledge-Base.git
cd Enterprise-RAG-Knowledge-Base
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python app.py
```

---

## Deployment

```bash
docker build -t rag-knowledge-base .
docker run -p 7860:7860 -e GROQ_API_KEY=... rag-knowledge-base
```

---

## Keywords

`RAG` `Retrieval-Augmented Generation` `Vector Database` `ChromaDB` `LangChain` `Hybrid Search` `BM25` `Cross-Encoder Reranking` `RAGAS Evaluation` `Hallucination Detection` `Sentence Transformers` `LLM` `Production AI` `Enterprise AI`
