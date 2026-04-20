# Error Navigation Tree

Use this tree to trace any error to its source file instantly.

---

## LAYER 1 — Is the error on startup or at runtime?

```
Error occurs
├── ON STARTUP (import errors, missing packages, env vars)
│   ├── ModuleNotFoundError         → requirements.txt (missing package)
│   ├── KeyError / missing env var  → .env file + pipeline/config.py
│   └── ChromaDB connection error   → vectorstore/chroma_manager.py:__init__
│
└── AT RUNTIME (after startup, during a request)
    ├── During document UPLOAD      → LAYER 2 (Ingestion)
    ├── During SEARCH / RETRIEVAL   → LAYER 3 (Retrieval)
    ├── During LLM ANSWER           → LAYER 4 (Generation)
    ├── During EVALUATION           → LAYER 5 (Evaluation)
    └── API / UI error              → LAYER 6 (API/UI)
```

---

## LAYER 2 — Ingestion Errors
**File:** `pipeline/ingestion.py`

```
Ingestion error
├── PDF parse fails / empty text    → ingestion.py → PDFLoader config
├── DOCX not loading                → ingestion.py → Docx2txtLoader
├── Chunks too large / too small    → ingestion.py → RecursiveCharacterTextSplitter params
├── Metadata missing                → ingestion.py → _extract_metadata()
└── File format not supported       → ingestion.py → SUPPORTED_EXTENSIONS dict
```

---

## LAYER 3 — Retrieval Errors
**Files:** `pipeline/retrieval.py`, `vectorstore/chroma_manager.py`, `pipeline/embeddings.py`

```
Retrieval error
├── No results returned             → retrieval.py → HybridRetriever.retrieve()
├── Embedding dimension mismatch    → embeddings.py → model name mismatch with ChromaDB collection
├── ChromaDB collection not found   → chroma_manager.py → get_or_create_collection()
├── BM25 index empty                → retrieval.py → BM25Retriever._build_index()
├── Reranker slow / failing         → retrieval.py → CrossEncoderReranker (check model download)
└── RRF fusion wrong scores         → retrieval.py → reciprocal_rank_fusion()
```

---

## LAYER 4 — Generation Errors
**File:** `pipeline/generation.py`

```
Generation error
├── LM Studio not responding        → generation.py → LMStudioLLM (is LM Studio running on port 1234?)
├── Groq API error                  → generation.py → GroqLLM (check GROQ_API_KEY in .env)
├── Context too long / token limit  → generation.py → _truncate_context()
├── No citations in response        → generation.py → _parse_citations()
├── Hallucination flag always ON    → guardrails.py → HallucinationDetector threshold
└── LLM returns empty string        → generation.py → check model is loaded in LM Studio
```

---

## LAYER 5 — Evaluation Errors
**Files:** `evaluation/ragas_eval.py`, `evaluation/metrics.py`

```
Evaluation error
├── RAGAS import error              → requirements.txt (ragas package)
├── Test dataset not found          → evaluation/test_dataset.json (run generator first)
├── Faithfulness score always 0     → ragas_eval.py → dataset format check
└── Metrics not saving              → evaluation/metrics.py → output path
```

---

## LAYER 6 — API / UI Errors
**Files:** `api/main.py`, `app.py`

```
API/UI error
├── FastAPI 422 Unprocessable       → api/models/ → Pydantic schema mismatch
├── CORS error in browser           → api/main.py → CORSMiddleware config
├── Streamlit crashes on upload     → app.py → file size / type validation
├── Slow response in UI             → retrieval.py top_k value (reduce it)
└── Port already in use             → change port in api/main.py or app.py
```

---

## QUICK REFERENCE — Which file owns what

| Concern | File |
|---|---|
| Config / env vars | `pipeline/config.py` |
| Document loading + chunking | `pipeline/ingestion.py` |
| Embedding model | `pipeline/embeddings.py` |
| Vector DB (ChromaDB) | `vectorstore/chroma_manager.py` |
| Hybrid search + reranking | `pipeline/retrieval.py` |
| LLM answer (LM Studio / Groq) | `pipeline/generation.py` |
| Hallucination detection | `pipeline/guardrails.py` |
| RAGAS evaluation | `evaluation/ragas_eval.py` |
| Custom metrics | `evaluation/metrics.py` |
| REST API endpoints | `api/main.py` |
| Streamlit UI | `app.py` |
| Docker / HuggingFace deploy | `Dockerfile` |
| Python dependencies | `requirements.txt` |

---

## LM Studio Checklist (local dev)
If generation fails locally, check these in order:
1. Is LM Studio app open?
2. Is a model loaded (green dot in LM Studio)?
3. Is the local server ON? (toggle in LM Studio → Local Server tab)
4. Is it running on port `1234`? (default)
5. Check `LM_STUDIO_URL=http://localhost:1234/v1` in your `.env`
