import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM provider ──────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio")  # "lmstudio" | "groq"

# LM Studio (local)
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF")

# Groq (deployment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── ChromaDB ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

# ── Embeddings ────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

# ── Retrieval ─────────────────────────────────────────────────
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "15"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
