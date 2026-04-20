import shutil
import tempfile
import os
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.ingestion import ingest_file
from pipeline.retrieval import HybridRetriever
from pipeline.generation import get_pipeline
from pipeline.guardrails import check_hallucination
from vectorstore.chroma_manager import ChromaManager

app = FastAPI(title="Enterprise RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = HybridRetriever()
chroma = ChromaManager()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    hallucination: dict
    provider: str


@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": chroma.count()}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = ingest_file(tmp_path)
        # patch source_file metadata to use original filename
        for c in chunks:
            c.metadata["source_file"] = file.filename
        added = chroma.add_documents(chunks)
        retriever.invalidate_bm25_cache()
    finally:
        os.unlink(tmp_path)

    return {"filename": file.filename, "chunks_indexed": added}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    hits = retriever.search(request.question)
    if not hits:
        raise HTTPException(status_code=404, detail="No relevant documents found. Upload documents first.")

    pipeline = get_pipeline()
    result = pipeline.generate(request.question, hits)
    hallucination = check_hallucination(result["answer"], hits)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        hallucination=hallucination,
        provider=result["provider"],
    )


@app.delete("/documents")
def clear_documents():
    chroma.delete_collection()
    retriever.invalidate_bm25_cache()
    return {"message": "All documents cleared."}
