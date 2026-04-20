import os
from pathlib import Path
from typing import List, Dict, Any

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from pipeline.config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS


def load_document(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)

    docs = loader.load()
    return _enrich_metadata(docs, file_path)


def _enrich_metadata(docs: List[Document], file_path: str) -> List[Document]:
    file_name = Path(file_path).name
    for doc in docs:
        doc.metadata["source_file"] = file_name
        doc.metadata["file_path"] = file_path
        doc.metadata.setdefault("page", 0)
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def ingest_file(file_path: str) -> List[Document]:
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    return chunks


def ingest_directory(dir_path: str) -> List[Document]:
    all_chunks = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            full_path = os.path.join(root, file)
            if Path(full_path).suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    all_chunks.extend(ingest_file(full_path))
                except Exception as e:
                    print(f"[ingestion] Skipping {full_path}: {e}")
    return all_chunks
