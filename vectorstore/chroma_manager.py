import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from langchain.schema import Document

from pipeline.embeddings import embed_texts, embed_query
from pipeline.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


class ChromaManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: List[Document]) -> int:
        texts = [c.page_content for c in chunks]
        embeddings = embed_texts(texts)
        ids = [f"{c.metadata.get('source_file','doc')}_{c.metadata.get('chunk_id', i)}" for i, c in enumerate(chunks)]
        metadatas = [c.metadata for c in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def similarity_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"text": doc, "metadata": meta, "score": 1 - dist})
        return hits

    def get_all_texts(self) -> List[str]:
        result = self.collection.get(include=["documents"])
        return result["documents"] or []

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self):
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
