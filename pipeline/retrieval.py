from typing import List, Dict
from functools import lru_cache

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from vectorstore.chroma_manager import ChromaManager
from pipeline.config import TOP_K_RETRIEVAL, TOP_K_RERANK, RERANKER_MODEL


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def reciprocal_rank_fusion(
    dense_hits: List[Dict], bm25_hits: List[Dict], k: int = 60
) -> List[Dict]:
    scores: Dict[str, float] = {}
    docs_map: Dict[str, Dict] = {}

    for rank, hit in enumerate(dense_hits):
        key = hit["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs_map[key] = hit  # dense hits carry full metadata — always preferred

    for rank, hit in enumerate(bm25_hits):
        key = hit["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in docs_map:  # don't overwrite dense hit's metadata with BM25's empty {}
            docs_map[key] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs_map[text] for text, _ in ranked]


class HybridRetriever:
    def __init__(self, chroma: ChromaManager = None):
        self.chroma = chroma or ChromaManager()
        self._bm25 = None
        self._bm25_corpus: List[str] = []

    def _build_bm25_index(self):
        corpus = self.chroma.get_all_texts()
        if not corpus:
            self._bm25 = None
            return
        tokenized = [doc.lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus = corpus

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        if self._bm25 is None:
            self._build_bm25_index()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"text": self._bm25_corpus[i], "metadata": {}, "score": float(scores[i])} for i in top_indices]

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        dense_hits = self.chroma.similarity_search(query, top_k=top_k)
        bm25_hits = self._bm25_search(query, top_k=top_k)
        fused = reciprocal_rank_fusion(dense_hits, bm25_hits)
        return fused[:top_k]

    def rerank(self, query: str, hits: List[Dict], top_k: int = TOP_K_RERANK) -> List[Dict]:
        if not hits:
            return []
        reranker = get_reranker()
        pairs = [[query, hit["text"]] for hit in hits]
        scores = reranker.predict(pairs)
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)
        reranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def search(self, query: str) -> List[Dict]:
        hits = self.retrieve(query)
        return self.rerank(query, hits)

    def invalidate_bm25_cache(self):
        self._bm25 = None
        self._bm25_corpus = []
