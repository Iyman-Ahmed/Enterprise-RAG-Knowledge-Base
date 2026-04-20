from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer
from pipeline.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False).tolist()


def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0]
