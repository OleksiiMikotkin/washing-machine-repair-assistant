import os

import numpy as np

from rag.embedder import embed_chunks
from vectordb.base import VectorDB


class Retriever:
    def __init__(self, db: VectorDB, model: str | None = None):
        self.db = db
        self.model = model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        vec: np.ndarray = embed_chunks([query], model_name=self.model)[0]
        hits = self.db.search(vec, top_k=top_k, filters=filters)
        return [{"payload": payload, "score": round(score, 4)} for payload, score in hits]
