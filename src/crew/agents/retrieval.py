import os

from crew.state import GraphState
from tools.retriever import Retriever
from vectordb.qdrant_client import QdrantDB

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        db = QdrantDB(
            collection_name=os.getenv("QDRANT_COLLECTION", "washing_machine"),
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6334")),
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        _retriever = Retriever(db)
    return _retriever


_EXACT_FILTER_KEYS = {"brand", "model", "error_code"}


def retrieval(state: GraphState) -> dict:
    query = state["query"]
    if state.get("image_description"):
        query = f"{query}\n{state['image_description']}"

    raw_filters = state.get("filters") or {}
    filters = {k: v for k, v in raw_filters.items() if k in _EXACT_FILTER_KEYS} or None

    results = _get_retriever().search(query=query, filters=filters)
    return {"retrieval_results": results}
