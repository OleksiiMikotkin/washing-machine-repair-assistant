import os
from pathlib import Path

from PIL import Image

from crew.state import GraphState
from rag.clip_embedder import embed_images, load_image
from vectordb.qdrant_client import QdrantDB

COLLECTION = "parts_images"
_SEARCH_K = 2
_MIN_SCORE = 0.65
_SCORE_MARGIN = 0.05  # include hits within this margin of the best score


def _db() -> QdrantDB:
    return QdrantDB(
        collection_name=COLLECTION,
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6334")),
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


def clip_retrieval(state: GraphState) -> dict:
    image_path = state.get("image_path")
    if not image_path:
        return {"parts_results": []}

    img: Image.Image = load_image(image_path)
    vec = embed_images([img])[0]

    hits = _db().search(vec, top_k=_SEARCH_K * 4)
    if not hits or hits[0][1] < _MIN_SCORE:
        return {"parts_results": [], "clip_score": None}

    best_score = hits[0][1]
    threshold = max(best_score - _SCORE_MARGIN, _MIN_SCORE)
    matched = [part for part, score in hits if score >= threshold][:_SEARCH_K]

    return {
        "parts_results": matched,
        "clip_score": best_score,
    }
