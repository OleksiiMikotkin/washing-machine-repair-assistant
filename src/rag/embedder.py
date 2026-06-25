from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=4)
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed_chunks(texts: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    model = _model(model_name)
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
