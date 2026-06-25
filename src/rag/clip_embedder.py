"""
CLIP embedder for multi-modal RAG.

Maps both images and text into the same vector space using a CLIP model
controlled by the CLIP_MODEL env var, enabling cross-modal search:
  - text query  → find similar images
  - image query → find matching SKU/text payload
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-16")


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(CLIP_MODEL)


def embed_text(texts: list[str]) -> np.ndarray:
    """Encode a list of strings into 512-dim CLIP vectors."""
    return _model().encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def embed_images(images: list[Image.Image]) -> np.ndarray:
    """Encode a list of PIL Images into 512-dim CLIP vectors."""
    return _model().encode(
        images,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def load_image(source: str | Path) -> Image.Image:
    """Load a PIL Image from a URL or local file path."""
    if isinstance(source, Path) or (isinstance(source, str) and not source.startswith("http")):
        return Image.open(source).convert("RGB")
    response = requests.get(source, timeout=10)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def embed_image_url(url: str) -> np.ndarray:
    """Convenience: download one image URL and return its 1×512 embedding."""
    img = load_image(url)
    return embed_images([img])
