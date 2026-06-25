"""
indexing_pipeline.py — abstract indexing pipeline: chunk → embed → index.

Accepts any VectorDB implementation. Call run_pipeline(db) from a concrete
runner script (e.g. build_qdrant_index.py).
"""

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parent          # scripts/indexing/
_ROOT = _SCRIPTS.parent.parent            # project root
sys.path.insert(0, str(_ROOT / "src"))   # rag/, vectordb/
sys.path.insert(0, str(_SCRIPTS))        # chunk_docs

import numpy as np

import chunk_docs
from rag.embedder import embed_chunks
from vectordb.base import VectorDB

CHUNKS_FILE = Path("storage/chunks/rag_chunks.jsonl")


def _load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def run_pipeline(
    db: VectorDB,
    chunks_file: Path = CHUNKS_FILE,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> None:
    print("=== Step 1: Chunking ===")
    chunk_docs.main()

    print("\n=== Step 2: Loading chunks ===")
    chunks = _load_chunks(chunks_file)
    print(f"  {len(chunks)} chunks")

    print(f"\n=== Step 3: Embedding  [{embedding_model}] ===")
    vectors: np.ndarray = embed_chunks([c["text"] for c in chunks], model_name=embedding_model)
    print(f"  vectors: {vectors.shape}")

    db.index(vectors, [c["id"] for c in chunks], chunks)
    print(f"  estimated size: {db.disk_size_mb()} MB")

    print("\nDone.")
