"""
build_parts_text_index.py — creates a QdrantDB instance and runs the full pipeline.

Run from project root:
    python scripts/indexing/build_parts_text_index.py

Requires Qdrant running locally:
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Settings are read from .env:
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION  (local)
    QDRANT_URL, QDRANT_API_KEY                    (Qdrant Cloud)
    EMBEDDING_MODEL                               (default: all-MiniLM-L6-v2)
"""

import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parent   # scripts/indexing/
_ROOT = _SCRIPTS.parent.parent     # project root
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_SCRIPTS))

from dotenv import load_dotenv

from vectordb.qdrant_client import QdrantDB
from indexing_pipeline import run_pipeline


def main() -> None:
    load_dotenv()

    db = QdrantDB(
        collection_name=os.getenv("QDRANT_COLLECTION", "washing_machine"),
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6334")),
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    run_pipeline(db, embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))


if __name__ == "__main__":
    main()
