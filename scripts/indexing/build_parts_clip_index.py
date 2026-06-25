"""
Builds a CLIP-based Qdrant collection for part images.

Reads:
  data/parts.csv                     — sku, name, brand, category, ...
  storage/index/part_images/{sku}/   — downloaded image files (any format)

For each image found on disk, encodes it with CLIP (512-dim) and upserts
into the 'parts_images' Qdrant collection with full part metadata as payload.

Run from project root:
    python scripts/indexing/build_parts_clip_index.py

Requires Qdrant running:
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
"""

import csv
import os
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
from PIL import Image

from rag.clip_embedder import embed_images
from vectordb.qdrant_client import QdrantDB

PARTS_CSV = _ROOT / "data/parts.csv"
IMAGES_DIR = _ROOT / "storage/index/part_images"
COLLECTION = "parts_images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _load_parts(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {row["sku"]: row for row in csv.DictReader(f)}


def main() -> None:
    load_dotenv()

    parts = _load_parts(PARTS_CSV)

    # Collect all image files grouped by SKU folder
    ready: list[dict] = []
    for sku_dir in sorted(IMAGES_DIR.iterdir()):
        if not sku_dir.is_dir():
            continue
        sku = sku_dir.name
        for img_path in sorted(sku_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_SUFFIXES:
                ready.append({"sku": sku, "local_path": img_path})

    if not ready:
        print("No images found. Run download_part_images.py first.")
        return

    skus_found = len({r["sku"] for r in ready})
    print(f"Found {len(ready)} images across {skus_found} SKUs.")

    # Load and encode all images with CLIP
    print("Encoding with CLIP...")
    pil_images = [Image.open(r["local_path"]).convert("RGB") for r in ready]
    vectors = embed_images(pil_images)

    # Build payloads joined with parts.csv metadata
    ids: list[str] = []
    payloads: list[dict] = []
    for r in ready:
        sku = r["sku"]
        meta = parts.get(sku, {})
        ids.append(str(uuid.uuid4()))
        payloads.append({
            "sku": sku,
            "image_path": str(r["local_path"]),
            "name": meta.get("name", ""),
            "brand": meta.get("brand", ""),
            "category": meta.get("category", ""),
        })

    # Upsert into Qdrant
    db = QdrantDB(
        collection_name=COLLECTION,
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6334")),
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    print(f"Upserting {len(ids)} vectors into '{COLLECTION}' collection...")
    db.index(vectors, ids, payloads)

    print(f"Done. {len(ids)} image vectors indexed across {skus_found} SKUs.")


if __name__ == "__main__":
    main()
