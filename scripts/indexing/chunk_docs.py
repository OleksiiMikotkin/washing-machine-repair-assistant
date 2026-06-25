"""
chunk_docs.py — reads all raw JSONL files from storage/raw/, splits long texts,
writes final chunks to storage/chunks/chunks.jsonl.

Run from project root:
    python scripts/chunk_docs.py
"""

import json
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR = Path("storage/raw")
OUT_FILE = Path("storage/chunks/rag_chunks.jsonl")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


_PARTSELECT_DROP = {"price", "source_file"}


def _normalize(record: dict) -> dict:
    """Build a unified 'text' field for records that don't have one (partselect format)."""
    if "text" in record:
        return record

    parts = []
    if record.get("name"):
        parts.append(record["name"])
    if record.get("description"):
        parts.append(record["description"])
    if record.get("troubleshooting"):
        parts.append(f"Fixes: {record['troubleshooting']}")

    cleaned = {k: v for k, v in record.items() if k not in _PARTSELECT_DROP}
    cleaned["text"] = ". ".join(parts)
    cleaned.setdefault("doc_type", "part_page")
    return cleaned


def split_record(record: dict, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    record = _normalize(record)
    text = record.get("text", "").strip()
    if not text:
        return []

    sub_texts = splitter.split_text(text)

    if len(sub_texts) == 1:
        # text fits in one chunk — keep original id
        return [{**record, "text": sub_texts[0]}]

    # multiple sub-chunks: inherit all metadata, assign new UUIDs
    results = []
    for i, sub in enumerate(sub_texts):
        chunk = {**record, "id": str(uuid.uuid4()), "text": sub}
        if i > 0:
            # mark that this is a continuation so retrieval agent can note it
            chunk["chunk_index"] = i
        results.append(chunk)
    return results


def main() -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        print(f"No JSONL files found in {RAW_DIR}")
        return

    total_in = 0
    total_out = 0

    with OUT_FILE.open("w", encoding="utf-8") as out_f:
        for raw_file in raw_files:
            file_in = 0
            file_out = 0
            with raw_file.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    chunks = split_record(record, splitter)
                    for chunk in chunks:
                        out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    file_in += 1
                    file_out += len(chunks)

            print(f"  {raw_file.name}: {file_in} records -> {file_out} chunks")
            total_in += file_in
            total_out += file_out

    print(f"\nTotal: {total_in} records -> {total_out} chunks")
    print(f"Written to {OUT_FILE}")


if __name__ == "__main__":
    main()
