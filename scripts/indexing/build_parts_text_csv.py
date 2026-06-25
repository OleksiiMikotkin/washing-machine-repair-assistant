"""
build_catalog.py — builds data/parts.csv from partselect_raw.jsonl.

Run from project root:
    python scripts/indexing/build_catalog.py

Source:  storage/raw/partselect_raw.jsonl  (produced by scrape_partselect.py)
Output:  data/parts.csv
"""

import csv
import json
from pathlib import Path

INPUT_FILE = Path("storage/raw/partselect_raw.jsonl")
OUTPUT_FILE = Path("data/parts.csv")

CSV_COLUMNS = ["sku", "name", "brand", "aliases", "category", "description", "compatibility"]


def build_row(record: dict) -> dict:
    aliases_raw = record.get("aliases", [])
    aliases = "|".join(aliases_raw) if isinstance(aliases_raw, list) else str(aliases_raw)

    compat_raw = record.get("compatibility", [])
    compatibility = "|".join(compat_raw) if isinstance(compat_raw, list) else str(compat_raw)

    # prefer manufacturer part number as the catalog key (e.g. DC32-00007A)
    # fall back to PartSelect PS-number if manufacturer_pn is missing
    sku = record.get("manufacturer_pn") or record.get("sku", "")

    return {
        "sku": sku,
        "name": record.get("name", ""),
        "brand": record.get("brand", ""),
        "aliases": aliases,
        "category": record.get("category", ""),
        "description": record.get("description", ""),
        "compatibility": compatibility,
    }


def main() -> None:
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with INPUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row = build_row(record)
            if row["sku"] or row["name"]:
                rows.append(row)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
