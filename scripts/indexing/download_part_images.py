"""
Populates storage/index/part_images/{sku}/ from "Webpage, Complete" browser saves.

For each .htm/.html file in storage/raw/partselect_html/:
  - extracts SKU from [itemprop='mpn']
  - copies all images from the matching *_files/ folder to part_images/{sku}/

Idempotent: skips files that already exist.
Run from project root:
    python scripts/indexing/download_part_images.py
"""

import shutil
from pathlib import Path

from bs4 import BeautifulSoup

HTML_DIR = Path("storage/raw/partselect_html")
OUT_DIR = Path("storage/index/part_images")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _extract_sku(html_path: Path) -> str:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.select_one("[itemprop='mpn']")
    return tag.get_text(strip=True) if tag else ""


def main() -> None:
    html_files = list(HTML_DIR.glob("*.htm")) + list(HTML_DIR.glob("*.html"))
    if not html_files:
        print(f"No HTML files found in {HTML_DIR}")
        return

    copied = skipped = 0
    for html_path in sorted(html_files):
        sku = _extract_sku(html_path)
        if not sku:
            print(f"  [SKIP] No SKU in {html_path.name}")
            continue

        files_dir = html_path.with_name(html_path.stem + "_files")
        if not files_dir.is_dir():
            print(f"  [SKIP] No _files/ folder for {html_path.name}")
            continue

        for img in sorted(files_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            dest = OUT_DIR / sku / img.name
            if dest.exists():
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            copied += 1
            print(f"  COPIED  {sku}  {img.name}")

    print(f"\nDone. {copied} copied, {skipped} already existed.")


if __name__ == "__main__":
    main()
