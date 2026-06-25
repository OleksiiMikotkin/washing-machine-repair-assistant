"""
Parses locally saved PartSelect HTML files into JSONL.

Workflow:
  1. Open part page in browser, save as HTML (Ctrl+S) into storage/raw/partselect_html/
  2. Run this script → storage/raw/partselect_raw.jsonl
  3. chunk_docs.py splits troubleshooting into Qdrant chunks
  4. build_parts_text_csv.py builds parts.csv from remaining fields
"""

import json
import re
import uuid
from pathlib import Path

from bs4 import BeautifulSoup

HTML_DIR = Path("storage/raw/partselect_html")
OUTPUT_FILE = Path("storage/raw/partselect_raw.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def find_section_text(soup: BeautifulSoup, *keywords: str) -> str:
    """Find text content of a section whose heading contains any of the keywords."""
    for tag in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
        tag_text = tag.get_text(strip=True).lower()
        if any(kw.lower() in tag_text for kw in keywords):
            parts = []
            for sibling in tag.find_next_siblings():
                if sibling.name in ("h2", "h3", "h4"):
                    break
                text = sibling.get_text(separator=" ", strip=True)
                if text:
                    parts.append(text)
            result = clean(" ".join(parts))
            if result:
                return result
    return ""


def find_text_near_label(soup: BeautifulSoup, *labels: str) -> str:
    """Find value next to a label like 'Part #:' or 'Brand:'."""
    for label in labels:
        pattern = re.compile(re.escape(label), re.IGNORECASE)
        tag = soup.find(string=pattern)
        if tag:
            parent = tag.parent
            # value may be in next sibling or same parent
            next_el = parent.find_next_sibling()
            if next_el:
                return clean(next_el.get_text())
            return clean(re.sub(re.escape(label), "", tag, flags=re.IGNORECASE))
    return ""


def parse_from_filename(filename: str) -> dict:
    """
    Extract basic info from filename pattern:
    PS1485646-Whirlpool-285753A-Direct-Drive-Motor-Coupling.htm
    """
    stem = Path(filename).stem
    parts = stem.split("-")

    ps_number = parts[0] if parts and parts[0].startswith("PS") else ""
    brand = parts[1] if len(parts) > 1 else ""
    manufacturer_pn = parts[2] if len(parts) > 2 else ""
    name = " ".join(parts[3:]).replace("-", " ").title() if len(parts) > 3 else ""

    return {
        "sku": ps_number,
        "manufacturer_pn": manufacturer_pn,
        "name": name,
        "brand": brand,
    }


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_part(html_path: Path) -> dict:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # --- base info from filename (reliable fallback) ---
    base = parse_from_filename(html_path.name)

    # --- name ---
    name = ""
    for sel in ["h1.title-main", "h1.pd__name", "h1"]:
        tag = soup.select_one(sel)
        if tag:
            name = clean(tag.get_text())
            break
    name = name or base["name"]

    # --- SKU / part numbers ---
    sku = base["sku"]
    manufacturer_pn = base["manufacturer_pn"]

    # itemprop attributes are the most reliable selectors on PartSelect pages
    tag = soup.select_one("[itemprop='productID']")
    if tag:
        sku = clean(tag.get_text())
    tag = soup.select_one("[itemprop='mpn']")
    if tag:
        manufacturer_pn = clean(tag.get_text())

    # --- brand ---
    brand = base["brand"]
    tag = soup.select_one("[itemprop='brand']")
    if tag:
        brand = clean(tag.get_text()) or brand

    # --- price ---
    price = ""
    for sel in ["[itemprop='price']", ".price", ".pd__price", "[class*='price']"]:
        tag = soup.select_one(sel)
        if tag:
            val = clean(tag.get_text())
            if "$" in val or re.search(r"\d+\.\d{2}", val):
                price = val
                break

    # --- category ---
    category = ""
    breadcrumbs = soup.select("ol.breadcrumb li, nav [aria-label='breadcrumb'] li, .breadcrumb li")
    if breadcrumbs:
        # usually: Home > Brand > Category > Part name
        texts = [clean(b.get_text()) for b in breadcrumbs if clean(b.get_text())]
        if len(texts) >= 3:
            category = texts[-2]  # second to last is category

    # --- description ---
    description = find_section_text(soup, "Product Description", "Description")
    if not description:
        tag = soup.select_one("[itemprop='description'], .pd__description, .product-description")
        if tag:
            description = clean(tag.get_text())

    # --- troubleshooting ---
    # PartSelect uses <div id="Troubleshooting"> (not a heading), so find_section_text misses it.
    # The symptoms are in <li> elements inside the next sibling div.
    troubleshooting = ""
    ts_div = soup.find("div", id=re.compile(r"^Troubleshooting$", re.I))
    if ts_div:
        wrap = ts_div.find_next_sibling("div")
        if wrap:
            symptoms_block = wrap.find(string=re.compile(r"fixes the following symptoms", re.I))
            if symptoms_block:
                ul = symptoms_block.find_parent().find_next_sibling("ul")
                if ul:
                    troubleshooting = ", ".join(
                        clean(li.get_text()) for li in ul.find_all("li") if li.get_text(strip=True)
                    )
    if not troubleshooting:
        troubleshooting = find_section_text(soup, "Troubleshooting", "Symptoms")

    # --- aliases (replaces) ---
    aliases: list[str] = []
    replaces_text = find_section_text(soup, "replaces", "Replaces These", "Also replaces")
    if replaces_text:
        # extract part-number-like tokens: letters+digits, 5-15 chars
        aliases = re.findall(r"\b[A-Z0-9]{5,15}\b", replaces_text)

    # --- compatibility ---
    compatibility: list[str] = []
    compat_section = find_section_text(soup, "Works With", "Compatible", "Fits")
    if compat_section:
        # model numbers typically look like: WW60J3047, WTW5000DW, etc.
        compatibility = re.findall(r"\b[A-Z]{2,4}\d{3,}[A-Z0-9]*\b", compat_section)

    # fallback: look for a compatibility table
    if not compatibility:
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                cells = [clean(td.get_text()) for td in row.find_all(["td", "th"])]
                for cell in cells:
                    models = re.findall(r"\b[A-Z]{2,4}\d{3,}[A-Z0-9]*\b", cell)
                    compatibility.extend(models)
        compatibility = list(dict.fromkeys(compatibility))  # deduplicate, preserve order

    # --- image URL ---
    image_url = ""
    for sel in ["[itemprop='image']", ".pd__image img", ".product-image img", "img.main-image"]:
        tag = soup.select_one(sel)
        if tag:
            image_url = tag.get("src") or tag.get("data-src") or ""
            if image_url:
                break

    return {
        "id": str(uuid.uuid4()),
        "sku": sku,
        "manufacturer_pn": manufacturer_pn,
        "name": name,
        "brand": brand,
        "category": category,
        "price": price,
        "description": description,
        "troubleshooting": troubleshooting,
        "aliases": aliases,
        "compatibility": compatibility,
        "image_url": image_url,
        "source_file": html_path.name,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not HTML_DIR.exists():
        print(f"Directory not found: {HTML_DIR}")
        print("Save PartSelect pages as HTML into that folder, then re-run.")
        return

    html_files = list(HTML_DIR.glob("*.htm")) + list(HTML_DIR.glob("*.html"))
    if not html_files:
        print(f"No HTML files found in {HTML_DIR}")
        return

    print(f"Found {len(html_files)} HTML files")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    parsed = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for path in sorted(html_files):
            print(f"  Parsing: {path.name}")
            try:
                record = parse_part(path)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                parsed += 1
            except Exception as e:
                print(f"    [ERROR] {e}")

    print(f"\nDone. Parsed: {parsed}/{len(html_files)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
