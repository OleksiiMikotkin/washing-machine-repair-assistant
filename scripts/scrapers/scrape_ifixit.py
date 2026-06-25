import json
import re
import time
import uuid
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.ifixit.com"
OUTPUT_FILE = Path("storage/raw/ifixit_raw.jsonl")

GUIDE_URLS = [
    "https://www.ifixit.com/Troubleshooting/Washing_Machine/Not+Spinning/484555",
    "https://www.ifixit.com/Troubleshooting/Washing_Machine/Shaking/500571",
    "https://www.ifixit.com/Troubleshooting/Washing_Machine/Not+Draining/523227",
    "https://www.ifixit.com/Troubleshooting/Washing_Machine/Leaking/484423",
    "https://www.ifixit.com/Troubleshooting/Washing_Machine/Locked+Washing+Machine/483069",
    "https://www.ifixit.com/Troubleshooting/Washing_Machine/Will+Not+Turn+On/493515",
    "https://www.ifixit.com/Troubleshooting/Washing_Machine/Has+No+Water/493438",
    "https://www.ifixit.com/Troubleshooting/Washing_Machine/Troubleshooting/536220",
]
DELAY_SECONDS = 2

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
}

PARTS_KEYWORDS = [
    "drain pump", "door seal", "door gasket", "drive belt", "belt",
    "bearing", "drum bearing", "capacitor", "motor", "control board",
    "lid switch", "door latch", "door lock", "pressure switch",
    "water valve", "inlet valve", "filter", "pump filter", "hose",
    "drain hose", "spider arm", "spider", "clutch", "coupler",
    "shock absorber", "suspension rod", "agitator", "tub seal",
    "shift actuator", "splutch",
]


def fetch(url: str) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [ERROR] {url}: {e}")
        return None



def detect_machine_types(text: str) -> list[str]:
    text_lower = text.lower()
    types = []
    if any(kw in text_lower for kw in ["front loader", "front-load", "front load"]):
        types.append("front_loader")
    if any(kw in text_lower for kw in ["top loader", "top-load", "top load"]):
        types.append("top_loader")
    return types or ["unknown"]


def extract_parts(text: str) -> list[str]:
    text_lower = text.lower()
    return [part for part in PARTS_KEYWORDS if part in text_lower]


def get_siblings_text(tag, stop_tags: tuple = ("h2", "h3")) -> str:
    parts = []
    for sibling in tag.find_next_siblings():
        if sibling.name in stop_tags:
            break
        text = sibling.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return " ".join(parts)


def get_chakra_text(heading_tag) -> str:
    """Fallback for Chakra UI pages: heading is in a wrapper div; text is in the sibling div."""
    parent = heading_tag.parent
    if not parent or parent.name != "div":
        return ""
    gp = parent.parent
    if not gp or gp.name != "div":
        return ""
    children = [c for c in gp.children if getattr(c, "name", None)]
    if len(children) >= 2:
        return children[-1].get_text(separator=" ", strip=True)
    return ""


def make_chunk(heading_tag, symptom: str, url: str, section: str | None) -> dict | None:
    cause_name = heading_tag.get_text(strip=True)
    cause_name = re.sub(r"\[.*?\]", "", cause_name).strip()  # remove wiki edit links

    if not cause_name or len(cause_name) < 4:
        return None

    stop = ("h2",) if heading_tag.name == "h2" else ("h2", "h3")
    text = get_siblings_text(heading_tag, stop_tags=stop) or get_chakra_text(heading_tag)
    if not text or len(text) < 30:
        return None

    full_text = f"{cause_name}. {text}"
    anchor_id = heading_tag.get("id", "")

    return {
        "id": str(uuid.uuid4()),
        "doc_type": "troubleshooting",
        "device_type": "washing_machine",
        "machine_type": detect_machine_types(full_text),
        "brand": None,
        "model": None,
        "symptom": symptom,
        "cause": None if section == "first_steps" else cause_name,
        "error_code": None,
        "section": section,
        "text": full_text,
        "parts_mentioned": extract_parts(full_text),
        "source_url": f"{url}#{anchor_id}" if anchor_id else url,
        "source_file": None,
    }


def extract_chunks(soup: BeautifulSoup, url: str) -> list[dict]:
    chunks = []

    h1 = soup.find("h1")
    page_title = h1.get_text(strip=True) if h1 else ""
    symptom = re.sub(r"^Washing Machine\s*", "", page_title, flags=re.IGNORECASE).strip().lower()

    # detect if page uses h2→h3 structure:
    # "Causes" h2 exists either by id or by text content
    causes_h2 = soup.find("h2", id=re.compile(r"causes", re.I)) or \
                next((h for h in soup.find_all("h2")
                      if h.get_text(strip=True).lower() == "causes"), None)
    # only True when actual causes are h3 under "Causes" h2 (not when causes are h2 themselves)
    has_nested_structure = False
    if causes_h2:
        for sib in causes_h2.find_next_siblings():
            if sib.name == "h2":
                break
            if sib.name == "h3":
                has_nested_structure = True
                break

    # page title words to skip if they appear as an h2 (id="top")
    page_title_lower = page_title.lower()

    for h2 in soup.find_all("h2"):
        h2_text = h2.get_text(strip=True).lower()

        # skip the page title duplicated as h2 (id="top")
        if h2_text == page_title_lower or h2.get("id") == "top":
            continue

        # First Steps → section flag, no cause
        if "first" in h2_text and "step" in h2_text:
            chunk = make_chunk(h2, symptom, url, section="first_steps")
            if chunk:
                chunks.append(chunk)
            continue

        # skip structural/nav headings
        if any(kw in h2_text for kw in ["causes", "related", "edit", "history", "flag"]):
            continue

        if has_nested_structure:
            # causes are h3 under the "Causes" h2
            for h3 in h2.find_next_siblings():
                if h3.name == "h2":
                    break
                if h3.name == "h3":
                    chunk = make_chunk(h3, symptom, url, section=None)
                    if chunk:
                        chunks.append(chunk)
        else:
            # causes are h2 directly
            chunk = make_chunk(h2, symptom, url, section=None)
            if chunk:
                chunks.append(chunk)

    return chunks


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for url in GUIDE_URLS:
            print(f"Scraping: {url}")
            soup = fetch(url)
            if not soup:
                continue

            chunks = extract_chunks(soup, url)
            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            print(f"  → {len(chunks)} chunks")
            total_chunks += len(chunks)
            time.sleep(DELAY_SECONDS)

    print(f"\nDone. Total chunks: {total_chunks}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
