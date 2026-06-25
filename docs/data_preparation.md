# Підготовка даних

Цей документ описує повний цикл збору даних та побудови RAG-індексу з нуля. Потрібен тільки якщо ви хочете зібрати власну базу або переіндексувати дані у своєму Qdrant-кластері.

---

## Вимоги

- Docker (для локального Qdrant) **або** обліковий запис [Qdrant Cloud](https://cloud.qdrant.io)
- Залежності встановлені: `pip install -r requirements.txt`

---

## Крок 1 — Запуск Qdrant

**Локально (Docker):**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Qdrant Cloud:** створіть кластер, скопіюйте URL та API-ключ у `.env` (`QDRANT_URL`, `QDRANT_API_KEY`) — Docker не потрібен.

---

## Крок 2 — Збір даних

```bash
# ifixit.com (8 сторінок) → storage/raw/ifixit_raw.jsonl  (~68 чанків)
python scripts/scrapers/scrape_ifixit.py

# storage/raw/partselect_html/*.htm → storage/raw/partselect_raw.jsonl  (див. примітку нижче)
python scripts/scrapers/scrape_partselect.py
```

> **Примітка щодо PartSelect:** відкрийте сторінку кожної деталі на partselect.com у браузері, збережіть як «Веб-сторінка, повністю» (Ctrl+S → «Webpage, Complete») і скопіюйте збережені файли до `storage/raw/partselect_html/`.

---

## Крок 3 — Побудова індексів

```bash
# --- Текстовий пошук ---

# partselect_raw.jsonl → data/parts.csv  (SKU, назва, бренд, опис, сумісність)
python scripts/indexing/build_parts_text_csv.py

# ifixit_raw.jsonl + partselect_raw.jsonl → Qdrant колекція washing_machine  (текстові вектори)
python scripts/indexing/build_parts_text_index.py

# --- CLIP (пошук за зображенням) ---

# partselect_html/*.htm + *_files/ → storage/index/part_images/{sku}/  (витягує SKU з HTML, копіює фото)
python scripts/indexing/download_part_images.py

# parts.csv + part_images/{sku}/ → Qdrant колекція parts_images  (CLIP-вектори зображень)
python scripts/indexing/build_parts_clip_index.py
```
