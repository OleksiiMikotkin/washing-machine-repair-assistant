# Мультиагентна система підтримки клієнтів для ремонту пральних машин

AI-асистент, який допомагає користувачам діагностувати несправності пральних машин та знаходити запасні частини. Користувач описує симптом текстом або додає фото — система визначає деталь, перевіряє каталог і повертає відповідь, обґрунтовану витягнутими даними.

---

## Чому цей проєкт

Проєкт обрано зі списку [*Portfolio Projects to Get You Hired in 2026*](https://roadmap.sh) як практична реалізація двох навичок, виділених у дорожній карті AI/ML-інженера 2026 року:

1. **Agentic AI / Мультиагентна оркестрація** — LangGraph-пайплайн, у якому спеціалізовані агенти (orchestrator, vision, clip_retrieval, retrieval, parts, stock, synthesizer) спільно обробляють один запит користувача, кожен з визначеною відповідальністю та набором інструментів.

2. **Multimodal RAG** — семантичний текстовий пошук у векторній базі Qdrant у поєднанні з CLIP-пошуком за зображенням, що дозволяє знайти запасну частину за описом *або* за фотографією.

**Постановка задачі.** Коли ламається пральна машина, користувач стикається з двома проблемами: зрозуміти, що саме зламалося (діагностика), та знайти правильну запасну частину (пошук у каталозі). Ця система вирішує обидва завдання в одному розмовному інтерфейсі, заземлюючи кожну відповідь на витягнутому контексті для уникнення галюцинацій.

**Існуючі рішення на ринку.** Для діагностики: [FixBot від iFixit](https://www.ifixit.com/News/114700/introducing-fixbot) — AI-асистент на базі 125 000 ремонтних керівництв; [RepairMan](https://apps.apple.com/us/app/ai-repair-guide-repairman/id6760124678) і [Reparo](https://reparo.blink42.com/) — мобільні застосунки для візуальної діагностики. Для пошуку деталей: [Synthavo](https://www.synthavo.de/en/) і [Nyris](https://www.nyris.io/products/spare-parts-search-suite) — пошук запчастин за фото. З публічно відомих продуктів жоден не об'єднує діагностику, пошук деталі та перевірку наявності на складі в єдиному розмовному пайплайні — саме це реалізує даний проєкт.

---

## Архітектура

```
              Запит користувача
                     │
                     ▼
            ┌─────────────────┐
            │   Orchestrator  │  intent + filters
            └────────┬────────┘
                     │
                     │ (якщо є фото)
                     ▼
            ┌─────────────────┐
            │     Vision      │  LLM_VISION_ENABLED=on
            │ CLIP Retrieval  │  LLM_VISION_ENABLED=off (Qdrant)
            └────────┬────────┘
                     │
          ┌──────────┴───────────┐
          │ troubleshooting /    │ part_lookup
          │ error_code           │
          ▼                      ▼
   ┌─────────────┐       ┌─────────────┐
   │  Retrieval  │       │    Parts    │
   │   (Qdrant)  │       │   (tools)   │
   └──────┬──────┘       └──────┬──────┘
          │              ┌──────┴──────┐
          │              │    Stock    │
          │              └──────┬──────┘
          └──────────┬──────────┘
                     ▼
            ┌─────────────────┐
            │   Synthesizer   │  → відповідь (streaming)
            └─────────────────┘
```

### Агенти

| Агент | Роль |
|---|---|
| **Orchestrator** | Класифікує intent (`troubleshooting` / `part_lookup` / `error_code` / `general`), витягує структуровані фільтри |
| **Vision** | Надсилає фото користувача до Gemini Vision; записує `image_description` у спільний стан |
| **CLIP Retrieval** | Вбудовує зображення запиту через CLIP (ViT-B/16), шукає в колекції `parts_images` у Qdrant |
| **Retrieval** | Семантичний пошук по керівництвах iFixit та документах PartSelect у колекції `washing_machine` |
| **Parts** | Tool-loop агент: викликає `search_parts`, `get_part_by_sku`, `find_compatible_parts` у локальному каталозі |
| **Stock** | Синхронний пошук — перевіряє наявність на складі за SKU (без LLM) |
| **Synthesizer** | Об'єднує весь витягнутий контекст у відповідь у вільній формі; підтримує потокову передачу токенів |

### LLM Gateway (`src/core/llm_gate.py`)

Усі LLM-виклики проходять через централізований шлюз, який забезпечує:
- **Fallback-маршрутизацію** — якщо основна модель не відповідає, запит автоматично повторюється з `LLM_FALLBACK_MODEL`
- **Метрики** — кожен LLM-виклик записується у `storage/llm_calls.db` (SQLite): модель, `session_id`, вхідні та вихідні токени, загальна затримка, TTFT (час до першого токена), TPOT (мс на токен), ознака використання fallback-моделі
- **Передачу session_id** — через `ContextVar` без змін у сигнатурах агентів

### Персистентність сесій

LangGraph `SqliteSaver` зберігає повний стан графу для кожної сесії (`thread_id = session_id`) у `storage/sessions.db`, що дозволяє вести багатоходову розмову та відновлювати сесії між запитами.

---

## Технологічний стек

| Рівень | Технологія | Обґрунтування |
|---|---|---|
| Оркестрація агентів | **LangGraph** | Підтримка циклічних, стейтфул мультиагентних графів з SQLite-чекпоінтингом |
| LLM-маршрутизація | **OpenRouter** | Єдиний API-ключ для Gemini, GPT-4o та резервних моделей без прив'язки до вендора |
| Мовні моделі | **Gemini 2.5 Flash** (текст), **GPT-4o / Gemini Vision** (зображення) | Найкраще співвідношення ціни та якості на рівні продакшну |
| Векторна база | **Qdrant** | Швидкий ANN-пошук; підтримує dense (текст) та CLIP (зображення) вектори |
| Текстові ембедінги | **all-MiniLM-L6-v2** | Швидка, компактна модель із хорошим recall для технічних текстів |
| Зображення ембедінги | **CLIP ViT-B/16** | Стандартний мультимодальний ембедінг для зіставлення фото з каталожними зображеннями |
| API | **FastAPI** | Асинхронний, Pydantic-валідація вхідних даних, підтримка SSE-стримінгу |
| UI | **Gradio** | Швидке прототипування чат-інтерфейсу з підтримкою завантаження зображень |
| Eval | **власний eval-пайплайн** | 4 чекери (PII, faithfulness, hallucination, refusal/injection) на золотому наборі |

---

## Джерела даних

| Джерело | Скрипт | Результат |
|---|---|---|
| Керівництва з усунення несправностей iFixit | `scripts/scrapers/scrape_ifixit.py` | `storage/raw/ifixit_raw.jsonl` |
| Каталог запасних частин PartSelect | `scripts/scrapers/scrape_partselect.py` | `storage/raw/partselect_raw.jsonl` |
| Структурований каталог деталей | `scripts/indexing/build_parts_text_csv.py` | `data/parts.csv` |
| Зображення деталей (CLIP-індекс) | `scripts/indexing/download_part_images.py` | `storage/index/part_images/` |

Усі сирі JSONL-файли розбиваються на чанки (`chunk_size=500`), вбудовуються через `all-MiniLM-L6-v2` і завантажуються до Qdrant через `scripts/indexing/build_parts_text_index.py`.

---

## Eval-пайплайн (`eval/`)

Автоматизована перевірка якості відповідей на 20 тестових випадках чотирьох типів:

| Чекер | Тип кейсу | Метод |
|---|---|---|
| `check_pii` | усі типи | Регулярні вирази: email, телефон, номери карток |
| `check_faithfulness` | `normal` | Перевірка наявності `expected_facts` у відповіді |
| `check_hallucination` | `normal` (із контекстом) | LLM-класифікатор: чи суперечить відповідь витягнутому контексту? |
| `check_refusal` | `out_of_scope`, `pii_probe` | Список фраз (fast-path) → LLM-класифікатор (fallback) |
| `check_injection` | `injection` | Список фраз капітуляції → LLM-класифікатор (fallback) |

Класифікатор за замовчуванням використовує OpenRouter (той самий ключ, що й основний LLM). Як безкоштовна альтернатива без API-витрат — локальний [Ollama](https://ollama.com/download): встановіть, завантажте модель і перемкніть відповідний блок у `.env`.

Вкладка **Debug / Eval** у Gradio UI: запуск усіх 20 кейсів із живим оновленням таблиці результатів, фільтрація за ID (`n001,o002`). У тій же вкладці є чекбокс **Show pipeline info in chat** — вмикає відображення графу виконання агентів (які вузли були активні, intent, latency, TTFT) безпосередньо в бульбашці чату.

Або через CLI:

```bash
# Ollama (опціонально, замість OpenRouter для eval)
ollama pull mistral
```

```bash
python eval/run_eval.py               # повний набір
python eval/run_eval.py --ids n001    # один кейс
```

---

## Приклади запитів

| Тип | Запит |
|---|---|
| Наявність деталі | Do you have the drain pump DD31-00016A in stock? |
| Підбір деталі | I need a water inlet valve for a Samsung washing machine — what do you have? |
| Діагностика | My washing machine is leaking water from the bottom — what could be the cause? |
| Поза темою | Can you recommend a good pasta recipe? |
| Ціна | What is the price for DC31-00054D? |
| За фото | Do you have this drain pump part? *(завантажте `examples/img01.jpg` або `examples/img02.jpg` з папки `examples/`)* |

> Зображення у `examples/` взяті з відкритих джерел і не збігаються з фото у CLIP-індексі — результат пошуку демонструє роботу з незнайомими зображеннями.

Повний набір тестових запитів — [`eval/golden_set.json`](eval/golden_set.json).

---

## Локальне розгортання

### Вимоги

- Python 3.11+
- [Ключ OpenRouter API](https://openrouter.ai) (для LLM-викликів)
- Ollama (опціонально — для локального eval-класифікатора)

### 1. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 2. Налаштування середовища

Для зручності перевірки надається **готова проіндексована база** у Qdrant Cloud та **готовий каталог деталей** `parts.csv` — збирати дані та запускати індексацію не потрібно.

```bash
cp .env.example .env        # macOS / Linux
copy .env.example .env      # Windows
```

Відредагуйте `.env` — замініть два плейсхолдери:

| Змінна | Дія |
|---|---|
| `OPENROUTER_API_KEY` | вставте ваш ключ OpenRouter |
| `EVAL_CLASSIFIER_API_KEY` | те саме значення |

Решта (Qdrant Cloud, моделі, ембедінги) вже заповнено.

### 3. Каталог деталей

Файл `data/parts.csv` вже включено до репозиторію — додаткових дій не потрібно.

> Для самостійного збору даних та побудови власного RAG-індексу з нуля — див. [docs/data_preparation.md](docs/data_preparation.md).

### 4. Запуск

```bash
# API-сервер
uvicorn src.api.main:app --reload

# UI (окремий термінал)
python src/ui/app.py
```

Відкрийте Gradio UI у браузері за URL-адресою, що виводиться в термінал.

---

## Структура проєкту

```
src/
├── api/            # FastAPI-застосунок — маршрути, схеми, підключення графу
├── core/           # LLM-шлюз (fallback, метрики, SQLite-персистентність)
├── crew/
│   ├── agents/     # orchestrator, vision, retrieval, clip_retrieval, parts, stock, synthesizer
│   ├── workflow.py # визначення LangGraph-графу та логіка маршрутизації
│   ├── state.py    # GraphState TypedDict
│   └── agent_loop.py  # спільний runner для tool-loop агентів
├── rag/            # ембедер (текст + CLIP)
├── tools/          # retriever, parts_catalog
├── vectordb/       # обгортка клієнта Qdrant
└── ui/             # Gradio чат-інтерфейс

scripts/
├── scrapers/       # scrape_ifixit.py, scrape_partselect.py
└── indexing/       # build_* скрипти для Qdrant та каталожних CSV-файлів

eval/
├── checkers.py     # PII, faithfulness, hallucination, refusal, injection
├── run_eval.py     # CLI-ранер
└── golden_set.json # 20 розмічених тестових кейсів

examples/           # зразки фото для тестування CLIP-пошуку за зображенням

data/
└── parts.csv       # каталог запасних частин (git-tracked)

storage/            # runtime-дані (gitignored)
├── raw/            # scraped JSONL-файли
├── index/          # CLIP-індекс зображень (part_images/{sku}/)
├── sessions.db     # стан сесій LangGraph (SQLite)
└── llm_calls.db    # метрики LLM-викликів (SQLite)

docs/
└── data_preparation.md  # повний цикл збору даних та побудови індексу з нуля
```

---

## Залежності

Дивіться [requirements.txt](requirements.txt) — усі версії зафіксовані.

Основні runtime-залежності: `langgraph`, `langchain-openai`, `qdrant-client`, `sentence-transformers`, `fastapi`, `gradio`, `openai`.
