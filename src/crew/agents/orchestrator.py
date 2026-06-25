import json
import os

from core.llm_gate import chat_completion
from crew.utils import messages_to_openai
from crew.state import GraphState

_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")

_SYSTEM = """\
You are an intent classifier for a washing machine customer support system.

Classify the LATEST user message into EXACTLY ONE intent and extract any filters present.
Do NOT infer filters from earlier messages in the conversation — use only what is explicitly stated in the latest message.

Intent rules:
- "troubleshooting" — user describes a symptom or problem with the machine
- "part_lookup"     — user wants to find, buy, or get info about a spare part
- "error_code"      — user mentions an error code shown on the machine display
- "general"         — anything else: usage questions, general how-to, comparisons

CRITICAL RULE: If the user asks to find, order, buy, or check availability of a part — even if they also
mention a symptom — use "part_lookup". Reserve "troubleshooting" only when the user wants diagnosis or repair advice.

FOLLOW-UP RULE: If the conversation already discussed a specific part and the latest message asks about
price, cost, availability, or stock — use "part_lookup" with no filters (the agent will reuse the context).

Filter keys (omit if not explicitly stated in the latest message):
- "brand"      — appliance brand, e.g. "Samsung", "LG"
- "model"      — model number, e.g. "WW60J", "WF45R"
- "symptom"    — described problem, e.g. "not spinning", "leaking"
- "error_code" — code from display, e.g. "E2", "5E"
- "sku"        — exact part number or SKU, e.g. "DC31-00187A", "WP89503"

Examples:
  "My washer won't spin"                            → "troubleshooting", symptom: "won't spin"
  "Machine is leaking from the bottom"              → "troubleshooting", symptom: "leaking"
  "Find me a drain pump for Samsung"                → "part_lookup", brand: "Samsung"
  "I need a new drive belt for WW60J"               → "part_lookup", model: "WW60J"
  "Where can I get part DC32-00007A?"               → "part_lookup", sku: "DC32-00007A"
  "Do you have part DC31-00187A?"                   → "part_lookup", sku: "DC31-00187A"
  "How many units of DC97-19289H are available?"    → "part_lookup", sku: "DC97-19289H"
  "What is the price for DC31-00054D?"              → "part_lookup", sku: "DC31-00054D"
  "What is the price?" (after discussing a part)    → "part_lookup", no extra filters
  "Is it in stock?" (after discussing a part)       → "part_lookup", no extra filters
  "Error code 5E on my Samsung"                     → "error_code", brand: "Samsung", error_code: "5E"
  "How often should I clean the filter?"            → "general"

Return a SINGLE JSON object — NOT an array. No markdown, no explanation:
{"intent": "part_lookup", "filters": {"brand": "Samsung"}}\
"""

VALID_INTENTS = {"troubleshooting", "part_lookup", "error_code", "general"}
VALID_FILTER_KEYS = {"brand", "model", "symptom", "error_code", "sku"}

_CONFIRM_WORDS = {"yes", "yeah", "yep", "yup", "sure", "correct", "right", "ok", "okay", "exactly", "that's right", "that's it"}
_DENY_WORDS    = {"no", "nope", "nah", "wrong", "incorrect", "not right", "not that", "different"}


def orchestrator(state: GraphState) -> dict:
    if state.get("confirmation_pending"):
        reply = state.get("query", "").strip().lower()
        if any(reply == w or reply.startswith(w + " ") for w in _CONFIRM_WORDS):
            return {"intent": "confirm", "confirmation_pending": False}
        if any(reply == w or reply.startswith(w + " ") for w in _DENY_WORDS):
            return {
                "intent": "clarify",
                "confirmation_pending": False,
                "parts_results": [],
                "clip_score": None,
            }
        # Ambiguous — treat as new query, fall through to LLM classification


    history = messages_to_openai(state.get("messages", []))
    response = chat_completion(
        model=_MODEL,
        messages=[{"role": "system", "content": _SYSTEM}, *history],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}

    intent = parsed.get("intent", "general")
    if intent not in VALID_INTENTS:
        intent = "general"

    filters = {
        k: v for k, v in parsed.get("filters", {}).items()
        if k in VALID_FILTER_KEYS and isinstance(v, str) and v.strip()
    }

    # Clear cached parts when the user is asking about a new part (has explicit search terms).
    # Empty filters = bare follow-up ("what's the price?") → keep cached parts_results.
    clear_parts = intent == "part_lookup" and bool(filters)

    return {
        "intent": intent,
        "filters": filters,
        "image_description": None,
        **({"parts_results": []} if clear_parts else {}),
    }
