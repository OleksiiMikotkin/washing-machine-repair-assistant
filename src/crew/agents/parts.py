import json
import os

from crew.agent_loop import run_tool_loop
from crew.state import GraphState
from tools.parts_catalog import CatalogSearch

_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")

_SYSTEM = """\
You are a parts lookup specialist for washing machines.
Your ONLY job is to call tools and return results. Never write explanations, apologies, or summaries.

Strategy:
1. If a specific SKU is mentioned or previously found, call get_part_by_sku.
2. Otherwise call search_parts with relevant keywords (name, brand, category, model).
3. If a machine model is mentioned, also call find_compatible_parts.
4. Call list_categories only if you are unsure which category to use.

After the tool calls are done, output nothing — stop immediately.\
"""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_parts",
            "description": "Search the parts catalog by name, category, SKU, or machine model compatibility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":     {"type": "string", "description": "Part name keyword, e.g. 'drain pump'"},
                    "brand":    {"type": "string", "description": "Appliance brand, e.g. 'Samsung', 'LG'"},
                    "category": {"type": "string", "description": "Part category, e.g. 'Pump', 'Seal', 'Belt'"},
                    "sku":      {"type": "string", "description": "Exact SKU if the user mentioned one"},
                    "model":    {"type": "string", "description": "Machine model for compatibility filter, e.g. 'WW60J'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_part_by_sku",
            "description": "Retrieve a specific part by its exact SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "The part SKU, e.g. 'DC31-00054A'"},
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_categories",
            "description": "List all available part categories in the catalog.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_compatible_parts",
            "description": "Find all parts compatible with a specific washing machine model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Machine model, e.g. 'WW60J'"},
                },
                "required": ["model"],
            },
        },
    },
]

_catalog = CatalogSearch()

_TOOL_REGISTRY = {
    "search_parts":          lambda **kw: _catalog.search(filters=kw or None),
    "get_part_by_sku":       lambda sku: _catalog.get_by_sku(sku),
    "list_categories":       lambda: _catalog.list_categories(),
    "find_compatible_parts": lambda model: _catalog.find_compatible_parts(model),
}


def parts(state: GraphState) -> dict:
    context = f"User query: {state['query']}"
    if state.get("filters"):
        context += f"\nKnown filters: {json.dumps(state['filters'])}"
    if state.get("image_description"):
        context += f"\nImage analysis: {state['image_description']}"
    prev_skus = [r["sku"] for r in state.get("parts_results", []) if r.get("sku")]
    if prev_skus:
        context += f"\nPreviously found SKUs (re-fetch if the query is a follow-up): {', '.join(prev_skus)}"

    result = run_tool_loop(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": context},
        ],
        tools=_TOOLS,
        tool_registry=_TOOL_REGISTRY,
        model=_MODEL,
    )

    # Collect all part dicts from tool call results, de-duplicate by SKU
    seen_skus: set[str] = set()
    found_parts: list[dict] = []
    for call in result["tool_calls"]:
        res = call["result"]
        rows = res if isinstance(res, list) else ([res] if isinstance(res, dict) else [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            sku = row.get("sku", "")
            if sku and sku not in seen_skus:
                seen_skus.add(sku)
                found_parts.append(row)
            elif not sku:
                found_parts.append(row)

    return {"parts_results": found_parts}
