import json
import os
from collections.abc import Iterator

from langchain_core.messages import AIMessage

from core.llm_gate import chat_completion, chat_completion_stream
from crew.utils import messages_to_openai
from crew.state import GraphState

_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")

_LOW_CONF = 0.75   # below this threshold: ask user to confirm the image match

_SYSTEM = """\
You are a customer support assistant for washing machine repair and spare parts.

You will receive a user question and structured context gathered by specialist agents.

CRITICAL RULES:
- Answer ONLY from the provided context. NEVER add information from your own knowledge.
- If "Parts found" is in the context, always state the part name, price, and stock status.
- If price is shown in the context, always include it in the answer.
- Do NOT add troubleshooting advice unless "Knowledge base results" is present in the context.
- If the context does not contain the answer, say "I don't have that information" — nothing more.
- If the issue requires physical inspection or hands-on diagnosis, tell the user to contact our support team directly.

IMAGE CONFIDENCE RULES:
- If context contains "Image confidence: LOW", the part was identified from an image but with low certainty.
  Phrase your answer as a question: "I think this might be [part name] — does that look right? It is [stock status] at $[price]."
  Set confirmation_pending to true.
- If context contains "Image confidence: HIGH", state the part directly without asking.
  Set confirmation_pending to false.
- If context contains "Intent: clarify", apologize briefly and ask the user to provide a part number or text description.
  Set confirmation_pending to false.

Return a JSON object with:
- "answer": your response to the user (plain text, no markdown)
- "confirmation_pending": true or false (default false)\
"""


def _build_context(state: GraphState) -> str:
    parts = [f"User question: {state['query']}"]

    intent = state.get("intent", "")
    if intent in ("confirm", "clarify"):
        parts.append(f"Intent: {intent}")

    clip_score = state.get("clip_score")
    if clip_score is not None:
        label = "LOW" if clip_score < _LOW_CONF else "HIGH"
        parts.append(f"Image confidence: {label} (score {clip_score:.2f})")

    if state.get("image_description"):
        parts.append(f"Image analysis: {state['image_description']}")

    if state.get("retrieval_results"):
        lines = []
        for r in state["retrieval_results"]:
            p = r["payload"]
            meta = ", ".join(filter(None, [p.get("symptom"), p.get("cause")]))
            prefix = f"[{meta}] " if meta else ""
            lines.append(f"- {prefix}{p.get('text', '')}")
        parts.append(f"Knowledge base results:\n" + "\n".join(lines))

    if state.get("parts_results"):
        stock = state.get("stock_results", {})
        lines = []
        for row in state["parts_results"]:
            sku = row.get("sku", "")
            availability = stock.get(sku, {})
            in_stock = availability.get("in_stock", False)
            qty = availability.get("qty", 0)
            price = availability.get("price")
            status = f"in stock ({qty} pcs)" if in_stock else "out of stock"
            price_str = f", ${price:.2f}" if price is not None else ""
            lines.append(f"- {row.get('name', sku)} [{sku}]: {status}{price_str}")
        parts.append(f"Parts found:\n" + "\n".join(lines))

    return "\n\n".join(parts)


_SYSTEM_STREAM = """\
You are a customer support assistant for washing machine repair and spare parts.

You will receive a user question and structured context gathered by specialist agents.

CRITICAL RULES:
- Answer ONLY from the provided context. NEVER add information from your own knowledge.
- If "Parts found" is in the context, always state the part name, price, and stock status.
- If price is shown in the context, always include it in the answer.
- Do NOT add troubleshooting advice unless "Knowledge base results" is present in the context.
- If the context does not contain the answer, say "I don't have that information" — nothing more.
- If the issue requires physical inspection or hands-on diagnosis, tell the user to contact our support team directly.

IMAGE CONFIDENCE RULES:
- If context contains "Image confidence: LOW", phrase your answer as a question:
  "I think this might be [part name] — does that look right? It is [stock status] at $[price]."
- If context contains "Image confidence: HIGH", state the part directly without asking.
- If context contains "Intent: clarify", apologize briefly and ask for part number or text description.

Respond with plain text only. No JSON, no markdown.\
"""


def get_confirmation_pending(state: GraphState) -> bool:
    clip_score = state.get("clip_score")
    return clip_score is not None and clip_score < _LOW_CONF


def synthesizer_stream(state: GraphState) -> Iterator[str]:
    history = messages_to_openai(state.get("messages", []))
    context = _build_context(state)
    yield from chat_completion_stream(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_STREAM},
            *history,
            {"role": "user", "content": context},
        ],
        temperature=0.2,
    )


def synthesizer(state: GraphState) -> dict:
    history = messages_to_openai(state.get("messages", []))
    context = _build_context(state)

    response = chat_completion(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            *history,
            {"role": "user", "content": context},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    answer = parsed.get("answer", "")
    confirmation_pending = bool(parsed.get("confirmation_pending", False))

    return {
        "final_answer": answer,
        "confirmation_pending": confirmation_pending,
        "messages": [AIMessage(content=answer)],
    }
