import operator
from typing import Annotated

from langgraph.graph import MessagesState


class GraphState(MessagesState):
    # --- input ---
    query: str
    image_path: str | None

    # --- orchestrator ---
    intent: str                    # troubleshooting | part_lookup | error_code | general
    filters: dict                  # brand, model, symptom → passed to Qdrant filter

    # --- vision ---
    image_description: str | None

    # --- parallel lookup agents ---
    retrieval_results: list[dict]  # [{payload: {...}, score: float}, ...]
    parts_results: list[dict]      # matching rows from parts.csv
    stock_results: dict            # {sku: {in_stock: bool, qty: int}}

    # --- image confidence ---
    clip_score: float | None          # cosine similarity from CLIP search
    confirmation_pending: bool        # True when synthesizer asked user to confirm image match

    # --- synthesizer ---
    final_answer: str

    # --- observability ---
    agents_used: Annotated[list[str], operator.add]
