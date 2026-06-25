import os
import sqlite3
from pathlib import Path

_LLM_VISION_ENABLED = os.getenv("LLM_VISION_ENABLED", "true").lower() == "true"

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from crew.agents.clip_retrieval import clip_retrieval
from crew.agents.orchestrator import orchestrator
from crew.agents.parts import parts
from crew.agents.retrieval import retrieval
from crew.agents.stock import stock
from crew.agents.synthesizer import synthesizer
from crew.agents.vision import vision
from crew.state import GraphState


# --- routing helpers ---

def _after_orchestrator(state: GraphState) -> str:
    intent = state.get("intent", "general")
    if intent in ("confirm", "clarify"):
        return "synthesizer"
    if state.get("image_path"):
        return "vision" if _LLM_VISION_ENABLED else "clip_retrieval"
    return _by_intent(state)


def _by_intent(state: GraphState) -> str:
    intent = state.get("intent", "general")
    if intent in ("troubleshooting", "error_code"):
        return "retrieval"
    if intent == "part_lookup":
        clip_failed = (
            state.get("image_path")
            and state.get("clip_score") is None
            and not state.get("parts_results")
            and not state.get("filters")
        )
        if clip_failed:
            return "synthesizer"
        return "stock" if state.get("parts_results") else "parts"
    return "synthesizer"


def _track(name: str, fn):
    """Wraps an agent so its name is appended to agents_used in graph state."""
    def wrapper(state: GraphState) -> dict:
        result = fn(state) or {}
        result["agents_used"] = [name]
        return result
    return wrapper


# --- graph assembly ---

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    g.add_node("orchestrator",   _track("orchestrator",   orchestrator))
    g.add_node("vision",         _track("vision",         vision))
    g.add_node("clip_retrieval", _track("clip_retrieval", clip_retrieval))
    g.add_node("retrieval",      _track("retrieval",      retrieval))
    g.add_node("parts",          _track("parts",          parts))
    g.add_node("stock",          _track("stock",          stock))
    g.add_node("synthesizer",    _track("synthesizer",    synthesizer))

    g.add_edge(START, "orchestrator")

    g.add_conditional_edges(
        "orchestrator",
        _after_orchestrator,
        {
            "synthesizer":    "synthesizer",
            "vision":         "vision",
            "clip_retrieval": "clip_retrieval",
            "retrieval":      "retrieval",
            "parts":          "parts",
            "stock":          "stock",
        },
    )

    g.add_conditional_edges(
        "vision",
        _by_intent,
        {"retrieval": "retrieval", "parts": "parts", "stock": "stock", "synthesizer": "synthesizer"},
    )

    g.add_conditional_edges(
        "clip_retrieval",
        _by_intent,
        {"retrieval": "retrieval", "parts": "parts", "stock": "stock", "synthesizer": "synthesizer"},
    )

    g.add_edge("retrieval", "synthesizer")
    g.add_edge("parts",     "stock")
    g.add_edge("stock",     "synthesizer")
    g.add_edge("synthesizer", END)

    return g


def make_checkpointer() -> SqliteSaver:
    _root = Path(__file__).parent.parent.parent
    _env = os.getenv("SESSIONS_DB", "storage/sessions.db")
    db_file = Path(_env) if Path(_env).is_absolute() else _root / _env
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    return SqliteSaver(conn)


def compile_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = make_checkpointer()
    return build_graph().compile(checkpointer=checkpointer)


def compile_pipeline(checkpointer=None):
    """Same graph but pauses before synthesizer — used by the streaming endpoint."""
    if checkpointer is None:
        checkpointer = make_checkpointer()
    return build_graph().compile(checkpointer=checkpointer, interrupt_before=["synthesizer"])
