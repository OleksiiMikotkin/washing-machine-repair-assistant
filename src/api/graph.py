"""
Singleton compiled graph and pipeline shared across all API requests.
Both share one checkpointer so session state is consistent.
"""

from crew.workflow import compile_graph, compile_pipeline, make_checkpointer

_checkpointer = None
_graph = None
_pipeline = None


def _get_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = make_checkpointer()
    return _checkpointer


def get_graph():
    global _graph
    if _graph is None:
        _graph = compile_graph(_get_checkpointer())
    return _graph


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = compile_pipeline(_get_checkpointer())
    return _pipeline
