import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from api.schemas import QueryResponse, SessionResponse, MessageItem
from api.graph import get_graph, get_pipeline
from crew.agents.synthesizer import synthesizer_stream, get_confirmation_pending
from core.llm_gate import current_session_id

router = APIRouter()

_UPLOADS_DIR = Path(__file__).parent.parent.parent.parent / "storage" / "uploads"
_MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _save_upload(image: UploadFile) -> str:
    contents = image.file.read()
    if len(contents) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 5 MB)")
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    dest = _UPLOADS_DIR / f"{uuid.uuid4()}{suffix}"
    dest.write_bytes(contents)
    return str(dest)


def _serialize_messages(messages: list) -> list[MessageItem]:
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append(MessageItem(role="human", content=str(msg.content)))
        elif isinstance(msg, AIMessage):
            result.append(MessageItem(role="ai", content=str(msg.content)))
    return result


@router.post("/query", response_model=QueryResponse)
def query(
    session_id: str = Form(...),
    message: str = Form(...),
    image: UploadFile | None = File(None),
) -> QueryResponse:
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    if len(message) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 chars)")

    image_path = _save_upload(image) if image is not None else None

    input_state = {
        "query": message,
        "image_path": image_path,
        "messages": [HumanMessage(content=message)],
    }

    current_session_id.set(session_id)
    t_start = time.perf_counter()
    try:
        result = graph.invoke(input_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = round((time.perf_counter() - t_start) * 1000)

    return QueryResponse(
        session_id=session_id,
        final_answer=result["final_answer"],
        agents_used=result.get("agents_used", []),
        latency_ms=latency_ms,
        context={
            "parts": result.get("parts_results", []),
            "stock": result.get("stock_results", {}),
            "retrieval": [r.get("payload", {}) for r in result.get("retrieval_results", [])],
        },
    )


@router.post("/query/stream")
def query_stream(
    session_id: str = Form(...),
    message: str = Form(...),
    image: UploadFile | None = File(None),
) -> StreamingResponse:
    if len(message) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 chars)")

    pipeline = get_pipeline()
    config = {"configurable": {"thread_id": session_id}}

    image_path = _save_upload(image) if image is not None else None

    input_state = {
        "query": message,
        "image_path": image_path,
        "messages": [HumanMessage(content=message)],
    }

    current_session_id.set(session_id)

    def generate():
        t_start = time.perf_counter()

        # Run the heavy pipeline (orchestrator → parts → stock) in a background
        # thread so we can send SSE keepalives while it works.  Without this the
        # client socket goes silent for the full pipeline duration and the UI's
        # read timeout fires before the first content token arrives.
        _state: list = [None]
        _error: list = [None]
        _done = threading.Event()

        def _run_pipeline():
            try:
                pipeline.invoke(input_state, config=config)
                _state[0] = pipeline.get_state(config).values
            except Exception as exc:
                _error[0] = exc
            finally:
                _done.set()

        threading.Thread(target=_run_pipeline, daemon=True).start()

        # Yield SSE comment lines every 15 s — the UI filters them out (they
        # don't start with "data: ") but the socket read succeeds, preventing
        # the client's read-timeout from firing.
        while not _done.wait(timeout=15):
            yield ": keepalive\n\n"

        if _error[0] is not None:
            yield f"data: {json.dumps({'error': str(_error[0])})}\n\n"
            return

        state = _state[0]
        try:
            full_answer = ""
            t_first: float | None = None
            for chunk in synthesizer_stream(state):
                if t_first is None:
                    t_first = time.perf_counter()
                full_answer += chunk
                yield f"data: {json.dumps(chunk)}\n\n"

            t_end = time.perf_counter()

            pipeline.update_state(config, {
                "final_answer": full_answer,
                "confirmation_pending": get_confirmation_pending(state),
                "messages": [AIMessage(content=full_answer)],
            }, as_node="synthesizer")

            meta = {
                "type": "metadata",
                "agents_used": state.get("agents_used", []) + ["synthesizer"],
                "intent": state.get("intent"),
                "latency_ms": round((t_end - t_start) * 1000),
                "ttft_ms": round((t_first - t_start) * 1000) if t_first else None,
                "clip_score": state.get("clip_score"),
            }
            yield f"data: {json.dumps(meta)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    snapshot = graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Session not found")

    values = snapshot.values

    return SessionResponse(
        session_id=session_id,
        intent=values.get("intent"),
        final_answer=values.get("final_answer"),
        messages=_serialize_messages(values.get("messages", [])),
    )


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> Response:
    conn = get_graph().checkpointer.conn
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
    conn.commit()
    return Response(status_code=204)
