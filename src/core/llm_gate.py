import concurrent.futures
import logging
import os
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable

from openai import OpenAI
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

current_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
current_agent_name: ContextVar[str | None] = ContextVar("agent_name", default=None)

# Hard per-call deadline: httpx read-timeout only resets per-chunk, so a model
# that trickles thinking tokens can hold the call for minutes.  This thread-based
# deadline fires regardless of data flow.
_CALL_DEADLINE = float(os.getenv("LLM_CALL_TIMEOUT", "30"))
_llm_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=20, thread_name_prefix="llm-call"
)


@dataclass
class CallRecord:
    call_id: str
    session_id: str | None
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    fallback_used: bool
    error: str | None
    agent_name: str | None = None
    input_messages: list[dict] | None = None
    output_text: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LLMStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    calls_per_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMGateway:
    def __init__(self, store=None) -> None:
        self._client: OpenAI | None = None
        self._hooks: list[Callable[[CallRecord], None]] = []
        self._stats_lock = threading.Lock()
        self.stats = LLMStats()
        self._store = store  # MetricsStore or None

    def register_hook(self, fn: Callable[[CallRecord], None]) -> None:
        self._hooks.append(fn)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set.")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=60.0,
            )
        return self._client

    def _finalize(self, rec: CallRecord) -> None:
        with self._stats_lock:
            self.stats.calls += 1
            self.stats.calls_per_model[rec.model] += 1
            if rec.input_tokens:
                self.stats.input_tokens += rec.input_tokens
            if rec.output_tokens:
                self.stats.output_tokens += rec.output_tokens
        if self._store:
            self._store.save(rec)
        for hook in self._hooks:
            try:
                hook(rec)
            except Exception as exc:
                logger.warning("llm hook %s raised: %s", hook, exc)

    def _fallback_list(self, model: str, fallbacks: list[str] | None) -> list[str]:
        env_fallback = os.getenv("LLM_FALLBACK_MODEL")
        extra = fallbacks or ([env_fallback] if env_fallback else [])
        return [model] + extra

    def complete(
        self,
        model: str,
        messages: list[dict],
        fallbacks: list[str] | None = None,
        **kwargs,
    ) -> ChatCompletion:
        session_id = current_session_id.get()
        agent_name = current_agent_name.get()
        models_to_try = self._fallback_list(model, fallbacks)
        last_exc: Exception | None = None
        used_model = model
        fallback_used = False

        t_start = perf_counter()
        response: ChatCompletion | None = None

        for i, m in enumerate(models_to_try):
            logger.info("llm call → %s  agent=%s", m, agent_name)
            try:
                client = self._get_client()
                future = _llm_executor.submit(
                    client.chat.completions.create,
                    model=m, messages=messages, **kwargs,
                )
                try:
                    response = future.result(timeout=_CALL_DEADLINE)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(
                        f"LLM call to {m} exceeded {_CALL_DEADLINE:.0f}s deadline"
                    )
                used_model = m
                fallback_used = i > 0
                break
            except Exception as exc:
                logger.warning("model %s failed: %s", m, exc)
                last_exc = exc

        latency_ms = (perf_counter() - t_start) * 1000
        logger.info("llm done  ← %s  %.1fs  agent=%s", used_model, latency_ms / 1000, agent_name)

        if response is None:
            rec = CallRecord(
                call_id=str(uuid.uuid4()),
                session_id=session_id,
                model=used_model,
                input_tokens=None,
                output_tokens=None,
                latency_ms=latency_ms,
                ttft_ms=None,
                tpot_ms=None,
                fallback_used=fallback_used,
                error=str(last_exc),
                agent_name=agent_name,
                input_messages=messages,
            )
            self._finalize(rec)
            raise last_exc  # type: ignore[misc]

        usage = response.usage
        rec = CallRecord(
            call_id=str(uuid.uuid4()),
            session_id=session_id,
            model=used_model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_ms=latency_ms,
            ttft_ms=None,
            tpot_ms=None,
            fallback_used=fallback_used,
            error=None,
            agent_name=agent_name,
            input_messages=messages,
            output_text=response.choices[0].message.content,
        )
        self._finalize(rec)
        return response

    def stream(
        self,
        model: str,
        messages: list[dict],
        fallbacks: list[str] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        session_id = current_session_id.get()
        agent_name = current_agent_name.get()
        models_to_try = self._fallback_list(model, fallbacks)
        last_exc: Exception | None = None
        used_model = model
        fallback_used = False

        t_start = perf_counter()
        stream_obj = None

        for i, m in enumerate(models_to_try):
            try:
                stream_obj = self._get_client().chat.completions.create(
                    model=m,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    **kwargs,
                )
                used_model = m
                fallback_used = i > 0
                break
            except Exception as exc:
                logger.warning("model %s failed to open stream: %s", m, exc)
                last_exc = exc

        if stream_obj is None:
            raise last_exc  # type: ignore[misc]

        t_first: float | None = None
        t_last: float | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        chunks: list[str] = []

        try:
            for chunk in stream_obj:
                if not chunk.choices:
                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    now = perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunks.append(delta.content)
                    yield delta.content
        except Exception as exc:
            latency_ms = (perf_counter() - t_start) * 1000
            rec = CallRecord(
                call_id=str(uuid.uuid4()),
                session_id=session_id,
                model=used_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                ttft_ms=None,
                tpot_ms=None,
                fallback_used=fallback_used,
                error=str(exc),
                agent_name=agent_name,
                input_messages=messages,
                output_text="".join(chunks) or None,
            )
            self._finalize(rec)
            raise

        latency_ms = (perf_counter() - t_start) * 1000
        ttft_ms = (t_first - t_start) * 1000 if t_first else None
        tpot_ms = (
            (t_last - t_first) * 1000 / output_tokens
            if t_first and t_last and output_tokens
            else None
        )
        rec = CallRecord(
            call_id=str(uuid.uuid4()),
            session_id=session_id,
            model=used_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            fallback_used=fallback_used,
            error=None,
            agent_name=agent_name,
            input_messages=messages,
            output_text="".join(chunks) or None,
        )
        self._finalize(rec)


def _make_gateway() -> LLMGateway:
    from crew.metrics_store import MetricsStore
    return LLMGateway(store=MetricsStore())


_gateway = _make_gateway()


def chat_completion(model: str, messages: list[dict], **kwargs) -> ChatCompletion:
    return _gateway.complete(model, messages, **kwargs)


def chat_completion_stream(model: str, messages: list[dict], **kwargs) -> Iterator[str]:
    yield from _gateway.stream(model, messages, **kwargs)


def get_stats() -> LLMStats:
    return _gateway.stats
