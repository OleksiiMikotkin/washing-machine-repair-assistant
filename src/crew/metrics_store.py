import sqlite3
import threading
from pathlib import Path

from core.llm_gate import CallRecord

_DEFAULT_DB = str(Path(__file__).parent.parent.parent / "storage" / "metrics.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id       TEXT PRIMARY KEY,
    session_id    TEXT,
    agent_name    TEXT,
    created_at    TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    latency_ms    REAL NOT NULL,
    ttft_ms       REAL,
    tpot_ms       REAL,
    cache_hit     INTEGER NOT NULL DEFAULT 0,
    fallback_used INTEGER NOT NULL DEFAULT 0,
    error         TEXT
)
"""


class MetricsStore:
    def __init__(self, db_path: str = _DEFAULT_DB) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(_CREATE_TABLE)
        self._migrate()
        self._db.commit()
        self._lock = threading.Lock()

    def _migrate(self) -> None:
        existing = {row[1] for row in self._db.execute("PRAGMA table_info(llm_calls)")}
        if "agent_name" not in existing:
            self._db.execute("ALTER TABLE llm_calls ADD COLUMN agent_name TEXT")

    def save(self, rec: CallRecord) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO llm_calls
                    (call_id, session_id, agent_name, created_at, model,
                     input_tokens, output_tokens,
                     latency_ms, ttft_ms, tpot_ms,
                     fallback_used, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.call_id,
                    rec.session_id,
                    rec.agent_name,
                    rec.created_at.isoformat(),
                    rec.model,
                    rec.input_tokens,
                    rec.output_tokens,
                    rec.latency_ms,
                    rec.ttft_ms,
                    rec.tpot_ms,
                    int(rec.fallback_used),
                    rec.error,
                ),
            )
            self._db.commit()
