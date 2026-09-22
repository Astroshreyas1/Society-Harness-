import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def attempt_key(run_id: str, agent: str, dtype: str, attempt: int) -> str:
    return f"run:{run_id}:{agent}:{dtype}:attempt_{attempt}"


def current_key(run_id: str, agent: str, dtype: str) -> str:
    return f"run:{run_id}:{agent}:{dtype}:current"


class KVStore:
    """Live handoff state between agents.

    Workers write versioned attempts; only a gate promotes one to `:current`.
    Downstream nodes read `:current`, never a raw attempt.
    """

    def __init__(self, path: Path | str = ":memory:"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.conn.commit()

    def put(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO kv (key, value, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        self.conn.commit()

    def get(self, key: str) -> Any | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def exists(self, key: str) -> bool:
        return self.conn.execute("SELECT 1 FROM kv WHERE key = ?", (key,)).fetchone() is not None

    def keys(self, prefix: str = "") -> list[str]:
        rows = self.conn.execute(
            "SELECT key FROM kv WHERE key LIKE ? ORDER BY created_at", (prefix + "%",)
        ).fetchall()
        return [r[0] for r in rows]

    def next_attempt(self, run_id: str, agent: str, dtype: str) -> int:
        prefix = f"run:{run_id}:{agent}:{dtype}:attempt_"
        nums = [int(k.rsplit("_", 1)[1]) for k in self.keys(prefix)]
        return max(nums, default=0) + 1

    def put_attempt(self, run_id: str, agent: str, dtype: str, attempt: int, value: Any) -> str:
        key = attempt_key(run_id, agent, dtype, attempt)
        if self.exists(key):
            raise ValueError(f"attempt already exists and is immutable: {key}")
        self.put(key, value)
        return key

    def get_attempt(self, run_id: str, agent: str, dtype: str, attempt: int) -> Any | None:
        return self.get(attempt_key(run_id, agent, dtype, attempt))

    def promote(self, run_id: str, agent: str, dtype: str, attempt: int) -> str:
        target = attempt_key(run_id, agent, dtype, attempt)
        if not self.exists(target):
            raise KeyError(f"cannot promote missing attempt: {target}")
        pointer = current_key(run_id, agent, dtype)
        self.put(pointer, {"points_to": target, "attempt": attempt})
        return pointer

    def get_current(self, run_id: str, agent: str, dtype: str) -> Any | None:
        pointer = self.get(current_key(run_id, agent, dtype))
        return self.get(pointer["points_to"]) if pointer else None

    def current_attempt(self, run_id: str, agent: str, dtype: str) -> int | None:
        pointer = self.get(current_key(run_id, agent, dtype))
        return pointer["attempt"] if pointer else None

    def close(self) -> None:
        self.conn.close()
