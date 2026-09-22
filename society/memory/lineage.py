import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["approved", "rejected", "abstained", "escalated", "error"]


class LineageRecord(BaseModel):
    ts: float = Field(default_factory=time.time)
    run_id: str
    node_id: str
    agent: str
    attempt: int
    provider: str = ""
    model: str = ""
    fallback_from: str | None = None
    inputs_read: list[str] = Field(default_factory=list)
    outputs_written: list[str] = Field(default_factory=list)
    files_written: list[str] = Field(default_factory=list)
    gate: str = ""
    verdict: Verdict | None = None
    confidence: float | None = None
    reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    text_tail: str = ""


class LineageLog:
    """Append-only audit trail. Agents never read it; humans and dashboards do."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def record(self, rec: LineageRecord) -> None:
        with self.path.open("a") as f:
            f.write(rec.model_dump_json() + "\n")

    def read(self, run_id: str | None = None) -> list[LineageRecord]:
        out: list[LineageRecord] = []
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    rec = LineageRecord.model_validate(json.loads(line))
                    if run_id is None or rec.run_id == run_id:
                        out.append(rec)
        return out

    def total_cost(self, run_id: str) -> float:
        return sum(r.cost_usd for r in self.read(run_id))
