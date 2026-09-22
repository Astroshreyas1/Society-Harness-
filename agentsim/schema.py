"""OTel-GenAI-shaped span records: the single trace format for real and synthetic data.

Field names follow the OpenTelemetry GenAI semantic conventions where one exists
(`op` = gen_ai.operation.name; `name` = gen_ai.tool.name / model name). Everything the
controller may observe is here; nothing about the future (sampled durations, outcomes)
is visible to a policy before it happens (causal observation boundary).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

OPS = ("invoke_agent", "invoke_workflow", "chat", "execute_tool", "retrieval", "wait", "think")
TIERS = ("model", "tool", "service")
OUTCOMES = ("ok", "429", "timeout", "evicted", "aborted", "deadlock", "session_end", "budget")   # budget: a tenant budget was exhausted


@dataclass
class Span:
    trace_id: str                 # session / workflow id
    span_id: str
    parent_span_id: str | None
    op: str                       # gen_ai.operation.name
    name: str                     # tool kind, model name, or phase label
    tier: str | None
    resource: str | None
    t_start: float
    t_end: float
    step_idx: int
    tokens_in: int = 0            # prefix + append presented to the model
    tokens_out: int = 0
    tokens_fresh: int = 0         # tokens actually prefilled (cache misses)
    duration: float = 0.0         # service time, excluding waits
    wait: float = 0.0             # admission / queueing wait
    mem: float = 0.0              # GB for sandboxes, tokens for KV
    outcome: str = "ok"
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"unknown op {self.op!r}")
        if self.tier is not None and self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}")
        if self.t_end < self.t_start:
            raise ValueError(f"span ends before it starts: {self}")


def write_jsonl(path: Path, spans: list[Span]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in spans:
            f.write(json.dumps(asdict(s), separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> Iterator[Span]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Span(**json.loads(line))
