"""Rung 0 Observer: turns OTel-shaped spans (real or synthetic) into the two things every
later component reads — per-workflow node sequences and per-resource occupancy series.
No decisions are made here.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentsim.schema import Span, read_jsonl  # noqa: E402


def load(paths: list[str | Path]) -> list[Span]:
    """Load spans from one or more JSONL files. Session ids (`trace_id`) are only unique within a
    run, so with several files each id is namespaced by its file index (`<i>:<trace_id>`) — without
    this, sessions from different seeds merge into interleaved sequences (ISSUES F15)."""
    spans: list[Span] = []
    for i, p in enumerate(paths):
        batch = list(read_jsonl(Path(p)))
        if len(paths) > 1:
            for s in batch:
                s.trace_id = f"{i}:{s.trace_id}"
        spans.extend(batch)
    if not spans:
        raise ValueError("no spans loaded")
    return spans


def node_token(s: Span, with_phase: bool, phase_attr: str | None = None) -> str | None:
    """The node-type vocabulary a predictor sees. Phases are hidden in real traces unless a
    semantic labeler (e.g. a System One model) supplies them; `with_phase` shows that upper bound,
    `phase_attr` uses a labeler's annotation (e.g. `jev_phase` from `agentsim annotate`) instead."""
    if s.op == "chat":
        if phase_attr is not None:
            return f"chat:{s.attrs[phase_attr]}" if phase_attr in s.attrs else "chat"
        return f"chat:{s.name}" if with_phase else "chat"
    if s.op == "execute_tool":
        return f"tool:{s.name}"
    if s.op == "retrieval":
        return "retrieval"
    if s.op == "think":
        return "user"
    return None                                        # invoke_agent roots and wait spans are not nodes


def sequences(spans: list[Span], with_phase: bool = False, ok_only: bool = True, phase_attr: str | None = None) -> dict[str, list[str]]:
    """trace_id -> ordered node tokens (a 'user' token marks each new request)."""
    by: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for s in spans:
        if ok_only and s.outcome != "ok" and s.op != "think":
            continue
        tok = node_token(s, with_phase, phase_attr)
        if tok is not None:
            by[s.trace_id].append((s.t_start, tok))
    return {tid: ["user"] + [t for _, t in sorted(v)] for tid, v in by.items()}


def occupancy(spans: list[Span], resource: str, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """In-flight count on `resource` over time, from span [t_start, t_end) intervals."""
    xs = [s for s in spans if s.resource == resource and s.op != "wait" and s.t_end > s.t_start]
    if not xs:
        raise ValueError(f"no spans on resource {resource!r}")
    t_max = max(s.t_end for s in xs)
    grid = np.arange(0.0, t_max + dt, dt)
    occ = np.zeros_like(grid)
    for s in xs:
        a, b = int(s.t_start // dt), int(np.ceil(s.t_end / dt))
        occ[a:b] += 1
    return grid, occ


def roots(spans: list[Span]) -> dict[str, Span]:
    return {s.trace_id: s for s in spans if s.op == "invoke_agent"}
