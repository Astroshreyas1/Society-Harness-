"""Outcome metrics computed from spans only (so the same code scores real traces).

Primary failure metric is per *request* (A2): requests that ended in an agent abort ÷ requests
started, over sessions that arrived after the warm-up. Session-level counts are secondary.
Fairness (A3): Jain over per-session goodput (completed requests ÷ active wall-clock), including
failed and censored sessions, plus the survivor-only variant for comparison.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .schema import Span


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def _jain(xs: list[float]) -> float:
    if not xs or sum(xs) <= 0:
        return float("nan")
    return (sum(xs) ** 2) / (len(xs) * sum(x * x for x in xs))


def summarize(spans: list[Span], horizon_s: float, warmup_s: float, capacities: dict[str, float]) -> dict:
    """Window = everything that starts at or after warmup_s. Requests are `invoke_workflow` spans."""
    window_h = (horizon_s - warmup_s) / 3600.0
    reqs = [s for s in spans if s.op == "invoke_workflow" and s.t_start >= warmup_s and not s.attrs.get("child")]
    flat = [s for s in spans if s.op not in ("invoke_agent", "invoke_workflow") and s.t_start >= warmup_s]
    roots = {s.trace_id: s for s in spans if s.op == "invoke_agent" and s.parent_span_id is None}
    children = [s for s in spans if s.op == "invoke_agent" and s.parent_span_id is not None and s.t_start >= warmup_s]

    started = len(reqs)
    done = [r for r in reqs if r.outcome == "ok"]
    failed = [r for r in reqs if r.outcome == "aborted"]
    tct = [r.t_end - r.t_start for r in done]

    chats = [s for s in flat if s.op == "chat"]
    tools = [s for s in flat if s.op == "execute_tool"]
    waits = [s for s in flat if s.op == "wait"]
    chat_waits = [s.wait for s in chats if s.outcome == "ok"]
    tool_waits = [s.wait for s in tools if s.outcome == "ok"]
    block_waits = [s.wait for s in waits]

    failed_reqs = {(r.trace_id, r.attrs["request_idx"]) for r in failed}
    req_of: dict[str, list[Span]] = defaultdict(list)
    for r in reqs:
        req_of[r.trace_id].append(r)

    def request_of(s: Span):
        for r in req_of.get(s.trace_id, ()):
            if r.t_start <= s.t_start <= r.t_end:
                return (s.trace_id, r.attrs["request_idx"])
        return None

    wasted_fresh = sum(s.tokens_fresh for s in chats if request_of(s) in failed_reqs or s.outcome in ("evicted", "timeout"))
    wasted_tool_s = sum(s.duration for s in tools if request_of(s) in failed_reqs or s.outcome == "timeout")
    total_fresh = sum(s.tokens_fresh for s in chats)
    total_tool_s = sum(s.duration for s in tools)

    # fairness: per session, service ratio = time actually served / request wall-clock, over all
    # in-window requests; a failed request contributes its wall-clock and no service (A3).
    service_by_req: dict[tuple[str, int], float] = defaultdict(float)
    for s_ in flat:
        if s_.op in ("chat", "execute_tool", "retrieval") and s_.outcome == "ok":
            k = request_of(s_)
            if k is not None:
                service_by_req[k] += s_.duration
    ratio_all, ratio_ok = [], []
    for tid, rs in req_of.items():
        elapsed = sum(r.t_end - r.t_start for r in rs)
        if elapsed <= 0:
            continue
        served = sum(service_by_req.get((tid, r.attrs["request_idx"]), 0.0) for r in rs if r.outcome == "ok")
        ratio_all.append(served / elapsed)
        if all(r.outcome != "aborted" for r in rs):
            ratio_ok.append(served / elapsed)

    usd_ok = sum(float(s.attrs.get("usd", 0.0)) for s in flat if s.op in ("chat", "execute_tool") and s.outcome == "ok")
    usd_failed = sum(float(s.attrs.get("usd", 0.0)) for s in flat if s.op in ("chat", "execute_tool") and s.outcome == "ok" and request_of(s) in failed_reqs)
    busy_model = sum(s.duration for s in chats if s.outcome == "ok")
    busy_cpu = sum(s.duration for s in tools if s.mem > 0 and s.outcome == "ok")
    n_ok = sum(1 for r in roots.values() if r.outcome == "ok" and r.t_start >= warmup_s)
    n_abort = sum(1 for r in roots.values() if r.outcome == "aborted" and r.t_start >= warmup_s)

    return {
        "window": {"warmup_s": warmup_s, "horizon_s": horizon_s, "requests_in_window": started},
        "requests": {"started": started, "completed": len(done), "failed": len(failed), "censored": started - len(done) - len(failed)},
        "failure_rate": len(failed) / (len(done) + len(failed)) if (done or failed) else float("nan"),
        "sessions": {"born_in_window_ok": n_ok, "born_in_window_aborted": n_abort,
                     "failure_rate_finished": n_abort / (n_ok + n_abort) if (n_ok + n_abort) else float("nan")},
        "throughput_requests_per_hour": len(done) / window_h if window_h > 0 else float("nan"),
        "request_tct_s": {"p50": _pct(tct, 50), "p90": _pct(tct, 90), "p99": _pct(tct, 99),
                          "mean": float(np.mean(tct)) if tct else float("nan"), "n": len(tct)},
        "chat_wait_s": {"p50": _pct(chat_waits, 50), "p99": _pct(chat_waits, 99), "max": max(chat_waits, default=0.0)},
        "tool_wait_s": {"p50": _pct(tool_waits, 50), "p99": _pct(tool_waits, 99), "max": max(tool_waits, default=0.0)},
        "blocking_wait_s": {"n": len(block_waits), "p99": _pct(block_waits, 99), "max": max(block_waits, default=0.0)},
        "spend": {"usd": round(usd_ok, 4), "usd_per_completed_request": round(usd_ok / len(done), 5) if done else float("nan"),
                  "usd_on_failed_requests": round(usd_failed, 4), "usd_waste_frac": (usd_failed / usd_ok) if usd_ok > 0 else 0.0},
        "errors": {
            "429": sum(1 for s in flat if s.outcome == "429"),
            "budget": sum(1 for s in flat if s.outcome == "budget"),
            "timeout": sum(1 for s in flat if s.outcome == "timeout" and s.op != "wait"),
            "evicted": sum(1 for s in flat if s.outcome == "evicted"),
            "deadlock_cycles_seen": sum(1 for s in waits if s.attrs.get("deadlock_cycle")),
            "deadlock_breaks": sum(1 for s in waits if s.outcome == "deadlock"),
        },
        "waste": {
            "fresh_tokens_wasted": wasted_fresh, "fresh_tokens_total": total_fresh,
            "fresh_token_waste_frac": wasted_fresh / total_fresh if total_fresh else float("nan"),
            "tool_seconds_wasted": wasted_tool_s, "tool_seconds_total": total_tool_s,
            "tool_waste_frac": wasted_tool_s / total_tool_s if total_tool_s else float("nan"),
        },
        "fairness": {"jain_service_ratio_all": _jain(ratio_all), "jain_survivors_only": _jain(ratio_ok)},
        "utilisation": {
            "model.slots": busy_model / (capacities["model.slots"] * (horizon_s - warmup_s)),
            "sandbox.cpu": busy_cpu / (capacities["sandbox.cpu"] * (horizon_s - warmup_s)),
        },
        "steps": {"chat": len(chats), "tool": len(tools), "retrieval": sum(1 for s in flat if s.op == "retrieval"),
                  "parallel_groups": sum(1 for s in tools if s.name == "parallel"),
                  "parallel_groups_run_parallel": sum(1 for s in tools if s.name == "parallel" and s.attrs.get("parallel")),
                  "spawns": sum(1 for s in chats if s.attrs.get("spawn")),
                  "children": len(children), "children_aborted": sum(1 for s in children if s.outcome == "aborted"),
                  "join_wait_s_p50": _pct([s.attrs["join_wait_s"] for s in chats if "join_wait_s" in s.attrs], 50),
                  "join_wait_s_p99": _pct([s.attrs["join_wait_s"] for s in chats if "join_wait_s" in s.attrs], 99)},
    }


def fidelity(spans: list[Span]) -> dict:
    """Marginal self-check against the published targets the seeds were shaped to (v2 §3.8-1).
    Per-request counts are over completed requests of completed sessions (F12: the shorter ones)."""
    tools = [m for s in spans if s.op == "execute_tool" and s.outcome == "ok" for m in (s.attrs.get("members") or [s.name])]
    durs = np.array([s.duration for s in spans if s.op == "execute_tool" and s.outcome == "ok" and s.name != "parallel"]) \
        if spans else np.array([0.0])
    d = durs if len(durs) else np.array([0.0])
    total = d.sum() if d.sum() > 0 else 1.0
    chats = [s for s in spans if s.op == "chat" and s.outcome == "ok"]
    out = np.array([s.tokens_out for s in chats]) if chats else np.array([0])
    ok_traces = {s.trace_id for s in spans if s.op == "invoke_agent" and s.outcome == "ok" and s.parent_span_id is None}
    req = sum(int(s.attrs.get("requests_done", 0)) for s in spans if s.op == "invoke_agent" and s.outcome == "ok" and s.parent_span_id is None)
    ok_chats = sum(1 for s in chats if s.trace_id in ok_traces)
    ok_tools = sum(len(s.attrs.get("members") or [s.name]) for s in spans
                   if s.op == "execute_tool" and s.outcome == "ok" and s.trace_id in ok_traces)
    return {
        "tool_calls": len(tools),
        "tool_share_calls_lt_1s": float((d < 1).mean()), "tool_share_time_lt_1s": float(d[d < 1].sum() / total),
        "tool_share_calls_gt_60s": float((d > 60).mean()), "tool_share_time_gt_60s": float(d[d > 60].sum() / total),
        "tool_mean_s": float(d.mean()), "tool_p50_s": float(np.percentile(d, 50)), "tool_p99_s": float(np.percentile(d, 99)),
        "targets": {"calls_lt_1s": 0.70, "time_lt_1s": "<0.01", "calls_gt_60s": 0.049, "time_gt_60s": 0.92, "tool_mean_s": 16.8,
                    "out_median": 252, "out_p99": 6571, "chats_per_request": 8.8, "tools_per_request": 10.8},
        "out_median": float(np.percentile(out, 50)), "out_p99": float(np.percentile(out, 99)),
        "chats_per_request": ok_chats / req if req else float("nan"),
        "tools_per_request": ok_tools / req if req else float("nan"),
    }
