"""Convert the TraceLab coding-agent trace (Zhu et al. 2026, CC BY 4.0; github.com/uw-syfi/TraceLab) into the
OTel-GenAI-shaped span JSONL the rest of this repo reads (agentsim.schema.Span) — rung 0 on real data.

  python rung0/tracelab_to_spans.py data/real/tracelab/syfi_coding_trace.jsonl.gz data/real/tracelab/spans.jsonl [--provider claude|codex|all]

Mapping (one session = one trace):
  invoke_agent      per session_id; name "coding" (both Claude Code and Codex are coding agents; provider in attrs)
  invoke_workflow   per request: starts at a round whose first input event is a user_message; ends at the last event before the next
  chat              per round: t_start = last input event (user message / tool result) of the round, t_end = last assistant event;
                    tokens_in = input_tokens_total, tokens_out = output_tokens (+ reasoning), tokens_fresh = newly_append_tokens
  execute_tool      per tool record: [emitted_at, result_at]; kinds mapped onto this repo's vocabulary (Bash→bash, Read/Glob→read,
                    Edit/Write/MultiEdit/NotebookEdit→edit, Grep→grep, WebSearch→search, WebFetch→web, everything else→other);
                    tools of one round whose intervals overlap are one "parallel" group span (attrs members, parallel=True)
  think             the gap from a request's last event to the next user message; attrs recipe / phase_end / request_idx,
                    where phase_end is the OBSERVABLE proxy "after:<last tool kind>" (real traces have no hidden phases)
Rounds without assistant events (aborted / empty) are skipped and counted. Everything the schema does not have is dropped.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentsim.schema import Span, write_jsonl  # noqa: E402

KIND = {"Bash": "bash", "Read": "read", "Glob": "read", "Edit": "edit", "Write": "edit", "MultiEdit": "edit", "NotebookEdit": "edit",
        "Grep": "grep", "WebSearch": "search", "WebFetch": "web"}
HUMAN_TOOLS = {"AskUserQuestion", "ExitPlanMode"}          # the "tool" waits for the human: think time, not tool time
INPUT_EVENTS = ("user_message", "tool_result")
ASSISTANT_EVENTS = ("reasoning", "text", "tool_call")


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def kind_of(name: str) -> str:
    return KIND.get(name, "other")


def _bucket(n) -> int:
    """log2 bucket of a character count (0 for missing)."""
    n = int(n or 0)
    return 0 if n <= 0 else min(16, n.bit_length())


def content_of(t: dict) -> str:
    """The observable content cue of a tool call: for Bash the sanitised command skeleton (binary names and pipes,
    as TraceLab publishes it), for every tool the input size bucket; tokens are space-separated for hashing."""
    name = t["tool_name"]
    toks = [f"tn:{name}", f"in:{_bucket(t.get('input_chars'))}"]
    sk = t.get("command_skeleton")
    if name == "Bash" and sk:
        toks += [w for w in sk.replace(";", " ; ").replace("|", " | ").split() if w]
    return " ".join(toks)


def convert(path: Path, provider: str) -> tuple[list[Span], Counter]:
    sessions: dict[str, list[dict]] = defaultdict(list)
    stats: Counter = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if provider != "all" and r["provider"] != provider:
                continue
            sessions[r["session_id"]].append(r)
            stats["rounds"] += 1
    spans: list[Span] = []
    for sid, rounds in sessions.items():
        def first_input(r: dict) -> float:                        # round_index is not time order in resumed/forked sessions
            xs = [ts(e["timestamp"]) for e in r["timing_events"] if e["event_type"] in INPUT_EVENTS]
            return max(xs) if xs else float("inf")
        rounds.sort(key=first_input)
        t0 = None
        out: list[Span] = []
        req_idx, req_start, req_last_end, req_steps, last_tool = -1, None, None, 0, "none"
        step_idx = 0
        prev_think, req_user_chars = None, 0

        last_chat: Span | None = None

        def close_request(t_end: float) -> None:
            if last_chat is not None:
                last_chat.attrs["is_final"] = True                   # the turn closed after this round: observable
            out.append(Span(trace_id=sid, span_id=f"{sid}-r{req_idx}", parent_span_id=f"{sid}-root", op="invoke_workflow", name="coding",
                            tier=None, resource=None, t_start=req_start, t_end=t_end, step_idx=0, outcome="ok",
                            attrs={"request_idx": req_idx, "steps": req_steps}))

        for r in rounds:
            ev = r["timing_events"]
            inputs = [ts(e["timestamp"]) for e in ev if e["event_type"] in INPUT_EVENTS]
            assist = [ts(e["timestamp"]) for e in ev if e["event_type"] in ASSISTANT_EVENTS]
            if not inputs or not assist:
                stats["rounds_skipped_no_events"] += 1
                continue
            t_in, t_out = max(inputs), max(assist)
            if t_out < t_in:
                stats["rounds_skipped_negative"] += 1
                continue
            if req_last_end is not None and t_in < req_last_end:  # overlapping rounds (concurrent sub-agents): clip to sequence
                stats["rounds_clipped_overlap"] += 1
                t_in = req_last_end
                t_out = max(t_out, t_in)
            if t0 is None:
                t0 = t_in
            if r["first_input_event_type"] == "user_message" or req_idx < 0:
                if req_idx >= 0:
                    close_request(req_last_end)
                    think = t_in - req_last_end
                    if think > 0:
                        dt = datetime.fromtimestamp(req_last_end, tz=timezone.utc)      # observable: who and when the request ended
                        out.append(Span(trace_id=sid, span_id=f"{sid}-think{req_idx}", parent_span_id=f"{sid}-root", op="think", name="human",
                                        tier=None, resource=None, t_start=req_last_end, t_end=t_in, step_idx=step_idx, duration=think,
                                        attrs={"recipe": "coding", "phase_end": f"after:{last_tool}", "request_idx": req_idx, "last_tool": last_tool,
                                               "user": rounds[0]["user"], "hour": dt.hour, "weekday": dt.weekday(), "prev_think": prev_think,
                                               "user_message_chars": req_user_chars}))
                        stats["thinks"] += 1
                        prev_think = think
                req_idx += 1
                req_start, req_steps, last_tool = t_in, 0, "none"
                req_user_chars = int(r.get("current_user_message_chars") or 0)
            out_tokens = int(r["output_tokens"] or 0) + int(r.get("reasoning_output_tokens") or 0)
            tools = [t for t in r["tools"] if t.get("emitted_at") and t.get("result_at") and t["tool_name"] not in HUMAN_TOOLS]
            chat_span = Span(trace_id=sid, span_id=f"{sid}-{r['round_id']}", parent_span_id=f"{sid}-root", op="chat", name="round", tier="model",
                             resource="model.slots", t_start=t_in, t_end=t_out, step_idx=step_idx, tokens_in=int(r["input_tokens_total"] or 0),
                             tokens_out=out_tokens, tokens_fresh=int(r.get("newly_append_tokens") or 0), duration=t_out - t_in,
                             attrs={"request_idx": req_idx, "n_tools": len(r["tools"]), "model": r["model"], "provider": r["provider"],
                                    "prefix_tokens": r.get("prefix_tokens"), "is_final": False,
                                    # observable content: the size of what the user / the tools just put in front of the model
                                    "content": f"pc:uc{_bucket(r.get('current_user_message_chars'))} pc:tr{_bucket(r.get('current_tool_result_chars'))} "
                                               f"pc:first:{r.get('first_input_event_type')}",
                                    "plan": "pl:" + ("tools" if tools else "none") + f" pl:n{min(len(tools), 6)}",
                                    "tool_kinds": [kind_of(t["tool_name"]) for t in tools],
                                    "tool_resources": ["sandbox.cpu" if kind_of(t["tool_name"]) not in ("search", "web") else f"ext.{kind_of(t['tool_name'])}" for t in tools],
                                    "tool_content": [content_of(t) for t in tools]})
            out.append(chat_span)
            last_chat = chat_span
            step_idx += 1
            req_steps += 1
            req_last_end = t_out
            stats["human_tools_dropped"] += sum(1 for t in r["tools"] if t["tool_name"] in HUMAN_TOOLS)
            tools.sort(key=lambda t: ts(t["emitted_at"]))
            groups: list[list[dict]] = []
            for t in tools:
                if groups and ts(t["emitted_at"]) < max(ts(x["result_at"]) for x in groups[-1]):
                    groups[-1].append(t)                              # overlaps the running group: parallel
                else:
                    groups.append([t])
            for g in groups:
                a, b = min(ts(t["emitted_at"]) for t in g), max(ts(t["result_at"]) for t in g)
                members = [kind_of(t["tool_name"]) for t in g]
                stats["tool_calls"] += len(g)
                stats["parallel_groups"] += int(len(g) > 1)
                local = any(m not in ("search", "web") for m in members)
                out.append(Span(trace_id=sid, span_id=f"{sid}-{g[0]['tool_call_id']}", parent_span_id=f"{sid}-root", op="execute_tool",
                                name="parallel" if len(g) > 1 else members[0], tier="tool",
                                resource=("sandbox.cpu" if local else f"ext.{members[0]}") if len(g) == 1 else "parallel",
                                t_start=a, t_end=b, step_idx=step_idx, duration=b - a, mem=0.0,
                                attrs={"members": members, "parallel": len(g) > 1, "errors": sum(1 for t in g if t.get("is_error")),
                                       "tool_names": [t["tool_name"] for t in g], "request_idx": req_idx,
                                       "content": [content_of(t) for t in g],
                                       "durations": [round(max(0.0, ts(t["result_at"]) - ts(t["emitted_at"])), 4) for t in g]}))
                step_idx += 1
                req_steps += 1
                req_last_end = max(req_last_end, b)
                last_tool = members[-1]
        if req_idx >= 0:
            close_request(req_last_end)
            out.append(Span(trace_id=sid, span_id=f"{sid}-root", parent_span_id=None, op="invoke_agent", name="coding", tier=None, resource=None,
                            t_start=t0, t_end=req_last_end, step_idx=0, outcome="ok",
                            attrs={"requests_done": req_idx + 1, "n_requests": req_idx + 1, "steps": step_idx, "provider": rounds[0]["provider"],
                                   "user": rounds[0]["user"], "key": 0, "slot": -1, "requests_started": req_idx + 1}))
            spans.extend(out)
            stats["sessions"] += 1
            stats["requests"] += req_idx + 1
    return spans, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--provider", default="all", choices=("all", "claude", "codex"))
    args = ap.parse_args()
    spans, stats = convert(Path(args.src), args.provider)
    write_jsonl(Path(args.out), spans)
    print(f"wrote {args.out}: {len(spans)} spans; " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
