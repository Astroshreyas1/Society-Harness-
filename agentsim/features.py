"""The Observer of the Needs Predictor (PREDICTOR_DESIGN §2): typed feature records from what a gateway can see.

One code path for training and serving. Online, the engine feeds `Observer.on_event` (the policy hook); offline,
`replay_events(spans)` reconstructs the same event stream from OTel-shaped spans and feeds it to the same Observer.
The Observer keeps one `SessionView` per session (observable state only), builds a `Record` at every decision
point, and emits `(record, labels)` pairs when the outcomes those records predicted have been realised — so the
model learns online from its own stream and offline from replayed traces with identical features.

Decision points (`Record.dp`) and what is predicted there:
  chat_ready   the request body is at the gate         -> Q_out (output tokens), G (spawn width), R (risk)
  tool_ready   a tool call is at the gate               -> Q_tool (duration of each member; one record per member), R
  chat_end     the response arrived, tools revealed     -> S2/S3 (the two nodes after the revealed tools), T (gap to
                                                           the next chat), R
  request_end  the turn closed                          -> I (idle gap; P(idle >= cold start))

Observable inputs (nothing else is read — the honesty selftest mutates hidden Program fields and asserts identical
records): recipe, request index, counts, the history of node tokens, the last tool kind, tokens_in of the submitted
step, the step's content cue and the response's plan cue, the revealed tool calls of the last chat, the fan-out
declared by a spawn, occupancy / waiters of every resource, the crude trackers' quantiles, and the latest typed
annotation from a System One model (jev.py) with its age. Chat phases are never read (they are hidden labels).
"""
from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .predict import QuantileTracker
from .schema import Span
from .workload import MAX_SPAWN, think_bucket

N_FEATS = 2 ** 16
HIST = 8                                   # node tokens of history kept per session
QUANTILES = (0.05, 0.2, 0.5, 0.8, 0.95)    # the knots every quantile head predicts (median, two up, two down)
DENSE_NAMES = ("log_tokens_in", "log_context", "chats_in_req", "tools_in_req", "req_idx", "log_age", "n_revealed", "n_local_revealed",
               "n_ext_revealed", "parallel", "is_child", "join_width", "log_prior_tool_q50", "log_prior_tool_q90", "log_prior_out_q50",
               "log_prior_idle_q50", "log_wait_so_far", "jev_present", "log_jev_age", "jev_p_final", "jev_p_long", "jev_p_spawn", "jev_p_fail",
               "occ_model", "occ_mem", "occ_cpu", "occ_ext", "wait_model", "wait_mem", "wait_cpu", "wait_ext")
N_DENSE = len(DENSE_NAMES)


_HASHES: dict[str, int] = {}


def fhash(s: str) -> int:
    """Deterministic feature id (blake2b, memoised: the same few thousand tokens recur at every decision point)."""
    h = _HASHES.get(s)
    if h is None:
        h = _HASHES[s] = int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little") % N_FEATS
        if len(_HASHES) > 500_000:
            _HASHES.clear()
    return h


def node_vocab(tool_kinds: list[str]) -> list[str]:
    """Structure targets: what the next node is. `final` = the request's closing chat, `spawn` = an orchestrator turn."""
    return ["chat", "final", "spawn", "retrieval"] + [f"tool:{k}" for k in sorted(tool_kinds)]


@dataclass
class Record:
    dp: str
    sid: str
    t: float
    ids: list[int]
    dense: np.ndarray
    key: tuple                              # coarse conformal key (e.g. (tool kind,) or (recipe,))
    meta: dict = field(default_factory=dict)


@dataclass
class SessionView:
    """Everything the gateway knows about one session. Observable only."""
    sid: str
    recipe: str
    is_child: bool
    depth: int
    first_seen: float
    request_idx: int = 0
    chats_in_request: int = 0
    tools_in_request: int = 0
    hist: deque = field(default_factory=lambda: deque(maxlen=HIST))
    last_tool: str = "none"
    last_plan: str = ""
    context_tokens: int = 0
    revealed: list[tuple[str, str, str]] = field(default_factory=list)   # (kind, resource, content) still to run from the last chat
    revealed_parallel: bool = False         # the revealed tool calls were declared as one parallel group
    ready_parallel: bool = False            # the step at the gate is a parallel group
    join_width: int = 0
    ready_kind: str = ""                    # the step at the gate now: "chat" / "tool" / "retrieval" / ""
    ready_since: float = -1.0
    ready_content: list[str] = field(default_factory=list)
    ready_tokens_in: int = 0
    jev: object = None                      # latest JevRecord (jev.py) or None
    in_request: bool = False
    # open predictions awaiting their outcome
    open_out: Record | None = None
    open_spawn: Record | None = None
    open_tools: list[Record] = field(default_factory=list)
    open_gap: Record | None = None
    open_idle: Record | None = None
    open_struct: list[tuple[Record, int, list[str]]] = field(default_factory=list)   # (record, nodes to skip, collected)
    open_risk: list[tuple[Record, float]] = field(default_factory=list)              # (record, deadline)
    open_phase: Record | None = None        # offline only: the chat_end record whose next chat's hidden phase labels PHASE
    open_remaining: list[tuple[Record, int]] = field(default_factory=list)   # (record, chats completed when it was made) -> REMAIN at request end
    open_next_res: list[Record] = field(default_factory=list)                # records awaiting the resource of the next node (NEXT_RES)
    open_join: Record | None = None
    last_revealed_sig: str = ""             # the previous chat's revealed tool kinds + cues (for the REPEAT label)
    open_repeat: Record | None = None       # the chat_end record whose "next turn repeats" label is pending


class Observer:
    """Builds records at decision points and emits (record, labels) pairs once outcomes are realised.
    `resources` is any mapping name -> object with .used / .capacity / .waiters (the engine's, or a replayed stand-in)."""

    def __init__(self, tool_kinds: list[str], tool_resources: dict[str, str], resources, cold_start_s: float, horizon_risk_s: float,
                 use_content: bool = True, keep_state: bool = False):
        self.tool_kinds = sorted(tool_kinds)
        self.tool_resources = tool_resources
        self.res = resources
        self.cold_start_s, self.H = float(cold_start_s), float(horizon_risk_s)
        self.use_content = use_content            # ablation: drop every content cue (skeletons, prompts, plans)
        self.keep_state = keep_state              # offline trainers: keep the System One state dict on each record
        self.vocab = node_vocab(self.tool_kinds)
        self.views: dict[str, SessionView] = {}
        self.prior_tool = QuantileTracker(lazy=True)   # the crude trackers: fed in as features (lazy: recomputed as the window grows)
        self.prior_out = QuantileTracker(lazy=True)
        self.prior_idle = QuantileTracker(lazy=True)
        self.emitted: list[tuple[Record, dict]] = []   # labelled pairs since the last drain()
        self.n_records = 0

    # ---- views ------------------------------------------------------------------------------
    def view(self, sid: str, recipe: str = "", is_child: bool = False, depth: int = 0, now: float = 0.0) -> SessionView:
        v = self.views.get(sid)
        if v is None:
            if not recipe:
                raise KeyError(f"no view for {sid} and no recipe to create one")
            v = self.views[sid] = SessionView(sid=sid, recipe=recipe, is_child=is_child, depth=depth, first_seen=now)
        return v

    def drain(self) -> list[tuple[Record, dict]]:
        out, self.emitted = self.emitted, []
        return out

    # ---- resource state (observable: the gateway's own counters) --------------------------------
    def _occ(self, name: str) -> float:
        r = self.res.get(name)
        return 0.0 if r is None or r.capacity <= 0 else min(2.0, r.used / r.capacity)

    def _waiters(self, name: str) -> float:
        r = self.res.get(name)
        return 0.0 if r is None else float(len(r.waiters))

    def _ext_state(self) -> tuple[float, float]:
        ext = [n for n in self.res if n.startswith("ext.")]
        if not ext:
            return 0.0, 0.0
        return max(self._occ(n) for n in ext), sum(self._waiters(n) for n in ext)

    # ---- featurisation ----------------------------------------------------------------------
    def featurize(self, v: SessionView, dp: str, now: float, extra_tokens: list[str], tokens_in: int, content: list[str],
                  key: tuple, meta: dict | None = None) -> Record:
        if not self.use_content:
            content, extra_tokens = [], [t for t in extra_tokens if not t.startswith("revc:")]
        toks = [f"dp:{dp}", f"recipe:{v.recipe}", f"bucket:{think_bucket(v.request_idx)}", f"child:{int(v.is_child)}",
                f"last_tool:{v.last_tool}", f"chats:{min(v.chats_in_request, 12)}", f"tools:{min(v.tools_in_request, 20)}",
                f"nrev:{min(len(v.revealed), 6)}", f"jw:{v.join_width}"]
        h = list(v.hist)
        for i, tok in enumerate(reversed(h)):
            toks.append(f"h{i}:{tok}")
        for a, b in zip(h, h[1:]):
            toks.append(f"bi:{a}>{b}")
        for a, b, c in zip(h, h[1:], h[2:]):
            toks.append(f"tri:{a}>{b}>{c}")
        if v.last_plan and self.use_content:
            toks += [f"plan:{v.last_plan}", f"rp:{v.recipe}|{v.last_plan}"]
        for kind, _, ct in v.revealed[:6]:
            toks.append(f"rev:{kind}")
        for ct in content:
            for piece in ct.split():
                toks += [f"ct:{piece}", f"ctd:{dp}|{piece}"]
        toks += extra_tokens
        for name in ("model.slots", "sandbox.mem", "sandbox.cpu"):
            toks.append(f"occ:{name}:{int(min(10, self._occ(name) * 10))}")
        if v.jev is not None:
            for q, (ans, prob) in v.jev.answers.items():
                toks += [f"jev:{q}={ans}", f"jevc:{q}={ans}|{int(prob * 4)}"]
            toks.append(f"jev_age:{int(min(6, math.log1p(max(0.0, now - v.jev.t_ready))))}")
        ids = sorted({fhash(t) for t in toks})
        jev = v.jev
        occ_ext, wait_ext = self._ext_state()
        dense = np.array([
            math.log1p(tokens_in), math.log1p(v.context_tokens), min(v.chats_in_request, 20), min(v.tools_in_request, 40),
            min(v.request_idx, 30), math.log1p(max(0.0, now - v.first_seen)), len(v.revealed),
            sum(1 for _, r, _ in v.revealed if r.startswith("sandbox")), sum(1 for _, r, _ in v.revealed if not r.startswith("sandbox")),
            float(meta.get("parallel", False)) if meta else 0.0, float(v.is_child), float(v.join_width),
            self._log_prior(self.prior_tool, key if dp == "tool_ready" else (), 0.5), self._log_prior(self.prior_tool, key if dp == "tool_ready" else (), 0.9),
            self._log_prior(self.prior_out, (v.recipe,), 0.5), self._log_prior(self.prior_idle, (v.recipe,), 0.5),
            math.log1p(max(0.0, now - v.ready_since)) if v.ready_since >= 0 else 0.0,
            float(jev is not None), math.log1p(max(0.0, now - jev.t_ready)) if jev is not None else 0.0,
            jev.prob("next_chat", "final") if jev is not None else 0.0, jev.prob("duration_class", "long") if jev is not None else 0.0,
            1.0 - jev.prob("spawn_width", "0") if jev is not None else 0.0, jev.prob("fail_soon", "yes") if jev is not None else 0.0,
            self._occ("model.slots"), self._occ("sandbox.mem"), self._occ("sandbox.cpu"), occ_ext,
            math.log1p(self._waiters("model.slots")), math.log1p(self._waiters("sandbox.mem")), math.log1p(self._waiters("sandbox.cpu")),
            math.log1p(wait_ext)], dtype=np.float64)
        if len(dense) != N_DENSE:
            raise AssertionError(f"dense vector has {len(dense)} entries, DENSE_NAMES has {N_DENSE}")
        self.n_records += 1
        rec = Record(dp=dp, sid=v.sid, t=now, ids=ids, dense=dense, key=key, meta=meta or {})
        if self.keep_state:
            rec.meta["state"] = state_of(v, dp, now, content, {n: self._occ(n) for n in ("model.slots", "sandbox.mem", "sandbox.cpu")})
        return rec

    @staticmethod
    def _log_prior(tr: QuantileTracker, key: tuple, tau: float) -> float:
        q = tr.quantile(key, tau)
        return 0.0 if q is None else math.log1p(q)

    # ---- the event stream (online: the policy hook; offline: replay_events) ----------------------
    def on_event(self, kind: str, sid: str, now: float, recipe: str = "", is_child: bool = False, depth: int = 0, **info) -> None:
        v = self.view(sid, recipe, is_child, depth, now)
        self._expire_risk(v, now)
        getattr(self, f"_ev_{kind}")(v, now, **info)

    def _ev_request_start(self, v: SessionView, now: float, request_idx: int, **_) -> None:
        v.request_idx, v.chats_in_request, v.tools_in_request, v.last_tool, v.in_request = request_idx, 0, 0, "none", True
        v.revealed, v.join_width = [], 0
        v.hist.append("user")
        if v.open_idle is not None:
            gap = max(0.05, now - v.open_idle.t)
            self.prior_idle.observe((v.recipe,), gap)
            self._emit(v.open_idle, {"I_gap": math.log(gap), "I_cold": float(gap >= self.cold_start_s)})
            v.open_idle = None

    def _ev_request_end(self, v: SessionView, now: float, **_) -> None:
        v.in_request = False
        v.open_phase = None
        for rec, n_at in v.open_remaining:
            self._emit(rec, {"REMAIN": float(v.chats_in_request - n_at)})
        v.open_remaining = []
        for rec in v.open_next_res:
            self._emit(rec, {"NEXT_RES": "none"})
        v.open_next_res = []
        self._close_open_struct(v, "final")
        if v.open_gap is not None:                                   # the request ended before another chat: the gap is the rest
            self._emit(v.open_gap, {"T_gap": math.log(max(0.1, now - v.open_gap.t))})
            v.open_gap = None
        for rec, _ in v.open_risk:
            self._emit(rec, {"R": 0.0, "BUDGET_FAIL": 0.0})
        v.open_risk = []
        if v.open_repeat is not None:
            self._emit(v.open_repeat, {"REPEAT": 0.0})
            v.open_repeat = None
        if not v.is_child:
            v.open_idle = self.featurize(v, "request_end", now, [], 0, [], (v.recipe,))

    def _ev_request_abort(self, v: SessionView, now: float, **_) -> None:
        v.in_request = False
        v.open_remaining, v.open_next_res = [], []
        self._close_open_struct(v, "final")
        for rec, _ in v.open_risk:
            self._emit(rec, {"R": 1.0})
        v.open_risk = []
        v.open_out = v.open_spawn = v.open_gap = None
        v.open_tools = []

    def _ev_session_end(self, v: SessionView, now: float, status: str, **_) -> None:
        self.views.pop(v.sid, None)

    def _ev_step_ready(self, v: SessionView, now: float, step_kind: str, name: str, tokens_in: int, members: list[str], content: list[str],
                       parallel_group: bool, **_) -> None:
        v.ready_kind, v.ready_since, v.ready_content, v.ready_tokens_in = step_kind, now, list(content), int(tokens_in)
        v.ready_parallel = bool(parallel_group)
        if step_kind == "chat":
            v.context_tokens = int(tokens_in)
            rec = self.featurize(v, "chat_ready", now, [], tokens_in, content, (v.recipe,))
            v.open_out = v.open_spawn = rec
            v.open_risk.append((rec, now + self.H))
            v.open_remaining.append((rec, v.chats_in_request))
            self._resolve_next_res(v, "model.slots")
            if v.open_gap is not None:                               # the gap this chat's predecessor predicted has closed
                self._emit(v.open_gap, {"T_gap": math.log(max(0.1, now - v.open_gap.t))})
                v.open_gap = None
        elif step_kind == "tool":
            self._resolve_next_res(v, self.tool_resources.get(members[0], "sandbox.cpu"))
            v.open_tools = []
            for i, (m, ct) in enumerate(zip(members, content)):
                rec = self.featurize(v, "tool_ready", now, [f"kind:{m}", f"res:{self.tool_resources.get(m, '?')}"], tokens_in, [ct],
                                     (m,), {"member": i, "kind": m, "parallel": parallel_group, "n_members": len(members)})
                v.open_tools.append(rec)
            if v.open_tools:
                v.open_risk.append((v.open_tools[0], now + self.H))
                v.open_next_res.append(v.open_tools[0])
        elif step_kind == "retrieval":
            self._resolve_next_res(v, "svc.retrieval")

    def _ev_step_start(self, v: SessionView, now: float, **_) -> None:
        pass

    def _ev_step_end(self, v: SessionView, now: float, step_kind: str, name: str, duration: float, **info) -> None:
        v.ready_kind, v.ready_since, v.ready_parallel = "", -1.0, False
        if step_kind == "chat":
            self._chat_end(v, now, duration, **info)
        elif step_kind == "tool":
            self._tool_end(v, now, duration, **info)
        else:
            v.hist.append("retrieval")
            self._advance_struct(v, "retrieval")

    def _chat_end(self, v: SessionView, now: float, duration: float, tokens_out: int, tool_kinds: list[str], tool_resources: list[str],
                  tool_content: list[str], plan: str, spawn: int, is_final: bool, parallel_group: list[str], outcome: str = "ok",
                  hidden_phase: str = "", usd: float = 0.0, **_) -> None:
        if outcome != "ok":
            return
        sig = "|".join(f"{k}:{c}" for k, c in zip(tool_kinds, tool_content))
        if v.open_repeat is not None:
            self._emit(v.open_repeat, {"REPEAT": float(bool(sig) and sig == v.last_revealed_sig)})
            v.open_repeat = None
        v.last_revealed_sig = sig
        if hidden_phase and v.open_phase is not None:                # replayed traces only: the annotation label (jev.py, point A)
            self._emit(v.open_phase, {"PHASE": hidden_phase})
        v.open_phase = None
        v.chats_in_request += 1
        v.last_plan = plan
        tok = "final" if is_final else ("spawn" if spawn else "chat")
        v.hist.append(tok)
        self._advance_struct(v, tok)
        self.prior_out.observe((v.recipe,), float(tokens_out))
        if v.open_out is not None:
            self._emit(v.open_out, {"Q_out": math.log1p(tokens_out), "G": min(spawn, MAX_SPAWN), "FINAL": float(is_final), "USD": float(usd)})
            v.open_out = v.open_spawn = None
        v.revealed = list(zip(tool_kinds, tool_resources, tool_content))
        v.revealed_parallel = bool(parallel_group)
        v.join_width = spawn
        rec = self.featurize(v, "chat_end", now, [f"tok:{tok}", f"spawn:{spawn}"] + [f"revc:{c}" for c in tool_content[:4]],
                             v.context_tokens, [], (v.recipe,), {"parallel": bool(parallel_group), "tok": tok})
        if not is_final:
            v.open_gap = rec
            v.open_struct.append((rec, len(tool_kinds), []))
            v.open_risk.append((rec, now + self.H))
            v.open_phase = rec
            v.open_remaining.append((rec, v.chats_in_request))
            v.open_repeat = rec
            if not tool_kinds and not spawn:                             # the next node is not revealed: what capacity will it need?
                v.open_next_res.append(rec)
            elif spawn:
                self._emit(rec, {"NEXT_RES": "model.slots"})
            else:
                self._emit(rec, {"NEXT_RES": tool_resources[0]})

    def _tool_end(self, v: SessionView, now: float, duration: float, members: list[str], durations: list[float] | None = None,
                  outcome: str = "ok", errors: int = 0, usd: float = 0.0, **_) -> None:
        durs = durations if durations is not None else [duration] * len(members)
        dmax = max(1e-3, max(float(d) for d in durs)) if durs else max(1e-3, duration)
        for i, (m, d, rec) in enumerate(zip(members, durs, v.open_tools)):
            d = max(1e-3, float(d))
            if outcome == "ok":
                self.prior_tool.observe((m,), d)
            lab = {"Q_tool": math.log(d), "censored": float(outcome != "ok")}
            if i == 0:
                lab.update({"Q_tool_max": math.log(dmax), "E_err": float(errors or (outcome != "ok")), "USD": float(usd)})
            self._emit(rec, lab)
        v.open_tools = []
        if outcome != "ok":
            return
        v.tools_in_request += len(members)
        v.last_tool = members[-1]
        for m in members:
            v.hist.append(f"tool:{m}")
            self._advance_struct(v, f"tool:{m}")
        v.revealed = v.revealed[len(members):]

    def _ev_budget_refused(self, v: SessionView, now: float, **_) -> None:
        for rec, _ in v.open_risk:                                        # the budget failure this request just hit
            self._emit(rec, {"BUDGET_FAIL": 1.0})
        v.open_risk = []

    def _ev_spawn(self, v: SessionView, now: float, width: int, **_) -> None:
        v.join_width = width
        v.open_join = self.featurize(v, "spawn", now, [f"width:{width}"], v.context_tokens, [], (v.recipe,), {"width": width})

    def _ev_join(self, v: SessionView, now: float, wait: float = 0.0, **_) -> None:
        v.join_width = 0
        v.hist.append("join")
        if v.open_join is not None:
            self._emit(v.open_join, {"JOIN": float(wait)})
            v.open_join = None

    def _resolve_next_res(self, v: SessionView, res: str) -> None:
        for rec in v.open_next_res:
            self._emit(rec, {"NEXT_RES": res})
        v.open_next_res = []

    def _ev_jev(self, v: SessionView, now: float, record, **_) -> None:
        v.jev = record

    # ---- label bookkeeping --------------------------------------------------------------------
    def _emit(self, rec: Record, labels: dict) -> None:
        self.emitted.append((rec, labels))

    def _advance_struct(self, v: SessionView, tok: str) -> None:
        done = []
        for i, (rec, skip, got) in enumerate(v.open_struct):
            if skip > 0:
                v.open_struct[i] = (rec, skip - 1, got)
                continue
            got.append(tok)
            if len(got) == 2:
                self._emit(rec, {"S2": got[0], "S3": got[1]})
                done.append(i)
        for i in reversed(done):
            v.open_struct.pop(i)

    def _close_open_struct(self, v: SessionView, tok: str) -> None:
        for rec, skip, got in v.open_struct:
            got = got + [tok] * (2 - len(got))
            self._emit(rec, {"S2": got[0], "S3": got[1]})
        v.open_struct = []

    def _expire_risk(self, v: SessionView, now: float) -> None:
        if not v.open_risk:
            return
        keep = []
        for rec, deadline in v.open_risk:
            if now >= deadline:
                self._emit(rec, {"R": 0.0})
            else:
                keep.append((rec, deadline))
        v.open_risk = keep


def state_of(v: SessionView, dp: str, now: float, content: list[str], occupancy: dict[str, float]) -> dict:
    """The unstructured-plus-structured state a System One model reads (jev.py): observable only."""
    return {"dp": dp, "sid": v.sid.split("/")[0], "recipe": v.recipe, "child": v.is_child, "request_idx": v.request_idx, "chats": v.chats_in_request,
            "tools": v.tools_in_request, "history": list(v.hist), "last_tool": v.last_tool, "plan": v.last_plan,
            "content": list(content), "revealed": [k for k, _, _ in v.revealed], "revealed_content": [c for _, _, c in v.revealed],
            "join_width": v.join_width, "tokens_in": v.ready_tokens_in, "age_s": round(now - v.first_seen, 1), "occupancy": occupancy}


# ---- offline replay: spans -> the same event stream ---------------------------------------------------
class ReplayResource:
    """Occupancy stand-in for replayed traces: `used` follows span intervals, `waiters` follows wait spans."""

    def __init__(self, name: str, capacity: float):
        self.name, self.capacity, self.used = name, float(capacity), 0.0
        self.waiters: dict[str, float] = {}


def replay_events(spans: list[Span], capacities: dict[str, float]):
    """Yield (t, kind, sid, recipe, is_child, depth, info) in time order, plus resource updates applied to
    `resources` as a side effect. Returns (events generator, resources). Only 'ok' spans describe structure;
    timed-out tools are replayed as censored step_end events, exactly as the engine reports them."""
    res = {n: ReplayResource(n, c) for n, c in capacities.items()}
    roots = {s.span_id: s for s in spans if s.op == "invoke_agent"}
    by_sid: dict[str, list[Span]] = {}
    for s in spans:
        if s.op in ("invoke_agent",):
            continue
        sid = s.span_id.rsplit("-", 1)[0] if s.op in ("invoke_workflow",) else s.span_id.split("-w")[0] if s.op == "wait" else s.span_id.rsplit("-", 1)[0]
        by_sid.setdefault(sid, []).append(s)
    ev: list[tuple] = []
    seq = 0
    # same-instant order, exactly as the engine emits: releases / step_end < request_end|abort < spawn < session_end
    # < request_start | join < step_ready | takes | waits < step_start
    ORDER = {"_res_give": 0, "step_end": 1, "request_end": 2, "request_abort": 2, "spawn": 3, "session_end": 4, "request_start": 5,
             "join": 5, "step_ready": 6, "_res_take": 6, "_wait_on": 6, "_wait_off": 0, "step_start": 7}

    def add(t, kind, sid, info, order=None):
        nonlocal seq
        seq += 1
        ev.append((t, ORDER[kind], seq, kind, sid, info))

    meta: dict[str, tuple[str, bool, int]] = {}
    for sid, root in ((sp.span_id[:-5], sp) for sp in roots.values()):
        meta[sid] = (root.name, root.parent_span_id is not None, int(root.attrs.get("depth", 0)))
        add(root.t_start, "_res_take", sid, {"res": "_session", "amt": 0.0}, 0)
        add(root.t_end, "session_end", sid, {"status": root.outcome}, 3)
    for sid, ss in by_sid.items():
        if sid not in meta:
            raise KeyError(f"spans of {sid} without an invoke_agent root")
        for s in sorted(ss, key=lambda x: x.t_start):
            if s.op == "invoke_workflow":
                add(s.t_start, "request_start", sid, {"request_idx": s.attrs["request_idx"]}, 0)
                if s.outcome == "ok":
                    add(s.t_end, "request_end", sid, {"request_idx": s.attrs["request_idx"], "last_tool": "", "chats": 0}, 2)
                elif s.outcome == "aborted":
                    add(s.t_end, "request_abort", sid, {}, 2)
            elif s.op == "wait":
                add(s.t_start, "_wait_on", sid, {"res": s.resource}, 0)
                add(s.t_end, "_wait_off", sid, {"res": s.resource}, 0)
            elif s.op in ("chat", "execute_tool", "retrieval"):
                ready = s.t_start - s.wait
                kind = {"chat": "chat", "execute_tool": "tool", "retrieval": "retrieval"}[s.op]
                members = s.attrs.get("members") or [s.name]
                content = s.attrs.get("content", [""] * len(members)) if kind != "chat" else [s.attrs.get("content", "")]
                if isinstance(content, str):
                    content = [content]
                if s.outcome in ("ok", "timeout") and s.op != "chat" or s.outcome == "ok":
                    add(ready, "step_ready", sid, {"step_kind": kind, "name": "chat" if kind == "chat" else s.name, "tokens_in": s.tokens_in,
                                                  "members": members, "content": content, "parallel_group": s.name == "parallel"}, 0)
                    add(s.t_start, "_res_take", sid, {"res": s.resource, "amt": 1.0}, 0)
                    add(s.t_start, "step_start", sid, {"step_kind": kind, "name": s.name, "tokens_in": s.tokens_in, "members": members,
                                                       "parallel": bool(s.attrs.get("parallel"))}, 0)
                    add(s.t_end, "_res_give", sid, {"res": s.resource, "amt": 1.0}, 0)
                if s.op == "chat" and s.outcome == "ok":
                    add(s.t_end, "step_end", sid, {"step_kind": "chat", "name": s.name, "duration": s.duration, "tokens_in": s.tokens_in,
                                                   "tokens_out": s.tokens_out, "n_tools": s.attrs.get("n_tools", 0),
                                                   "parallel_group": [], "tool_resources": s.attrs.get("tool_resources", []),
                                                   "tool_kinds": s.attrs.get("tool_kinds", []), "tool_content": s.attrs.get("tool_content", []),
                                                   "plan": s.attrs.get("plan", ""), "spawn": int(s.attrs.get("spawn", 0)),
                                                   "is_final": bool(s.attrs.get("is_final", s.name == "final")), "outcome": "ok",
                                                   "hidden_phase": s.attrs.get("phase", ""), "usd": float(s.attrs.get("usd", 0.0))}, 1)
                elif s.op == "execute_tool" and s.outcome in ("ok", "timeout"):
                    add(s.t_end, "step_end", sid, {"step_kind": "tool", "name": s.name, "duration": s.duration, "members": members,
                                                   "parallel": bool(s.attrs.get("parallel")), "content": content, "errors": int(s.attrs.get("errors", 0)),
                                                   "usd": float(s.attrs.get("usd", 0.0)),
                                                   "durations": s.attrs.get("durations", [s.duration] * len(members)), "outcome": s.outcome}, 1)
                elif s.op == "retrieval" and s.outcome == "ok":
                    add(s.t_end, "step_end", sid, {"step_kind": "retrieval", "name": "retrieval", "duration": s.duration, "outcome": "ok"}, 1)
                if s.op == "chat" and s.outcome == "ok" and int(s.attrs.get("spawn", 0)):
                    add(s.t_end, "spawn", sid, {"width": int(s.attrs["spawn"])}, 2)
                    if "join_wait_s" in s.attrs:
                        add(s.t_end + float(s.attrs["join_wait_s"]), "join", sid, {"wait": float(s.attrs["join_wait_s"]),
                                                                                     "failed": int(s.attrs.get("children_failed", 0)),
                                                                                     "width": int(s.attrs["spawn"])}, 0)
    ev.sort(key=lambda e: (e[0], e[1], e[2]))

    def gen():
        for t, _, _, kind, sid, info in ev:
            if kind == "_res_take":
                r = res.get(info["res"])
                if r is not None:
                    r.used += info["amt"]
                continue
            if kind == "_res_give":
                r = res.get(info["res"])
                if r is not None:
                    r.used = max(0.0, r.used - info["amt"])
                continue
            if kind == "_wait_on":
                r = res.get(info["res"])
                if r is not None:
                    r.waiters[sid] = t
                continue
            if kind == "_wait_off":
                r = res.get(info["res"])
                if r is not None:
                    r.waiters.pop(sid, None)
                continue
            recipe, is_child, depth = meta[sid]
            yield t, kind, sid, recipe, is_child, depth, info

    return gen(), res


def chat_tool_lists(spans: list[Span]) -> None:
    """Chat spans written before Phase 6 carry no revealed tool lists; reconstruct `tool_kinds` / `tool_resources` /
    `tool_content` from the tool spans that follow each chat in the same session (in place)."""
    by_sid: dict[str, list[Span]] = {}
    for s in spans:
        if s.op in ("chat", "execute_tool"):
            by_sid.setdefault(s.span_id.rsplit("-", 1)[0], []).append(s)
    for ss in by_sid.values():
        ss.sort(key=lambda x: x.t_start)
        for i, s in enumerate(ss):
            if s.op != "chat" or "tool_kinds" in s.attrs:
                continue
            kinds, ress, cts = [], [], []
            for nxt in ss[i + 1:]:
                if nxt.op != "execute_tool":
                    break
                members = nxt.attrs.get("members") or [nxt.name]
                kinds += members
                ress += [nxt.resource or "parallel"] * len(members)
                cts += list(nxt.attrs.get("content", [""] * len(members)))
            s.attrs["tool_kinds"], s.attrs["tool_resources"], s.attrs["tool_content"] = kinds, ress, cts
