"""Reserver v2 (PREDICTOR_DESIGN §5): `policy.type = "needs"`.

The reactive gate plus a Needs Predictor (features.py + needs.py) and, optionally, a System One annotator (jev.py),
consumed by a fixed reservation rule with the same two knobs as rung 3 (tau, h):

  occupancy forecast   every live session contributes its predicted demand curve per resource over the next h seconds
                       (current holds until their predicted release, revealed and predicted steps at their predicted
                       times, sub-agents from spawn to the predicted join) plus an arrival term; `pressure(res, t0, t1)`
                       = forecast excess over capacity, normalised — the *price* of holding a unit in that window.
  expected-value gate  a lease, prewarm or park is issued only when its benefit (latency-seconds this session saves)
                       exceeds its cost (latency-seconds others lose = hold x pressure). Free pools cost nothing to
                       reserve; saturated ones cost a lot. No new knob: prices are measured.
  leases               gang CPU for a revealed parallel group (sized by content-conditioned duration quantiles), budget
                       units for revealed external calls (calibrated margin, as rung 3), k model slots + k sandboxes for
                       the sub-agents of a spawn (family-owned: the children draw on the parent's lease), and, from the
                       G head, a *predicted* spawn gang issued at chat submission, starting at the predicted chat end.
  ordering             shortest-predicted-first on the model queue only while it is deeper than the slot count
                       (conformal median of predicted service time), virtual-time fair queuing otherwise, and a
                       starvation guard: anything that has waited past half the client timeout goes first.
  KV                   admission reserve = conformal q_tau of predicted output tokens (not a fixed 4096);
                       retention = q_tau of the predicted gap to the next chat (+ grace), Belady eviction on it.
  idle sandboxes       park iff E[idle] x pressure(sandbox) > cold start; prewarm iff P(hit) x cold start > hold x pressure.
  Jev                  asked asynchronously at every decision point through `ext.jev` (rate-limited, latency); the
                       answer lands via a policy timer and enters the next records as features with its age.

Honesty: every input is the observation stream (`on_event`), the gateway's own counters and the ledger. Hidden
values in hook signatures (think, gap, Step.duration / tokens_out / follow_tools / phase) are never read; the
selftest feeds contradictory ones and asserts identical decisions.

Ablation switches (`policy.ablate`, '+'-joined): vtfq park prewarm kv gang budget spawn srpt aging content jev conformal
ev forecast learn.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .features import Observer, Record, node_vocab
from .jev import JevChannel, Judge, LocalSystemOne, RemoteSystemOne, answers_from_labels, catalogue, questions_at
from .needs import NeedsModel, NeedsPredictor, Prediction
from .policies import LeaseController, ReactiveGate
from .control import AllocationLoop
from .predict import QuantileTracker
from .resources import Lease, Resource
from .workload import Program, Recipes, Step

ROOT = Path(__file__).resolve().parent.parent
SWITCHES = {"vtfq", "park", "prewarm", "kv", "gang", "budget", "spawn", "srpt", "aging", "content", "jev", "conformal", "ev", "forecast", "learn", "feedback", "pacing"}


class Forecast:
    """Service-graph demand horizon: per resource, expected units in use per bucket over the next H seconds."""

    def __init__(self, resources: dict[str, Resource], dt: float, H: float):
        if dt <= 0 or H < dt:
            raise ValueError("forecast needs 0 < dt <= H")
        self.res, self.dt, self.H = resources, float(dt), float(H)
        self.n = int(math.ceil(H / dt))
        self.seg: dict[str, list[tuple[str, float, float, float]]] = {}      # sid -> [(res, start, end, amt)]
        self._sum: dict[str, np.ndarray] = {}
        self._sum_t = -1.0
        self.arrivals = 0.0                                                   # sessions / s (EWMA)
        self._last_arrival = -1.0
        self.hold: dict[str, tuple[float, float]] = {}                        # res -> (sum of unit-seconds per session, n sessions)

    def set(self, sid: str, segments: list[tuple[str, float, float, float]]) -> None:
        self.seg[sid] = [(r, s, e, a) for r, s, e, a in segments if e > s and a > 0]
        self._sum_t = -1.0

    def drop(self, sid: str) -> None:
        if self.seg.pop(sid, None) is not None:
            self._sum_t = -1.0

    def saw_arrival(self, now: float) -> None:
        if self._last_arrival >= 0:
            gap = max(1e-3, now - self._last_arrival)
            self.arrivals = 0.9 * self.arrivals + 0.1 / gap if self.arrivals > 0 else 1.0 / gap
        self._last_arrival = now

    def saw_hold(self, res: str, unit_s: float) -> None:
        s, n = self.hold.get(res, (0.0, 0.0))
        self.hold[res] = (s + unit_s, n + 1)

    def _rebuild(self, now: float) -> None:
        sums = {name: np.zeros(self.n) for name in self.res}
        bias = getattr(self, "bias", {})
        for segs in self.seg.values():
            for r, s, e, a in segs:
                arr = sums.get(r)
                if arr is None:
                    continue
                i0 = max(0, int((s - now) // self.dt))
                i1 = min(self.n, int(math.ceil((e - now) / self.dt)))
                if i1 > i0:
                    arr[i0:i1] += a
        for name, arr in sums.items():                                        # newcomers: arrival rate x mean hold per session
            s, n = self.hold.get(name, (0.0, 0.0))
            if n > 0 and self.arrivals > 0:
                arr += self.arrivals * s / n
            arr *= bias.get(name, 1.0)                                        # the feedback loop's correction of this resource's forecast
        self._sum, self._sum_t = sums, now

    def demand(self, res: str, now: float) -> np.ndarray:
        if self._sum_t != now:
            self._rebuild(now)
        return self._sum[res]

    def excess(self, res: str, t0: float, t1: float, now: float) -> float:
        """Expected units short of capacity over [t0, t1] (queue that would form), plus the queue standing now."""
        r = self.res[res]
        d = self.demand(res, now)
        i0 = max(0, min(self.n - 1, int((t0 - now) // self.dt)))
        i1 = max(i0 + 1, min(self.n, int(math.ceil((t1 - now) / self.dt))))
        return float(np.maximum(0.0, d[i0:i1] - r.capacity).mean()) + len(r.waiters)

    def pressure(self, res: str, t0: float, t1: float, now: float) -> float:
        r = self.res[res]
        return self.excess(res, t0, t1, now) / r.capacity if r.capacity > 0 else 0.0


class NeedsController(LeaseController):
    name = "needs"
    leases_at_ready = True

    def __init__(self, spec: dict, rng: np.random.Generator, cfg: dict, recipes: Recipes):
        m, fw = cfg["model"], cfg["framework"]
        ReactiveGate.__init__(self, spec, rng, float(m["prefill_rate"]), float(m["decode_base"]))
        self.tau, self.h = float(spec["tau"]), float(spec["h"])
        if not 0.5 <= self.tau < 1.0:
            raise ValueError("needs.tau must be in [0.5, 1)")
        if not self.h > 0:
            raise ValueError("needs.h must be > 0")
        self.off = set(x for x in str(spec.get("ablate", "")).split("+") if x)
        unknown = self.off - SWITCHES
        if unknown:
            raise ValueError(f"needs.ablate: unknown switches {sorted(unknown)}; known: {sorted(SWITCHES)}")
        if "needs" not in spec:
            raise KeyError("policy.needs block missing (model, jev, learn, replay)")
        ns = spec["needs"]
        self.cold_start_s = float(fw["sandbox_cold_start_s"])
        self.step_timeout = float(fw["step_timeout_s"])
        self.kv_default = int(m["kv_reserve_out"])
        self.tool_kinds = sorted(recipes.tool_resources)
        self.phases = sorted({ph for r in recipes.recipes.values() for ph in list(r["transitions"]) + ["final"]})
        self.vocab = node_vocab(self.tool_kinds)
        self.observer = Observer(self.tool_kinds, recipes.tool_resources, None, self.cold_start_s, self.h, use_content="content" not in self.off)
        model = NeedsModel.load(ROOT / ns["model"]) if ns["model"] else None
        seed = int(cfg["seed"])
        self.pred = NeedsPredictor(self.vocab, self.tau, seed, model, replay=int(ns["replay"]), learn=bool(ns["learn"]) and "learn" not in self.off)
        self.conformal = "conformal" not in self.off
        self.jev: JevChannel | None = None
        self.cat = catalogue(self.tool_kinds, self.phases)
        self.judge = Judge(self.cat)
        js = ns["jev"]
        if bool(js["enabled"]) and "jev" not in self.off:
            if js.get("remote"):
                s1 = RemoteSystemOne(self.cat)
            else:
                s1 = LocalSystemOne.load(ROOT / js["model"]) if js["model"] else LocalSystemOne(self.cat, seed)
            schema = {q.name: q.options for q in self.cat.values()}
            if set(s1.schema) != set(schema) or any(tuple(s1.schema[q]) != tuple(schema[q]) for q in schema):
                raise ValueError("the loaded System One model answers a different question schema than this society's")
            self.jev = JevChannel(s1, js, np.random.default_rng([seed, 11]))
        self.jev_recs: dict[tuple[str, str], object] = {}   # (sid, dp) -> the latest delivered record for that decision point
        self.jev_veto: dict[str, float] = {}                 # sid -> P(the reserved units will be used) when Jev judged it low
        self._flush_key = None
        self.forecast: Forecast | None = None
        self.loop: AllocationLoop | None = None
        self.tick_s = 30.0
        self.srpt_depth = 1.0           # shortest-predicted-first once the model queue exceeds the slot count (depth swept in NEEDS_REPORT: no gain below 1)
        # the starvation guard (oldest waiter first past half the client timeout) is opt-in: on the saturated societies it costs
        # ~30% throughput and prevents no failures (NEEDS_REPORT); `policy.ablate=aging` turns it off when it is on
        self.aging = bool(ns["starvation_guard"]) and "aging" not in self.off
        self.cost_chat: dict[str, tuple[int, float]] = {}       # recipe -> (n, sum usd per chat) from provider usage headers
        self.req_time: dict[str, tuple[int, float]] = {}        # recipe -> (n requests, sum wall-clock seconds) — how long spend is spread over
        self.cost_tool: dict[str, tuple[int, float]] = {}       # recipe -> (n, sum usd per tool step)
        self.chats_per_req: dict[str, tuple[int, float]] = {}   # recipe -> (n requests, sum chats)
        self.calls_seen: dict[str, tuple[int, int]] = {}
        self.join_wait = QuantileTracker()
        self.gb_seen: tuple[float, int] = (0.0, 0)
        self.tags: dict[str, float] = {}
        self.v_clock = 0.0
        # latest predictions per session
        self.p_chat: dict[str, Prediction] = {}
        self.p_tools: dict[str, list[Prediction]] = {}
        self.p_end: dict[str, Prediction] = {}
        self.p_idle: dict[str, Prediction] = {}
        self.revealed_q: dict[str, list[tuple[str, str, float, float]]] = {}   # sid -> [(kind, resource, q50, q_tau)] of revealed tools
        self.spawn_kids: dict[str, list[tuple[str, float]]] = {}
        self.decisions = {"park": 0, "hold": 0, "prewarm": 0, "prewarm_skipped": 0, "gang": 0, "gang_skipped": 0, "gang_paid": 0, "spawn_gang": 0,
                          "spawn_predicted": 0, "srpt_orders": 0, "vtfq_orders": 0, "aging_orders": 0, "jev_asked": 0, "jev_delivered": 0}
        self.first_ready: dict[str, float] = {}     # sid -> when the client submitted the step at the gate (retries keep it)
        self.gang_open: dict[str, bool] = {}        # sid -> a gang lease was issued for the group now at the gate
        self.timers: dict[int, tuple[str, object]] = {}
        self._timer_seq = 0
        self.request_started: dict[str, float] = {}
        self.now = 0.0

    # ---- wiring ---------------------------------------------------------------------------
    def replace_leases(self, sid: str) -> None:              # replaced by the engine: re-issue this session's leases now
        raise RuntimeError("replace_leases is available only inside an Engine")

    def _ensure(self) -> None:
        if self.forecast is None:
            if self.resources is None:
                raise RuntimeError("needs controller used before the engine attached its resources")
            self.observer.res = self.resources
            self.forecast = Forecast(self.resources, dt=10.0, H=self.h)
            self.loop = AllocationLoop(self.resources, self.budgets, self.tick_s, enabled="feedback" not in self.off)
            self.timer(self.now + self.tick_s, ("tick",))

    def _pressure(self, res: str, t0: float, t1: float, now: float) -> float:
        if "forecast" in self.off:                                         # no future: the standing queue only
            r = self.resources[res]
            return (len(r.waiters) + max(0.0, r.used - r.capacity)) / r.capacity
        return self.forecast.pressure(res, t0, t1, now)

    def _q(self, pred: Prediction | None, head: str, tau: float, key: tuple | None = None) -> float | None:
        if pred is None:
            return None
        return pred.q(head, tau, key, conformal=self.conformal)

    def _service_s(self, tokens_in: int, out_tokens: float) -> float:
        return tokens_in / self.prefill_rate + out_tokens / self.decode_base

    # ---- observation stream ------------------------------------------------------------------
    def on_event(self, kind: str, p: Program, now: float, **info) -> None:
        self.now = now
        self._ensure()
        self.loop.sample(now)
        sid = p.sid
        if kind == "request_start" and sid not in self.observer.views:
            self.forecast.saw_arrival(now)
            g, n = self.gb_seen
            self.gb_seen = (g + p.sandbox_gb, n + 1)
        self.observer.on_event(kind, sid, now, recipe=p.recipe, is_child=p.parent is not None, depth=p.depth, **info)
        for rec, labels in self.observer.drain():
            self.pred.learn(rec, labels)
            self._judge(rec, labels)
        v = self.observer.views.get(sid)
        if kind == "request_start":
            self.request_started[sid] = now
            self.p_idle.pop(sid, None)
        elif kind == "step_ready":
            self.first_ready[sid] = now
            if info["step_kind"] == "chat" and v.open_out is not None:
                self.p_chat[sid] = self.pred.predict(v.open_out)
                self._ask_jev(v, "chat_ready", now, info["content"])
            elif info["step_kind"] == "tool" and v.open_tools:
                self.p_tools[sid] = [self.pred.predict(r) for r in v.open_tools]
                self._ask_jev(v, "tool_ready", now, info["content"])
            self._refresh_segments(p, v, now)
        elif kind == "step_start":
            if "vtfq" not in self.off:
                start = max(self.tags.get(sid, self.v_clock), self.v_clock)
                w = self.loop.weight(sid)
                p_loop = self._jev_p(sid, "chat_end", "stuck_in_loop")
                if p_loop is not None and p_loop > 0.7:
                    w *= 2.0                                                  # a looping agent yields to others
                self.tags[sid] = start + self._predicted_cost(p, v, info["step_kind"], info["members"], info["tokens_in"]) * w
                self.v_clock = start
        elif kind == "step_end":
            if info.get("outcome", "ok") != "ok":
                return
            self.loop.served_add(sid, float(info["duration"]), now)
            if info["step_kind"] in ("chat", "tool") and "usd" in info:
                d = self.cost_chat if info["step_kind"] == "chat" else self.cost_tool
                n, tot = d.get(p.recipe, (0, 0.0))
                d[p.recipe] = (n + 1, tot + float(info["usd"]))
            if info["step_kind"] == "chat":
                for key in (p.recipe, ""):                                    # the srpt fallback estimate (ReactiveGate.mean_out)
                    n, tot = self.out_seen.get(key, (0, 0.0))
                    self.out_seen[key] = (n + 1, tot + float(info["tokens_out"]))
                self.p_chat.pop(sid, None)
                self.revealed_q[sid] = self._quantiles_for_revealed(v, now, info["tool_kinds"], info["tool_resources"], info["tool_content"])
                if v.open_gap is not None:
                    self.p_end[sid] = self.pred.predict(v.open_gap)
                    local = [k for k, r, _, _ in self.revealed_q[sid] if r.startswith("sandbox")]
                    cand = {"resource": "sandbox.cpu", "units": len(local) - 1, "horizon_s": int(self.h)} if v.revealed_parallel and len(local) >= 2 else None
                    self._ask_jev(v, "chat_end", now, [], cand)
            elif info["step_kind"] == "tool":
                self.p_tools.pop(sid, None)
                if self.gang_open.pop(sid, False):                            # did the reserved units get used?
                    paid = int(bool(info.get("parallel")))
                    self.decisions["gang_paid"] += paid
                    jr = self.jev_recs.get((sid, "chat_end"))
                    if jr is not None:
                        self.judge.observe(jr, "will_use_reservation", "yes" if paid else "no")
                dur = float(info["duration"])
                if any(r.startswith("sandbox") for r in (self.observer.tool_resources.get(mm, "") for mm in info["members"])):
                    self.forecast.saw_hold("sandbox.cpu", dur)
            self._refresh_segments(p, v, now)
        elif kind == "request_end":
            n, tot = self.chats_per_req.get(p.recipe, (0, 0.0))
            self.chats_per_req[p.recipe] = (n + 1, tot + float(info.get("chats", 0)))
            t0 = self.request_started.pop(sid, now)
            n, tot = self.req_time.get(p.recipe, (0, 0.0))
            self.req_time[p.recipe] = (n + 1, tot + (now - t0))
            self.forecast.saw_hold("model.slots", 0.35 * (now - t0))      # a session in a request holds a slot ~a third of the time
            self.forecast.saw_hold("sandbox.mem", p.sandbox_gb * (now - t0))
            if v is not None and v.open_idle is not None:
                self.p_idle[sid] = self.pred.predict(v.open_idle)
                self._ask_jev(v, "request_end", now, [])
            self.revealed_q.pop(sid, None)
            self._refresh_segments(p, v, now)
        elif kind == "spawn":
            self.spawn_kids[sid] = list(info["children"])
            self._ask_jev(v, "spawn", now, [], {"resource": "model.slots+sandbox", "units": len(info["children"]), "horizon_s": int(self.h)})
            self._refresh_segments(p, v, now)
        elif kind == "join":
            self.join_wait.observe((min(int(info["width"]), 4),), float(info["wait"]))
            self.spawn_kids.pop(sid, None)
        elif kind in ("session_end", "request_abort"):
            self.loop.forget(sid)
            for d in (self.p_chat, self.p_tools, self.p_end, self.p_idle, self.revealed_q, self.spawn_kids, self.tags, self.request_started,
                      self.first_ready, self.gang_open, self.jev_veto):
                d.pop(sid, None)
            for k in [k for k in self.jev_recs if k[0] == sid]:
                del self.jev_recs[k]
            self.forecast.drop(sid)

    # ---- Jev: speculative fan-out, asynchronous delivery, online judging ---------------------------------------
    def _ask_jev(self, v, dp: str, now: float, content: list[str], candidate: dict | None = None) -> None:
        """Ask every question this boundary can use in one call (the API evaluates them in parallel); the answer lands
        via a timer at its ready time. `candidate` describes the action the Reserver is about to take, for the judge questions."""
        if self.jev is None:
            return
        from .features import state_of
        occ = {n: self.observer._occ(n) for n in ("model.slots", "sandbox.mem", "sandbox.cpu")}
        st = state_of(v, dp, now, content, occ)
        if candidate:
            st["candidate"] = candidate
        qs = tuple(q for q in questions_at(self.cat, dp) if self.judge.trusted(q))     # a question whose ECE drifted is not asked
        if not qs:
            return
        rec = self.jev.ask(now, st, qs, v.sid)
        self.decisions["jev_asked"] += 1
        if rec is not None:
            self._deliver_later(v.sid, rec)
        elif self.jev.batch_window > 0 and self._flush_key is None:
            self._timer_seq += 1
            self._flush_key = ("flush", self._timer_seq)
            self.timer(self.jev.flush_at(), self._flush_key)

    def _deliver_later(self, sid: str, rec) -> None:
        self._timer_seq += 1
        self.timers[self._timer_seq] = (sid, rec)
        self.timer(rec.t_ready, self._timer_seq)

    def on_timer(self, key, now: float) -> None:
        if isinstance(key, tuple) and key[0] == "tick":                    # the feedback loop's clock
            self.now = now
            self._ensure()
            self.loop.tick_now(now)
            self.forecast.bias = {n: self.loop.scale[n].x for n in self.resources} if "feedback" not in self.off else {}
            self.forecast._sum_t = -1.0
            k = max(1, int(self.tick_s // self.forecast.dt))
            for n in self.resources:
                self.loop.set_prediction(n, float(self.forecast.demand(n, now)[:k].mean()))
            self.loop.fairness_update(now)
            self.timer(now + self.tick_s, ("tick",))
            return
        if isinstance(key, tuple) and key[0] == "flush":
            self._flush_key = None
            for sid, rec in self.jev.flush(now):
                self._deliver_later(sid, rec)
            if self.jev.pending and self._flush_key is None:
                self._timer_seq += 1
                self._flush_key = ("flush", self._timer_seq)
                self.timer(max(now, self.jev.flush_at()), self._flush_key)
            return
        sid, rec = self.timers.pop(key)
        if sid not in self.observer.views:
            return
        self.observer.on_event("jev", sid, now, record=rec)
        self.jev_recs[(sid, rec.dp)] = rec
        self.decisions["jev_delivered"] += 1
        p_use = rec.noul("will_use_reservation")
        if p_use is not None and self.judge.trusted("will_use_reservation") and rec.dp in ("chat_end", "spawn"):
            if p_use < 0.3 and (self.gang_open.get(sid) or sid in self.spawn_kids):
                self.jev_veto[sid] = p_use                             # the judge says the reservation will not be used: withdraw it
                self.decisions["jev_vetoes"] = self.decisions.get("jev_vetoes", 0) + 1
                self.replace_leases(sid)
            else:
                self.jev_veto.pop(sid, None)

    def _judge(self, rec, labels: dict) -> None:
        """Score the System One answers for this decision point against what actually happened."""
        if self.jev is None:
            return
        jr = self.jev_recs.get((rec.sid, rec.dp))
        if jr is None:
            return
        for q, a in answers_from_labels(rec.dp, labels, self.cold_start_s, self.h).items():
            self.judge.observe(jr, q, a)

    def _jev_p(self, sid: str, dp: str, q: str) -> float | None:
        """A judge question's P(yes) for this session, if answered, confident and trusted."""
        jr = self.jev_recs.get((sid, dp))
        if jr is None or not self.judge.trusted(q) or not jr.ok(q):
            return None
        return jr.noul(q)

    # ---- predictions used by several hooks -------------------------------------------------------
    def _quantiles_for_revealed(self, v, now: float, kinds: list[str], resources: list[str], contents: list[str]) -> list[tuple[str, str, float, float]]:
        out = []
        for k, r, c in zip(kinds, resources, contents):
            rec = self.observer.featurize(v, "tool_ready", now, [f"kind:{k}", f"res:{r}"], v.context_tokens, [c], (k,), {"kind": k})
            pr = self.pred.predict(rec)
            out.append((k, r, pr.q("Q_tool", 0.5, (k,), self.conformal), pr.q("Q_tool", self.tau, (k,), self.conformal)))
        return out

    def _chat_service(self, p: Program, tokens_in: int, tau: float) -> float:
        pr = self.p_chat.get(p.sid)
        out = self._q(pr, "Q_out", tau, (p.recipe,))
        if out is None:
            out = self.mean_out(p.recipe)
        return self._service_s(tokens_in, out)

    def _predicted_cost(self, p: Program, v, step_kind: str, members: list[str], tokens_in: int) -> float:
        """Memory-centric service cost for fair queuing, from predicted (not realised) durations."""
        if step_kind == "chat":
            return tokens_in * self._chat_service(p, tokens_in, self.tau) / 1e6
        preds = self.p_tools.get(p.sid, [])
        dur = 0.0
        for i, m in enumerate(members):
            q = preds[i].q("Q_tool", self.tau, (m,), self.conformal) if i < len(preds) else self.observer.prior_tool.quantile((m,), self.tau)
            dur = max(dur, q if q is not None else 0.0)
        return p.sandbox_gb * dur

    def _refresh_segments(self, p: Program, v, now: float) -> None:
        """This session's contribution to the demand horizon."""
        if "forecast" in self.off or v is None:
            return
        segs: list[tuple[str, float, float, float]] = []
        H = self.h
        gap = self._q(self.p_end.get(p.sid), "T_gap", 0.8, (p.recipe,))
        tin = v.ready_tokens_in or v.context_tokens
        svc = self._chat_service(p, tin, 0.8)
        if v.ready_kind == "chat":
            segs.append(("model.slots", now, now + min(H, svc), 1.0))
        elif v.in_request:
            start = now + (gap if gap is not None else 30.0)
            segs.append(("model.slots", start, start + min(H, svc), 1.0))
        if "sandbox.mem" in p.holds:
            until = now + (min(H, (gap or 0.0) + svc) if v.in_request else min(H, self._q(self.p_idle.get(p.sid), "I_gap", 0.5, (p.recipe,)) or H))
            segs.append(("sandbox.mem", now, until, p.sandbox_gb))
            segs.append(("sandbox.cpu", now, until, 1.0))
        t = now
        for k, r, q50, qt in self.revealed_q.get(p.sid, []):
            if r.startswith("ext.") or r.startswith("svc."):
                segs.append((r, t, t + min(H, qt), 1.0))
            t += q50
        kids = self.spawn_kids.get(p.sid)
        if kids:
            jw = self.join_wait.quantile((min(len(kids), 4),), 0.5) or 300.0
            segs.append(("model.slots", now, now + min(H, jw), 0.5 * len(kids)))
            segs.append(("sandbox.mem", now, now + min(H, jw), sum(g for _, g in kids)))
            segs.append(("sandbox.cpu", now, now + min(H, jw), float(len(kids))))
        self.forecast.set(p.sid, segs)

    # ---- budgets: forecast-aware pacing ------------------------------------------------------------------
    def _remaining_spend(self, sid: str, recipe: str) -> float:
        """Predicted spend to complete this session's current request: remaining chats x mean cost per chat (+ tools)."""
        v = self.observer.views.get(sid)
        n, tot = self.chats_per_req.get(recipe, (0, 0.0))
        per_req = tot / n if n else 8.0
        rem = max(1.0, per_req - (v.chats_in_request if v is not None else 0))
        jr = self.jev_recs.get((sid, "chat_end")) or self.jev_recs.get((sid, "chat_ready"))
        if jr is not None and self.judge.trusted("remaining_work") and jr.ok("remaining_work"):
            lvl = jr.score("remaining_work")                                 # levels: done, one, a few, many, long
            rem = float(np.interp(lvl, [0, 1, 2, 3, 4], [0.5, 1, 3, 7, 15]))
        nc, tc = self.cost_chat.get(recipe, (0, 0.0))
        nt, tt = self.cost_tool.get(recipe, (0, 0.0))
        return rem * ((tc / nc if nc else 0.0) + 1.3 * (tt / nt if nt else 0.0))

    def budget_admit(self, p: Program, step: Step, usd: float, tokens: float, now: float) -> bool:
        if "pacing" in self.off or self.loop is None:
            return ReactiveGate.budget_admit(self, p, step, usd, tokens, now)
        v = self.observer.views.get(p.sid)
        new_request = v is None or (v.chats_in_request == 0 and step.kind == "chat")
        for name, amt in (("usd", usd), ("tokens", tokens)):
            b = self.budgets.get(name)
            if b is None or amt <= 0:
                continue
            floor = self.loop.floor[name] * b.capacity
            level = b.remaining_frac(now) * b.capacity
            if new_request and name == "usd":                                # admit new work only if what is started can still finish
                p_ex = self._jev_p(p.sid, "chat_ready", "budget_will_exceed")
                if p_ex is not None and p_ex > 0.7 and level < 0.5 * b.capacity:
                    self.decisions["paced_by_judge"] = self.decisions.get("paced_by_judge", 0) + 1   # the judge says this request cannot finish: wait
                    return False
                inflight, horizon = 0.0, 0.0
                for q in self.observer.views.values():
                    if q.sid != p.sid and q.in_request and q.chats_in_request > 0 and not q.is_child:
                        inflight += self._remaining_spend(q.sid, q.recipe)
                        n, tot = self.req_time.get(q.recipe, (0, 0.0))
                        nc, tc = self.chats_per_req.get(q.recipe, (0, 0.0))
                        per_chat_s = (tot / n) / max(1.0, tc / nc) if n and nc else 20.0
                        horizon = max(horizon, per_chat_s * max(1.0, (tc / nc if nc else 8.0) - q.chats_in_request))
                refill = b.per_hour / 3600.0 * min(horizon, self.h)          # what arrives while the in-flight requests finish
                if level - amt - max(0.0, inflight - refill) < floor:
                    self.decisions["paced"] = self.decisions.get("paced", 0) + 1
                    return False
            elif level - amt < 0.5 * floor:                                  # in-flight work may dip to half the floor
                self.decisions["paced"] = self.decisions.get("paced", 0) + 1
                return False
        return True

    def budget_retry_after(self, p: Program, usd: float, tokens: float, now: float) -> float:
        b = self.budgets.get("usd")
        if b is None or "pacing" in self.off:
            return ReactiveGate.budget_retry_after(self, p, usd, tokens, now)
        deficit = max(usd, self.loop.floor["usd"] * b.capacity + usd - b.remaining_frac(now) * b.capacity)
        base = min(30.0, max(1.0, b.wait_for(now, deficit)))                # short, frequent re-checks: waiting sessions must not age into timeouts
        share = self._remaining_spend(p.sid, p.recipe) / max(1e-6, b.capacity)
        return base * (1.0 + min(1.0, share))                                # the more a session still needs, the later it retries

    # ---- ordering ----------------------------------------------------------------------------------
    def priority(self, p: Program, step: Step, ready_at: float, now: float) -> float:
        first = self.first_ready.get(p.sid, ready_at)                       # since the client submitted the step, retries included
        if self.aging and now - first > 0.5 * self.step_timeout:             # starvation guard (opt-in): oldest first past half the timeout
            self.decisions["aging_orders"] += 1
            return -1e12 + first
        model = self.resources["model.slots"]
        if step.kind == "chat" and "srpt" not in self.off and len(model.waiters) > self.srpt_depth * model.capacity:
            self.decisions["srpt_orders"] += 1
            return self._chat_service(p, step.tokens_in, 0.5)              # shortest predicted first while the queue is deep
        if "vtfq" in self.off:
            return ready_at
        self.decisions["vtfq_orders"] += 1
        return self.tags.get(p.sid, self.v_clock)

    # ---- KV ------------------------------------------------------------------------------------------
    def kv_reserve(self, p: Program, step: Step, default: int) -> int:
        """A predicted (smaller) reserve admits more chats — worth its eviction risk only while KV admission binds."""
        if "kv" in self.off or not self.resources["model.kv"].waiters:
            return default
        out = self._q(self.p_chat.get(p.sid), "Q_out", self.tau, (p.recipe,))
        if out is None:
            return default
        return int(min(4 * default, max(128, out)))

    def retention_ttl(self, p: Program, step: Step | None, now: float, gap: float | None, default: float) -> float:
        if "kv" in self.off:
            return default
        if step is not None:                                                 # mid-request: the predicted gap to the next chat
            q = self._q(self.p_end.get(p.sid), "T_gap", self.tau, (p.recipe,))
            return q + default if q is not None else default
        q = self._q(self.p_idle.get(p.sid), "I_gap", self.tau, (p.recipe,))
        return q + default if q is not None else default

    def kv_evict_key(self, sid: str, entry, now: float):
        if "kv" in self.off:
            return (entry.retained_until >= now, entry.last_access)
        return (entry.retained_until >= now, -entry.retained_until)

    # ---- idle sandboxes --------------------------------------------------------------------------------
    def idle_timeout(self, p: Program, think: float, default: float) -> float:
        if "park" in self.off:
            return default
        if "ev" in self.off:
            return 0.0                                                       # the shipped rule: park at once
        pr = self.p_idle.get(p.sid)
        q50 = self._q(pr, "I_gap", 0.5, (p.recipe,))
        if q50 is None:
            self.decisions["park"] += 1
            return 0.0                                                       # nothing known yet: the cheap safe default
        now = self.now
        cost = min(q50, self.h) * self._pressure("sandbox.mem", now, now + min(q50, self.h), now)
        p_safe = self._jev_p(p.sid, "request_end", "safe_to_park")            # judge: will the sandbox go unused past a cold start?
        if p_safe is None:
            p_safe = pr.p_true("I_cold")
        if cost > (1.0 - p_safe) * self.cold_start_s:                         # parking costs a cold start only if the idle is short
            self.decisions["park"] += 1
            return 0.0
        self.decisions["hold"] += 1
        return default

    def join_timeout(self, p: Program, default: float) -> float:
        if "park" in self.off:
            return default
        if "ev" in self.off:
            return 0.0
        now = self.now
        width = len(self.spawn_kids.get(p.sid, []))
        jw = self.join_wait.quantile((min(width, 4),), 0.5) or 300.0
        p_safe = self._jev_p(p.sid, "spawn", "safe_to_park")
        keep_cost = (1.0 - p_safe) * self.cold_start_s if p_safe is not None else self.cold_start_s
        return 0.0 if min(jw, self.h) * self._pressure("sandbox.mem", now, now + min(jw, self.h), now) > keep_cost else default

    def prewarm_delay(self, p: Program, think: float) -> float | None:
        if "prewarm" in self.off:
            return None
        pr = self.p_idle.get(p.sid)
        lo, hi = self._q(pr, "I_gap", 1.0 - self.tau, (p.recipe,)), self._q(pr, "I_gap", self.tau, (p.recipe,))
        if lo is None or hi is None or lo <= self.cold_start_s:
            return None
        hold = min(self.h, max(0.0, hi - lo))
        if hold <= 0:
            return None
        t0 = pr.rec.t
        if "ev" not in self.off:
            benefit = (2 * self.tau - 1) * self.cold_start_s                 # P(arrival inside the interval) x cold start saved
            cost = hold * self._pressure("sandbox.mem", t0 + lo - self.cold_start_s, t0 + lo + hold, t0)
            if benefit <= cost:
                self.decisions["prewarm_skipped"] += 1
                return None
        self.decisions["prewarm"] += 1
        return lo - self.cold_start_s

    def prewarm_hold(self, p: Program) -> float:
        pr = self.p_idle.get(p.sid)
        lo, hi = self._q(pr, "I_gap", 1.0 - self.tau, (p.recipe,)), self._q(pr, "I_gap", self.tau, (p.recipe,))
        return min(self.h, max(1.0, hi - lo)) if lo is not None and hi is not None else self.h

    # ---- leases ----------------------------------------------------------------------------------------
    def _fits(self, res: str, amt: float, start: float, expiry: float) -> float:
        """The largest part of `amt` that I1 still allows on `res` throughout [start, expiry) (0 if none)."""
        return max(0.0, min(amt, self.resources[res].capacity - self.ledger.active_max(res, start, expiry)))

    def leases(self, p: Program, step: Step | None, now: float) -> list:
        self._ensure()
        out: list[Lease] = []
        v = self.observer.views.get(p.sid)
        if v is None:
            return out
        if step is None:
            return out
        p_loop = self._jev_p(p.sid, "chat_end", "stuck_in_loop")
        if p_loop is not None and p_loop > 0.7:                              # a looping agent gets no reservations
            self.decisions["loop_no_lease"] = self.decisions.get("loop_no_lease", 0) + 1
            return out
        # (1) gang CPU for the revealed parallel group at the head of the revealed list (at chat end / at its submission)
        rq = self.revealed_q.get(p.sid, [])
        local = [(k, q50, qt) for k, r, q50, qt in rq if r.startswith("sandbox")]
        parallel = v.revealed_parallel if step.kind == "chat" else v.ready_parallel   # observable: declared by the response / the submission
        if "gang" not in self.off and parallel and len(local) >= 2:
            qs = [qt for _, _, qt in local[:6]]
            worth = min(self.h, sum(qs) - max(qs))
            extra = len(qs) - 1
            if worth > 0:
                pressure = self._pressure("sandbox.cpu", now, now + worth, now)
                issued, paid = self.decisions["gang"], self.decisions["gang_paid"]
                p_pay = (paid + 1.0) / (issued + 2.0)                        # learned: how often a reserved gang actually ran in parallel
                p_judge = self.jev_veto.get(p.sid)                            # the judge's verdict on this very candidate, if it arrived
                if p_judge is None:
                    jp = self._jev_p(p.sid, "chat_end", "will_use_reservation")
                    p_judge = jp
                if p_judge is not None:
                    p_pay = 0.5 * p_pay + 0.5 * p_judge
                if "ev" in self.off or p_pay > pressure * extra:              # E[gain] = p_pay x worth  >  others' loss = extra x worth x pressure
                    amt = self._fits("sandbox.cpu", float(extra + 1), now, now + worth)   # family holds include the session's own unit
                    if amt >= 2.0:
                        out.append(Lease(p.sid, "sandbox.cpu", amt, now, now + worth))
                        self.decisions["gang"] += 1
                        self.gang_open[p.sid] = True
                else:
                    self.decisions["gang_skipped"] += 1
        # (2) budget units for the revealed external calls (rung 3 slice 3, calibrated margin)
        if "budget" not in self.off:
            for name in sorted({r for _, r, _, _ in rq if self.resources[r].calls is not None}):
                key = f"{name}@rpm"
                if self.ledger.active_sum(key, now) + 1.0 <= self.resources[name].rpm:
                    out.append(Lease(p.sid, key, 1.0, now, now + self.h))
        # (3) the sub-agents of a spawn: k slots + their sandboxes, family-owned, while the pool is contended
        kids = self.spawn_kids.get(p.sid)
        if "spawn" not in self.off and kids and p.join_left > 0 and self.jev_veto.get(p.sid, 1.0) >= 0.3:
            model, mem, cpu = self.resources["model.slots"], self.resources["sandbox.mem"], self.resources["sandbox.cpu"]
            contended = len(model.waiters) > 0 or len(mem.waiters) > 0 or mem.used + sum(g for _, g in kids) > mem.capacity
            k = len(kids)
            window = min(self.h, 3.0 * self._chat_service(p, v.context_tokens, 0.8) + 30.0)
            pressure = max(self._pressure("model.slots", now, now + window, now), self._pressure("sandbox.mem", now, now + window, now))
            # the same rule as the parallel gang: the parent's gain (its join shortened by ~window) must exceed others' loss (k x window x pressure)
            if "ev" in self.off or (contended and pressure * k < 1.0):
                # the gang covers the children's *start* (first chat, first sandbox), not their lifetime: a child holds a model
                # slot only while it chats, so a lease that lasted the whole join would re-block others at every release
                for res, amt in (("model.slots", float(k)), ("sandbox.mem", sum(g for _, g in kids)), ("sandbox.cpu", float(k))):
                    fit = self._fits(res, amt, now, now + window)
                    if fit > 0 and not any(l.res == res for l in out):
                        out.append(Lease(p.sid, res, fit, now, now + window))
                self.decisions["spawn_gang"] += 1
            elif contended:
                self.decisions["spawn_gang_skipped"] = self.decisions.get("spawn_gang_skipped", 0) + 1
        # (4) a *predicted* spawn at chat submission: reserve from the predicted chat end
        elif "spawn" not in self.off and step.kind == "chat" and v.ready_kind == "chat" and p.parent is None:
            pr = self.p_chat.get(p.sid)
            if pr is not None:
                pg = pr.probs("G")
                p_spawn = float(1.0 - pg[0])
                if p_spawn > 0.5 and len(self.resources["model.slots"].waiters) > 0:
                    k = int(np.argmax(pg[1:]) + 1)
                    start = now + self._chat_service(p, step.tokens_in, 0.5)
                    window = min(self.h, 3.0 * self._chat_service(p, step.tokens_in, 0.8) + 30.0)
                    fit = self._fits("model.slots", float(k), start, start + window)
                    if fit > 0 and not any(l.res == "model.slots" for l in out):
                        out.append(Lease(p.sid, "model.slots", fit, start, start + window))
                        self.decisions["spawn_predicted"] += 1
        return out

    # ---- reporting ---------------------------------------------------------------------------------------
    def report(self) -> dict:
        rep = {"predictor": self.pred.report(), "decisions": dict(self.decisions), "records": self.observer.n_records,
               "ablate": sorted(self.off), "feedback": self.loop.report() if self.loop is not None else {}}
        if self.jev is not None:
            rep["jev"] = self.jev.report()
            rep["jev_judge"] = self.judge.report()                             # online ECE / accuracy per question vs realised outcomes
            rep["jev_untrusted"] = [q for q in self.cat if not self.judge.trusted(q)]
        return rep
