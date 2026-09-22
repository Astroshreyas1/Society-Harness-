"""The harness in wall-clock: one gateway in front of the society's shared resources, four admission policies.

Every DAG node's model call, every shell tool call and every gate's test run asks the gateway for its resource unit
(`acquire`) and hands it back with what it consumed (`release`). What differs between the policies is only what they
do with the same questions; the world (world.py) and the observation stream are identical:

  off      nobody coordinates: the agents call providers directly. An api-like tier (the cloud model) answers 429 when
           its concurrency, request or token rate is exceeded and the worker's SDK retries with backoff, then gives up
           (a burned attempt); OS-like pools (the local model, the sandboxes) queue first-come; the platform bills at
           the end of a call and refuses what the tenant cannot pay (a burned attempt, the money spent anyway).
  gate     the reactive gate of the original project's `local_slots` semaphore, generalised: everything queues at the
           gateway in submission order, nothing is issued that the tier's headers say would 429, a tenant's calls pause
           while its balance is under a fixed floor. No prediction.
  needs    the Reserver of agentsim/reserver.py on the society's service graph. Per (role, provider, retry) quantiles of
           duration, tokens and cost learned from the observation stream (warm-started from the deployment's own
           lineage log) give each in-flight run a predicted schedule of its remaining nodes; the schedules sum to a
           demand forecast per resource (`Forecast`), whose excess over capacity is the *pressure* that prices every
           reservation. Rules, all from the simulator: a lease on the downstream tier for the window a run is predicted
           to need it, issued when the queueing it saves exceeds the hold it costs others; a gang lease co-scheduling
           the parallel branches of a run; shortest-predicted-first on a model queue deeper than its slots, weighted
           virtual-time fair queuing otherwise; budget pacing that reserves the predicted spend-to-completion of every
           in-flight run so a run is started only when it can be finished ("finish what you started"); the feedback
           loop (`AllocationLoop`) correcting the forecast's bias per resource, tightening the pacing floor after a
           refusal, and weighting tenants by their served share; optionally System One (jev.py) as an annotator whose
           answers are used only while the online judge finds them calibrated.
  oracle   the same rules with the sampled futures read through `ReplayTable.peek` — the clairvoyant bound.

Honesty: `needs` sees only `on_event` facts, the gateway's counters, the ledger and the lineage history; the
`peek` guard raises if anyone but the oracle reads a future (tests/test_harness.py).
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np

from agentsim.control import AllocationLoop
from agentsim.predict import QuantileTracker
from agentsim.reserver import Forecast
from agentsim.resources import Ledger, Lease, Resource
from society.providers.replay import Draw, ProviderError, ReplayTable, cost_usd

from .world import World

POLICIES = ("off", "gate", "needs", "oracle")
GATE_BUDGET_FLOOR = 0.10           # the reactive gate's fixed pause: a tenant's calls wait under this fraction of its budget
MIN_CHARGE_USD = 0.02              # what the platform refuses a call under when nobody coordinates (one prompt's worth)
SDK_RETRIES = 2                    # the client SDK's default retries on 429 / overloaded
MAX_WAIT_S = 900.0                 # an agent framework gives up on a call queued this long (society seconds)


@dataclass
class NodeRef:
    id: str
    agent: str
    res: str                       # the model-tier resource its provider maps to
    deps: list[str]
    paid: bool                     # costs money (cloud tier)


@dataclass
class RunState:
    run_id: str
    tenant: str
    nodes: dict[str, NodeRef]
    started: float
    done: set[str] = field(default_factory=set)
    attempts: dict[str, int] = field(default_factory=dict)
    spent: float = 0.0
    inflight: set[str] = field(default_factory=set)
    judging: set[str] = field(default_factory=set)        # released, gates still running: scheduled as done until the verdict says otherwise
    status: str = "running"
    gang_open: bool = False
    service: dict[str, float] = field(default_factory=dict)   # node -> last attempt's slot time (for the judge's labels)


@dataclass
class Grant:
    sid: str
    res: str
    amt: float
    kind: str
    t_ready: float
    t_start: float
    run_id: str
    node_id: str
    agent: str
    attempt: int
    tenant: str
    est_tokens: float = 0.0
    est_usd: float = 0.0
    prepaid: float = 0.0


@dataclass
class Waiter:
    grant: Grant
    fut: asyncio.Future
    est_service: float = 0.0
    tag: float = 0.0


# ---- prediction --------------------------------------------------------------------------------------------
class Brain:
    """What `needs` knows: per (role, provider, retry) quantiles learned from the observation stream."""

    tau = 0.8

    def __init__(self, warm: list[dict] | None = None, tau: float = 0.8):
        if not 0.5 <= tau < 1.0:
            raise ValueError("tau must be in [0.5, 1)")
        self.tau = tau
        self.dur = QuantileTracker(min_n=4)
        self.usd = QuantileTracker(min_n=4)
        self.tok = QuantileTracker(min_n=4)
        self.tool = QuantileTracker(min_n=4)
        self.rej: dict[tuple, tuple[int, int]] = {}                   # (agent, provider) -> (n, rejected)
        for r in warm or []:
            self.learn_usage(r["agent"], r["provider"], int(r.get("attempt", 1)), r["duration_ms"] / 1000.0, float(r["cost_usd"]), int(r["input_tokens"]))
            self.learn_verdict(r["agent"], r["provider"], r["verdict"] == "rejected")

    @staticmethod
    def key(agent: str, provider: str, attempt: int) -> tuple:
        return (agent, provider, "retry" if attempt > 1 else "first")

    def learn_usage(self, agent: str, provider: str, attempt: int, dur: float, usd: float, tokens_in: int) -> None:
        k = self.key(agent, provider, attempt)
        self.dur.observe(k, max(0.0, dur))
        self.usd.observe(k, max(0.0, usd))
        self.tok.observe(k, max(0.0, float(tokens_in)))

    def learn_verdict(self, agent: str, provider: str, rejected: bool) -> None:
        n, r = self.rej.get((agent, provider), (0, 0))
        self.rej[(agent, provider)] = (n + 1, r + int(rejected))

    def service(self, agent: str, provider: str, attempt: int) -> tuple[float, float]:
        k = self.key(agent, provider, attempt)
        q50 = self.dur.quantile(k, 0.5)
        qt = self.dur.quantile(k, self.tau)
        return (q50 if q50 is not None else 60.0, qt if qt is not None else 120.0)

    def cost(self, agent: str, provider: str, attempt: int) -> tuple[float, float]:
        k = self.key(agent, provider, attempt)
        q50, qt = self.usd.quantile(k, 0.5), self.usd.quantile(k, self.tau)
        return (q50 if q50 is not None else 0.0, qt if qt is not None else 0.0)

    def tokens(self, agent: str, provider: str, attempt: int) -> float:
        q = self.tok.quantile(self.key(agent, provider, attempt), 0.5)
        return q if q is not None else 0.0

    def p_reject(self, agent: str, provider: str) -> float:
        n, r = self.rej.get((agent, provider), (0, 0))
        return (r + 0.5) / (n + 1.0)                                    # Laplace-smoothed

    def expected_attempts(self, agent: str, provider: str, max_attempts: int = 3) -> float:
        p = self.p_reject(agent, provider)
        return float(sum(p ** i for i in range(max_attempts)))          # 1 + p + p^2 ... (capped by the retry policy)

    def tool_s(self, agent: str) -> float:
        q = self.tool.quantile((agent,), 0.5)
        return q if q is not None else 10.0


class OracleBrain(Brain):
    """The clairvoyant view: the sampled future of every attempt, read through the table's peek."""

    def __init__(self, table: ReplayTable, seed: int, providers: dict[str, str], tau: float = 0.8):
        super().__init__(None, tau)
        self.table, self.seed, self.providers = table, seed, providers
        self.max_attempts = 3

    def _draw(self, run_id: str, node_id: str, agent: str, attempt: int) -> Draw:
        return self.table.peek(self.seed, run_id, node_id, attempt, agent, self.providers[agent])

    def true_service(self, run_id: str, node_id: str, agent: str, attempt: int) -> float:
        return self._draw(run_id, node_id, agent, attempt).duration_s

    def true_cost(self, run_id: str, node_id: str, agent: str, attempt: int) -> float:
        d = self._draw(run_id, node_id, agent, attempt)
        return cost_usd(self.providers[agent], d.tokens_in, d.tokens_out)

    def true_remaining(self, run_id: str, node_id: str, agent: str, attempt: int) -> tuple[float, float, int]:
        """(service seconds, usd, attempts) this node still takes from `attempt` on, faults included."""
        s = u = 0.0
        for a in range(attempt, self.max_attempts + 1):
            d = self._draw(run_id, node_id, agent, a)
            s += d.duration_s
            u += cost_usd(self.providers[agent], d.tokens_in, d.tokens_out)
            if d.fault is None:
                return s, u, a - attempt + 1
        return s, u, self.max_attempts - attempt + 1


# ---- System One as the annotator (optional) --------------------------------------------------------------------
def harness_catalogue() -> dict:
    from agentsim.jev import Question

    Q = [
        Question("will_be_rejected", "noul",
                 "Will this node attempt be rejected by a quality gate? `role`, `provider`, `attempt` and `prior_rejections` describe it; "
                 "`history` lists how attempts of this role on this provider ended recently.",
                 ("no", "yes"), "yes when the attempt will come back in a shape a gate rejects", ("node_ready",), "rejected", "feature"),
        Question("service_class", "score",
                 "How long will this node attempt hold its model slot? `role`, `provider`, `attempt` and `history` describe it.",
                 ("under a minute", "one to two minutes", "several minutes", "very long"), None, ("node_ready",), "service", "feature"),
    ]
    return {q.name: q for q in Q}


SERVICE_EDGES = (60.0, 120.0, 600.0)


def service_class(seconds: float) -> str:
    opts = tuple(harness_catalogue()["service_class"].options)
    for i, edge in enumerate(SERVICE_EDGES):
        if seconds < edge:
            return opts[i]
    return opts[-1]


class SocietyJev:
    """Asks System One at every node ready, judges every answer against the realised outcome, and hands the brain a
    correction only for questions the judge trusts (agentsim.jev.Judge: rolling ECE under the kill rule)."""

    def __init__(self, model, cat: dict):
        from agentsim.jev import Judge

        self.model, self.cat = model, cat
        self.judge = Judge(cat, window=200, min_n=20)
        self.pending: dict[str, Any] = {}
        self.asked = self.used = 0

    def ask(self, sid: str, state: dict, now: float) -> None:
        try:
            rec = self.model.decide(state, ("will_be_rejected", "service_class"), now)
        except Exception:  # noqa: BLE001 — an unreachable model is a missing feature, never a failed admission
            return
        self.asked += 1
        self.pending[sid] = rec

    def p_reject(self, sid: str) -> float | None:
        rec = self.pending.get(sid)
        if rec is None or not self.judge.trusted("will_be_rejected"):
            return None
        self.used += 1
        return rec.noul("will_be_rejected")

    def settle(self, sid: str, rejected: bool, service_s: float) -> None:
        rec = self.pending.pop(sid, None)
        if rec is None:
            return
        self.judge.observe(rec, "will_be_rejected", "yes" if rejected else "no")
        self.judge.observe(rec, "service_class", service_class(service_s))

    def report(self) -> dict:
        return {"asked": self.asked, "used": self.used, "judge": self.judge.report()}


# ---- the gateway ----------------------------------------------------------------------------------------------
class Gateway:
    def __init__(self, world: World, policy: str = "needs", *, brain: Brain | None = None, h: float = 600.0,
                 jev: SocietyJev | None = None, max_wait_s: float = MAX_WAIT_S, providers: dict[str, str] | None = None):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.world, self.policy, self.h = world, policy, float(h)
        self.res = world.resources
        self.brain = brain
        self.jev = jev if policy == "needs" else None
        self.max_wait = float(max_wait_s)
        self.providers = providers or {}
        self.ledger = Ledger()
        self.forecast = Forecast(self.res, dt=5.0, H=self.h) if policy in ("needs", "oracle") else None
        self.loop = AllocationLoop(self.res, world.budgets, tick_s=30.0, enabled=policy in ("needs", "oracle"))
        self.waiters: dict[str, list[Waiter]] = {name: [] for name in self.res}
        self.runs: dict[str, RunState] = {}
        self.tags: dict[str, float] = {}
        self.v_clock = 0.0
        self.decisions = {"leases": 0, "leases_skipped": 0, "gang": 0, "gang_skipped": 0, "srpt_orders": 0, "vtfq_orders": 0,
                          "budget_waits": 0, "refused_429": 0, "refused_budget": 0, "timeouts": 0, "run_deferred": 0}
        self.events: list[tuple[str, float, dict]] = []
        self.observers: list[Callable[[str, float, dict], None]] = []
        self._ticker: asyncio.Task | None = None
        self._last_tick = 0.0
        self.closed = False
        if policy in ("needs", "oracle") and brain is None:
            raise ValueError(f"policy {policy!r} needs a Brain")

    # ---- lifecycle --------------------------------------------------------------------------------------------
    def now(self) -> float:
        return self.world.now()

    def start(self) -> None:
        if self._ticker is None:
            self._ticker = asyncio.get_running_loop().create_task(self._tick_forever())

    async def close(self) -> None:
        self.closed = True
        if self._ticker is not None:
            self._ticker.cancel()
            try:
                await self._ticker
            except asyncio.CancelledError:
                pass
            self._ticker = None

    async def _tick_forever(self) -> None:
        period = max(0.005, 0.5 / self.world.clock.scale)                  # every half a society second
        while not self.closed:
            await asyncio.sleep(period)
            now = self.now()
            self.ledger.expire(now)
            if now - self._last_tick >= self.loop.tick:
                if self.forecast is not None:
                    k = max(1, int(self.loop.tick // self.forecast.dt))
                    for name in self.res:
                        self.forecast.bias = getattr(self.forecast, "bias", {})
                        self.loop.set_prediction(name, float(self.forecast.demand(name, now)[:k].mean()))
                self.loop.tick_now(now)
                if self.forecast is not None:
                    self.forecast.bias = {n: p.x for n, p in self.loop.scale.items()}
                    self.forecast._sum_t = -1.0
                self.loop.fairness_update(now)
                self._last_tick = now
            self._dispatch_all(now)

    def _emit(self, event: str, now: float, **info: Any) -> None:
        if len(self.events) < 200_000:
            self.events.append((event, now, info))
        for cb in self.observers:
            cb(event, now, info)

    # ---- the service graph ------------------------------------------------------------------------------------
    def run_start(self, run_id: str, tenant: str, nodes: list[NodeRef]) -> None:
        now = self.now()
        self.runs[run_id] = RunState(run_id, tenant, {n.id: n for n in nodes}, now)
        if self.forecast is not None:
            self.forecast.saw_arrival(now)
            self._refresh(self.runs[run_id], now)
            self._gang(self.runs[run_id], now)
        self._emit("run_start", now, run_id=run_id, tenant=tenant, nodes=[n.id for n in nodes])

    async def admit_run(self, run_id: str, tenant: str, nodes: list[NodeRef]) -> float:
        """Budget pacing at the workflow level (needs / oracle): a run starts only when the tenant's balance, net of what
        the runs already in flight are predicted to still spend and plus the refill over this run's predicted makespan,
        covers this run's predicted spend-to-completion above the loop's floor. Returns the seconds it was deferred."""
        b = self.world.budgets.get(tenant)
        if self.policy not in ("needs", "oracle") or b is None or not any(n.paid for n in nodes):
            return 0.0
        probe = RunState(run_id, tenant, {n.id: n for n in nodes}, self.now())
        deferred = 0.0
        first = True
        while True:
            now = self.now()
            reserved = sum(self._remaining_spend(o, now, tau=False) for o in self.runs.values()
                           if o.tenant == tenant and o.status == "running" and (o.inflight or o.attempts))
            need = self._remaining_spend(probe, now, tau=False)
            horizon = max((e for _, e, _ in self._schedule(probe, now).values()), default=now) - now
            floor = self.loop.floor.get(tenant, GATE_BUDGET_FLOOR) * b.capacity
            level = b.remaining_frac(now) * b.capacity
            if level - reserved + b.per_hour * horizon / 3600.0 >= need + floor:
                return deferred
            if first:
                self.decisions["run_deferred"] += 1
                first = False
            self.decisions["budget_waits"] += 1
            await self.world.clock.sleep(5.0)
            deferred += 5.0

    def run_end(self, run_id: str, status: str) -> None:
        now = self.now()
        run = self.runs.get(run_id)
        if run is None:
            return
        run.status = status
        if self.forecast is not None:
            self.forecast.drop(run_id)
            self.ledger.set_session(run_id, [], now)
        self.tags.pop(run_id, None)
        self._emit("run_end", now, run_id=run_id, status=status, spent=run.spent, duration=now - run.started)
        self._dispatch_all(now)

    def _schedule(self, run: RunState, now: float) -> dict[str, tuple[float, float, float]]:
        """Predicted (start, q50 end, q_tau end) per remaining node: a critical-path schedule from the predicted durations."""
        order = list(run.nodes)
        out: dict[str, tuple[float, float, float]] = {}
        for _ in range(len(order)):
            for nid in order:
                if nid in out or nid in run.done or nid in run.judging:
                    continue
                if any(d not in run.done and d not in run.judging and d not in out for d in run.nodes[nid].deps):
                    continue
                n = run.nodes[nid]
                attempt = run.attempts.get(nid, 0) + 1
                q50, qt = self._service_of(run, n, attempt)
                if nid in run.inflight:
                    start = now
                else:
                    start = max([now] + [out[d][1] for d in n.deps if d in out])
                out[nid] = (start, start + q50, start + qt)
        return out

    def _service_of(self, run: RunState, n: NodeRef, attempt: int) -> tuple[float, float]:
        if isinstance(self.brain, OracleBrain):
            s, _, _ = self.brain.true_remaining(run.run_id, n.id, n.agent, attempt)
            return s, s
        assert self.brain is not None
        q50, qt = self.brain.service(n.agent, self.providers.get(n.agent, ""), attempt)
        ea = self.brain.expected_attempts(n.agent, self.providers.get(n.agent, ""))
        return q50 * ea, qt * ea

    def _cost_of(self, run: RunState, n: NodeRef, attempt: int) -> tuple[float, float]:
        if not n.paid:
            return 0.0, 0.0
        if isinstance(self.brain, OracleBrain):
            _, u, _ = self.brain.true_remaining(run.run_id, n.id, n.agent, attempt)
            return u, u
        assert self.brain is not None
        prov = self.providers.get(n.agent, "")
        q50, qt = self.brain.cost(n.agent, prov, attempt)
        ea = self.brain.expected_attempts(n.agent, prov)
        pj = self.jev.p_reject(f"{run.run_id}/{n.id}") if self.jev is not None else None
        if pj is not None:                                                 # a trusted System One answer replaces the base rate for this attempt
            ea = 1.0 + pj * (ea - 1.0) / max(1e-6, self.brain.p_reject(n.agent, prov))
        return q50 * ea, qt * ea

    def _refresh(self, run: RunState, now: float) -> None:
        """Re-issue the run's forecast segments from its predicted schedule."""
        if self.forecast is None:
            return
        segs = []
        for nid, (s, e50, et) in self._schedule(run, now).items():          # the mean demand (q50): the loop corrects its bias; q_tau sizes the leases
            n = run.nodes[nid]
            segs.append((n.res, s, e50, 1.0))
            if n.agent in ("backend", "integration", "ship"):              # these roles run the tests: a sandbox unit near the end
                t = self.brain.tool_s(n.agent) if self.brain is not None else 10.0
                segs.append(("sandbox.cpu", max(s, e50 - t), e50 + t, 1.0))
        self.forecast.set(run.run_id, segs)

    # ---- leases -------------------------------------------------------------------------------------------------
    def _pressure(self, res: str, t0: float, t1: float, now: float) -> float:
        assert self.forecast is not None
        return self.forecast.pressure(res, t0, t1, now)

    def _issue(self, run: RunState, leases: list[Lease], now: float) -> None:
        keep = []
        for lease in leases:
            cap = Ledger.limit(self.res, lease.res)
            others = self.ledger.active_max(lease.res, lease.start, lease.expiry) - (self.ledger.own(lease.res, run.run_id, now) if lease.start <= now else 0.0)
            if others + lease.amt <= cap + 1e-9:
                keep.append(lease)
        self.ledger.set_session(run.run_id, keep, now)

    def _downstream_leases(self, run: RunState, now: float) -> None:
        """After a node ends: reserve the next tier for the window the run is predicted to need it, if worth it (EV gate)."""
        if self.forecast is None:
            return
        sched = self._schedule(run, now)
        by_res: dict[str, list[tuple[float, float]]] = {}
        for nid, (s, e50, et) in sched.items():
            n = run.nodes[nid]
            if nid in run.inflight or s <= now + 1e-9:                     # ready now or running: a waiter, not a reservation
                continue
            by_res.setdefault(n.res, []).append((s, et))
        leases = []
        for res, windows in by_res.items():
            t0 = min(w[0] for w in windows)
            t1 = min(now + self.h, max(w[1] for w in windows))
            if t1 <= t0:
                continue
            r = self.res[res]
            pressure = self._pressure(res, t0, t1, now)
            mean_service = float(np.mean([e - s for s, e in windows]))
            benefit = pressure * mean_service                              # the queueing this run would otherwise face
            cost = (t1 - t0) * pressure / max(1.0, r.capacity)              # the hold others lose, priced by the same pressure
            if pressure >= 0.5 and benefit > 2.0 * cost:                    # only a clearly saturated tier is worth holding
                leases.append(Lease(run.run_id, res, 1.0, t0, t1))
                self.decisions["leases"] += 1
            else:
                self.decisions["leases_skipped"] += 1
        self._issue(run, leases, now)

    def _gang(self, run: RunState, now: float) -> None:
        """A run whose first nodes are parallel branches on the same tier gets them co-scheduled: k slots for their start window."""
        if self.forecast is None:
            return
        roots = [n for n in run.nodes.values() if not n.deps]
        by_res: dict[str, list[NodeRef]] = {}
        for n in roots:
            by_res.setdefault(n.res, []).append(n)
        for res, ns in by_res.items():
            k = len(ns)
            if k < 2:
                continue
            q50 = max(self._service_of(run, n, 1)[0] for n in ns)
            window = min(self.h, 3 * q50 + 30.0)
            pressure = self._pressure(res, now, now + window, now)
            if pressure * k < 1.0 and self.ledger.active_max(res, now, now + window) + k <= self.res[res].capacity + 1e-9:
                self.ledger.set_session(run.run_id, [Lease(run.run_id, res, float(k), now, now + window)], now)
                run.gang_open = True
                self.decisions["gang"] += 1
            else:
                self.decisions["gang_skipped"] += 1

    # ---- admission ------------------------------------------------------------------------------------------------
    async def acquire(self, sid: str, res: str, kind: str, *, run_id: str, node_id: str, agent: str, attempt: int, tenant: str,
                      amt: float = 1.0) -> Grant:
        if res not in self.res:
            raise KeyError(f"unknown resource {res!r}")
        now = self.now()
        run = self.runs.get(run_id)
        g = Grant(sid, res, amt, kind, now, now, run_id, node_id, agent, attempt, tenant)
        if kind == "chat" and self.brain is not None:
            prov = self.providers.get(agent, "")
            if isinstance(self.brain, OracleBrain):
                g.est_tokens = float(self.brain._draw(run_id, node_id, agent, attempt).tokens_in)
                g.est_usd = self.brain.true_cost(run_id, node_id, agent, attempt)
            else:
                g.est_tokens = self.brain.tokens(agent, prov, attempt)
                g.est_usd = self.brain.cost(agent, prov, attempt)[1]
        if run is not None and kind == "chat":
            run.attempts[node_id] = attempt
            if self.jev is not None:
                self.jev.ask(sid, self._jev_state(run, node_id, agent, attempt), now)
        r = self.res[res]
        if self.policy == "off" and r.api_like:                             # the provider answers now: a 429 or the slot
            why = self._provider_refusal(r, g, now)
            if why is not None:
                self.decisions["refused_429" if why.kind != "budget" else "refused_budget"] += 1
                self._emit("refused", now, sid=sid, res=res, kind=why.kind)
                raise why
        w = Waiter(g, asyncio.get_running_loop().create_future(), est_service=self._est_service(g))
        self.waiters[res].append(w)
        self._emit("step_ready", now, sid=sid, res=res, kind=kind, run_id=run_id, node_id=node_id, agent=agent, attempt=attempt, tenant=tenant)
        self._dispatch(res, now)
        try:
            await asyncio.wait_for(w.fut, timeout=self.max_wait / self.world.clock.scale)
        except asyncio.TimeoutError:
            self._remove(w)
            self.decisions["timeouts"] += 1
            self._emit("timeout", self.now(), sid=sid, res=res)
            raise ProviderError("timeout", 0.0, f"queued {self.max_wait:.0f} s on {res}") from None
        except asyncio.CancelledError:
            self._remove(w)
            raise
        return w.fut.result()

    def _remove(self, w: Waiter) -> None:
        try:
            self.waiters[w.grant.res].remove(w)
        except ValueError:
            pass

    def _provider_refusal(self, r: Resource, g: Grant, now: float) -> ProviderError | None:
        if r.used + g.amt > r.capacity + 1e-9:
            return ProviderError("overloaded", 2.0, f"{r.name} at capacity")
        if not r.can_call(now, 1.0):
            return ProviderError("rate_limit", r.budget_wait(now), f"{r.name} requests per minute exceeded")
        tpm = self.world.tpm.get(r.name)
        if tpm is not None:
            tpm._refill(now)
            if tpm.level <= 0:
                return ProviderError("rate_limit", tpm.wait_for(1.0, now, 1.0), f"{r.name} tokens per minute exceeded")
        b = self.world.budgets.get(g.tenant)
        if b is not None and g.kind == "chat" and r.name in self.world.prices and not b.can_pay(now, MIN_CHARGE_USD):
            b.refused += 1
            return ProviderError("budget", 0.0, f"{g.tenant} has insufficient credits")
        return None

    def _est_service(self, g: Grant) -> float:
        if g.kind != "chat" or self.brain is None:
            return 0.0
        if isinstance(self.brain, OracleBrain):
            return self.brain.true_service(g.run_id, g.node_id, g.agent, g.attempt)
        return self.brain.service(g.agent, self.providers.get(g.agent, ""), g.attempt)[0]

    def _jev_state(self, run: RunState, node_id: str, agent: str, attempt: int) -> dict:
        prov = self.providers.get(agent, "")
        hist = [e[2] for e in self.events[-400:] if e[0] == "step_end" and e[2].get("agent") == agent and e[2].get("kind") == "chat"]
        return {"dp": "node_ready", "role": agent, "provider": prov, "attempt": attempt,
                "prior_rejections": [e.get("reject_gate") for e in hist[-40:] if e.get("run_id") == run.run_id and e.get("node_id") == node_id and e.get("rejected")],
                "history": [{"attempt": e.get("attempt"), "rejected": e.get("rejected"), "service_s": round(e.get("duration", 0.0))} for e in hist[-8:]],
                "queue_depth": len(self.waiters[run.nodes[node_id].res]) if node_id in run.nodes else 0}

    # ---- ordering ---------------------------------------------------------------------------------------------------
    def _order(self, res: str, now: float) -> list[Waiter]:
        ws = self.waiters[res]
        r = self.res[res]
        if self.policy in ("off", "gate") or self.brain is None:
            return sorted(ws, key=lambda w: w.grant.t_ready)                                            # FIFO
        own = [w for w in ws if self.ledger.own(res, w.grant.run_id, now) > 0]                           # a live reservation first
        rest = [w for w in ws if w not in own]
        if r.tier == "model" and len(rest) > r.capacity:                                                 # shortest predicted first while the queue is deeper than the slots
            self.decisions["srpt_orders"] += 1
            rest.sort(key=lambda w: (w.est_service, w.grant.t_ready))
        else:
            self.decisions["vtfq_orders"] += 1
            for w in rest:
                if w.tag == 0.0:
                    start = max(self.tags.get(w.grant.run_id, self.v_clock), self.v_clock)
                    w.tag = start + max(1.0, w.est_service) * self.loop.weight(w.grant.tenant)
            rest.sort(key=lambda w: (w.tag, w.grant.t_ready))
        own.sort(key=lambda w: w.grant.t_ready)
        return own + rest

    def _fits(self, w: Waiter, now: float) -> tuple[bool, str]:
        g, r = w.grant, self.res[w.grant.res]
        if self.policy in ("needs", "oracle"):
            reserved = self.ledger.reserved_for_others(r.name, g.run_id, now, r.holders)
        else:
            reserved = 0.0
        if r.used + g.amt > r.capacity - reserved + 1e-9:
            return False, "capacity"
        if g.kind != "chat":
            return True, ""
        if not r.can_call(now, 1.0):
            return False, "rpm"
        tpm = self.world.tpm.get(r.name)
        if tpm is not None:
            tpm._refill(now)
            need = g.est_tokens if self.policy in ("needs", "oracle") else (0.1 * tpm.tpm if self.policy == "gate" else 0.0)
            if tpm.level < need or tpm.level <= 0:
                return False, "tpm"
        b = self.world.budgets.get(g.tenant)
        if b is not None and r.name in self.world.prices:
            ok = self._budget_ok(b, g, now)
            if not ok:
                return False, "budget"
        return True, ""

    def _budget_ok(self, b, g: Grant, now: float) -> bool:
        if self.policy == "off":
            if not b.can_pay(now, MIN_CHARGE_USD):
                b.refused += 1
                return True                                                # the platform's refusal lands at grant time (below)
            return True
        if self.policy == "gate":
            return b.remaining_frac(now) >= GATE_BUDGET_FLOOR
        # needs / oracle: the run's spend-to-completion was covered when it was admitted (admit_run); a node waits only
        # while the balance cannot pay its own attempt — never long enough to burn it (the refill is on its way)
        run = self.runs.get(g.run_id)
        this = g.est_usd if run is None else self._cost_of(run, run.nodes[g.node_id], g.attempt)[1]
        refill = b.per_hour * self._est_service(g) / 3600.0
        level = b.remaining_frac(now) * b.capacity
        ok = level + refill >= this
        if not ok:
            self.decisions["budget_waits"] += 1
        return ok

    def _remaining_spend(self, run: RunState, now: float, tau: bool = True) -> float:
        """Predicted USD the run still spends: q_tau (tau=True) or q50 per remaining node, times the expected attempts."""
        total = 0.0
        for nid, n in run.nodes.items():
            if nid in run.done or nid in run.judging:
                continue
            attempt = run.attempts.get(nid, 0) + (0 if nid in run.inflight else 1)
            total += self._cost_of(run, n, max(1, attempt))[1 if tau else 0]
        return total

    def _dispatch_all(self, now: float) -> None:
        for res in list(self.res):
            self._dispatch(res, now)

    def _dispatch(self, res: str, now: float) -> None:
        if not self.waiters[res]:
            return
        for w in self._order(res, now):
            if w.fut.done():
                self._remove(w)
                continue
            ok, why = self._fits(w, now)
            if not ok:
                if why == "capacity":
                    break
                continue
            self._grant(w, now)

    def _grant(self, w: Waiter, now: float) -> None:
        g, r = w.grant, self.res[w.grant.res]
        self._remove(w)
        r.take(g.sid, g.amt)
        g.t_start = now
        if g.kind == "chat":
            r.take_call(now, 0.0)
            tpm = self.world.tpm.get(r.name)
            if tpm is not None and g.est_tokens > 0:
                tpm.try_take(g.est_tokens, now, 1.0)
            b = self.world.budgets.get(g.tenant)
            if b is not None and r.name in self.world.prices and self.policy in ("needs", "oracle") and g.est_usd > 0:
                b._refill(now)
                g.prepaid = max(0.0, min(g.est_usd, b.level))              # reserved for this attempt; settled at release
                b.pay(now, g.prepaid)
        if self.policy in ("needs", "oracle") and self.ledger.own(r.name, g.run_id, now) > 0:
            run = self.runs.get(g.run_id)
            if run is not None and run.gang_open:
                run.gang_open = False
            self.ledger.consume(g.run_id, r.name, now)
        run = self.runs.get(g.run_id)
        if run is not None and g.kind == "chat":
            run.inflight.add(g.node_id)
            if self.forecast is not None:
                self._refresh(run, now)
        self._emit("step_start", now, sid=g.sid, res=r.name, kind=g.kind, wait=now - g.t_ready, run_id=g.run_id, node_id=g.node_id,
                   agent=g.agent, attempt=g.attempt, tenant=g.tenant, est_service=w.est_service)
        w.fut.set_result(g)

    # ---- release ----------------------------------------------------------------------------------------------------
    def release(self, g: Grant, *, tokens_in: int = 0, tokens_out: int = 0, outcome: str = "ok", rejected: bool = False,
                reject_gate: str | None = None) -> float:
        """Hand the unit back; bills the tenant for a chat. Returns the USD charged. Raises ProviderError("budget") when
        the platform could not collect the whole amount (the work is lost; the money that was there is gone)."""
        now = self.now()
        r = self.res[g.res]
        duration = now - g.t_start
        r.give(g.sid, g.amt)
        usd = 0.0
        run = self.runs.get(g.run_id)
        budget_failed = False
        if g.kind == "chat":
            price = self.world.prices.get(r.name)
            tpm = self.world.tpm.get(r.name)
            if tpm is not None:
                tpm.debit(max(0.0, tokens_in - g.est_tokens), now)         # what the reservation did not cover
            if price is not None:
                usd = (tokens_in * price[0] + tokens_out * price[1]) / 1e6
                b = self.world.budgets.get(g.tenant)
                if b is not None and usd > 0:
                    due = usd - g.prepaid
                    if due <= 0:                                            # refund what the reservation did not use
                        b._refill(now)
                        b.level = min(b.capacity, b.level - due)
                        b.spent += due
                    elif b.can_pay(now, due):
                        b.pay(now, due)
                    else:
                        b._refill(now)
                        paid = max(0.0, b.level)
                        b.pay(now, paid)
                        b.refused += 1
                        budget_failed = True
                        outcome = "budget"
            if run is not None:
                run.spent += usd
                run.inflight.discard(g.node_id)
                run.service[g.node_id] = duration
                if outcome == "ok" and not rejected:
                    run.judging.add(g.node_id)                              # the gates decide; until then it is scheduled as done
            self.loop.served_add(g.tenant, duration, now)
            if self.brain is not None and not isinstance(self.brain, OracleBrain) and outcome in ("ok", "budget"):
                self.brain.learn_usage(g.agent, self.providers.get(g.agent, ""), g.attempt, duration, usd, tokens_in)
            if (rejected or outcome != "ok") and run is not None:          # a worker-level failure is a verdict already
                self.note_verdict(g.run_id, g.node_id, False, gate="no_error" if outcome == "ok" else outcome)
        elif self.brain is not None and not isinstance(self.brain, OracleBrain):
            self.brain.tool.observe((g.agent,), duration)
        if self.forecast is not None and g.kind == "chat" and run is not None:
            self.forecast.saw_hold(r.name, duration)
            self._refresh(run, now)
            if run.status == "running":
                self._downstream_leases(run, now)
        self._emit("step_end", now, sid=g.sid, res=r.name, kind=g.kind, duration=duration, tokens_in=tokens_in, tokens_out=tokens_out, usd=usd,
                   outcome=outcome, rejected=rejected, reject_gate=reject_gate, run_id=g.run_id, node_id=g.node_id, agent=g.agent,
                   attempt=g.attempt, tenant=g.tenant)
        self._dispatch_all(now)
        if budget_failed:
            self.decisions["refused_budget"] += 1
            raise ProviderError("budget", 0.0, f"{g.tenant} could not pay ${usd:.3f} for {g.sid}")
        return usd

    def note_verdict(self, run_id: str, node_id: str, approved: bool, gate: str | None = None) -> None:
        """The executor's verdict on an attempt: a rejected node repeats (the predicted schedule moves), the brain and the
        judge learn the outcome."""
        run = self.runs.get(run_id)
        if run is None:
            return
        run.judging.discard(node_id)
        if approved:
            run.done.add(node_id)
        else:
            run.done.discard(node_id)
        n = run.nodes.get(node_id)
        if n is not None:
            prov = self.providers.get(n.agent, "")
            if self.brain is not None and not isinstance(self.brain, OracleBrain):
                self.brain.learn_verdict(n.agent, prov, not approved)
            if self.jev is not None:
                self.jev.settle(f"{run_id}/{node_id}", not approved, run.service.get(node_id, 0.0))
        self._emit("verdict", self.now(), run_id=run_id, node_id=node_id, approved=approved, gate=gate)
        if self.forecast is not None:
            now = self.now()
            self._refresh(run, now)
            self._downstream_leases(run, now)

    # ---- tools ------------------------------------------------------------------------------------------------------
    async def tool(self, sid: str, kind: str, fn: Callable[[], str], *, run_id: str, node_id: str, agent: str, attempt: int, tenant: str) -> str:
        g = await self.acquire(sid, "sandbox.cpu", kind, run_id=run_id, node_id=node_id, agent=agent, attempt=attempt, tenant=tenant)
        try:
            return await asyncio.to_thread(fn)
        finally:
            self.release(g)

    # ---- report -----------------------------------------------------------------------------------------------------
    def report(self) -> dict:
        out = {"policy": self.policy, "decisions": dict(self.decisions),
               "ledger": {"issued": self.ledger.issued_total, "expired": self.ledger.expired_total,
                          "unit_s_issued": round(self.ledger.unit_s_issued, 1), "unit_s_unused": round(self.ledger.unit_s_unused, 1)},
               "budgets": {n: {"spent": round(b.spent, 4), "refused": b.refused, "level_frac": round(b.remaining_frac(self.now()), 3)}
                           for n, b in self.world.budgets.items()}}
        if self.policy in ("needs", "oracle"):
            out["loop"] = self.loop.report()
        if self.jev is not None:
            out["jev"] = self.jev.report()
        return out
