"""Admission policies. One interface; the engine asks a policy:
  cap(res, now)                    effective capacity it may use right now
  on_unavailable(res)              "wait" (queue at the gate) or "fail" (429 to the agent)
  priority(p, step, ready_at, now) waiter ordering; lower runs first (FIFO = ready_at)
  sdk_retries()                    how many times the client SDK retries a transient error
  retry_delay(attempt)             backoff before a retry
  release_model_for_tools(hold)    whether a chat's model slot is released across its tool calls
  tpm_fraction()                   share of the TPM budget it lets itself use
  external_retry_after(res, now, p, n_calls) how long a paused external step waits before asking again
  can_issue_external(res, now, u, p, n_calls) header-based pause (gate), provider clairvoyance (oracle), or budget
                                   leases (lease controller); `u` is the provider's hidden draw — only the oracle reads it
  idle_timeout(p, think, default)  when to park an idle session's sandbox
  prewarm_delay(p, think)          seconds after the request ends to re-acquire it (None: never); `think` is
                                   hidden — only the oracle may read it
  prewarm_hold(p)                  how long a prewarmed sandbox stays warm unused before it re-parks (lease
                                   expiry; inf = until the request arrives)
  retention_ttl(p, step, now, gap, default)
                                   how long a session's idle KV prefix stays retained after a chat; `gap` is
                                   the engine's hidden knowledge of the time to the session's next chat
                                   (tool durations, or the think time once known) — only the oracle may read it
  kv_evict_key(sid, entry, now)    order in which idle KV prefixes are evicted under pressure (lower first);
                                   default expired-first-then-LRU; the oracle evicts the farthest next use (Belady)
  on_call_cost(res, now, cost)     the budget units a call turned out to cost (from the provider's headers);
                                   the lease controller learns each provider's collision rate from it
  on_external_429 / on_external_ok AIMD hooks
  leases(p, step, now)             the session's reservations after a step ends (v1 §6.5 ledger.replace):
                                   a list of Lease (one per resource); [] = none. Consulted by the engine
                                   before every acquisition: others may not take what a live lease keeps.
  on_event(kind, p, now, **info)   observation stream a gateway legitimately sees (request_start /
                                   request_end / step_ready / step_start / step_end / spawn / join with
                                   realised facts only: step kind, size and content cue at submission,
                                   realised duration, tokens once produced, the response's plan cue and
                                   revealed tool calls, last tool kind, request index). Never a Step's
                                   sampled future.
  kv_reserve(p, step, default)     output-token reserve a chat is admitted with (KV admission and TPM);
                                   `step.tokens_out` is hidden — only tokens_in / content may be read
  join_timeout(p, default)         seconds until an orchestrator's idle sandbox parks while it waits for
                                   its sub-agents (B10)
  on_timer(key, now)               a POLICY_TIMER the policy set with self.timer(t, key) fired
  budget_admit(p, step, usd, tokens, now)   pacing against the tenant budgets (self.budgets: name -> BudgetBucket);
                                   False = wait budget_retry_after(...) seconds and ask again. The provider still refuses what
                                   the budget cannot pay for (outcome "budget").

Uncoordinated   = agents call providers directly with SDK-default retries (jittered backoff);
                  API-like resources 429, OS-like resources block; the framework may pin the model
                  slot across tool calls (hold-and-wait). `jitter=false` reproduces the herd.
ReactiveGate    = HiveMind's primitives without prediction: everything queues at the gate, external
                  caps learned by AIMD, header-based pause, TPM used at a fraction, forced release.
                  `queue` = "fifo" (ready time), "vtfq" (virtual-time fair queuing on the *realised*
                  memory-centric cost of each step, Justitia — a fixed rule) or "srpt" (shortest predicted
                  service first on the model queue: tokens_in / prefill_rate, which a gateway knows exactly,
                  plus the running mean of *observed* tokens_out per recipe / decode_base; tools stay FIFO).
                  RUNG0_REPORT §12–§13: vtfq pays where the sandbox pool binds, srpt where the model tier does.
ClairvoyantGate = the oracle upper bound (Hermes-Oracle style): sees true durations, tokens and
                  provider state; SRPT ordering; never issues a call that would 429; parks a sandbox
                  the moment a session goes idle and re-acquires it exactly one cold start before
                  the next request. Not the optimum — a bound on what prediction could buy.
LeaseController = rung 3, slice 1: the reactive gate plus predictions a gateway can make from its own
                  observation stream (predict.py): parks a sandbox when the predicted idle lower quantile
                  exceeds a cold start, prewarms it at that quantile and re-parks after h if unused, sets
                  KV retention to the predicted gap, and orders waiters by virtual-time fair queuing on a
                  memory-centric cost (Justitia). Two knobs: tau (quantile), h (horizon seconds).
                  Reads no hidden value: the honesty selftest feeds it contradictory ones.
"""
from __future__ import annotations

import math

import numpy as np

from .predict import QuantileTracker
from .resources import Lease, Resource
from .workload import Program, Step, think_bucket


class Policy:
    name = "base"
    leases_at_ready = False      # a Reserver that forecasts re-issues its leases at step submission and at spawn too
    budgets: dict = {}           # set by the engine: tenant BudgetBuckets by name (usd, tokens)

    def __init__(self, spec: dict, rng: np.random.Generator):
        self.spec, self.rng = spec, rng
        self.retry_base_s, self.retry_cap_s = float(spec["retry_base_s"]), float(spec["retry_cap_s"])
        self.jitter = bool(spec["jitter"])
        self._sdk_retries = int(spec["sdk_retries"])

    def cap(self, res: Resource, now: float) -> float:
        return res.capacity

    def on_unavailable(self, res: Resource) -> str:
        raise NotImplementedError

    def priority(self, p: Program, step: Step, ready_at: float, now: float) -> float:
        return ready_at

    def sdk_retries(self) -> int:
        return self._sdk_retries

    def retry_delay(self, attempt: int) -> float:
        d = min(self.retry_cap_s, self.retry_base_s * (2 ** attempt))
        return float(self.rng.uniform(0, d)) if self.jitter else d          # full jitter vs herd

    def release_model_for_tools(self, framework_holds: bool) -> bool:
        raise NotImplementedError

    def tpm_fraction(self) -> float:
        return 1.0

    def can_issue_external(self, res: Resource, now: float, u: float, p: Program, n_calls: int) -> bool:
        return True

    def external_retry_after(self, res: Resource, now: float, p: Program, n_calls: int) -> float:
        return res.budget_wait(now)                                          # one budget unit's refill time

    def idle_timeout(self, p: Program, think: float, default: float) -> float:
        return default

    def prewarm_delay(self, p: Program, think: float) -> float | None:
        return None

    def prewarm_hold(self, p: Program) -> float:
        return math.inf

    def retention_ttl(self, p: Program, step: Step | None, now: float, gap: float | None, default: float) -> float:
        return default

    def kv_evict_key(self, sid: str, entry, now: float):
        return (entry.retained_until >= now, entry.last_access)

    def on_event(self, kind: str, p: Program, now: float, **info) -> None:
        pass

    def kv_reserve(self, p: Program, step: Step, default: int) -> int:
        return default

    def join_timeout(self, p: Program, default: float) -> float:
        return default

    def on_timer(self, key, now: float) -> None:
        raise RuntimeError(f"{self.name}: a timer fired ({key!r}) but the policy handles none")

    def budget_admit(self, p: Program, step: Step, usd: float, tokens: float, now: float) -> bool:
        return True

    def budget_retry_after(self, p: Program, usd: float, tokens: float, now: float) -> float:
        return 1.0

    def timer(self, t: float, key) -> None:            # replaced by the engine with its scheduler
        raise RuntimeError("policy timers are available only inside an Engine")

    def leases(self, p: Program, step: Step | None, now: float) -> list:
        return []

    def on_call_cost(self, res: Resource, now: float, cost: float) -> None:
        pass

    def on_external_429(self, res: Resource, now: float) -> None:
        pass

    def on_external_ok(self, res: Resource, now: float) -> None:
        pass


class Uncoordinated(Policy):
    name = "uncoordinated"

    def on_unavailable(self, res: Resource) -> str:
        return "fail" if res.api_like else "wait"

    def release_model_for_tools(self, framework_holds: bool) -> bool:
        return not framework_holds


class ReactiveGate(Policy):
    name = "reactive_gate"

    def __init__(self, spec: dict, rng: np.random.Generator, prefill_rate: float = 1.0, decode_base: float = 1.0):
        super().__init__(spec, rng)
        self.bucket_fraction, self.aimd_min = float(spec["bucket_fraction"]), float(spec["aimd_min"])
        self.header_pause_frac = float(spec["header_pause_frac"])
        if not 0 < self.bucket_fraction <= 1:
            raise ValueError("bucket_fraction must be in (0, 1]")
        self.queue = spec["queue"]
        if self.queue not in ("fifo", "vtfq", "srpt"):
            raise ValueError(f"policy.queue must be 'fifo', 'vtfq' or 'srpt', got {self.queue!r}")
        self.prefill_rate, self.decode_base = prefill_rate, decode_base
        self.out_seen: dict[str, tuple[int, float]] = {}   # SRPT: recipe -> (n, sum of observed tokens_out)
        self.aimd: dict[str, float] = {}
        self.ok_streak: dict[str, int] = {}
        self.tags: dict[str, float] = {}       # VTFQ: virtual finish tag per session
        self.starts: dict[str, float] = {}     # VTFQ: virtual start of the step in service
        self.v_clock = 0.0

    @staticmethod
    def memory_cost(p: Program, step_kind: str, members: list[str], tokens_in: int, duration: float) -> float:
        """Memory-centric cost of a step (Justitia): memory footprint x realised time. GB·s for local tools,
        KV-GB·s for chats (~1 GB per 10k tokens), a nominal 0.1 GB for external tools and retrieval."""
        if step_kind == "chat":
            return tokens_in / 1e4 * duration
        if step_kind == "tool" and any(not m.startswith(("search", "web", "api")) for m in members):
            return p.sandbox_gb * duration
        return 0.1 * duration

    def mean_out(self, recipe: str) -> float:
        n, tot = self.out_seen.get(recipe, (0, 0.0))
        if n == 0:
            n, tot = self.out_seen.get("", (0, 0.0))                 # global fallback, then 0 before any completion
        return tot / n if n else 0.0

    def on_event(self, kind: str, p: Program, now: float, **info) -> None:
        if info.get("outcome", "ok") != "ok":                         # a timed-out call: not a realised service time
            return
        if self.queue == "srpt":
            if kind == "step_end" and info["step_kind"] == "chat":
                for key in (p.recipe, ""):
                    n, tot = self.out_seen.get(key, (0, 0.0))
                    self.out_seen[key] = (n + 1, tot + float(info["tokens_out"]))
            return
        if self.queue != "vtfq":
            return
        if kind == "step_start":                                   # start-time fair queuing: no credit for idling
            start = max(self.tags.get(p.sid, self.v_clock), self.v_clock)
            self.starts[p.sid], self.tags[p.sid], self.v_clock = start, start, start
        elif kind == "step_end":
            self.tags[p.sid] = self.starts.get(p.sid, self.v_clock) + self.memory_cost(
                p, info["step_kind"], info.get("members", []), info.get("tokens_in", 0), info["duration"])

    def priority(self, p: Program, step: Step, ready_at: float, now: float) -> float:
        if self.queue == "vtfq":
            return self.tags.get(p.sid, self.v_clock)
        if self.queue == "srpt" and step.kind == "chat":
            return step.tokens_in / self.prefill_rate + self.mean_out(p.recipe) / self.decode_base   # observable estimate
        return ready_at

    def cap(self, res: Resource, now: float) -> float:
        return min(res.capacity, self.aimd.setdefault(res.name, res.capacity))

    def on_unavailable(self, res: Resource) -> str:
        return "wait"

    def release_model_for_tools(self, framework_holds: bool) -> bool:
        return True

    def tpm_fraction(self) -> float:
        return self.bucket_fraction

    def can_issue_external(self, res: Resource, now: float, u: float, p: Program, n_calls: int) -> bool:
        return res.remaining_frac(now) > self.header_pause_frac         # HiveMind: pause below 10% remaining

    def budget_admit(self, p: Program, step: Step, usd: float, tokens: float, now: float) -> bool:
        """HiveMind's header pause applied to spend: hold new spend while any budget is below the pause fraction."""
        for name, amt in (("usd", usd), ("tokens", tokens)):
            b = self.budgets.get(name)
            if b is not None and amt > 0 and b.remaining_frac(now) <= self.header_pause_frac:
                return False
        return True

    def budget_retry_after(self, p: Program, usd: float, tokens: float, now: float) -> float:
        waits = [b.wait_for(now, self.header_pause_frac * b.capacity - b.level + amt) for name, amt in (("usd", usd), ("tokens", tokens))
                 for b in [self.budgets.get(name)] if b is not None and amt > 0]
        return max(0.5, min(waits)) if waits and min(waits) < math.inf else 30.0

    def external_retry_after(self, res: Resource, now: float, p: Program, n_calls: int) -> float:
        """Wait until the bucket refills past the pause threshold (F19: a one-unit wait spun every 1 ms
        whenever the threshold was more than one unit)."""
        deficit = self.header_pause_frac * res.rpm - res.remaining_frac(now) * res.rpm
        return max(0.05, deficit * 60.0 / res.rpm)

    def on_external_429(self, res: Resource, now: float) -> None:
        self.aimd[res.name] = max(self.aimd_min, self.cap(res, now) * 0.5)
        self.ok_streak[res.name] = 0

    def on_external_ok(self, res: Resource, now: float) -> None:
        self.ok_streak[res.name] = self.ok_streak.get(res.name, 0) + 1
        if self.ok_streak[res.name] >= 10:
            self.aimd[res.name] = min(res.capacity, self.cap(res, now) + 1.0)
            self.ok_streak[res.name] = 0


class ClairvoyantGate(Policy):
    name = "clairvoyant"

    def __init__(self, spec: dict, rng: np.random.Generator, cold_start_s: float, prefill_rate: float, decode_base: float):
        super().__init__(spec, rng)
        self.cold_start_s, self.prefill_rate, self.decode_base = cold_start_s, prefill_rate, decode_base

    def on_unavailable(self, res: Resource) -> str:
        return "wait"

    def priority(self, p: Program, step: Step, ready_at: float, now: float) -> float:
        if step.kind == "chat":                                              # SRPT on true service time
            return step.tokens_in / self.prefill_rate + step.tokens_out / self.decode_base
        return step.duration

    def release_model_for_tools(self, framework_holds: bool) -> bool:
        return True

    def can_issue_external(self, res: Resource, now: float, u: float, p: Program, n_calls: int) -> bool:
        return res.can_call(now, u)                                          # knows the provider's bucket and the draw

    def idle_timeout(self, p: Program, think: float, default: float) -> float:
        return 0.0                                                           # park the moment the session idles

    def join_timeout(self, p: Program, default: float) -> float:
        return 0.0                                                           # an orchestrator waiting on a join is idle

    def budget_admit(self, p: Program, step: Step, usd: float, tokens: float, now: float) -> bool:
        """Knows the budget's level and the step's cost: never issues a call the budget would refuse."""
        return all(self.budgets[n].can_pay(now, amt) for n, amt in (("usd", usd), ("tokens", tokens)) if n in self.budgets and amt > 0)

    def budget_retry_after(self, p: Program, usd: float, tokens: float, now: float) -> float:
        waits = [self.budgets[n].wait_for(now, amt) for n, amt in (("usd", usd), ("tokens", tokens)) if n in self.budgets and amt > 0]
        return max(0.05, min(waits)) if waits and min(waits) < math.inf else 30.0

    def prewarm_delay(self, p: Program, think: float) -> float | None:
        return think - self.cold_start_s if think > self.cold_start_s else None   # knows the think time

    def retention_ttl(self, p: Program, step: Step | None, now: float, gap: float | None, default: float) -> float:
        # exact gap until the next chat is *ready*, plus the provider's TTL grace for the time it then queues
        # for a slot (which even an oracle of durations does not set); never less than the gate's retention
        return default if gap is None else gap + default

    def kv_evict_key(self, sid: str, entry, now: float):
        return (entry.retained_until >= now, -entry.retained_until)          # Belady: farthest next use first


class LeaseController(ReactiveGate):
    name = "lease"

    def __init__(self, spec: dict, rng: np.random.Generator, cold_start_s: float, prefill_rate: float, decode_base: float):
        super().__init__(spec, rng, prefill_rate, decode_base)
        self.tau, self.h = float(spec["tau"]), float(spec["h"])
        if not 0.0 < self.tau < 1.0:
            raise ValueError("lease.tau must be in (0, 1)")
        if not self.h > 0:
            raise ValueError("lease.h must be > 0 seconds")
        # ablation switches (E6): '+'-joined names of features to turn OFF, e.g. "vtfq+prewarm"
        self.off = set(x for x in str(spec.get("ablate", "")).split("+") if x)
        unknown = self.off - {"vtfq", "park", "prewarm", "kv", "gang", "budget"}
        if unknown:
            raise ValueError(f"lease.ablate: unknown features {sorted(unknown)}")
        self.cold_start_s, self.prefill_rate, self.decode_base = cold_start_s, prefill_rate, decode_base
        self.idle = QuantileTracker()          # realised idle gaps, keyed (recipe, last tool kind, request bucket)
        self.tool_dur = QuantileTracker()      # realised single-tool durations, keyed (recipe, kind)
        self.chat_out = QuantileTracker()      # realised output tokens, keyed (recipe,)
        self.pending_idle: dict[str, tuple[float, tuple]] = {}      # sid -> (request end time, state key)
        self.tags: dict[str, float] = {}       # virtual finish tag per session (VTFQ)
        self.v_clock = 0.0
        self.gang: dict[str, tuple[int, float]] = {}   # sid -> (extra units, wait worth paying) declared at the last chat end
        self.upcoming: dict[str, list[str]] = {}       # sid -> external resources the chat's tool calls will hit (slice 3)
        self.calls_seen: dict[str, tuple[int, int]] = {}   # provider -> (calls observed, calls that cost an extra unit)
        self.ledger = None                     # set by the engine: what is already leased (I1)
        self.resources = None

    # ---- observation stream ------------------------------------------------------------
    def on_event(self, kind: str, p: Program, now: float, **info) -> None:
        if info.get("outcome", "ok") != "ok":
            return
        if kind == "request_end":
            self.pending_idle[p.sid] = (now, (p.recipe, info["last_tool"], think_bucket(info["request_idx"])))
        elif kind == "request_start":
            if p.sid in self.pending_idle:
                t0, key = self.pending_idle.pop(p.sid)
                self.idle.observe(key, now - t0)
        elif kind == "step_end":
            if info["step_kind"] == "tool" and len(info["members"]) == 1:
                self.tool_dur.observe((p.recipe, info["members"][0]), info["duration"])
            elif info["step_kind"] == "chat":
                self.chat_out.observe((p.recipe,), float(info["tokens_out"]))
                self.gang.pop(p.sid, None)
                self.upcoming[p.sid] = sorted({r for r in info["tool_resources"] if self.resources[r].calls is not None})
                members = info["parallel_group"]
                if len(members) >= 2:                                  # the framework declared a fan-out of local tools
                    qs = [self.tool_dur.quantile((p.recipe, m), self.tau) for m in members]
                    if all(q is not None for q in qs):
                        worth = min(self.h, sum(qs) - max(qs))          # waiting longer than this beats nothing
                        if worth > 0:
                            self.gang[p.sid] = (len(members) - 1, worth)
        elif kind == "step_start":
            start = max(self.tags.get(p.sid, self.v_clock), self.v_clock)     # start-time fair queuing
            self.tags[p.sid] = start + self._cost(p, info["step_kind"], info["members"], info["tokens_in"])
            self.v_clock = start

    # ---- predictions (observable inputs only) -------------------------------------------
    def _cost(self, p: Program, step_kind: str, members: list[str], tokens_in: int) -> float:
        """Memory-centric service cost: GB·s for tools, token·s for chats (predicted at the tau quantile)."""
        if step_kind == "chat":
            out = self.chat_out.quantile((p.recipe,), self.tau) or 0.0
            return tokens_in * (tokens_in / self.prefill_rate + out / self.decode_base) / 1e6
        dur = 0.0
        for kind in members:
            q = self.tool_dur.quantile((p.recipe, kind), self.tau)
            dur = max(dur, q if q is not None else 0.0)
        return p.sandbox_gb * dur

    def _idle_key(self, p: Program) -> tuple:
        return self.pending_idle[p.sid][1] if p.sid in self.pending_idle else (p.recipe, p.last_tool, think_bucket(max(0, p.request_idx - 1)))

    def _idle_lower(self, p: Program) -> float | None:
        return self.idle.quantile(self._idle_key(p), 1.0 - self.tau)

    def _idle_width(self, p: Program) -> float | None:
        """Predicted interval of the next arrival, q_tau - q_(1-tau): how uncertain the forecast is."""
        key = self._idle_key(p)
        lo, hi = self.idle.quantile(key, 1.0 - self.tau), self.idle.quantile(key, self.tau)
        return None if lo is None or hi is None else max(0.0, hi - lo)

    def priority(self, p: Program, step: Step, ready_at: float, now: float) -> float:
        if "vtfq" in self.off:
            return ready_at                                       # FIFO, as the gate
        return self.tags.get(p.sid, self.v_clock)

    def idle_timeout(self, p: Program, think: float, default: float) -> float:
        if "park" in self.off:
            return default
        q = self._idle_lower(p)
        if q is None:
            return default                                        # nothing observed yet: the gate's behaviour
        return 0.0 if q > self.cold_start_s else default          # worth a cold start with probability tau

    def prewarm_delay(self, p: Program, think: float) -> float | None:
        if "prewarm" in self.off:
            return None
        q, width = self._idle_lower(p), self._idle_width(p)
        if q is None or width is None or q <= self.cold_start_s or width > self.h:
            return None                                           # too uncertain to hold a warm sandbox for the interval
        return q - self.cold_start_s

    def prewarm_hold(self, p: Program) -> float:
        width = self._idle_width(p)
        return min(self.h, width) if width is not None else self.h   # warm for the predicted interval, then re-park

    def retention_ttl(self, p: Program, step: Step | None, now: float, gap: float | None, default: float) -> float:
        if "kv" in self.off:
            return default
        if step is not None and step.follow_tools:                # mid-request: the tools the chat just issued
            total = self.cold_start_s if p.sandbox_cold else 0.0
            for t in step.follow_tools:
                qs = [self.tool_dur.quantile((p.recipe, m.name), self.tau) for m in t.members()]
                total += max((q for q in qs if q is not None), default=0.0)
            return total + default if total > 0 else default       # predicted gap + TTL grace for queueing
        if step is None:                                          # request ended: the idle gap
            q = self.idle.quantile(self._idle_key(p), self.tau)
            return q + default if q is not None else default
        return default

    def kv_evict_key(self, sid: str, entry, now: float):
        if "kv" in self.off:
            return (entry.retained_until >= now, entry.last_access)
        return (entry.retained_until >= now, -entry.retained_until)          # Belady on the predicted next use

    def leases(self, p: Program, step: Step | None, now: float) -> list:
        """Slice 2: reserve the extra CPU units a declared parallel group needs, for as long as waiting for
        them still beats running the group sequentially. All-or-nothing against what is already leased (I1)."""
        if step is None or step.kind != "chat":
            return []
        out = []
        if p.sid in self.gang:
            extra, worth = self.gang.pop(p.sid)
            if "gang" not in self.off and self.ledger.active_sum("sandbox.cpu", now) + extra <= self.resources["sandbox.cpu"].capacity:
                out.append(Lease(p.sid, "sandbox.cpu", float(extra), now, now + worth))
        upcoming = self.upcoming.pop(p.sid, [])
        for name in ([] if "budget" in self.off else upcoming):   # slice 3: one budget unit per provider the calls need
            key = f"{name}@rpm"
            if self.ledger.active_sum(key, now) + 1.0 <= self.resources[name].rpm:
                out.append(Lease(p.sid, key, 1.0, now, now + self.h))
        return out

    def on_call_cost(self, res: Resource, now: float, cost: float) -> None:
        n, k = self.calls_seen.get(res.name, (0, 0))
        self.calls_seen[res.name] = (n + 1, k + (1 if cost > 1.0 else 0))

    def collision_rate(self, name: str) -> float:
        """Observed share of calls that cost an extra unit (another tenant landed first); 1.0 until 20 calls were seen."""
        n, k = self.calls_seen.get(name, (0, 0))
        return k / n if n >= 20 else 1.0

    def _margin(self, res: Resource, n_calls: int) -> float:
        """Budget units to require for n calls: n plus the tau-quantile of Binomial(n, p̂) extra units —
        the worst case (2n) only while p̂ is unknown or tau demands it."""
        p_hat = self.collision_rate(res.name)
        cum, extra = 0.0, 0
        for k in range(n_calls + 1):
            cum += math.comb(n_calls, k) * p_hat ** k * (1 - p_hat) ** (n_calls - k)
            if cum >= self.tau - 1e-12:
                extra = k
                break
        else:
            extra = n_calls
        return float(n_calls + extra)

    def can_issue_external(self, res: Resource, now: float, u: float, p: Program, n_calls: int) -> bool:
        """Issue only what the bucket can pay for after others' reservations, with a margin for the extra units
        other tenants' calls may cost (learned collision rate, tau quantile). A reservation keeps others off the
        units; it does not create them."""
        if "budget" in self.off:
            return ReactiveGate.can_issue_external(self, res, now, u, p, n_calls)
        key = f"{res.name}@rpm"
        level = res.remaining_frac(now) * res.rpm
        return level - self.ledger.reserved_for_others(key, p.sid, now, {}) >= self._margin(res, n_calls)

    def external_retry_after(self, res: Resource, now: float, p: Program, n_calls: int) -> float:
        """Time until the bucket refills to what this step needs on top of others' reservations (>= 1 s)."""
        if "budget" in self.off:
            return ReactiveGate.external_retry_after(self, res, now, p, n_calls)
        key = f"{res.name}@rpm"
        level = res.remaining_frac(now) * res.rpm
        need = self._margin(res, n_calls) + self.ledger.reserved_for_others(key, p.sid, now, {}) - level
        return max(1.0, need * 60.0 / res.rpm)


def make_policy(spec: dict, rng: np.random.Generator, cfg: dict) -> Policy:
    t = spec["type"]
    if t == "uncoordinated":
        return Uncoordinated(spec, rng)
    if t == "reactive_gate":
        m = cfg["model"]
        return ReactiveGate(spec, rng, float(m["prefill_rate"]), float(m["decode_base"]))
    if t == "clairvoyant":
        m = cfg["model"]
        return ClairvoyantGate(spec, rng, float(cfg["framework"]["sandbox_cold_start_s"]),
                               float(m["prefill_rate"]), float(m["decode_base"]))
    if t == "lease":
        m = cfg["model"]
        return LeaseController(spec, rng, float(cfg["framework"]["sandbox_cold_start_s"]),
                               float(m["prefill_rate"]), float(m["decode_base"]))
    if t == "needs":
        from pathlib import Path
        from .reserver import NeedsController
        from .workload import Recipes
        root = Path(__file__).resolve().parent.parent
        return NeedsController(spec, rng, cfg, Recipes.load(root / cfg["recipes"]))
    raise ValueError(f"unknown policy type {t!r}")
