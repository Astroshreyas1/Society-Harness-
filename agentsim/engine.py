"""Discrete-event engine: program-centric (AgentServeSim), causal successor release,
three resource tiers, a pluggable admission policy, OTel-shaped span output.

Events carry (sid, epoch, uid, attempt); an event whose program epoch, step uid or attempt has
moved on is stale and ignored. Kinds:
  ARRIVE, REQUEST_START, TRY_RUN, DECODE_START, CHAT_END, TOOL_END, RETRIEVAL_END,
  WAIT_TIMEOUT (client timeout on a step), DEADLOCK_BREAK, SANDBOX_PARK, SANDBOX_PREWARM,
  LEASE_EXPIRE (a reservation ended: waiters on that resource are re-tried),
  POLICY_TIMER (a policy's own alarm, e.g. an asynchronous annotation arriving; never stale)

Multi-agent (B10): a chat whose response launches k sub-agents spawns k child programs (own recipe, sandbox,
context, RNG streams keyed by the parent's identity) that run their one request concurrently; the parent holds
no step while it waits for the join and continues with its next chat when the last child has ended (done or
aborted). Child spans carry the parent's trace id, so a request's metrics include its sub-agents' work. A parent
waiting on a join counts as "waiting for its children" in the hold-and-wait probe: parents holding sandboxes while
their children queue for sandboxes is a real cycle and is detected as one.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .marginals import Marginals
from .policies import Policy
from .resources import Ledger, ModelPhysics, Resource, TokenBucket, build_budgets
from .schema import Span
from .workload import Arrivals, Program, Recipes, Step, Workload

ORDER = ["model.slots", "sandbox.mem", "sandbox.cpu"]          # global acquisition order; ext.* and svc.* follow


@dataclass
class Pending:
    uid: int
    step: Step
    ready_at: float
    started_at: float = -1.0
    attempt: int = 0
    sdk_tries: int = 0
    tools_left: list[Step] = field(default_factory=list)
    prefill_s: float = 0.0
    fresh: int = 0
    waiting_for: str | None = None
    waiting_amt: float = 0.0
    partial: list[tuple[str, float]] = field(default_factory=list)   # taken in this attempt, not yet committed
    own: dict[str, float] = field(default_factory=dict)              # committed holds that belong to this step only
    running: bool = False
    parallel_ok: bool = False                                        # a parallel group got its extra CPU units
    gang_wait: bool = False                                          # waiting for CPU units a lease reserved (bounded by its expiry)
    reserve: int = 0                                                 # KV / TPM output reserve this chat was admitted with


class Engine:
    def __init__(self, cfg: dict, recipes: Recipes, marginals: Marginals, resources: dict[str, Resource], policy: Policy):
        self.cfg, self.res, self.policy = cfg, resources, policy
        fw = cfg["framework"]
        seed = int(cfg["seed"])
        gen = cfg["generator"]
        self.wl = Workload(recipes, marginals, cfg["recipe_mix"], bool(fw["hold_model_during_tool"]),
                           int(fw["context_limit"]), int(fw["compaction_reset"]), seed, gen["recipe_temperature"],
                           gen["think_snr"], gen["think_state_seed"], gen["tool_tail_scale"], gen["p_parallel_scale"],
                           gen["content_snr"])
        self.arrivals = Arrivals(cfg["arrivals"], seed)
        m = cfg["model"]
        self.model = ModelPhysics(int(resources["model.kv"].capacity), float(m["prefill_rate"]),
                                  float(m["decode_base"]), float(m["decode_capacity"]), float(m["kv_ttl_s"]),
                                  evict_key=policy.kv_evict_key)
        self.reserve_out = int(m["kv_reserve_out"])
        self.price_in, self.price_out = float(m["price_in_per_mtok"]) / 1e6, float(m["price_out_per_mtok"]) / 1e6
        self.bucket = TokenBucket(resources["model.tpm"].capacity)
        self.budgets = build_budgets(cfg)                                   # tenant budgets (usd, tokens); {} = unmetered
        self.budget_refusals = 0
        self.step_timeout = float(fw["step_timeout_s"])
        self.idle_default = float(fw["sandbox_idle_timeout_s"])
        self.cold_start = float(fw["sandbox_cold_start_s"])
        self.rp = cfg["reaction"]
        self.horizon = float(cfg["horizon_s"])
        self.now, self.seq, self.heap = 0.0, 0, []
        self.programs: dict[str, Program] = {}
        self.pending: dict[str, Pending] = {}
        self.epoch: dict[str, int] = {}
        self.spans: list[Span] = []
        self.roots: dict[str, Span] = {}
        self.wait_open: dict[str, Span] = {}
        self.req_open: dict[str, Span] = {}
        self.n_sessions, self.next_uid, self.arrival_index = 0, 0, 0
        self.slot_count: dict[int, int] = {}
        self.deadlocks_detected = 0
        self.events = 0
        self.prewarm_idle_s = 0.0          # sandbox-seconds held warm by a prewarm before (or without) use
        self.ledger = Ledger()             # reservations (rung 3); empty unless the policy issues leases
        self._waking: set[str] = set()     # resources whose wake loop is on the stack (re-entrancy guard, F21)
        self.children: dict[str, list[str]] = {}   # parent sid -> child sids (B10)
        self.join_span: dict[str, Span] = {}       # parent sid -> the chat span that launched the fan-out in progress
        self.spawns, self.joins = 0, 0
        policy.ledger, policy.resources = self.ledger, self.res    # a Reserver must see what is already leased (I1)
        policy.timer = self._policy_timer                          # a policy may set its own alarms (POLICY_TIMER)
        policy.budgets = self.budgets                              # a gate sees the tenant's spend headers
        if hasattr(policy, "replace_leases"):
            policy.replace_leases = self._policy_replace_leases    # a Reserver may withdraw / re-issue a session's leases off-cycle

    # ---- scheduling --------------------------------------------------------------------
    def _at(self, t: float, kind: str, sid: str | None = None, **payload) -> None:
        self.seq += 1
        pd = self.pending.get(sid) if sid else None
        stamp = (self.epoch.get(sid, 0) if sid else 0, pd.uid if pd else -1, pd.attempt if pd else -1)
        heapq.heappush(self.heap, (t, self.seq, kind, sid, stamp, payload))

    def _stale(self, sid: str | None, stamp, kind: str) -> bool:
        if sid is None:
            return False
        if self.epoch.get(sid, 0) != stamp[0]:
            return True
        if kind in ("ARRIVE", "REQUEST_START", "SANDBOX_PARK", "SANDBOX_PREWARM"):
            return False
        pd = self.pending.get(sid)
        return pd is None or pd.uid != stamp[1] or pd.attempt != stamp[2]

    def run(self) -> None:
        if self.arrivals.closed:
            for slot in range(int(self.arrivals.spec["population"])):
                self._at(self.arrivals.rng.uniform(0, float(self.arrivals.spec["stagger_s"])), "ARRIVE", slot=slot)
        else:
            self._at(self.arrivals.next_interarrival(0.0), "ARRIVE", slot=-1)
        last, same = (None, None), 0
        while self.heap:
            t, _, kind, sid, stamp, payload = heapq.heappop(self.heap)
            if t > self.horizon:
                break
            self.now = t
            if self._stale(sid, stamp, kind):
                continue
            same = same + 1 if (t, sid) == last else 0
            last = (t, sid)
            if same > 10000:
                raise RuntimeError(f"zero-time event loop at t={t} for {sid} ({kind})")
            self.events += 1
            getattr(self, f"_ev_{kind}")(sid, **payload)
            if self.events % 500 == 0:
                self.check_accounting()
        self._censor_open_sessions()
        self.check_accounting()

    def check_accounting(self) -> None:
        for r in self.res.values():
            r.check()
        self.ledger.expire(self.now)              # lazy expiry, as before every acquisition: a lease whose expiry equals `now`
        self.ledger.check(self.res, self.now)     # and whose LEASE_EXPIRE event is later in this instant's queue is not a violation

    def _free(self, res: Resource, sid: str) -> float:
        """Capacity `sid` may take now: free under the policy's cap, minus what live leases keep for others."""
        return res.free(self.policy.cap(res, self.now)) - self.ledger.reserved_for_others(res.name, sid, self.now, res.holders)

    def _replace_leases(self, p: Program, step: Step | None) -> None:
        leases = self.policy.leases(p, step, self.now)
        for lease in leases:
            if lease.amt > Ledger.limit(self.res, lease.res):
                raise ValueError(f"{p.sid}: lease of {lease.amt} exceeds what {lease.res} can ever hold")
        self.ledger.set_session(p.sid, leases, self.now)
        for lease in leases:
            if self.ledger.active_sum(lease.res, self.now) > Ledger.limit(self.res, lease.res) + 1e-9:
                raise AssertionError(f"I1: {p.sid}'s lease on {lease.res} over-commits the resource (policy must check ledger.active_sum)")
            self._at(lease.expiry, "LEASE_EXPIRE", None, res=lease.base)

    def _ev_LEASE_EXPIRE(self, _sid: str | None, res: str) -> None:
        if self.ledger.expire(self.now):
            self._wake(self.res[res])

    def _policy_timer(self, t: float, key) -> None:
        if t < self.now:
            raise ValueError(f"policy timer in the past: {t} < {self.now}")
        self._at(t, "POLICY_TIMER", None, key=key)

    def _ev_POLICY_TIMER(self, _sid: str | None, key) -> None:
        self.policy.on_timer(key, self.now)

    def _policy_replace_leases(self, sid: str) -> None:
        p = self.programs.get(sid)
        if p is None or p.status != "active":
            return
        pd = self.pending.get(sid)
        self._replace_leases(p, pd.step if pd is not None else None)
        for name in list(self.res):
            if self.res[name].waiters:
                self._wake(self.res[name])

    # ---- session lifecycle -------------------------------------------------------------
    def _ev_ARRIVE(self, _sid: str | None, slot: int) -> None:
        if self.arrivals.closed:
            key = slot * 1_000_000 + self.slot_count.get(slot, 0)
            self.slot_count[slot] = self.slot_count.get(slot, 0) + 1
        else:
            key = self.arrival_index
            self.arrival_index += 1
            self._at(self.now + self.arrivals.next_interarrival(self.now), "ARRIVE", slot=-1)
        self.n_sessions += 1
        sid = f"s{self.n_sessions:06d}"
        p = self.wl.new_program(sid, key, slot, self.now)
        self.programs[sid], self.epoch[sid] = p, 0
        self.roots[sid] = Span(trace_id=sid, span_id=f"{sid}-root", parent_span_id=None, op="invoke_agent",
                               name=p.recipe, tier=None, resource=None, t_start=self.now, t_end=self.now, step_idx=0,
                               attrs={"key": key, "slot": slot, "n_requests": p.n_requests, "sandbox_gb": round(p.sandbox_gb, 2),
                                      "context0": p.context_tokens, "requests_started": 0})
        self._ev_REQUEST_START(sid)

    def _ev_REQUEST_START(self, sid: str) -> None:
        p = self.programs[sid]
        self.roots[sid].attrs["requests_started"] += 1
        attrs = {"request_idx": p.request_idx, "slot": p.slot, "key": p.key}
        if p.parent is not None:
            attrs["child"] = True                                     # a sub-agent's task: counted inside its parent's request
        self.req_open[sid] = Span(trace_id=p.trace, span_id=f"{sid}-r{p.request_idx}", parent_span_id=f"{sid}-root", op="invoke_workflow",
                                  name=p.recipe, tier=None, resource=None, t_start=self.now, t_end=self.now, step_idx=p.step_idx, attrs=attrs)
        self.wl.start_request(p)
        if p.prewarmed_at >= 0:                                       # warm on arrival: the idle part was the price
            self.prewarm_idle_s += self.now - p.prewarmed_at
            p.prewarm_hits += 1
            p.prewarmed_at = -1.0
        p.last_tool, p.chats_in_request = "none", 0
        self.policy.on_event("request_start", p, self.now, request_idx=p.request_idx)
        self._ready_chat(p)

    def _close_request(self, sid: str, outcome: str) -> None:
        r = self.req_open.pop(sid, None)
        if r is None:
            raise RuntimeError(f"{sid}: closing a request that is not open")
        r.t_end, r.outcome = self.now, outcome
        r.attrs["steps"] = self.programs[sid].step_idx - r.step_idx
        self.spans.append(r)

    def _end_request(self, p: Program) -> None:
        if "model.slots" in p.holds:
            self._release(p, "model.slots")
        self._close_request(p.sid, "ok")
        self.policy.on_event("request_end", p, self.now, request_idx=p.request_idx, last_tool=p.last_tool,
                             chats=p.chats_in_request)
        self._replace_leases(p, None)
        p.request_idx += 1
        if p.request_idx >= p.n_requests:
            self._end_session(p, "done")
            return
        think = self.wl.think_time(p) * float(self.arrivals.spec.get("think_scale", 1.0))
        self._span(p, op="think", name="human", tier=None, resource=None, t_start=self.now, t_end=self.now + think, duration=think,
                   attrs={"recipe": p.recipe, "phase_end": p.phase_end, "request_idx": p.request_idx})
        self.model.set_ttl(p.sid, self.now, self.policy.retention_ttl(p, None, self.now, think, self.model.kv_ttl_s))
        self._at(self.now + think, "REQUEST_START", p.sid)
        if "sandbox.mem" in p.holds:
            t_idle = self.policy.idle_timeout(p, think, self.idle_default)
            if t_idle < think:
                self._at(self.now + t_idle, "SANDBOX_PARK", p.sid, request_idx=p.request_idx)
                delay = self.policy.prewarm_delay(p, think)                   # seconds after now; the policy's forecast
                if delay is not None and t_idle < delay < think:
                    self._at(self.now + delay, "SANDBOX_PREWARM", p.sid, request_idx=p.request_idx)
                elif delay is not None and delay >= think:                    # forecast late: arrival pays the cold start
                    pass

    def _ev_SANDBOX_PARK(self, sid: str, request_idx: int) -> None:
        p = self.programs[sid]
        if p.request_idx != request_idx or sid in self.pending or "sandbox.mem" not in p.holds:
            return
        if p.prewarmed_at >= 0:                                       # a prewarmed sandbox nobody used: waste
            self.prewarm_idle_s += self.now - p.prewarmed_at
            p.prewarmed_at = -1.0
        self._release(p, "sandbox.mem")
        self._release(p, "sandbox.cpu")
        p.sandbox_cold = True

    def _ev_SANDBOX_PREWARM(self, sid: str, request_idx: int) -> None:
        p = self.programs[sid]
        if p.request_idx != request_idx or sid in self.pending or "sandbox.mem" in p.holds:
            return
        mem, cpu = self.res["sandbox.mem"], self.res["sandbox.cpu"]
        if self._free(mem, sid) >= p.sandbox_gb and self._free(cpu, sid) >= 1:
            mem.take(sid, p.sandbox_gb)
            cpu.take(sid, 1.0)
            p.holds["sandbox.mem"], p.holds["sandbox.cpu"] = p.sandbox_gb, 1.0
            p.sandbox_cold = False                                    # cold start paid while idle
            p.prewarmed_at = self.now
            hold = self.policy.prewarm_hold(p)                        # lease expiry: re-park if the forecast was stale
            if hold < math.inf:
                self._at(self.now + hold, "SANDBOX_PARK", p.sid, request_idx=request_idx)

    def _end_session(self, p: Program, status: str) -> None:
        p.status = status
        for r in list(p.holds):
            self._release(p, r)
        self.ledger.set_session(p.sid, [], self.now)
        self.model.drop(p.sid, self.now)
        self._wake_kv()
        self._dequeue(p.sid, "aborted")
        self.pending.pop(p.sid, None)
        self.epoch[p.sid] += 1
        root = self.roots.pop(p.sid)
        root.t_end = self.now
        root.outcome = {"done": "ok", "aborted": "aborted"}[status]
        root.attrs.update(requests_done=p.request_idx, requests_failed=p.requests_failed, steps=p.step_idx, episodes=p.episodes,
                          spend_usd=round(p.spend_usd, 6), spend_tokens=round(p.spend_tokens))
        self.spans.append(root)
        self.policy.on_event("session_end", p, self.now, status=status)
        if self.arrivals.closed and p.parent is None:
            self._at(self.now, "ARRIVE", slot=p.slot)
        if p.parent is not None:
            self._child_ended(p)

    # ---- multi-agent spawn / join (B10) --------------------------------------------------
    def _spawn(self, p: Program, k: int, chat: Span, step: Step) -> None:
        """The chat's response launched k sub-agents: create them, start their requests now, and park the
        orchestrator's sandbox after the policy's join timeout (it holds no step while it waits)."""
        p.spawn_seq += 1
        p.join_left, p.join_started = k, self.now
        self.spawns += 1
        self.join_span[p.sid] = chat
        self.children[p.sid] = []
        kids = [self.wl.new_child(p, j, self.now) for j in range(k)]
        # the framework declares the sub-agents and their sandbox sizes before they run: observable
        self.policy.on_event("spawn", p, self.now, width=k, children=[(c.sid, c.sandbox_gb) for c in kids])
        if self.policy.leases_at_ready:
            self._replace_leases(p, step)
        for j, c in enumerate(kids):
            self.programs[c.sid], self.epoch[c.sid] = c, 0
            self.children[p.sid].append(c.sid)
            self.roots[c.sid] = Span(trace_id=p.trace, span_id=f"{c.sid}-root", parent_span_id=chat.span_id, op="invoke_agent",
                                     name=c.recipe, tier=None, resource=None, t_start=self.now, t_end=self.now, step_idx=0,
                                     attrs={"key": c.key, "slot": c.slot, "n_requests": 1, "sandbox_gb": round(c.sandbox_gb, 2),
                                            "context0": c.context_tokens, "requests_started": 0, "child": True, "parent": p.sid, "depth": c.depth})
            self._ev_REQUEST_START(c.sid)
        if "sandbox.mem" in p.holds:
            t_park = self.policy.join_timeout(p, self.idle_default)
            if not t_park >= 0:
                raise ValueError(f"join_timeout must be >= 0, got {t_park}")
            self._at(self.now + t_park, "SANDBOX_PARK", p.sid, request_idx=p.request_idx)

    def _child_ended(self, c: Program) -> None:
        parent = self.programs[c.parent]
        parent.join_left -= 1
        if c.status == "aborted":
            parent.children_failed += 1
        if parent.join_left > 0 or parent.status != "active" or parent.sid not in self.req_open:
            return
        self.joins += 1
        chat = self.join_span.pop(parent.sid)
        chat.attrs["join_wait_s"] = round(self.now - parent.join_started, 3)
        chat.attrs["children_failed"] = sum(1 for sid in self.children[parent.sid] if self.programs[sid].status == "aborted")
        self.policy.on_event("join", parent, self.now, wait=self.now - parent.join_started,
                             failed=chat.attrs["children_failed"], width=len(self.children[parent.sid]))
        self._ready_chat(parent)                                      # the orchestrator's next turn

    def _censor_open_sessions(self) -> None:
        self.now = self.horizon
        for sid in list(self.wait_open):
            self._close_wait(sid, "session_end")
        for sid in list(self.req_open):
            self._close_request(sid, "session_end")
        for sid, root in list(self.roots.items()):
            p = self.programs[sid]
            p.status = "censored"
            root.t_end = self.horizon
            root.outcome = "session_end"
            root.attrs.update(requests_done=p.request_idx, requests_failed=p.requests_failed, steps=p.step_idx,
                              episodes=p.episodes, censored=True)
            self.spans.append(root)
        self.roots.clear()

    # ---- steps ---------------------------------------------------------------------------
    def _ready_chat(self, p: Program) -> None:
        r = self.wl.wants_retrieval(p)
        self._ready(p, r if r is not None else self.wl.next_chat(p), [])

    def _ready(self, p: Program, step: Step, tools_left: list[Step]) -> None:
        self.next_uid += 1
        self.pending[p.sid] = Pending(uid=self.next_uid, step=step, ready_at=self.now, tools_left=tools_left)
        self._at(self.now + self.step_timeout, "WAIT_TIMEOUT", p.sid)
        # the request body is at the gateway: kind, size and content cue are observable before admission; nothing sampled is
        self.policy.on_event("step_ready", p, self.now, step_kind=step.kind, name="chat" if step.kind == "chat" else step.name,
                             tokens_in=step.tokens_in, members=[m.name for m in step.members()],
                             content=[m.content for m in step.members()], parallel_group=bool(step.siblings))
        if self.policy.leases_at_ready:                               # a Reserver that forecasts re-issues its leases at submission too
            self._replace_leases(p, step)
        self._try_run(p)

    def _needs(self, p: Program, step: Step) -> dict[str, float]:
        """Resources this step must hold, in global order. Session-level sandbox holds are taken once."""
        if step.kind == "chat":
            return {} if "model.slots" in p.holds else {"model.slots": 1.0}
        needs: dict[str, float] = {}
        if step.kind == "retrieval":
            needs[step.resource] = 1.0
            return needs
        local = [m for m in step.members() if m.resource.startswith("sandbox")]
        if local and "sandbox.mem" not in p.holds:
            needs["sandbox.mem"] = p.sandbox_gb
            needs["sandbox.cpu"] = 1.0
        for m in step.members():
            if not m.resource.startswith("sandbox"):
                needs[m.resource] = needs.get(m.resource, 0.0) + 1.0
        return needs

    def _ev_TRY_RUN(self, sid: str) -> None:
        self._try_run(self.programs[sid])

    def _extra_cpu(self, step: Step) -> int:
        """Extra CPU units a parallel group of local tools would use; never waited for."""
        if step.kind != "tool":
            return 0
        local = [m for m in step.members() if m.resource.startswith("sandbox")]
        return len(local) - 1 if len(local) > 1 else 0

    def _try_run(self, p: Program) -> None:
        pd = self.pending[p.sid]
        step = pd.step
        self.ledger.expire(self.now)                                  # I3, lazily, before any acquisition
        taken = {n for n, _ in pd.partial}
        for name, amt in self._needs(p, step).items():
            if name in taken:
                continue
            res = self.res[name]
            if self._free(res, p.sid) >= amt - 1e-9:
                res.take(p.sid, amt)
                pd.partial.append((name, amt))
                continue
            if self.policy.on_unavailable(res) == "wait":
                self._wait(p, res, amt)
                return
            self._fail(p, "429", res)
            return
        extra = self._extra_cpu(step)
        if extra and not pd.parallel_ok:                          # opportunistic: take them or run the group sequentially
            cpu = self.res["sandbox.cpu"]
            if self._free(cpu, p.sid) >= extra:
                cpu.take(p.sid, float(extra))
                pd.partial.append(("sandbox.cpu", float(extra)))
                pd.parallel_ok = True
                if self.ledger.own("sandbox.cpu", p.sid, self.now) > 0:
                    self.ledger.consume(p.sid, "sandbox.cpu", self.now)
            elif self.ledger.own("sandbox.cpu", p.sid, self.now) >= extra:    # a gang lease: wait for the units it reserved
                pd.gang_wait = True
                self._wait(p, cpu, float(extra))
                return
        if step.kind == "chat":
            pd.reserve = int(self.policy.kv_reserve(p, step, self.reserve_out))   # output reserve: the scenario's, or predicted
            if pd.reserve < 0:
                raise ValueError(f"kv_reserve must be >= 0, got {pd.reserve}")
            need_kv = step.tokens_in + pd.reserve                  # the whole context becomes active KV (F16: a reused
            if not self.model.fits(need_kv):                       # idle prefix of this session is not "active" yet)
                if self.policy.on_unavailable(self.res["model.kv"]) == "wait":
                    self._wait(p, self.res["model.kv"], float(need_kv))
                    return
                self._fail(p, "429", self.res["model.kv"])
                return
            miss = self.model.miss_tokens(p.sid, step.tokens_in, self.now)
            fresh = miss + pd.reserve                                       # TPM: fresh tokens + reserve
            usd, tok = miss * self.price_in + pd.reserve * self.price_out, float(fresh)   # prepaid: input + the output reserve
            if self.budgets and not self._budget_ok(p, usd, tok):           # the tenant's budget: pace (gate) or fail (uncoordinated)
                return
            if not self.bucket.try_take(fresh, self.now, self.policy.tpm_fraction()):
                if self.policy.on_unavailable(self.res["model.tpm"]) == "wait":
                    self._return_partial(p)
                    self._at(self.now + self.bucket.wait_for(fresh, self.now, self.policy.tpm_fraction()), "TRY_RUN", p.sid)
                    return
                self._fail(p, "429", self.res["model.tpm"])
                return
            self._pay(p, usd, tok)                                          # the reserve is paid up front; the actual output settles at chat end
        else:
            calls = [(self.res[m.resource], float(p.rng_env.random())) for m in step.members() if self.res[m.resource].calls is not None]
            per_res = {res.name: sum(1 for r, _ in calls if r.name == res.name) for res, _ in calls}
            for res, u in calls:                                      # headers first: nothing is issued if the gate pauses
                if not self.policy.can_issue_external(res, self.now, u, p, per_res[res.name]):
                    self._return_partial(p)
                    delay = self.policy.external_retry_after(res, self.now, p, per_res[res.name])
                    if not delay > 0:
                        raise ValueError(f"external_retry_after must be > 0, got {delay}")
                    self._at(self.now + delay, "TRY_RUN", p.sid)
                    return
            usd = sum(res.cost_per_call for res, _ in calls)
            tok = sum(res.tokens_per_call for res, _ in calls)
            if self.budgets and (usd > 0 or tok > 0) and not self._budget_ok(p, usd, tok):
                return
            for res, u in calls:                                      # the provider's verdict per member call
                if not res.can_call(self.now, u):
                    self.policy.on_external_429(res, self.now)
                    self._fail(p, "429", res)
                    return
            self._pay(p, usd, tok)
            for res, u in calls:
                res.take_call(self.now, u)
                self.policy.on_call_cost(res, self.now, res.call_cost(u))       # observable: the header delta this call caused
                if self.ledger.own(f"{res.name}@rpm", p.sid, self.now) > 0:      # the reserved budget unit is spent
                    self.ledger.consume(p.sid, f"{res.name}@rpm", self.now)
        self._start(p)

    def _budget_ok(self, p: Program, usd: float, tok: float) -> bool:
        """Admission against the tenant budgets: the policy may pace (wait), the provider refuses what it cannot pay for."""
        pd = self.pending[p.sid]
        if not self.policy.budget_admit(p, pd.step, usd, tok, self.now):
            self._return_partial(p)
            delay = self.policy.budget_retry_after(p, usd, tok, self.now)
            if not delay > 0:
                raise ValueError(f"budget_retry_after must be > 0, got {delay}")
            self._at(self.now + delay, "TRY_RUN", p.sid)
            return False
        short = [(n, amt) for n, amt in (("usd", usd), ("tokens", tok)) if n in self.budgets and amt > 0 and not self.budgets[n].can_pay(self.now, amt)]
        if short:
            self.budget_refusals += 1
            self.budgets[short[0][0]].refused += 1
            self.policy.on_event("budget_refused", p, self.now, budget=short[0][0], amount=short[0][1])
            self._fail(p, "budget", self.res[pd.step.resource] if pd.step.resource else self.res["model.slots"])
            return False
        return True

    def _pay(self, p: Program, usd: float, tok: float) -> None:
        """Debit (or refund, when negative) the tenant budgets and the session's counters."""
        if "usd" in self.budgets and usd != 0:
            self.budgets["usd"].pay(self.now, usd)
        if "tokens" in self.budgets and tok != 0:
            self.budgets["tokens"].pay(self.now, tok)
        p.spend_usd += usd
        p.spend_tokens += tok

    def _return_partial(self, p: Program) -> None:
        pd = self.pending[p.sid]
        for name, amt in pd.partial:
            self._release(p, name, amt)
        pd.partial.clear()
        pd.parallel_ok = False                                            # extra units, if any, went back too

    def _wait(self, p: Program, res: Resource, amt: float) -> None:
        pd = self.pending[p.sid]
        pd.waiting_for, pd.waiting_amt = res.name, amt
        res.waiters[p.sid] = pd.ready_at
        self.wait_open[p.sid] = Span(trace_id=p.trace, span_id=f"{p.sid}-w{pd.uid}-{pd.attempt}-{len(self.spans)}", parent_span_id=f"{p.sid}-root",
                                     op="wait", name=pd.step.name, tier=res.tier, resource=res.name, t_start=self.now,
                                     t_end=self.now, step_idx=p.step_idx,
                                     attrs={"holds": sorted(p.holds), "waits_for": res.name})
        if not pd.gang_wait and self._in_wait_cycle(p.sid):        # gang waits end at their lease's expiry: not a cycle
            self.deadlocks_detected += 1
            self.wait_open[p.sid].attrs["deadlock_cycle"] = True
            self._at(self.now + float(self.cfg["deadlock_timeout_s"]), "DEADLOCK_BREAK", p.sid, wid=self.wait_open[p.sid].span_id)

    def _in_wait_cycle(self, start: str) -> bool:
        def holders_of(sid: str) -> list[str]:
            pd = self.pending.get(sid)
            if pd is None:
                q = self.programs[sid]
                if q.join_left > 0:                                   # an orchestrator waits for its children (B10)
                    return [c for c in self.children[sid] if self.programs[c].status == "active"]
                return []
            if pd.waiting_for is None:
                return []
            return [h for h in self.res[pd.waiting_for].holders if h != sid]
        stack, seen = holders_of(start), set()
        while stack:
            s = stack.pop()
            if s == start:
                return True
            if s in seen:
                continue
            seen.add(s)
            stack.extend(holders_of(s))
        return False

    def _close_wait(self, sid: str, outcome: str) -> None:
        w = self.wait_open.pop(sid, None)
        if w is not None:
            w.t_end, w.outcome, w.wait = self.now, outcome, self.now - w.t_start
            self.spans.append(w)
        pd = self.pending.get(sid)
        if pd is not None:
            pd.waiting_for = None

    def _dequeue(self, sid: str, outcome: str) -> None:
        for res in self.res.values():
            res.waiters.pop(sid, None)
        pd = self.pending.get(sid)
        if pd is not None:
            pd.gang_wait = False
        self._close_wait(sid, outcome)

    def _release(self, p: Program, name: str, amt: float | None = None) -> None:
        res = self.res[name]
        res.give(p.sid, amt)
        if amt is None or p.holds.get(name, 0.0) - amt <= 1e-9:
            p.holds.pop(name, None)
        else:
            p.holds[name] -= amt
        self._wake(res)

    def _wake(self, res: Resource) -> None:
        """Hand freed capacity to the best fitting waiters. A release that happens *inside* this loop (a woken
        step took units, then bounced and returned them) does not recurse into a nested wake: the loop
        rescans anyway. Without this, a long queue bouncing on a header pause recursed once per waiter (F21)."""
        if res.name in self._waking:
            return
        self._waking.add(res.name)
        try:
            self._wake_loop(res)
        finally:
            self._waking.discard(res.name)

    def _wake_loop(self, res: Resource) -> None:
        self.ledger.expire(self.now)
        if res.name == "sandbox.cpu":                                 # gang waiters whose lease is gone run sequentially now
            for sid in list(res.waiters):
                pd = self.pending.get(sid)
                if pd is not None and pd.gang_wait and pd.waiting_for == res.name and self.ledger.own(res.name, sid, self.now) == 0.0:
                    del res.waiters[sid]
                    pd.gang_wait = False
                    self._close_wait(sid, "ok")
                    self._try_run(self.programs[sid])
        while res.waiters:
            best, best_sid = None, None
            for sid in res.waiters:
                pd = self.pending.get(sid)
                if pd is None or pd.waiting_for != res.name:
                    continue
                if res.name == "model.kv":
                    fits = self.model.fits(int(pd.waiting_amt))
                else:
                    fits = self._free(res, sid) >= pd.waiting_amt - 1e-9
                if not fits:
                    continue
                pr = self.policy.priority(self.programs[sid], pd.step, pd.ready_at, self.now)
                if best is None or pr < best:
                    best, best_sid = pr, sid
            if best_sid is None:
                for sid in [s for s in res.waiters if s not in self.pending]:
                    del res.waiters[sid]
                return
            del res.waiters[best_sid]
            self.pending[best_sid].gang_wait = False
            self._close_wait(best_sid, "ok")
            self._try_run(self.programs[best_sid])

    def _wake_kv(self) -> None:
        self._wake(self.res["model.kv"])

    # ---- failures, timeouts, reactions ------------------------------------------------
    def _fail(self, p: Program, outcome: str, res: Resource) -> None:
        pd = self.pending[p.sid]
        self._return_partial(p)
        self._span(p, op={"chat": "chat", "tool": "execute_tool", "retrieval": "retrieval"}[pd.step.kind],
                   name=pd.step.name, tier=res.tier, resource=res.name, t_start=self.now, t_end=self.now,
                   outcome=outcome, wait=self.now - pd.ready_at, attrs={"attempt": pd.attempt})
        self._react(p, outcome)

    def _react(self, p: Program, cause: str) -> None:
        pd = self.pending[p.sid]
        action = self.wl.react(p, pd.sdk_tries, self.policy.sdk_retries(), self.rp["after_sdk"], int(self.rp["max_episodes"]))
        if action == "retry":
            pd.sdk_tries += 1
            pd.attempt += 1
            pd.ready_at = self.now
            delay = self.policy.retry_delay(pd.sdk_tries)
            self._at(self.now + delay, "TRY_RUN", p.sid)
            self._at(self.now + delay + self.step_timeout, "WAIT_TIMEOUT", p.sid)
            return
        if action == "abort":
            p.requests_failed += 1
            self._close_request(p.sid, "aborted")
            self.policy.on_event("request_abort", p, self.now, cause=cause)
            self._end_session(p, "aborted")
            return
        extra = self.wl.next_chat(p)                                  # replan: think again, then redo the failed step
        if pd.step.kind != "chat":
            extra.follow_tools = [pd.step] + pd.tools_left
        self._ready(p, extra, [])

    def _ev_WAIT_TIMEOUT(self, sid: str) -> None:
        pd = self.pending[sid]
        if pd.running and pd.step.kind == "tool":
            return                                                    # running tools time out by their resource limit
        self._timeout_step(self.programs[sid], "timeout")

    def _ev_DEADLOCK_BREAK(self, sid: str, wid: str) -> None:
        w = self.wait_open.get(sid)
        if w is None or w.span_id != wid:
            return
        self._timeout_step(self.programs[sid], "deadlock")

    def _timeout_step(self, p: Program, cause: str) -> None:
        """The client gave up on this step (SDK timeout) or a deadlock was broken: everything the
        step holds is returned, the pinned request (if any) errors, and the agent reacts."""
        pd = self.pending[p.sid]
        step = pd.step
        if pd.waiting_for is not None:
            self._dequeue(p.sid, cause)
        self._return_partial(p)
        if pd.running:
            if step.kind == "chat":
                self.model.drop(p.sid, self.now)
                self._wake_kv()
            self._finish_step_holds(p, pd, ok=False)
        if step.kind == "chat" and "model.slots" in p.holds:
            self._release(p, "model.slots")
        if step.kind != "chat" and "model.slots" in p.holds:              # pinned request across the tool errored
            self._release(p, "model.slots")
        self._span(p, op={"chat": "chat", "tool": "execute_tool", "retrieval": "retrieval"}[step.kind], name=step.name,
                   tier=self.res[step.resource].tier if step.resource else "model", resource=step.resource or "model.slots",
                   t_start=self.now, t_end=self.now, outcome="timeout", wait=self.now - pd.ready_at,
                   attrs={"attempt": pd.attempt, "cause": cause})
        pd.attempt += 1                                                  # running/scheduled events become stale
        self._react(p, "timeout")

    # ---- running a step ----------------------------------------------------------------
    def _commit(self, p: Program) -> None:
        pd = self.pending[p.sid]
        session_level = {"sandbox.mem", "sandbox.cpu"}
        for name, amt in pd.partial:
            if name in session_level and name not in p.holds:
                p.holds[name] = 0.0
            p.holds[name] = p.holds.get(name, 0.0) + amt
            if name not in session_level and name != "model.slots":
                pd.own[name] = pd.own.get(name, 0.0) + amt
        if pd.parallel_ok:
            pd.own["sandbox.cpu"] = float(self._extra_cpu(pd.step))       # extra units belong to the step
        pd.partial.clear()
        pd.running = True

    def _start(self, p: Program) -> None:
        pd = self.pending[p.sid]
        step = pd.step
        self._commit(p)
        pd.started_at = self.now
        self.policy.on_event("step_start", p, self.now, step_kind=step.kind, name=step.name, tokens_in=step.tokens_in,
                             members=[m.name for m in step.members()], parallel=pd.parallel_ok)
        if step.kind == "chat":
            pd.fresh, pd.prefill_s, evicted = self.model.begin_prefill(p.sid, step.tokens_in, self.now)
            for v in evicted:
                self._evict(self.programs[v])
            self._at(self.now + pd.prefill_s, "DECODE_START", p.sid)
            return
        if step.kind == "retrieval":
            self._at(self.now + step.duration, "RETRIEVAL_END", p.sid)
            return
        cold = 0.0
        if any(m.resource.startswith("sandbox") for m in step.members()) and p.sandbox_cold:
            cold, p.sandbox_cold = self.cold_start, False
        sequential = bool(step.siblings) and not pd.parallel_ok           # thread pool full: run members one after another
        end, outcome, t = 0.0, "ok", 0.0
        for m in step.members():
            lim = self.res[m.resource].timeout_s
            d = m.duration + cold
            if d > lim:
                end, outcome = (t + lim if outcome == "ok" else min(end, t + lim)), "timeout"
            elif outcome == "ok":
                end = max(end, t + d)
            if sequential:
                t += d
        self._at(self.now + end, "TOOL_END", p.sid, outcome=outcome)

    def _ev_DECODE_START(self, sid: str) -> None:
        pd = self.pending[sid]
        dur, evicted = self.model.begin_decode(sid, pd.step.tokens_out, self.now)
        for v in evicted:
            self._evict(self.programs[v])
        self._at(self.now + dur, "CHAT_END", sid, decode_s=dur)

    def _evict(self, victim: Program) -> None:
        """A decoding request lost its KV to memory pressure: record it, restart it from prefill."""
        pd = self.pending[victim.sid]
        self._span(victim, op="chat", name=pd.step.name, tier="model", resource="model.kv", t_start=pd.started_at,
                   t_end=self.now, outcome="evicted", tokens_in=pd.step.tokens_in, tokens_fresh=pd.fresh,
                   duration=self.now - pd.started_at, wait=pd.started_at - pd.ready_at, attrs={"attempt": pd.attempt})
        pd.attempt += 1
        pd.running = False
        pd.ready_at = self.now
        self._at(self.now, "TRY_RUN", victim.sid)
        self._at(self.now + self.step_timeout, "WAIT_TIMEOUT", victim.sid)

    def _ev_CHAT_END(self, sid: str, decode_s: float) -> None:
        p, pd = self.programs[sid], self.pending[sid]
        step = pd.step
        gap = None                                                    # hidden: time until this session's next chat
        if step.follow_tools:
            gap = sum(t.duration for t in step.follow_tools) + (self.cold_start if p.sandbox_cold else 0.0)
        self.model.end_decode(sid, self.now, self.policy.retention_ttl(p, step, self.now, gap, self.model.kv_ttl_s))
        self.bucket.debit(max(0, step.tokens_out - pd.reserve), self.now)
        self._pay(p, (step.tokens_out - pd.reserve) * self.price_out, float(step.tokens_out - pd.reserve))   # settle the output against the reserve (may refund)
        self._wake_kv()
        dur = pd.prefill_s + decode_s
        p.attained_service += dur
        self.wl.after_chat(p, step)                                   # draws the next phase; sets the response's plan cue
        self._span(p, op="chat", name=step.name, tier="model", resource="model.slots", t_start=pd.started_at, t_end=self.now,
                   tokens_in=step.tokens_in, tokens_out=step.tokens_out, tokens_fresh=pd.fresh, duration=dur,
                   wait=pd.started_at - pd.ready_at,
                   attrs={"attempt": pd.attempt, "n_tools": sum(len(t.members()) for t in step.follow_tools), "phase": step.name,
                          "usd": round(pd.fresh * self.price_in + step.tokens_out * self.price_out, 6),
                          "content": step.content, "plan": step.plan, "spawn": step.spawn, "is_final": step.is_final,
                          "tool_kinds": [m.name for t in step.follow_tools for m in t.members()],
                          "tool_resources": [m.resource for t in step.follow_tools for m in t.members()],
                          "tool_content": [m.content for t in step.follow_tools for m in t.members()]})
        chat_span = self.spans[-1]
        p.step_idx += 1
        p.chats_in_request += 1
        pd.running = False
        group = step.follow_tools[0] if step.follow_tools and step.follow_tools[0].siblings else None
        self.policy.on_event("step_end", p, self.now, step_kind="chat", name=step.name, duration=dur, tokens_in=step.tokens_in,
                             tokens_out=step.tokens_out, n_tools=sum(len(t.members()) for t in step.follow_tools),
                             parallel_group=[m.name for m in group.members() if m.resource.startswith("sandbox")] if group else [],
                             tool_resources=[m.resource for t in step.follow_tools for m in t.members()],   # the calls the chat issued
                             tool_kinds=[m.name for t in step.follow_tools for m in t.members()],
                             tool_content=[m.content for t in step.follow_tools for m in t.members()],
                             plan=step.plan, spawn=step.spawn, is_final=step.is_final,
                             usd=pd.fresh * self.price_in + step.tokens_out * self.price_out)     # the provider's usage header
        self._replace_leases(p, step)
        has_tools = bool(step.follow_tools)
        if step.is_final or not has_tools or self.policy.release_model_for_tools(self.wl.hold_model_during_tool):
            self._release(p, "model.slots")
        self.pending.pop(sid)
        if step.is_final:
            self._end_request(p)
        elif step.spawn:
            self._spawn(p, step.spawn, chat_span, step)
        elif has_tools:
            self._ready(p, step.follow_tools[0], step.follow_tools[1:])
        else:
            self._ready_chat(p)

    def _finish_step_holds(self, p: Program, pd: Pending, ok: bool) -> None:
        """The one place a step's own holds go back (normal end, resource timeout, client timeout)."""
        for name, amt in list(pd.own.items()):
            self._release(p, name, amt)
            if ok and self.res[name].calls is not None:
                self.policy.on_external_ok(self.res[name], self.now)
        pd.own.clear()
        pd.parallel_ok = False
        pd.running = False

    def _ev_TOOL_END(self, sid: str, outcome: str) -> None:
        p, pd = self.programs[sid], self.pending[sid]
        step = pd.step
        ran_parallel = bool(step.siblings) and pd.parallel_ok          # read before the holds (and the flag) go back
        self._finish_step_holds(p, pd, outcome == "ok")
        local = any(m.resource.startswith("sandbox") for m in step.members())
        self._span(p, op="execute_tool", name=step.name, tier="tool", resource=step.resource or "parallel",
                   t_start=pd.started_at, t_end=self.now, duration=self.now - pd.started_at, wait=pd.started_at - pd.ready_at,
                   outcome=outcome, mem=p.sandbox_gb if local else 0.0,
                   attrs={"attempt": pd.attempt, "members": [m.name for m in step.members()], "parallel": ran_parallel,
                          "usd": round(sum(self.res[m.resource].cost_per_call for m in step.members()), 6),
                          "content": [m.content for m in step.members()], "durations": [round(m.duration, 4) for m in step.members()]})
        p.step_idx += 1
        if outcome == "timeout":                                      # observable: the call was killed at its resource limit
            self.policy.on_event("step_end", p, self.now, step_kind="tool", name=step.name, duration=self.now - pd.started_at,
                                 members=[m.name for m in step.members()], parallel=ran_parallel, outcome="timeout", errors=1,
                                 content=[m.content for m in step.members()],
                                 durations=[min(m.duration, self.res[m.resource].timeout_s) for m in step.members()])
            pd.attempt += 1
            self._react(p, "timeout")
            return
        p.last_tool = step.members()[-1].name
        self.policy.on_event("step_end", p, self.now, step_kind="tool", name=step.name, duration=self.now - pd.started_at,
                             members=[m.name for m in step.members()], parallel=ran_parallel, outcome="ok", errors=0,
                             content=[m.content for m in step.members()], durations=[m.duration for m in step.members()],
                             usd=sum(self.res[m.resource].cost_per_call for m in step.members()))
        self._replace_leases(p, step)
        self.wl.after_tool(p, step)
        tools_left = pd.tools_left
        self.pending.pop(sid)
        if tools_left:
            self._ready(p, tools_left[0], tools_left[1:])
        else:
            self._ready_chat(p)

    def _ev_RETRIEVAL_END(self, sid: str) -> None:
        p, pd = self.programs[sid], self.pending[sid]
        step = pd.step
        self._finish_step_holds(p, pd, True)
        self._span(p, op="retrieval", name="retrieval", tier="service", resource=step.resource, t_start=pd.started_at,
                   t_end=self.now, duration=self.now - pd.started_at, wait=pd.started_at - pd.ready_at)
        p.step_idx += 1
        self.policy.on_event("step_end", p, self.now, step_kind="retrieval", name="retrieval", duration=self.now - pd.started_at)
        self._replace_leases(p, step)
        self.wl.after_tool(p, step)
        self.pending.pop(sid)
        self._ready(p, self.wl.next_chat(p), [])

    # ---- spans -------------------------------------------------------------------------
    def _span(self, p: Program, **kw) -> None:
        kw.setdefault("step_idx", p.step_idx)
        self.spans.append(Span(trace_id=p.trace, span_id=f"{p.sid}-{len(self.spans)}", parent_span_id=f"{p.sid}-root", **kw))
