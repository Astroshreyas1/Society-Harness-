"""Resources and their physics: capacities, holders, waiters, provider rate limits with headers,
the model tier's KV/latency model (TTL prefix cache, admission on active KV, SIC least-progressed
eviction), and the TPM bucket.

Resource names (the three tiers of v1 §1.1):
  model.slots   concurrent requests at the endpoint (API concurrency or server slots)
  model.kv      KV-cache tokens on the instance (admission by the physics, not by take/give)
  model.tpm     tokens-per-minute bucket (API tier); debited on *fresh* tokens + output
  sandbox.mem   GB, held while a session's sandbox exists (parked on idle)
  sandbox.cpu   CPU slots, one per live sandbox plus extra units for parallel local tools
  ext.<tool>    external tool/API: concurrency cap + RPM bucket + background load; headers exposed
  mcp.<server>  an MCP tool server: the same physics as ext.* (concurrency, RPM, latency limit) — tool kinds map to it
  gpu.<pool>    a local accelerator pool for tools (embedding, local inference): concurrency units
  svc.retrieval service-tier retrieval in-flight calls
Every tool-tier resource may carry `cost_per_call` (USD) and `tokens_per_call`; the model tier carries prices per
million tokens. Calls and chats debit the tenant's `BudgetBucket`s (USD, tokens): a call the budget cannot pay for
fails with outcome "budget" — the failure mode the controller's pacing exists to prevent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


class CallBucket:
    """Requests-per-minute bucket with continuous refill; `remaining_frac` is the rate-limit header."""

    def __init__(self, rpm: float):
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.rpm, self.level, self.t = rpm, rpm, 0.0

    def _refill(self, now: float) -> None:
        self.level = min(self.rpm, self.level + (now - self.t) * self.rpm / 60.0)
        self.t = now

    def can_call(self, now: float, calls: float = 1.0) -> bool:
        self._refill(now)
        return self.level >= calls - 1e-9

    def take(self, now: float, calls: float = 1.0) -> None:
        self._refill(now)
        self.level -= calls

    def remaining_frac(self, now: float) -> float:
        self._refill(now)
        return max(0.0, self.level) / self.rpm

    def wait_for(self, now: float, calls: float = 1.0) -> float:
        self._refill(now)
        return max(0.001, (calls - self.level) * 60.0 / self.rpm)


@dataclass
class Resource:
    name: str
    capacity: float
    tier: str
    api_like: bool                       # exceeding it yields a 429 (True) or a blocking queue (False) when uncoordinated
    rpm: float = math.inf                # provider request budget; inf = none
    background_load: float = 0.0         # other tenants: probability that a call costs one extra budget unit
    timeout_s: float = math.inf
    cost_per_call: float = 0.0           # USD per call (external APIs, MCP servers)
    tokens_per_call: float = 0.0         # tokens a call consumes from a token budget (token-metered providers)
    used: float = 0.0
    holders: dict[str, float] = field(default_factory=dict)      # sid -> amount
    waiters: dict[str, float] = field(default_factory=dict)      # sid -> ready_at (policy orders them)
    calls: CallBucket | None = None

    def __post_init__(self) -> None:
        if not math.isinf(self.rpm):
            self.calls = CallBucket(self.rpm)
        if not 0 <= self.background_load < 1:
            raise ValueError(f"{self.name}: background_load must be in [0,1)")

    def free(self, cap: float) -> float:
        return cap - self.used

    def take(self, sid: str, amt: float) -> None:
        self.used += amt
        self.holders[sid] = self.holders.get(sid, 0.0) + amt

    def give(self, sid: str, amt: float | None = None) -> None:
        held = self.holders.get(sid, 0.0)
        if held <= 0:
            raise RuntimeError(f"{sid} releases {self.name} it does not hold")
        amt = held if amt is None else amt
        if amt > held + 1e-9:
            raise RuntimeError(f"{sid} releases {amt} of {self.name} but holds {held}")
        self.used -= amt
        if held - amt <= 1e-9:
            del self.holders[sid]
        else:
            self.holders[sid] = held - amt

    # -- provider budget (headers) -------------------------------------------------------
    def call_cost(self, u: float) -> float:
        """Budget units one call costs: 1, plus 1 when another tenant's call lands first (u < background_load)."""
        return 1.0 + (1.0 if u < self.background_load else 0.0)

    def can_call(self, now: float, u: float) -> bool:
        return self.calls is None or self.calls.can_call(now, self.call_cost(u))

    def take_call(self, now: float, u: float) -> None:
        if self.calls is not None:
            self.calls.take(now, self.call_cost(u))

    def remaining_frac(self, now: float) -> float:
        return 1.0 if self.calls is None else self.calls.remaining_frac(now)

    def budget_wait(self, now: float) -> float:
        return 0.0 if self.calls is None else self.calls.wait_for(now, 1.0)

    def check(self) -> None:
        """Accounting invariant: used equals the sum of holders, never negative."""
        s = sum(self.holders.values())
        if abs(s - self.used) > 1e-6 or self.used < -1e-6:
            raise AssertionError(f"{self.name}: used={self.used} but holders sum to {s}")


class BudgetBucket:
    """A tenant budget (USD or tokens) that refills continuously at `per_hour` up to `capacity` (the allowance of one
    period). `remaining_frac` is what a spend header would show; `can_pay` / `pay` are the provider's verdict."""

    def __init__(self, capacity: float, per_hour: float):
        if capacity <= 0 or per_hour < 0:
            raise ValueError("budget needs capacity > 0 and per_hour >= 0")
        self.capacity, self.per_hour, self.level, self.t = float(capacity), float(per_hour), float(capacity), 0.0
        self.spent, self.refused = 0.0, 0

    def _refill(self, now: float) -> None:
        self.level = min(self.capacity, self.level + (now - self.t) * self.per_hour / 3600.0)
        self.t = now

    def can_pay(self, now: float, amount: float) -> bool:
        self._refill(now)
        return self.level >= amount - 1e-9

    def pay(self, now: float, amount: float) -> None:
        self._refill(now)
        self.level -= amount
        self.spent += amount

    def remaining_frac(self, now: float) -> float:
        self._refill(now)
        return max(0.0, self.level) / self.capacity

    def wait_for(self, now: float, amount: float) -> float:
        self._refill(now)
        if self.per_hour <= 0:
            return math.inf
        return max(0.001, (amount - self.level) * 3600.0 / self.per_hour)


class TokenBucket:
    """Tokens-per-minute bucket with continuous refill (API-tier rate limit on fresh tokens)."""

    def __init__(self, tpm: float):
        if tpm <= 0:
            raise ValueError("tpm must be positive")
        self.tpm, self.level, self.t = tpm, tpm, 0.0

    def _refill(self, now: float) -> None:
        self.level = min(self.tpm, self.level + (now - self.t) * self.tpm / 60.0)
        self.t = now

    def try_take(self, tokens: float, now: float, fraction: float = 1.0) -> bool:
        self._refill(now)
        if self.level - tokens >= (1.0 - fraction) * self.tpm - 1e-6:   # keep (1-fraction) as headroom
            self.level -= tokens
            return True
        return False

    def debit(self, tokens: float, now: float) -> None:
        """Unconditional debit (actual output tokens at completion); may go negative → later waits."""
        self._refill(now)
        self.level -= tokens

    def wait_for(self, tokens: float, now: float, fraction: float) -> float:
        self._refill(now)
        deficit = tokens - (self.level - (1.0 - fraction) * self.tpm)
        return max(0.001, deficit * 60.0 / self.tpm)          # never a zero-time retry


@dataclass(frozen=True)
class Lease:
    """A reservation (v1 §6.2): `amt` units of `res` for session `sid`, usable from `start`, gone at `expiry`.
    Backfillable by construction: only the *unconsumed* part (amt minus what the holder already holds) is
    kept from others, and nothing survives its expiry (I3)."""
    sid: str
    res: str
    amt: float
    start: float
    expiry: float

    def __post_init__(self) -> None:
        if self.amt <= 0 or self.expiry <= self.start:
            raise ValueError(f"lease must have amt > 0 and expiry > start: {self}")
        if self.res in ("model.kv", "model.tpm"):
            raise ValueError(f"leases on {self.res} are not allowed (admission stays in ModelPhysics / TokenBucket)")

    @property
    def is_budget(self) -> bool:
        """A reservation of provider call-budget units (`<resource>@rpm`) rather than of concurrency units."""
        return self.res.endswith("@rpm")

    @property
    def base(self) -> str:
        return self.res[:-4] if self.is_budget else self.res


class Ledger:
    """All live leases, by resource and session. Single-node; invariants I1 (sum of active leases <= capacity)
    and I3 (nothing past expiry) are asserted by `check` / enforced by `expire`."""

    def __init__(self) -> None:
        self.by_res: dict[str, dict[str, Lease]] = {}
        self.expired_total = 0
        self.issued_total = 0
        self.unit_s_issued = 0.0          # sum of amt x (expiry - start) over every lease issued
        self.unit_s_unused = 0.0          # reserved unit-seconds that were never consumed (expired, replaced, or the wait before use)
        self.by_kind: dict[str, dict[str, float]] = {"cpu": {"issued": 0.0, "unused": 0.0}, "budget": {"issued": 0.0, "unused": 0.0}}

    def _retire(self, lease: Lease, now: float) -> None:
        """A lease leaves the ledger unconsumed: what it reserved until now (or its expiry) was never used."""
        wasted = lease.amt * max(0.0, min(now, lease.expiry) - lease.start)
        self.unit_s_unused += wasted
        self.by_kind["budget" if lease.is_budget else "cpu"]["unused"] += wasted

    def set_session(self, sid: str, leases: list[Lease], now: float = 0.0) -> None:
        """Replace every lease of `sid` with `leases` (the policy's whole current forecast for that session)."""
        for res in self.by_res.values():
            old = res.pop(sid, None)
            if old is not None:
                self._retire(old, now)
        seen = set()
        for lease in leases:
            if lease.sid != sid:
                raise ValueError(f"lease for {lease.sid} handed in under session {sid}")
            if lease.res in seen:
                raise ValueError(f"{sid}: two leases on {lease.res}; one per resource")
            seen.add(lease.res)
            self.by_res.setdefault(lease.res, {})[sid] = lease
            self.issued_total += 1
            self.unit_s_issued += lease.amt * (lease.expiry - lease.start)
            self.by_kind["budget" if lease.is_budget else "cpu"]["issued"] += lease.amt * (lease.expiry - lease.start)

    def consume(self, sid: str, res: str, now: float) -> None:
        """The holder took what it reserved: the lease is fulfilled; the wait before use was the reservation's cost."""
        lease = self.by_res.get(res, {}).pop(sid, None)
        if lease is None:
            raise RuntimeError(f"{sid}: consuming a lease on {res} it does not hold")
        waited = lease.amt * max(0.0, now - lease.start)
        self.unit_s_unused += waited
        self.by_kind["budget" if lease.is_budget else "cpu"]["unused"] += waited

    def active_sum(self, res: str, now: float) -> float:
        return sum(lease.amt for lease in self.by_res.get(res, {}).values() if lease.start <= now < lease.expiry)

    def active_max(self, res: str, t0: float, t1: float) -> float:
        """The largest sum of leases on `res` active at any instant of [t0, t1): what I1 leaves for a new lease there."""
        leases = [l for l in self.by_res.get(res, {}).values() if l.expiry > t0 and l.start < t1]
        if not leases:
            return 0.0
        points = {t0} | {l.start for l in leases if t0 <= l.start < t1}
        return max(sum(l.amt for l in leases if l.start <= t < l.expiry) for t in points)

    def expire(self, now: float) -> int:
        n = 0
        for res in self.by_res.values():
            for sid in [sid for sid, lease in res.items() if lease.expiry <= now]:
                self._retire(res.pop(sid), now)
                n += 1
        self.expired_total += n
        return n

    def next_expiry(self) -> float:
        return min((lease.expiry for res in self.by_res.values() for lease in res.values()), default=math.inf)

    @staticmethod
    def in_family(owner: str, sid: str) -> bool:
        """A lease held by an orchestrator covers its sub-agents (B10): child ids are `<parent>/<k>.<j>`."""
        return sid == owner or sid.startswith(owner + "/")

    @staticmethod
    def family_holds(owner: str, holders: dict[str, float]) -> float:
        return sum(amt for h, amt in holders.items() if h == owner or h.startswith(owner + "/"))

    def own(self, res: str, sid: str, now: float) -> float:
        """The session's own active lease (strict: a child's eligibility for its parent's lease is expressed only
        through `reserved_for_others`, never as a gang it waits for)."""
        lease = self.by_res.get(res, {}).get(sid)
        return lease.amt if lease is not None and lease.start <= now < lease.expiry else 0.0

    def reserved_for_others(self, res: str, sid: str, now: float, holders: dict[str, float]) -> float:
        """Capacity `sid` may not touch: other families' active leases minus what those families already hold."""
        total = 0.0
        for other, lease in self.by_res.get(res, {}).items():
            if not self.in_family(other, sid) and lease.start <= now < lease.expiry:
                total += max(0.0, lease.amt - self.family_holds(other, holders))
        return total

    @staticmethod
    def limit(resources: dict[str, "Resource"], res: str) -> float:
        """What I1 bounds a resource's leases by: its capacity, or its RPM bucket size for budget leases."""
        base = res[:-4] if res.endswith("@rpm") else res
        if base not in resources:
            raise KeyError(f"lease on unknown resource {res!r}")
        if res.endswith("@rpm"):
            if resources[base].calls is None:
                raise ValueError(f"{base} has no call budget to lease")
            return resources[base].rpm
        return resources[base].capacity

    def check(self, resources: dict[str, "Resource"], now: float) -> None:
        for res, leases in self.by_res.items():
            lim = self.limit(resources, res)
            active = sum(lease.amt for lease in leases.values() if lease.start <= now < lease.expiry)
            if active > lim + 1e-9:
                raise AssertionError(f"I1 violated on {res}: active leases {active} > limit {lim}")
            if any(lease.expiry <= now for lease in leases.values()):
                raise AssertionError(f"I3 violated on {res}: an expired lease is still in the ledger")


@dataclass
class KVEntry:
    tokens: int
    retained_until: float
    last_access: float


class ModelPhysics:
    """One model endpoint/instance: prefix cache with TTL retention, KV admission on active state
    (vLLM semantics), least-progressed-first eviction when generation outgrows the admitted reserve
    (Service-Induced Congestion), and a shared decode capacity."""

    def __init__(self, kv_capacity: int, prefill_rate: float, decode_base: float,
                 decode_capacity: float, kv_ttl_s: float, evict_key=None):
        for k, v in dict(kv_capacity=kv_capacity, prefill_rate=prefill_rate, decode_base=decode_base,
                         decode_capacity=decode_capacity).items():
            if v <= 0:
                raise ValueError(f"{k} must be positive")
        self.kv_capacity, self.prefill_rate = kv_capacity, prefill_rate
        self.decode_base, self.decode_capacity, self.kv_ttl_s = decode_base, decode_capacity, kv_ttl_s
        # idle-entry eviction order under pressure: the policy's view of "next use" (default: expired first, then LRU)
        self.evict_key = evict_key or (lambda sid, e, now: (e.retained_until >= now, e.last_access))
        self.retained: dict[str, KVEntry] = {}
        self.decoding: dict[str, tuple[int, float, float]] = {}   # sid -> (kv tokens, t_start, duration)
        self.evictions = 0
        self.retained_token_s_reused = 0.0      # idle KV that a later chat of the session did reuse (token·s)
        self.retained_token_s_unused = 0.0      # idle KV removed (expired, evicted, dropped) without reuse (token·s)

    def _account_idle_removal(self, sid: str, now: float) -> None:
        e = self.retained.get(sid)
        if e is not None and e.retained_until < math.inf:
            self.retained_token_s_unused += e.tokens * max(0.0, min(now, e.retained_until) - e.last_access)

    def used(self) -> int:
        return sum(e.tokens for e in self.retained.values()) + sum(t for t, _, _ in self.decoding.values())

    def active_tokens(self) -> int:
        """KV that cannot be reclaimed: prefixes of running requests plus decoding state."""
        return sum(e.tokens for e in self.retained.values() if e.retained_until == math.inf) + \
            sum(t for t, _, _ in self.decoding.values())

    def fits(self, need: int) -> bool:
        return self.active_tokens() + need <= self.kv_capacity

    def miss_tokens(self, sid: str, tokens_in: int, now: float) -> int:
        """Tokens that must be prefilled: everything unless a fresh retained entry is a prefix."""
        e = self.retained.get(sid)
        if e is None or e.retained_until < now or e.tokens > tokens_in:      # expired or context compacted
            return tokens_in
        return tokens_in - e.tokens

    def _make_room(self, need: int, now: float, keep: str) -> list[str]:
        evicted: list[str] = []
        if self.kv_capacity - self.used() >= need:
            return evicted
        idle = [s for s, e in self.retained.items() if e.retained_until < math.inf and s != keep]
        for sid in sorted(idle, key=lambda s: self.evict_key(s, self.retained[s], now)):
            self._account_idle_removal(sid, now)
            del self.retained[sid]
            if self.kv_capacity - self.used() >= need:
                return evicted
        while self.kv_capacity - self.used() < need and self.decoding:
            victim = min(self.decoding, key=lambda s: (now - self.decoding[s][1]) / self.decoding[s][2])
            self.decoding.pop(victim)
            self.retained.pop(victim, None)
            self.evictions += 1
            evicted.append(victim)
        if self.kv_capacity - self.used() < need:
            raise RuntimeError("KV capacity smaller than a single request; raise model.kv capacity")
        return evicted

    def begin_prefill(self, sid: str, tokens_in: int, now: float) -> tuple[int, float, list[str]]:
        miss = self.miss_tokens(sid, tokens_in, now)
        if miss == tokens_in:
            self._account_idle_removal(sid, now)
            self.retained.pop(sid, None)
        else:
            e = self.retained[sid]
            self.retained_token_s_reused += e.tokens * (now - e.last_access)
        evicted = self._make_room(miss, now, keep=sid)
        e = self.retained.setdefault(sid, KVEntry(tokens=0, retained_until=math.inf, last_access=now))
        e.tokens += miss
        e.retained_until, e.last_access = math.inf, now
        return miss, miss / self.prefill_rate, evicted

    def begin_decode(self, sid: str, tokens_out: int, now: float) -> tuple[float, list[str]]:
        evicted = self._make_room(tokens_out, now, keep=sid)
        n_active = len(self.decoding) + 1
        rate = min(self.decode_base, self.decode_capacity / n_active)
        dur = tokens_out / rate
        entry = self.retained.pop(sid)
        self.decoding[sid] = (entry.tokens + tokens_out, now, dur)
        return dur, evicted

    def end_decode(self, sid: str, now: float, ttl: float) -> None:
        """Generation finished: the prefix stays as an idle, evictable entry for `ttl` seconds (the policy's
        retention_ttl; the scenario's kv_ttl_s by default)."""
        if not ttl >= 0:
            raise ValueError(f"retention ttl must be >= 0, got {ttl}")
        tok, _, _ = self.decoding.pop(sid)
        self.retained[sid] = KVEntry(tokens=tok, retained_until=now + ttl, last_access=now)

    def set_ttl(self, sid: str, now: float, ttl: float) -> bool:
        """Re-set the idle entry's expiry once more is known (a final chat: the think time). Returns False
        when memory pressure already reclaimed the entry in between; an *active* entry is a logic error."""
        if not ttl >= 0:
            raise ValueError(f"retention ttl must be >= 0, got {ttl}")
        e = self.retained.get(sid)
        if e is None:
            return False
        if e.retained_until == math.inf:
            raise RuntimeError(f"{sid}: KV entry is active, not idle")
        e.retained_until = now + ttl
        return True

    def drop(self, sid: str, now: float) -> None:
        self.decoding.pop(sid, None)
        self._account_idle_removal(sid, now)
        self.retained.pop(sid, None)


def build_resources(cfg: dict) -> dict[str, Resource]:
    out: dict[str, Resource] = {}
    for name, r in cfg["resources"].items():
        out[name] = Resource(name=name, capacity=float(r["capacity"]), tier=r["tier"], api_like=bool(r["api_like"]),
                             rpm=float(r.get("rpm", math.inf)), background_load=float(r.get("background_load", 0.0)),
                             timeout_s=float(r.get("timeout_s", math.inf)), cost_per_call=float(r.get("cost_per_call", 0.0)),
                             tokens_per_call=float(r.get("tokens_per_call", 0.0)))
    for required in ("model.slots", "model.kv", "model.tpm", "sandbox.mem", "sandbox.cpu", "svc.retrieval"):
        if required not in out:
            raise KeyError(f"resources missing {required!r}")
    return out


def build_budgets(cfg: dict) -> dict[str, BudgetBucket]:
    """Tenant budgets from cfg["budgets"]: {"usd": {"capacity": 5.0, "per_hour": 5.0}, "tokens": {...}}; {} = none."""
    out = {}
    for name, b in cfg["budgets"].items():
        if name not in ("usd", "tokens"):
            raise KeyError(f"budgets: unknown budget {name!r} (usd | tokens)")
        out[name] = BudgetBucket(float(b["capacity"]), float(b["per_hour"]))
    return out
