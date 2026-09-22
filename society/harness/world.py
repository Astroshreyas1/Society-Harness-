"""The world the society runs in: a clock in *society seconds* and the shared resources every DAG run contends for.

Society seconds are the recorded runs' seconds; `Clock.scale` divides them into wall-clock seconds so a bench of
dozens of runs finishes in minutes while every rate (requests per minute, dollars per hour) keeps its recorded
meaning. The resources reuse the simulator's physics classes unchanged (agentsim/resources.py):

  model.anthropic   the cloud model tier: concurrent requests (capacity), requests per minute (a CallBucket), tokens
                    per minute (a TokenBucket) — api-like: exceeding it is a 429 when nobody coordinates
  model.ollama      the local coder: one resident model, one generation at a time — OS-like: callers queue
  sandbox.cpu       the pool of sandboxes that run the agents' shell tools and the gates' test runs — OS-like
  budgets           one BudgetBucket per tenant (USD, refilled per hour): the platform refuses what it cannot pay for
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field

from agentsim.resources import BudgetBucket, Resource, TokenBucket

DEFAULT_WORLD = {
    "anthropic": {"concurrency": 3, "rpm": 6.0, "tpm": 150_000.0, "price_in": 5.0, "price_out": 25.0},
    "ollama": {"slots": 1},
    "sandbox": {"cpu": 2},
    "budget": {"usd": 1.5, "per_hour": 6.0, "tenants": 3},
    "scale": 20.0,
}

PROVIDER_RESOURCE = {"anthropic": "model.anthropic", "ollama": "model.ollama"}


class Clock:
    def __init__(self, scale: float = 20.0):
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.scale = float(scale)
        self.t0 = time.monotonic()

    def now(self) -> float:
        return (time.monotonic() - self.t0) * self.scale

    async def sleep(self, society_s: float) -> None:
        if society_s > 0:
            await asyncio.sleep(society_s / self.scale)


@dataclass
class World:
    clock: Clock
    resources: dict[str, Resource]
    budgets: dict[str, BudgetBucket]
    tpm: dict[str, TokenBucket] = field(default_factory=dict)
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)      # resource -> (usd per Mtok in, out)
    spec: dict = field(default_factory=dict)

    def resource_of(self, provider: str) -> str:
        try:
            return PROVIDER_RESOURCE[provider]
        except KeyError:
            raise KeyError(f"no resource for provider {provider!r}") from None

    def tenant_of(self, run_index: int) -> str:
        n = max(1, int(self.spec["budget"]["tenants"]))
        return f"tenant{run_index % n}"

    def now(self) -> float:
        return self.clock.now()


def build_world(spec: dict | None = None, **overrides) -> World:
    """`spec` follows DEFAULT_WORLD; keyword overrides take dotted keys (anthropic.rpm=12)."""
    s = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULT_WORLD.items()}
    for k, v in (spec or {}).items():
        if isinstance(v, dict):
            s.setdefault(k, {}).update(v)
        else:
            s[k] = v
    for key, val in overrides.items():
        sec, _, name = key.partition(".")
        if name:
            s[sec][name] = val
        else:
            s[sec] = val
    a = s["anthropic"]
    resources = {
        "model.anthropic": Resource("model.anthropic", float(a["concurrency"]), "model", api_like=True, rpm=float(a["rpm"])),
        "model.ollama": Resource("model.ollama", float(s["ollama"]["slots"]), "model", api_like=False),
        "sandbox.cpu": Resource("sandbox.cpu", float(s["sandbox"]["cpu"]), "sandbox", api_like=False),
    }
    b = s["budget"]
    budgets = {}
    if b.get("usd", 0) and b["usd"] > 0:
        budgets = {f"tenant{i}": BudgetBucket(float(b["usd"]), float(b["per_hour"])) for i in range(int(b["tenants"]))}
    tpm = {"model.anthropic": TokenBucket(float(a["tpm"]))} if a.get("tpm") and not math.isinf(float(a["tpm"])) else {}
    return World(Clock(float(s["scale"])), resources, budgets, tpm, {"model.anthropic": (float(a["price_in"]), float(a["price_out"]))}, s)
