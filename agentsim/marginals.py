"""Quantity marginals: distributions for tokens, durations, think times, memory.

Specs are plain dicts (JSON) so they can be seeded from published statistics now and
replaced by `fit` from real OTel-shaped traces at rung 0. Provenance lives next to
each seed in data/marginals.json.

Spec forms (anything else throws):
  {"type": "lognormal", "median": m, "p99": q}            # or "p90"
  {"type": "mixture", "weights": [...], "parts": [spec, ...]}
  {"type": "const", "value": v}
  {"type": "one_or", "p_one": p, "then": spec}            # 1 with prob p, else 1 + spec (integers)
Optional on any spec: "cap": upper bound applied after sampling.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Z = {"p90": 1.2815515655446004, "p99": 2.3263478740408408}


def lognormal_from_quantile(median: float, key: str, value: float) -> tuple[float, float]:
    if median <= 0 or value <= median:
        raise ValueError(f"need 0 < median < {key}: median={median}, {key}={value}")
    return math.log(median), math.log(value / median) / Z[key]


@dataclass(frozen=True)
class Dist:
    kind: str
    mu: float = 0.0
    sigma: float = 0.0
    value: float = 0.0
    weights: tuple[float, ...] = ()
    parts: tuple["Dist", ...] = ()
    p_one: float = 0.0
    cap: float = math.inf

    def sample(self, rng: np.random.Generator) -> float:
        if self.kind == "lognormal":
            x = float(rng.lognormal(self.mu, self.sigma))
        elif self.kind == "const":
            x = self.value
        elif self.kind == "mixture":
            i = int(rng.choice(len(self.parts), p=self.weights))
            x = self.parts[i].sample(rng)
        elif self.kind == "one_or":
            x = 1.0 if rng.random() < self.p_one else 1.0 + self.parts[0].sample(rng)
        else:
            raise ValueError(f"unknown dist kind {self.kind!r}")
        return min(x, self.cap)

    def mean_estimate(self, rng: np.random.Generator, n: int = 20000) -> float:
        return float(np.mean([self.sample(rng) for _ in range(n)]))


def build(spec: dict) -> Dist:
    cap = float(spec.get("cap", math.inf))
    t = spec["type"]
    if t == "lognormal":
        key = "p99" if "p99" in spec else "p90"
        mu, sigma = lognormal_from_quantile(float(spec["median"]), key, float(spec[key]))
        return Dist("lognormal", mu=mu, sigma=sigma, cap=cap)
    if t == "const":
        return Dist("const", value=float(spec["value"]), cap=cap)
    if t == "mixture":
        w = tuple(float(x) for x in spec["weights"])
        if abs(sum(w) - 1.0) > 1e-9:
            raise ValueError(f"mixture weights must sum to 1, got {sum(w)}")
        return Dist("mixture", weights=w, parts=tuple(build(p) for p in spec["parts"]), cap=cap)
    if t == "one_or":
        return Dist("one_or", p_one=float(spec["p_one"]), parts=(build(spec["then"]),), cap=cap)
    raise ValueError(f"unknown spec type {t!r}")


class Marginals:
    """Named distributions. Missing names throw at lookup, not at sample time."""

    REQUIRED = (
        "tool_duration", "output_tokens", "append_tokens", "initial_context_tokens",
        "think_time", "session_requests", "sandbox_mem_gb", "retrieval_duration",
    )

    def __init__(self, spec: dict):
        self.spec = spec
        for k in self.REQUIRED:
            if k not in spec:
                raise KeyError(f"marginals.json missing {k!r}")
        self._d: dict[str, Dist] = {}
        for name, s in spec.items():
            if name == "provenance":
                continue
            if isinstance(s, dict) and "type" in s:
                self._d[name] = build(s)
            elif isinstance(s, dict):                      # keyed family, e.g. tool_duration[kind]
                for sub, ss in s.items():
                    self._d[f"{name}.{sub}"] = build(ss)
            else:
                raise ValueError(f"bad marginal entry {name!r}")

    def get(self, name: str) -> Dist:
        if name not in self._d:
            raise KeyError(f"no marginal {name!r}; have {sorted(self._d)}")
        return self._d[name]

    def sample(self, name: str, rng: np.random.Generator) -> float:
        return self.get(name).sample(rng)

    @classmethod
    def load(cls, path: Path) -> "Marginals":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
