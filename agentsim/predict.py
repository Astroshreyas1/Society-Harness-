"""Cheap online predictors for rung 3 (v2 §6.1: online Markov for structure, streaming quantiles for
quantities). They learn only from what a policy is told through `Policy.on_event` — realised step
durations, realised idle gaps, observed step kinds — never from a Step's hidden fields.

QuantileTracker  per-key bounded window of recent values; `quantile(key, tau)` backs off to shorter
                 key prefixes (e.g. (recipe, bucket) -> (recipe,) -> ()) until `min_n` values exist;
                 None when nothing was seen at any level.
MarkovNext       k-th order transition counts over tokens with backoff to shorter contexts and to the
                 unigram; `predict(hist)` returns a pmf over next tokens, `prob(hist, token)` one entry.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np


class QuantileTracker:
    def __init__(self, window: int = 512, min_n: int = 8, lazy: bool = False):
        if window < min_n or min_n < 1:
            raise ValueError("need window >= min_n >= 1")
        self.window, self.min_n, self.lazy = window, min_n, lazy
        self.vals: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=window))
        self._cache: dict[tuple[tuple, float], float] = {}
        self._seen: dict[tuple, int] = defaultdict(int)       # observations per key (lazy mode: recompute every +12.5%)
        self._at: dict[tuple[tuple, float], int] = {}

    def observe(self, key: tuple, x: float) -> None:
        if not isinstance(key, tuple):
            raise TypeError("key must be a tuple (prefixes are the back-off levels)")
        if not x >= 0:
            raise ValueError(f"observations must be >= 0, got {x}")
        for i in range(len(key), -1, -1):                      # the key and every prefix, down to ()
            self.vals[key[:i]].append(x)
            self._seen[key[:i]] += 1
        if not self.lazy:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != key[:len(k[0])]}   # drop stale prefixes

    def n(self, key: tuple) -> int:
        return len(self.vals.get(key, ()))

    def quantile(self, key: tuple, tau: float) -> float | None:
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be in (0, 1)")
        for i in range(len(key), -1, -1):
            k = key[:i]
            if self.n(k) >= self.min_n:
                ck = (k, tau)
                if ck not in self._cache or (self.lazy and self._seen[k] > self._at[ck] * 1.125 + 4):
                    self._cache[ck] = float(np.percentile(np.fromiter(self.vals[k], dtype=float), 100.0 * tau))
                    self._at[ck] = self._seen[k]
                return self._cache[ck]
        return None


class MarkovNext:
    def __init__(self, k: int = 1):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.ctx: dict[tuple, Counter] = defaultdict(Counter)
        self.uni: Counter = Counter()

    def observe(self, hist: tuple, nxt: str) -> None:
        h = tuple(hist)[-self.k:]
        for j in range(len(h) + 1):                              # every suffix context incl. ()
            self.ctx[h[j:]][nxt] += 1
        self.uni[nxt] += 1

    def predict(self, hist: tuple) -> dict[str, float]:
        h = tuple(hist)[-self.k:]
        for j in range(len(h) + 1):                              # longest context with data first
            c = self.ctx.get(h[j:])
            if c:
                tot = sum(c.values())
                return {t: n / tot for t, n in c.items()}
        if not self.uni:
            return {}
        tot = sum(self.uni.values())
        return {t: n / tot for t, n in self.uni.items()}

    def prob(self, hist: tuple, token: str) -> float:
        return self.predict(hist).get(token, 0.0)
