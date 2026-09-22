"""Feedback-loop allocation (Phase 7): the controller measures its own prediction error per resource and corrects.

Prediction alone is open-loop: a forecast that is biased low under-reserves, one biased high over-reserves, and a
budget pacer that never looks back drifts. `AllocationLoop` closes the loop every `tick_s` seconds:

  demand error       per resource: realised time-averaged occupancy over the tick versus the occupancy the forecast
                     predicted for that tick when it began -> e = (realised - predicted) / capacity
                     correction: a per-resource *scale* on the forecast (PI with anti-windup, bounded [0.5, 2]);
                     the Reserver prices every lease / park / prewarm with the corrected forecast, so a biased
                     predictor is corrected before it costs anything, not after it has.
  budget pacing      per tenant budget: the floor a step must leave in the bucket after paying, as a fraction of
                     the budget's capacity — raised multiplicatively when the provider refused a payment in the tick
                     (the failure the pacer exists to prevent), lowered additively on a quiet tick (AIMD) — and a
                     reservation for the predicted spend-to-completion of every in-flight request, so new work is
                     admitted only when what is already started can still be paid for ("finish what you started").
  fairness           per session: the served-time / wall-clock ratio versus the population's; sessions below their
                     share get a lower virtual-time weight (served sooner), those above a higher one — bounded, slow.

Every error signal and every correction is reported (`report()`), so a gain can be attributed to the loop or to the
predictor. Switch: `policy.ablate=feedback` freezes all scales at 1 and the floor at the gate's header pause.
"""
from __future__ import annotations

import math

import numpy as np


class PI:
    def __init__(self, kp: float, ki: float, lo: float, hi: float, x0: float = 1.0):
        self.kp, self.ki, self.lo, self.hi = kp, ki, lo, hi
        self.x, self.i = x0, 0.0
        self.n, self.abs_err = 0, 0.0

    def update(self, e: float) -> float:
        self.i = max(-2.0, min(2.0, self.i + e))                            # anti-windup
        self.x = max(self.lo, min(self.hi, 1.0 + self.kp * e + self.ki * self.i))
        self.n += 1
        self.abs_err += abs(e)
        return self.x


class AllocationLoop:
    def __init__(self, resources: dict, budgets: dict, tick_s: float = 30.0, enabled: bool = True, kp: float = 0.6, ki: float = 0.15):
        if tick_s <= 0:
            raise ValueError("tick_s must be > 0")
        self.res, self.budgets, self.tick, self.enabled = resources, budgets, float(tick_s), enabled
        self.scale = {n: PI(kp, ki, 0.5, 2.0) for n in resources}          # forecast scale per resource
        self.occ_int = {n: 0.0 for n in resources}                          # integral of used(t) over the current tick
        self.last_t = 0.0
        self.tick_start = 0.0
        self.predicted = {n: float("nan") for n in resources}               # the forecast's mean occupancy for this tick
        self.floor = {n: 0.10 for n in budgets}                             # pacing floor as a fraction of capacity (starts at the gate's pause)
        self.refused_seen = {n: 0 for n in budgets}
        self.spend_int = {n: 0.0 for n in budgets}
        self.fair_w: dict[str, float] = {}                                  # sid -> VTFQ weight
        self.served: dict[str, tuple[float, float]] = {}                    # sid -> (served seconds, since)
        self.history: list[dict] = []
        self.ticks = 0

    # ---- sampling (called on every event) ----------------------------------------------------
    def sample(self, now: float) -> None:
        dt = now - self.last_t
        if dt > 0:
            for n, r in self.res.items():
                self.occ_int[n] += r.used * dt
        self.last_t = now

    def set_prediction(self, name: str, mean_occ: float) -> None:
        self.predicted[name] = mean_occ

    # ---- the tick ------------------------------------------------------------------------------
    def tick_now(self, now: float) -> dict:
        """Close the tick: errors, corrections, a history row. Returns the row."""
        self.sample(now)
        length = max(1e-9, now - self.tick_start)
        row = {"t": round(now, 1), "res": {}, "budget": {}}
        for n, r in self.res.items():
            realised = self.occ_int[n] / length
            pred = self.predicted[n]
            self.occ_int[n] = 0.0
            if math.isnan(pred) or r.capacity <= 0:
                continue
            e = (realised - pred) / r.capacity
            sc = self.scale[n].update(e) if self.enabled else 1.0
            row["res"][n] = {"realised": round(realised, 3), "predicted": round(pred, 3), "err": round(e, 4), "scale": round(sc, 3)}
        for n, b in self.budgets.items():
            refused = b.refused - self.refused_seen[n]
            self.refused_seen[n] = b.refused
            if self.enabled:
                if refused > 0:
                    self.floor[n] = min(0.5, self.floor[n] * 1.5 + 0.02)      # a payment was refused: pace harder
                else:
                    self.floor[n] = max(0.02, self.floor[n] - 0.01)           # a quiet tick: release
            row["budget"][n] = {"refused": refused, "floor": round(self.floor[n], 3), "level_frac": round(b.remaining_frac(now), 3)}
        self.tick_start = now
        self.ticks += 1
        if len(self.history) < 5000:
            self.history.append(row)
        return row

    # ---- fairness ---------------------------------------------------------------------------------
    def served_add(self, sid: str, seconds: float, now: float) -> None:
        s, since = self.served.get(sid, (0.0, now))
        self.served[sid] = (s + seconds, since)

    def fairness_update(self, now: float, k: float = 0.5) -> None:
        if not self.enabled or len(self.served) < 4:
            return
        ratios = {sid: s / max(1.0, now - since) for sid, (s, since) in self.served.items() if now - since > 30.0}
        if len(ratios) < 4:
            return
        mean = float(np.mean(list(ratios.values())))
        if mean <= 0:
            return
        for sid, r in ratios.items():
            e = (r - mean) / mean                                            # above share -> positive -> heavier tag
            self.fair_w[sid] = max(0.5, min(2.0, 1.0 + k * max(-1.0, min(1.0, e))))

    def weight(self, sid: str) -> float:
        return self.fair_w.get(sid, 1.0)

    def forget(self, sid: str) -> None:
        self.fair_w.pop(sid, None)
        self.served.pop(sid, None)

    def report(self) -> dict:
        return {"ticks": self.ticks, "enabled": self.enabled,
                "scale": {n: round(p.x, 3) for n, p in self.scale.items() if p.n > 0},
                "mean_abs_err": {n: round(p.abs_err / p.n, 4) for n, p in self.scale.items() if p.n > 0},
                "budget_floor": {n: round(f, 3) for n, f in self.floor.items()},
                "fair_weights": {"n": len(self.fair_w), "min": round(min(self.fair_w.values()), 3) if self.fair_w else 1.0,
                                 "max": round(max(self.fair_w.values()), 3) if self.fair_w else 1.0}}
