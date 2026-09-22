"""E3 - heavy tails (H2/H3). Tail shares, quantiles and a quantile-matched lognormal for tool
durations, output tokens and think times. Sets the reservation quantile tau. Compared against the
TraceLab targets the seeds were shaped to. The Hill estimate is printed for completeness only:
on a capped lognormal mixture it is not a Pareto index (ISSUES F10) - use the shares.

  python rung0/e3_tails.py data/synthetic/hosted/*/traces.jsonl
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from observer import load


def hill(x: np.ndarray, frac: float = 0.05) -> float:
    x = np.sort(x[x > 0])[::-1]
    k = max(10, int(frac * len(x)))
    if len(x) <= k:
        return float("nan")
    return float(1.0 / np.mean(np.log(x[:k] / x[k])))


def describe(name: str, x: np.ndarray, unit: str) -> None:
    x = np.asarray(x, dtype=float)
    q = lambda p: float(np.percentile(x, p))
    mu, sigma = math.log(q(50)), math.log(q(99) / q(50)) / 2.3263478740408408
    print(f"\n{name}: n={len(x)} mean={x.mean():.2f}{unit} p50={q(50):.2f} p90={q(90):.2f} p99={q(99):.2f} max={x.max():.1f}"
          f" | lognormal(mu={mu:.2f}, sigma={sigma:.2f}) | Hill alpha(top5%)={hill(x):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    args = ap.parse_args()
    spans = load(args.traces)
    d = np.array([s.duration for s in spans if s.op == "execute_tool" and s.outcome == "ok" and s.name != "parallel"])
    if len(d) == 0:
        raise ValueError("no completed tool calls in traces")
    tot = d.sum()
    print(f"tool calls n={len(d)}")
    print(f"  calls < 1 s : {100 * (d < 1).mean():5.1f}% of calls, {100 * d[d < 1].sum() / tot:5.2f}% of tool time   (TraceLab Claude: 70%, <1%)")
    print(f"  calls > 60 s: {100 * (d > 60).mean():5.1f}% of calls, {100 * d[d > 60].sum() / tot:5.1f}% of tool time   (TraceLab Claude: 4.9%, 92%)")
    print(f"  mean {d.mean():.1f} s (TraceLab: 16.8 s)")
    describe("tool duration (all kinds)", d, "s")
    for k in sorted({s.name for s in spans if s.op == "execute_tool" and s.name != "parallel"}):
        dk = np.array([s.duration for s in spans if s.op == "execute_tool" and s.outcome == "ok" and s.name == k])
        if len(dk) >= 30:
            describe(f"  tool:{k}", dk, "s")
    out = np.array([s.tokens_out for s in spans if s.op == "chat" and s.outcome == "ok" and s.tokens_out > 0])
    if len(out):
        describe("output tokens (TraceLab Claude: median 252, p99 6,571)", out, "")
    think = np.array([s.duration for s in spans if s.op == "think"])
    if len(think):
        describe("think time (TraceLab: median 84 s, p90 1,236 s)", think, "s")
    share_time = 100 * d[d > 60].sum() / tot
    share_calls = 100 * (d > 60).mean()
    print(f"\nH2/H3 verdict: {share_time:.0f}% of tool time sits in the {share_calls:.1f}% of calls longer than a minute;"
          f" reservations must be on quantiles, never means. (Hill alpha = {hill(d):.2f}, for completeness only: not a Pareto"
          f" index on a capped lognormal mixture - ISSUES F10.)")


if __name__ == "__main__":
    main()
