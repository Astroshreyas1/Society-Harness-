"""E5 - how predictable is idle (think) time from observable state? (ISSUES B7)

Reads think spans (op="think") whose attrs carry the state a gateway can observe when a request
ends: recipe, phase_end (the last phase before the final chat) and request_idx. Trace-level
70/30 split. Reports, for the per-state predictor against the state-blind baseline:
  R²        of log think time (per-state mean vs global mean)
  pinball   loss at τ ∈ {0.5, 0.8, 0.9} for the *lower* quantile q_{1-τ} (what a park decision
            needs), per-state vs global, as a ratio (< 1 = the state helps)
  coverage  share of test think times >= the predicted lower quantile (target τ)
The generator's `think_snr` knob sets how much of the log-variance the state explains; on real
traces the number is whatever it is (Copilot reports 86-90% of idle time predictable).

  python rung0/e5_idle_predictability.py data/synthetic/hosted_snr/*snr=0.9*/traces.jsonl [--split 0.7]
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np

from observer import load

BUCKETS = ((0, 0, "r0"), (1, 2, "r1-2"), (3, 7, "r3-7"), (8, 10 ** 9, "r8+"))
TAUS = (0.5, 0.8, 0.9)


def bucket(request_idx: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= request_idx <= hi:
            return name
    raise ValueError(f"request_idx {request_idx} outside every bucket")


def think_records(spans) -> list[tuple[str, tuple[str, str, str], float]]:
    """(trace_id, state key, log think) for every think span with the observable state attrs."""
    out = []
    for s in spans:
        if s.op != "think":
            continue
        for k in ("recipe", "phase_end", "request_idx"):
            if k not in s.attrs:
                raise KeyError(f"think span {s.span_id} lacks attrs[{k!r}]; regenerate traces with the B7 generator")
        if s.duration <= 0:
            raise ValueError(f"think span {s.span_id} has non-positive duration {s.duration}")
        out.append((s.trace_id, (s.attrs["recipe"], s.attrs["phase_end"], bucket(int(s.attrs["request_idx"]))), math.log(s.duration)))
    if not out:
        raise ValueError("no think spans found")
    return out


def split(records, frac: float, seed: int = 0):
    traces = sorted({r[0] for r in records})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(traces))
    train_ids = {traces[i] for i in perm[: int(frac * len(traces))]}
    return [r for r in records if r[0] in train_ids], [r for r in records if r[0] not in train_ids]


def r2_from(train, test) -> float:
    by = defaultdict(list)
    for _, k, y in train:
        by[k].append(y)
    g = float(np.mean([y for _, _, y in train]))
    means = {k: float(np.mean(v)) for k, v in by.items()}
    ys = np.array([y for _, _, y in test])
    pred = np.array([means.get(k, g) for _, k, _ in test])
    ss_res, ss_tot = float(((ys - pred) ** 2).sum()), float(((ys - ys.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def think_r2(spans, frac: float = 0.7) -> float:
    """R² of log think time from the observable state, trace-level split (used by the selftest)."""
    train, test = split(think_records(spans), frac)
    return r2_from(train, test)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    records = think_records(load(args.traces))
    train, test = split(records, args.split, args.seed)
    print(f"think spans={len(records)} traces={len({r[0] for r in records})} states={len({r[1] for r in records})} "
          f"train/test={len(train)}/{len(test)}")
    print(f"R² of log think from (recipe, phase_end, request bucket): {r2_from(train, test):.3f}")
    by = defaultdict(list)
    for _, k, y in train:
        by[k].append(y)
    ytr = np.array([y for _, _, y in train])
    yte = np.array([y for _, _, y in test])
    print(f"\n{'τ':>4s} {'lower q':>8s} {'pinball state/global':>21s} {'coverage state':>15s} {'coverage global':>16s}")
    for tau in TAUS:
        lq = 100 * (1 - tau)
        g = float(np.percentile(ytr, lq))
        st = {k: float(np.percentile(v, lq)) for k, v in by.items() if len(v) >= 5}
        pred = np.array([st.get(k, g) for _, k, _ in test])
        ratio = pinball(yte, pred, 1 - tau) / pinball(yte, np.full_like(yte, g), 1 - tau)
        print(f"{tau:4.2f} {f'q{1 - tau:.1f}':>8s} {ratio:21.3f} {float(np.mean(yte >= pred)):15.3f} {float(np.mean(yte >= g)):16.3f}")


if __name__ == "__main__":
    main()
