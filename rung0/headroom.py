"""Where does prediction pay? Reads an `agentsim sweep` CSV (sampled scenarios x variants x seeds) and
reports, per scenario and overall:
  headroom   = clairvoyant - gate_ship, paired over seeds, per metric (failure_rate, tct_p99, throughput_rph),
               with the paired-t p (5 seeds cannot reach 0.05 on the exact Wilcoxon; compare.py F18)
  knob win   = gate_ship - gate_sampled: what the shipped fixed rules buy over the scenario's own config
  lease share= median over seeds of (lease - gate_ship) / (clairvoyant - gate_ship)
  Spearman rho between each sampled knob and the relative throughput / failure headroom across scenarios.

  python rung0/headroom.py data/synthetic/sweep/sweep.csv [--baseline gate_ship] [--oracle clairvoyant] [--policy lease] [--top 10]
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import numpy as np

from compare import ALPHA, paired_t, oracle_normalised

METRICS = {"failure_rate": True, "tct_p99": True, "throughput_rph": False}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--baseline", default="gate_ship")
    ap.add_argument("--sampled", default="gate_sampled")
    ap.add_argument("--oracle", default="clairvoyant")
    ap.add_argument("--policy", default="lease")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty sweep")
    knobs = sorted(k for k in rows[0] if k.startswith("knob:"))
    by: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by[r["scenario"]][r["variant"]][int(r["seed"])] = r
    per = []
    for sc, variants in sorted(by.items()):
        for v in (args.baseline, args.sampled, args.oracle, args.policy):
            if v not in variants:
                raise KeyError(f"{sc}: variant {v!r} missing")
        seeds = sorted(set.intersection(*(set(variants[v]) for v in (args.baseline, args.sampled, args.oracle, args.policy))))
        vec = lambda v, m: np.array([float(variants[v][s][m]) for s in seeds])
        rec = {"scenario": sc, "n": len(seeds), "knobs": {k: float(variants[args.baseline][seeds[0]][k]) if _isnum(variants[args.baseline][seeds[0]][k]) else variants[args.baseline][seeds[0]][k] for k in knobs}}
        for m in METRICS:
            b, o, p, sm = vec(args.baseline, m), vec(args.oracle, m), vec(args.policy, m), vec(args.sampled, m)
            rec[m] = {"base": float(b.mean()), "oracle": float(o.mean()), "policy": float(p.mean()), "sampled": float(sm.mean()),
                      "head": float((o - b).mean()), "head_p": paired_t(o - b), "knob_win": float((b - sm).mean()),
                      "score": oracle_normalised(p, b, o)}
        per.append(rec)
    n = len(per)
    print(f"scenarios={n} seeds/scenario={per[0]['n']} baseline={args.baseline} oracle={args.oracle} policy={args.policy}\n")
    print("## Overall")
    print(f"| metric | scenarios with significant oracle headroom over {args.baseline} (paired t < {ALPHA}) | median relative headroom | median {args.policy} score where significant | median knob win ({args.baseline} − {args.sampled}) |")
    print("|---|---|---|---|---|")
    for m, lower in METRICS.items():
        sig = [r for r in per if r[m]["head_p"] < ALPHA and (r[m]["head"] < 0 if lower else r[m]["head"] > 0)]
        rel = [r[m]["head"] / r[m]["base"] for r in per if r[m]["base"] != 0]
        scores = [r[m]["score"] for r in sig if not math.isnan(r[m]["score"])]
        wins = [r[m]["knob_win"] for r in per]
        print(f"| {m} | {len(sig)}/{n} | {100 * float(np.median(rel)):+.0f}% | {float(np.median(scores)) if scores else float('nan'):.2f} | {float(np.median(wins)):+.3f} |")
    print(f"\n## Which knobs go with headroom (Spearman rho across {n} scenarios; |rho| ≥ 0.3 shown)")
    numeric = [k for k in knobs if all(isinstance(r["knobs"][k], float) for r in per)]
    print("| knob | rho vs relative throughput headroom | rho vs failure headroom (oracle − base) |")
    print("|---|---|---|")
    rt = np.array([r["throughput_rph"]["head"] / r["throughput_rph"]["base"] for r in per])
    fh = np.array([r["failure_rate"]["head"] for r in per])
    shown = 0
    for k in numeric:
        x = np.array([r["knobs"][k] for r in per])
        a, b = spearman(x, rt), spearman(x, fh)
        if abs(a) >= 0.3 or abs(b) >= 0.3:
            shown += 1
            print(f"| {k[5:]} | {a:+.2f} | {b:+.2f} |")
    if not shown:
        print("| (none ≥ 0.3) | | |")
    print(f"\n## Top {args.top} scenarios by relative throughput headroom")
    print("| scenario | thr base → oracle (lease) | failure base → oracle (lease) | headroom p (thr) | cold start | idle t/o | rate/min | snr | CPU | mem | search rpm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(per, key=lambda r: -r["throughput_rph"]["head"] / r["throughput_rph"]["base"])[: args.top]:
        k = r["knobs"]
        t, fr = r["throughput_rph"], r["failure_rate"]
        print(f"| {r['scenario']} | {t['base']:.0f} → {t['oracle']:.0f} ({t['policy']:.0f}) | {fr['base']:.3f} → {fr['oracle']:.3f} ({fr['policy']:.3f}) | {t['head_p']:.3f} | "
              f"{k['knob:framework.sandbox_cold_start_s']:.0f} s | {k['knob:framework.sandbox_idle_timeout_s']:.0f} | {k['knob:arrivals.rate_per_min']:.2f} | {k['knob:generator.think_snr']:.2f} | "
              f"{k['knob:resources.sandbox.cpu.capacity']:.0f} | {k['knob:resources.sandbox.mem.capacity']:.0f} | {k['knob:resources.ext.search.rpm']:.0f} |")


def _isnum(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
