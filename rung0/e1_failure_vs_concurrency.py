"""E1 - failure vs concurrency (H1). Reads a grid_summary.csv produced by `agentsim grid`
(or one you assemble from real runs with the same columns) and reports, per policy and
population, failure rate with a 95% CI over seeds next to the model-tier utilisation.
H1 is confirmed when failures occur while utilisation < 1 (capacity was sufficient).

  python rung0/e1_failure_vs_concurrency.py data/synthetic/e1/grid_summary.csv [--plot out.png]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def ci95(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--axis", default="population", help="column holding the concurrency knob")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.axis not in rows[0]:
        raise KeyError(f"column {args.axis!r} not in {list(rows[0])}")
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["policy"], float(r[args.axis]))].append(r)

    print(f"{'policy':16s} {args.axis:>10s} {'fail':>8s} {'+-95%':>6s} {'util_model':>10s} {'429/run':>8s} {'tct_p99':>8s} {'jain':>6s} {'n':>3s}")
    table = defaultdict(list)
    for (pol, x), rs in sorted(groups.items()):
        f, e = ci95([float(r["failure_rate"]) for r in rs])
        u, _ = ci95([float(r["util_model"]) for r in rs])
        n429 = np.mean([float(r["n429"]) for r in rs])
        p99 = np.mean([float(r["tct_p99"]) for r in rs])
        jain = np.mean([float(r["jain_all"]) for r in rs])
        print(f"{pol:16s} {x:10.2f} {f:8.3f} {e:6.3f} {u:10.3f} {n429:8.0f} {p99:8.0f} {jain:6.3f} {len(rs):3d}")
        table[pol].append((x, f, e, u))
    verdict = any(f > 0.05 and u < 0.9 for pol in table for _, f, _, u in table[pol] if pol == "uncoordinated")
    print("\nH1 verdict:", "CONFIRMED - failures > 5% while model utilisation < 0.9 (capacity was sufficient)" if verdict
          else "NOT confirmed on this data - failures only appear once the model tier is saturated")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for pol, pts in table.items():
            pts.sort()
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", capsize=3, label=pol)
        ax.set_xlabel(args.axis); ax.set_ylabel("workflow failure rate"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend()
        ax.set_title("E1: failure vs concurrency (95% CI over seeds)")
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print("plot:", args.plot)


if __name__ == "__main__":
    main()
