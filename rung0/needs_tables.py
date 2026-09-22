"""Markdown tables for NEEDS_REPORT.md from the Phase 6–7 sweep CSVs (data/synthetic/needs/*.csv).

  python rung0/needs_tables.py data/synthetic/needs/hosted_content.csv --axis content_snr --baseline gate_vtfq0 --oracle oracle0

One table per axis value: every variant's mean over seeds for the report's metric families, paired wins against the
baseline, and a star where the paired difference is significant after Holm (exact Wilcoxon, α 0.05; compare.py).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import ALPHA, METRICS, compare, load_rows, pair  # noqa: E402

COLS = [("failure_rate", "failure", 3), ("tct_p50", "p50 s", 0), ("tct_p99", "p99 s", 0), ("throughput_rph", "req·h⁻¹", 0),
        ("timeouts", "timeouts", 0), ("jain_all", "Jain", 2)]
EXTRA = [("budget_errs", "refused", 0), ("usd_per_req", "$/req", 3), ("usd_waste_frac", "$ wasted", 3), ("children_aborted", "children aborted", 1),
         ("join_p99", "join p99 s", 0), ("refused_429", "429s", 0), ("provider_errs", "burned attempts", 0), ("leases", "leases", 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--axis", required=True)
    ap.add_argument("--baseline", default="gate_vtfq0")
    ap.add_argument("--oracle", default="oracle0")
    ap.add_argument("--variants", nargs="*", help="subset and order of variants (default: all, sorted)")
    args = ap.parse_args()
    rows = load_rows([args.csv])
    extra = [c for c in EXTRA if c[0] in rows[0] and any(r[c[0]] not in ("", "0", "0.0", "nan") for r in rows)]
    cols = COLS + extra
    metrics = {c[0]: METRICS.get(c[0], True) for c in cols if c[0] in rows[0]}
    for c in cols:
        if c[0] not in METRICS:
            METRICS[c[0]] = True
    cells = pair(rows, args.axis, list(metrics), "variant")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r[args.axis]][r["variant"]].append(r)
    for ax in sorted(cells, key=lambda v: float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v):
        cell = cells[ax]
        n = int(cell["_seeds"]["n"][0])
        variants = args.variants or sorted(v for v in cell if v != "_seeds")
        print(f"\n**{args.axis} = {ax}** (n = {n} paired seeds; * = significant after Holm vs `{args.baseline}`; wins = per-seed paired wins on failure / p50 / p99 / throughput)\n")
        print("| variant | " + " | ".join(c[1] for c in cols) + " | wins |")
        print("|---|" + "---|" * (len(cols) + 1))
        for v in variants:
            if v not in cell:
                continue
            sig = {}
            wins = ""
            if v != args.baseline:
                res = compare(cell, v, args.baseline, args.oracle if args.oracle in cell else None, metrics)
                sig = {r["metric"]: (r["p_holm"] < ALPHA, r["wins"], r["losses"]) for r in res}
                w = lambda m: f"{sig[m][1]}" if m in sig else "-"
                wins = f"{w('failure_rate')}/{w('tct_p50')}/{w('tct_p99')}/{w('throughput_rph')} of {n}"
            vals = []
            for key, _, nd in cols:
                m = float(np.mean([float(r[key]) for r in by[ax][v]]))
                star = "*" if sig.get(key, (False,))[0] else ""
                vals.append(f"{m:,.{nd}f}{star}" if key not in ("failure_rate", "usd_waste_frac", "jain_all", "usd_per_req") else f"{m:.{nd}f}{star}")
            print(f"| {v} | " + " | ".join(vals) + f" | {wins} |")


if __name__ == "__main__":
    main()
