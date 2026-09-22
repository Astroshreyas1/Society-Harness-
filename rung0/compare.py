"""A6 - paired comparison of policies over seeds, and the oracle-normalised score.

Reads grid_summary.csv files (from `agentsim grid`), pairs runs by (axis value, seed) across
policies (common random numbers make the pairing meaningful), and for every metric family
reports the paired difference policy - baseline with:
  mean difference and a bootstrap 95% CI, exact Wilcoxon signed-rank p (two-sided, enumerated),
  Cohen's dz, per-seed wins/ties/losses, and Holm-Bonferroni-adjusted p across metric families.
With --oracle, the oracle-normalised score (policy - baseline) / (oracle - baseline) per metric,
median over seeds, is added; "no headroom" where oracle vs baseline is not significant.

  python rung0/compare.py data/synthetic/e1/grid_summary.csv --axis population --baseline reactive_gate
  python rung0/compare.py data/synthetic/hosted/grid_summary.csv --axis rate_per_min --baseline reactive_gate --oracle clairvoyant
  python rung0/compare.py --selftest

Lower-is-better metrics are listed in METRICS; the direction decides what a "win" is.
Throughput is requests per hour *completed* (F13); sessions started are never reported.
numpy only.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS: dict[str, bool] = {                       # metric -> lower is better
    "failure_rate": True, "tct_p50": True, "tct_p99": True, "throughput_rph": False, "jain_all": False,
    "token_waste": True, "tool_waste": True, "timeouts": True, "block_wait_max": True,
}
ALPHA = 0.05


# ---- statistics --------------------------------------------------------------------------
def wilcoxon_exact(d: np.ndarray) -> tuple[float, int]:
    """Two-sided exact signed-rank p for paired differences d (zeros dropped, ties averaged).
    Enumerates all 2^n sign patterns; n <= 20."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0
    if n > 20:
        raise ValueError(f"exact Wilcoxon enumerates 2^n patterns; n={n} > 20")
    a = np.abs(d)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n)
    i = 0
    while i < n:                                                    # average ranks over ties
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    centre = n * (n + 1) / 4.0
    signs = np.array(list(itertools.product((0.0, 1.0), repeat=n)))  # 2^n x n
    dist = signs @ ranks
    p = float(np.mean(np.abs(dist - centre) >= abs(w_plus - centre) - 1e-12))
    return min(1.0, p), n


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz; Numerical Recipes 6.4)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            return h
    raise RuntimeError("incomplete beta continued fraction did not converge")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x in (0.0, 1.0):
        return x
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided(t: float, df: int) -> float:
    """Two-sided p-value of Student's t with df degrees of freedom."""
    if df < 1:
        raise ValueError("df must be >= 1")
    x = df / (df + t * t)
    return betainc(df / 2.0, 0.5, x)


def paired_t(d: np.ndarray) -> float:
    """Two-sided paired t-test p on differences d (assumes roughly normal differences)."""
    n = len(d)
    s = float(np.std(d, ddof=1)) if n > 1 else 0.0
    if n < 2:
        return float("nan")
    if s == 0.0:
        return 1.0 if float(np.mean(d)) == 0.0 else 0.0
    return t_two_sided(float(np.mean(d)) / (s / math.sqrt(n)), n - 1)


def wilcoxon_floor(n: int) -> float:
    """Smallest two-sided exact p attainable with n non-zero differences: 2 / 2^n."""
    return 2.0 / (2 ** n)


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, returned in the input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def bootstrap_ci(d: np.ndarray, b: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(b, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohen_dz(d: np.ndarray) -> float:
    s = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return float(np.mean(d) / s) if s > 0 else float("nan")


def oracle_normalised(policy: np.ndarray, base: np.ndarray, oracle: np.ndarray) -> float:
    """Median over seeds of (policy - base) / (oracle - base), over seeds where the oracle moved."""
    num, den = policy - base, oracle - base
    ok = den != 0
    return float(np.median(num[ok] / den[ok])) if ok.any() else float("nan")


# ---- data ----------------------------------------------------------------------------------
def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with Path(p).open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise ValueError("no rows loaded")
    return rows


def filter_rows(rows: list[dict], filters: list[str]) -> list[dict]:
    """Keep rows whose column equals the value for every 'column=value' filter (compared as strings, then as floats)."""
    out = rows
    for f in filters:
        col, _, val = f.partition("=")
        if not col or not val:
            raise ValueError(f"filter must be column=value, got {f!r}")
        if out and col not in out[0]:
            raise KeyError(f"filter column {col!r} not in CSV (have {sorted(out[0])})")

        def same(a: str) -> bool:
            try:
                return float(a) == float(val)
            except ValueError:
                return a == val
        out = [r for r in out if same(r[col])]
    if not out:
        raise ValueError(f"no rows left after filters {filters}")
    return out


def pair(rows: list[dict], axis: str, metric_names: list[str], policy_col: str = "policy") -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """axis value -> policy -> metric -> vector over seeds (seed order shared across policies). `policy_col` names the
    column that identifies a policy: `policy` in grid CSVs, `variant` in sweep CSVs."""
    for r in rows:
        for k in (policy_col, "seed", axis, *metric_names):
            if k not in r:
                raise KeyError(f"column {k!r} missing from CSV (have {sorted(r)})")
    cells: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        cells[r[axis]][r[policy_col]][r["seed"]] = r
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for ax, by_pol in cells.items():
        seeds = sorted(set.intersection(*(set(s) for s in by_pol.values())), key=lambda s: float(s))
        if not seeds:
            raise ValueError(f"{axis}={ax}: no seed shared by all policies")
        out[ax] = {pol: {m: np.array([float(by_pol[pol][s][m]) for s in seeds]) for m in metric_names} for pol in by_pol}
        out[ax]["_seeds"] = {"n": np.array([len(seeds)])}
    return out


# ---- report --------------------------------------------------------------------------------
def compare(cell: dict[str, dict[str, np.ndarray]], policy: str, baseline: str, oracle: str | None,
            metrics: dict[str, bool], test: str = "wilcoxon") -> list[dict]:
    """One row per metric. `test` ('wilcoxon' | 't') selects which p the Holm adjustment and the
    significance flag use; both p's are always reported."""
    if test not in ("wilcoxon", "t"):
        raise ValueError(f"test must be 'wilcoxon' or 't', got {test!r}")
    if policy not in cell or baseline not in cell:
        raise KeyError(f"policy {policy!r} or baseline {baseline!r} absent from this cell (have {sorted(cell)})")
    rows, ps = [], []
    for m, lower in metrics.items():
        a, b = cell[policy][m], cell[baseline][m]
        d = a - b
        better = (d < 0) if lower else (d > 0)
        worse = (d > 0) if lower else (d < 0)
        p, n_eff = wilcoxon_exact(d)
        lo, hi = bootstrap_ci(d)
        row = {"metric": m, "base_mean": float(b.mean()), "pol_mean": float(a.mean()), "diff": float(d.mean()),
               "ci": (lo, hi), "wins": int(better.sum()), "ties": int((d == 0).sum()), "losses": int(worse.sum()),
               "dz": cohen_dz(d), "p": p, "p_t": paired_t(d), "n_eff": n_eff}
        if oracle is not None:
            o = cell[oracle][m]
            p_head = wilcoxon_exact(o - b)[0] if test == "wilcoxon" else paired_t(o - b)
            row["headroom_p"] = p_head
            row["score"] = oracle_normalised(a, b, o) if p_head < ALPHA else float("nan")
        rows.append(row)
        ps.append(p if test == "wilcoxon" else row["p_t"])
    for row, adj in zip(rows, holm(ps)):
        row["p_holm"] = adj
    return rows


def fmt(x: float, nd: int = 3) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "—"
    return f"{x:.{nd}f}" if abs(x) < 1000 else f"{x:,.0f}"


def render(axis: str, ax: str, n: int, policy: str, baseline: str, oracle: str | None, rows: list[dict],
           test: str = "wilcoxon") -> str:
    head = (f"### {axis}={ax} · {policy} vs {baseline} · n={n} paired seeds · "
            f"exact Wilcoxon floor {wilcoxon_floor(n):.4f}; Holm on {test}")
    cols = ["metric", baseline, policy, "Δ mean [95% CI]", "W/T/L", "dz", "p_wilcoxon", "p_t", "p_holm"]
    if oracle:
        cols.append(f"score vs {oracle}")
    lines = [head, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    sig = []
    for r in rows:
        star = "**" if r["p_holm"] < ALPHA else ""
        cells = [r["metric"], fmt(r["base_mean"]), fmt(r["pol_mean"]),
                 f"{star}{fmt(r['diff'])} [{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]{star}",
                 f"{r['wins']}/{r['ties']}/{r['losses']}", fmt(r["dz"], 2), fmt(r["p"], 4), fmt(r["p_t"], 4),
                 fmt(r["p_holm"], 4)]
        if oracle:
            cells.append("no headroom" if r["headroom_p"] >= ALPHA else fmt(r["score"], 2))
        lines.append("| " + " | ".join(cells) + " |")
        if r["p_holm"] < ALPHA:
            direction = "better" if (r["wins"] > r["losses"]) else "worse"
            sig.append(f"{r['metric']} {direction}")
    lines.append("")
    lines.append(f"Significant after Holm on {test} (α={ALPHA}): " + (", ".join(sig) if sig else "none") + ".")
    if test == "wilcoxon" and wilcoxon_floor(n) > ALPHA:
        lines.append(f"Note: with n={n} the exact Wilcoxon cannot reach α={ALPHA} (floor {wilcoxon_floor(n):.4f}); "
                     "read the CI, W/T/L and p_t, or use --test t, or run ≥6 seeds.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*")
    ap.add_argument("--axis", help="grid axis column, e.g. population or rate_per_min")
    ap.add_argument("--baseline", default="reactive_gate")
    ap.add_argument("--policies", nargs="*", help="policies to compare against the baseline (default: all others)")
    ap.add_argument("--oracle", help="policy name of the clairvoyant bound (adds the oracle-normalised score)")
    ap.add_argument("--metrics", nargs="*", default=list(METRICS))
    ap.add_argument("--filter", action="append", default=[], help="column=value; keep only matching rows (repeatable)")
    ap.add_argument("--test", choices=("wilcoxon", "t"), default="wilcoxon", help="which p the Holm adjustment and the significance flag use")
    ap.add_argument("--out", help="write the markdown here as well")
    ap.add_argument("--policy-col", default="policy", help="column identifying a policy: policy (grid CSVs) or variant (sweep CSVs)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.csv or not args.axis:
        ap.error("csv files and --axis are required (or --selftest)")
    unknown = [m for m in args.metrics if m not in METRICS]
    if unknown:
        raise KeyError(f"unknown metrics {unknown}; known: {sorted(METRICS)}")
    metrics = {m: METRICS[m] for m in args.metrics}
    cells = pair(filter_rows(load_rows(args.csv), args.filter), args.axis, list(metrics), args.policy_col)
    out = []
    for ax in sorted(cells, key=lambda v: float(v) if v.replace(".", "", 1).isdigit() else v):
        cell = cells[ax]
        n = int(cell["_seeds"]["n"][0])
        policies = args.policies or [p for p in cell if p not in (args.baseline, "_seeds")]
        for pol in policies:
            out.append(render(args.axis, ax, n, pol, args.baseline, args.oracle,
                              compare(cell, pol, args.baseline, args.oracle, metrics, args.test), args.test))
    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


# ---- selftest --------------------------------------------------------------------------------
def selftest() -> None:
    checks = []
    p, n = wilcoxon_exact(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    checks.append(("exact Wilcoxon: 5 positive differences -> two-sided p = 2/32", abs(p - 0.0625) < 1e-12 and n == 5))
    p, n = wilcoxon_exact(np.array([0.0, 0.0, 0.0]))
    checks.append(("exact Wilcoxon: all-zero differences -> p = 1, n_eff = 0", p == 1.0 and n == 0))
    p, _ = wilcoxon_exact(np.array([3.0, -1.0, 2.0, 4.0, 5.0]))          # W+ = 14 of 15; P(|W-7.5|>=6.5) = 4/32
    checks.append(("exact Wilcoxon: one small negative -> p = 4/32", abs(p - 0.125) < 1e-12))
    checks.append(("Holm: [0.01, 0.04, 0.03] -> [0.03, 0.06, 0.06]",
                   np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])))
    s = oracle_normalised(np.array([7.0, 7.0]), np.array([10.0, 10.0]), np.array([4.0, 4.0]))
    checks.append(("oracle-normalised: base 10, policy 7, oracle 4 -> 0.5", abs(s - 0.5) < 1e-12))
    s = oracle_normalised(np.array([7.0]), np.array([10.0]), np.array([10.0]))
    checks.append(("oracle-normalised: oracle == base -> nan", math.isnan(s)))
    rows = []
    for pol, base in (("a", 1.0), ("b", 2.0)):
        for ax in ("5", "10"):
            for seed in (1, 2, 3):
                rows.append({"policy": pol, "seed": str(seed), "population": ax, "failure_rate": str(base + seed / 4 + (0.5 if ax == "10" else 0)),
                             "throughput_rph": str(100 - base)})
    checks.append(("filter_rows keeps only population=10 rows and compares numerically",
                   len(filter_rows(rows, ["population=10.0"])) == 6 and all(r["population"] == "10" for r in filter_rows(rows, ["population=10"]))))
    cells = pair(rows, "population", ["failure_rate", "throughput_rph"])
    r = compare(cells["5"], "b", "a", None, {"failure_rate": True, "throughput_rph": False})
    checks.append(("pairing: b - a on failure_rate = 1.0 at every seed, 0 wins / 3 losses",
                   abs(r[0]["diff"] - 1.0) < 1e-12 and r[0]["wins"] == 0 and r[0]["losses"] == 3 and r[0]["ties"] == 0))
    checks.append(("pairing: throughput lower for b counts as losses", r[1]["losses"] == 3))
    checks.append(("dz is nan when all differences are equal", math.isnan(r[0]["dz"])))
    checks.append(("Student t: two-sided p(t=2.776, df=4) = 0.05", abs(t_two_sided(2.776445, 4) - 0.05) < 1e-4))
    checks.append(("Student t: two-sided p(t=0, df=9) = 1", abs(t_two_sided(0.0, 9) - 1.0) < 1e-12))
    checks.append(("Student t: two-sided p(t=12.706, df=1) = 0.05", abs(t_two_sided(12.7062, 1) - 0.05) < 1e-4))
    p_t = paired_t(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    checks.append(("paired t on [1..5]: t = 3/(1.581/sqrt5) = 4.243, p = 0.0132", abs(p_t - 0.01324) < 2e-4))
    checks.append(("Wilcoxon floor for n=5 is 2/32", abs(wilcoxon_floor(5) - 0.0625) < 1e-12))
    cell = {"base": {"m": np.array([10.0, 10.5, 9.5, 10.2, 9.8])}, "pol": {"m": np.array([7.0, 7.5, 6.5, 7.2, 6.8])},
            "orc": {"m": np.array([4.0, 4.5, 3.5, 4.2, 3.8])}}
    rw = compare(cell, "pol", "base", "orc", {"m": True}, test="wilcoxon")[0]
    rt = compare(cell, "pol", "base", "orc", {"m": True}, test="t")[0]
    checks.append(("headroom gate follows --test: wilcoxon (n=5) -> no headroom; t -> score 0.5",
                   math.isnan(rw["score"]) and abs(rt["score"] - 0.5) < 1e-12))
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    if not all(ok for _, ok in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
