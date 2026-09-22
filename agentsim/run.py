"""CLI.

  python -m agentsim run      --scenario scenarios/api_coding.json --out data/synthetic/demo [--set policy.type=reactive_gate ...]
  python -m agentsim grid     --grid scenarios/grid_e1.json --out data/synthetic/e1
  python -m agentsim fit      --traces a.jsonl [b.jsonl ...] --out data/marginals.fitted.json [--recipes-out data/recipes.fitted.json]
  python -m agentsim sample-scenarios --base scenarios/hosted_mixed.json --ranges scenarios/knob_ranges.json --n 20 --out scenarios/sampled
  python -m agentsim sweep    --scenarios scenarios/sampled/*.json --variants scenarios/variants_ship.json --seeds 1-5 --out data/synthetic/sweep/sweep.csv
  python -m agentsim selftest
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .engine import Engine
from .marginals import Marginals
from .metrics import fidelity, summarize
from .policies import make_policy
from .resources import build_resources
from .schema import read_jsonl, write_jsonl
from .workload import Recipes

ROOT = Path(__file__).resolve().parent.parent


def set_path(cfg: dict, dotted: str, value) -> None:
    """Set cfg[a][b]... = value for a dotted path. Keys may themselves contain dots (resource names such
    as `ext.search`): at every level the longest prefix that is an existing key wins. Unknown paths throw."""
    parts = dotted.split(".")
    d, i = cfg, 0
    while True:
        match = None
        for j in range(len(parts), i, -1):                     # longest existing key first
            k = ".".join(parts[i:j])
            if isinstance(d, dict) and k in d:
                match = (k, j)
                break
        if match is None:
            raise KeyError(f"override path {dotted!r}: {'.'.join(parts[i:])!r} not in config")
        k, j = match
        if j == len(parts):
            d[k] = value
            return
        d, i = d[k], j


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_scenario(path: Path, overrides: list[str]) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for ov in overrides:
        k, _, v = ov.partition("=")
        set_path(cfg, k, parse_value(v))
    return cfg


def build_engine(cfg: dict) -> Engine:
    recipes = Recipes.load(ROOT / cfg["recipes"])
    marginals = Marginals.load(ROOT / cfg["marginals"])
    resources = build_resources(cfg)
    policy = make_policy(cfg["policy"], np.random.default_rng([int(cfg["seed"]), 4]), cfg)
    return Engine(cfg, recipes, marginals, resources, policy)


def summary_of(eng: Engine, cfg: dict, wall_s: float = 0.0) -> dict:
    """Span-derived metrics plus the engine-level counters (whole run, not windowed)."""
    caps = {n: r.capacity for n, r in eng.res.items()}
    summary = summarize(eng.spans, float(cfg["horizon_s"]), float(cfg["warmup_s"]), caps)
    summary["waste"]["kv_retained_token_s_unused"] = eng.model.retained_token_s_unused
    summary["waste"]["kv_retained_token_s_reused"] = eng.model.retained_token_s_reused
    summary["waste"]["sandbox_prewarm_idle_s"] = eng.prewarm_idle_s
    summary["waste"]["lease_unit_s_issued"] = eng.ledger.unit_s_issued
    summary["waste"]["lease_unit_s_unused"] = eng.ledger.unit_s_unused
    summary["waste"]["lease_by_kind"] = eng.ledger.by_kind
    summary["waste"]["prewarm_hits"] = sum(q.prewarm_hits for q in eng.programs.values())
    summary["budgets"] = {n: {"spent": round(b.spent, 4), "refused": b.refused, "level_frac_end": round(b.remaining_frac(eng.now), 3)} for n, b in eng.budgets.items()}
    summary["fidelity"] = fidelity(eng.spans)
    if hasattr(eng.policy, "report"):
        summary["controller"] = eng.policy.report()
    summary["sim"] = {"wall_s": round(wall_s, 2), "events": eng.events, "spans": len(eng.spans),
                      "sessions_started": eng.n_sessions, "kv_evictions": eng.model.evictions,
                      "deadlocks_detected": eng.deadlocks_detected, "policy": eng.policy.name, "seed": cfg["seed"],
                      "spawns": eng.spawns, "joins": eng.joins}
    return summary


def simulate(cfg: dict) -> tuple[list, dict]:
    eng = build_engine(cfg)
    t0 = time.perf_counter()
    eng.run()
    return eng.spans, summary_of(eng, cfg, time.perf_counter() - t0)


def _write_run(out: Path, spans, summary: dict, cfg: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "traces.jsonl", spans)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "scenario.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def cmd_run(args) -> None:
    cfg = load_scenario(args.scenario, args.set)
    spans, summary = simulate(cfg)
    _write_run(Path(args.out), spans, summary, cfg)
    print(json.dumps({k: summary[k] for k in ("requests", "failure_rate", "sessions", "request_tct_s", "errors", "waste",
                                               "fairness", "sim")}, indent=2))


FLAT = [("policy", "sim.policy"), ("seed", "sim.seed"), ("req_started", "requests.started"), ("req_done", "requests.completed"),
        ("req_failed", "requests.failed"), ("failure_rate", "failure_rate"), ("session_fail_finished", "sessions.failure_rate_finished"), ("tct_n", "request_tct_s.n"),
        ("tct_p50", "request_tct_s.p50"), ("tct_p99", "request_tct_s.p99"), ("chat_wait_p99", "chat_wait_s.p99"),
        ("block_wait_max", "blocking_wait_s.max"), ("n429", "errors.429"), ("budget_errs", "errors.budget"), ("usd", "spend.usd"), ("usd_per_req", "spend.usd_per_completed_request"), ("usd_waste_frac", "spend.usd_waste_frac"), ("timeouts", "errors.timeout"), ("evicted", "errors.evicted"),
        ("deadlock_breaks", "errors.deadlock_breaks"), ("cycles_seen", "errors.deadlock_cycles_seen"), ("token_waste", "waste.fresh_token_waste_frac"), ("tool_waste", "waste.tool_waste_frac"), ("kv_unused_token_s", "waste.kv_retained_token_s_unused"), ("prewarm_idle_s", "waste.sandbox_prewarm_idle_s"), ("prewarm_hits", "waste.prewarm_hits"),
        ("lease_cpu_issued", ("waste", "lease_by_kind", "cpu", "issued")), ("lease_cpu_unused", ("waste", "lease_by_kind", "cpu", "unused")),
        ("lease_budget_issued", ("waste", "lease_by_kind", "budget", "issued")), ("lease_budget_unused", ("waste", "lease_by_kind", "budget", "unused")),
        ("par_groups", "steps.parallel_groups"), ("par_groups_parallel", "steps.parallel_groups_run_parallel"),
        ("spawns", "steps.spawns"), ("children", "steps.children"), ("children_aborted", "steps.children_aborted"), ("join_p99", "steps.join_wait_s_p99"),
        ("jain_all", "fairness.jain_service_ratio_all"), ("jain_survivors", "fairness.jain_survivors_only"),
        ("util_model", ("utilisation", "model.slots")), ("util_cpu", ("utilisation", "sandbox.cpu")),
        ("throughput_rph", "throughput_requests_per_hour"), ("wall_s", "sim.wall_s")]


def _get(d: dict, path):
    for k in (path.split(".") if isinstance(path, str) else path):
        d = d[k]
    return d


def apply_axes(cfg: dict, names: list[str], combo: tuple) -> str:
    """Set one grid point. An axis name may tie several paths with '|' (all get the value); the run
    tag uses the first path's last component."""
    for n, v in zip(names, combo):
        for path in n.split("|"):
            set_path(cfg, path, v)
    return "_".join(f"{n.split('|')[0].split('.')[-1]}={str(v).replace('/', '~')}" for n, v in zip(names, combo))   # file-safe


def cmd_grid(args) -> None:
    spec = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    base = ROOT / spec["base"]
    axes: dict[str, list] = spec["axes"]
    seeds: list[int] = spec["seeds"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    names, values = list(axes), [axes[k] for k in axes]
    for combo in itertools.product(*values):
        for seed in seeds:
            cfg = load_scenario(base, [])
            tag = apply_axes(cfg, names, combo) + f"_seed={seed}"
            cfg["seed"] = seed
            spans, summary = simulate(cfg)
            if not args.no_traces:
                _write_run(out / tag, spans, summary, cfg)
            else:
                (out / tag).mkdir(parents=True, exist_ok=True)
                (out / tag / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            row = {"run": tag, **{n.split("|")[0].split(".")[-1]: v for n, v in zip(names, combo)}}
            row.update({k: _get(summary, path) for k, path in FLAT})
            rows.append(row)
            print(f"{tag:60s} fail={row['failure_rate']:.3f} tct_p99={row['tct_p99']:.0f}s 429={row['n429']} "
                  f"timeouts={row['timeouts']} dl={row['deadlock_breaks']} wall={row['wall_s']}s", flush=True)
    with (out / "grid_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'grid_summary.csv'} ({len(rows)} runs)")


def cmd_fit(args) -> None:
    """Fit marginals (and, with --recipes-out, observable-phase recipes) from OTel-shaped traces; see agentsim.fit."""
    from .fit import fit_all, load_files
    files = load_files(args.traces)
    seed_m = json.loads((ROOT / args.seed_marginals).read_text(encoding="utf-8"))
    seed_r = json.loads((ROOT / args.seed_recipes).read_text(encoding="utf-8"))
    marg, rec = fit_all(files, seed_m, seed_r)
    Marginals(marg)
    Recipes(rec)
    Path(args.out).write_text(json.dumps(marg, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: fitted {marg['provenance']['fitted_entries']}; kept seeds for {marg['provenance']['unfitted_entries']}")
    if args.recipes_out:
        Path(args.recipes_out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"wrote {args.recipes_out}: " + ", ".join(f"{n} ({r['_fitted']['requests']} requests, {r['_fitted']['phases']} phases)" for n, r in rec["recipes"].items()))


def cmd_train_needs(args) -> None:
    from .train import train_needs
    rep = train_needs(args.traces, Path(args.out), int(args.epochs), int(args.seed), use_content=not args.no_content,
                      scenario=Path(args.scenario) if args.scenario else None, max_sessions=args.max_sessions)
    print(json.dumps(rep, indent=2))


def cmd_train_jev(args) -> None:
    from .train import train_jev
    rep = train_jev(args.traces, Path(args.out), int(args.epochs), int(args.seed), scenario=Path(args.scenario) if args.scenario else None,
                    max_sessions=args.max_sessions)
    print(json.dumps(rep, indent=2))


def cmd_evaluate_jev(args) -> None:
    from .train import evaluate_jev
    print(json.dumps(evaluate_jev(args.traces, Path(args.model) if args.model else None, args.remote, args.max_n,
                                  Path(args.scenario) if args.scenario else None, args.max_sessions), indent=2))


def cmd_annotate(args) -> None:
    from .train import annotate
    print(json.dumps(annotate(Path(args.trace), Path(args.model), Path(args.out)), indent=2))


# ---- scenario sampling (A8) ------------------------------------------------------------------
def _draw_knob(spec: dict, rng: np.random.Generator):
    t = spec["type"]
    if t == "uniform":
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    if t == "loguniform":
        lo, hi = float(spec["low"]), float(spec["high"])
        if lo <= 0 or hi <= lo:
            raise ValueError(f"loguniform needs 0 < low < high, got {lo}, {hi}")
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if t == "int":
        return int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
    if t == "choice":
        opts = spec["options"]
        return opts[int(rng.integers(0, len(opts)))]
    raise ValueError(f"unknown knob type {t!r} (uniform | loguniform | int | choice)")


def sample_scenarios(base: Path, ranges: Path | dict, n: int, seed: int, out: Path) -> int:
    """Draw n scenarios from `base` with every knob in `ranges` sampled from its declared range
    (domain randomisation, A8). Each file records `sampled_knobs` and has its own scenario seed."""
    spec = ranges if isinstance(ranges, dict) else json.loads(Path(ranges).read_text(encoding="utf-8"))
    knobs = {k: v for k, v in spec.items() if not k.startswith("_")}
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng([seed, 8])
    out.mkdir(parents=True, exist_ok=True)
    base_cfg = load_scenario(base, [])
    for k, v in knobs.items():                                 # validate every path before drawing anything
        set_path(json.loads(json.dumps(base_cfg)), k, None)
        if "complement" in v:
            set_path(json.loads(json.dumps(base_cfg)), v["complement"], None)
    for i in range(n):
        cfg = load_scenario(base, [])
        drawn = {k: _draw_knob(v, rng) for k, v in knobs.items()}
        for k, v in list(drawn.items()):
            set_path(cfg, k, v)
            if "complement" in knobs[k]:                        # a probability whose sibling must make the row sum to 1
                if not isinstance(v, float) or not 0.0 <= v <= 1.0:
                    raise ValueError(f"{k}: complement knobs must draw a probability in [0, 1], got {v!r}")
                set_path(cfg, knobs[k]["complement"], 1.0 - v)
                drawn[knobs[k]["complement"]] = 1.0 - v
        cfg["seed"] = seed * 1000 + i
        cfg["sampled_knobs"] = drawn
        (out / f"scenario_{i:04d}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return n


def cmd_sample_scenarios(args) -> None:
    n = sample_scenarios(Path(args.base), Path(args.ranges), args.n, args.seed, Path(args.out))
    print(f"wrote {n} scenarios to {args.out}")


# ---- sweep over sampled scenarios (A8 in use) ---------------------------------------------------
def sweep(scenarios: list[Path], variants: dict[str, list[str]], seeds: list[int], out_csv: Path, horizon_s: float | None = None,
          axes: dict[str, list] | None = None, require_knobs: bool = True) -> int:
    """Run every scenario under every named variant (a list of dotted overrides), every axis combination and seed;
    one CSV row per run with the scenario's `sampled_knobs` as `knob:<path>` columns (when present), each axis as a
    column named by its path's last component, and the FLAT metrics. Summaries only; rows are appended to the CSV as
    they finish (a long sweep can be read while it runs)."""
    if not scenarios or not variants or not seeds:
        raise ValueError("sweep needs scenarios, variants and seeds")
    axes = axes or {}
    names, values = list(axes), [axes[k] for k in axes]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n, writer, f = 0, None, None
    try:
        for sc in scenarios:
            for combo in itertools.product(*values) if names else [()]:
                for name, sets in variants.items():
                    for seed in seeds:
                        cfg = load_scenario(sc, list(sets) + [f"seed={seed}"] + ([f"horizon_s={horizon_s}"] if horizon_s else []))
                        for path, v in zip(names, combo):
                            for one in path.split("|"):                       # '|' ties several paths to one axis value
                                set_path(cfg, one, v)
                        if require_knobs and "sampled_knobs" not in cfg:
                            raise KeyError(f"{sc}: not a sampled scenario (no sampled_knobs); use `sample-scenarios` first")
                        _, summary = simulate(cfg)
                        row = {"scenario": Path(sc).stem, "variant": name, "seed": seed}
                        row.update({path.split("|")[0].split(".")[-1]: v for path, v in zip(names, combo)})
                        row.update({f"knob:{k}": v for k, v in cfg.get("sampled_knobs", {}).items()})
                        row.update({k: _get(summary, path) for k, path in FLAT})
                        if writer is None:
                            f = out_csv.open("w", newline="", encoding="utf-8")
                            writer = csv.DictWriter(f, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        f.flush()
                        n += 1
                        print(f"{Path(sc).stem} {name:14s} {' '.join(f'{k}={v}' for k, v in zip(names, combo))} seed={seed} fail={row['failure_rate']:.3f} "
                              f"p50={row['tct_p50']:.0f} p99={row['tct_p99']:.0f} thr={row['throughput_rph']:.0f} wall={row['wall_s']}s", flush=True)
    finally:
        if f is not None:
            f.close()
    return n


def _parse_seeds(text: str) -> list[int]:
    if "-" in text and "," not in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def _parse_axes(items: list[str]) -> dict[str, list]:
    axes = {}
    for it in items:
        path, _, vals = it.partition("=")
        if not path or not vals:
            raise ValueError(f"--axis must be path=v1,v2,..., got {it!r}")
        axes[path] = [parse_value(v) for v in vals.split(",")]
    return axes


def cmd_sweep(args) -> None:
    variants = json.loads(Path(args.variants).read_text(encoding="utf-8"))
    variants = {k: v for k, v in variants.items() if not k.startswith("_")}
    n = sweep([Path(p) for p in args.scenarios], variants, _parse_seeds(args.seeds), Path(args.out), args.horizon,
              _parse_axes(args.axis), require_knobs=not args.any_scenario)
    print(f"wrote {args.out} ({n} runs)")


# ---- selftest ---------------------------------------------------------------------------
def _digest(spans) -> str:
    h = hashlib.sha256()
    for s in spans:
        h.update(f"{s.trace_id}|{s.op}|{s.name}|{s.t_start:.6f}|{s.t_end:.6f}|{s.tokens_out}|{s.outcome}".encode())
    return h.hexdigest()[:16]


def _phase_r1(spans) -> float:
    """Entropy reduction R_1 = 1 - H(next phase | phase) / H(phase) over chat spans within requests."""
    import math
    from collections import Counter, defaultdict
    seqs: dict[tuple[str, int], list[tuple[float, str]]] = defaultdict(list)
    for s in spans:
        if s.op == "chat" and s.outcome == "ok":
            seqs[(s.trace_id, s.attrs["request_idx"] if "request_idx" in s.attrs else -1)].append((s.t_start, s.name))
    uni, pair = Counter(), defaultdict(Counter)
    for v in seqs.values():
        names = [n for _, n in sorted(v)]
        uni.update(names)
        for a, b in zip(names, names[1:]):
            pair[a][b] += 1
    n = sum(uni.values())
    h0 = -sum(c / n * math.log2(c / n) for c in uni.values())
    m = sum(sum(c.values()) for c in pair.values())
    h1 = -sum(v / m * math.log2(v / sum(c.values())) for c in pair.values() for v in c.values())
    return 1.0 - h1 / h0 if h0 > 0 else float("nan")


def _first_steps(spans) -> dict[tuple[int, int], tuple]:
    """(slot, key) -> exogenous facts drawn at program creation, which no policy can influence."""
    return {(s.attrs["slot"], s.attrs["key"]): (s.attrs["n_requests"], s.attrs["sandbox_gb"], s.attrs["context0"])
            for s in spans if s.op == "invoke_agent" and s.parent_span_id is None}


def cmd_selftest(args) -> None:
    base = ROOT / "scenarios/api_coding.json"
    checks = []
    cfg = load_scenario(base, ["horizon_s=1200", "warmup_s=0"])
    a, sa = simulate(cfg)
    b, sb = simulate(load_scenario(base, ["horizon_s=1200", "warmup_s=0"]))
    checks.append(("determinism: same seed -> identical spans", _digest(a) == _digest(b)))
    g, sg = simulate(load_scenario(base, ["horizon_s=1200", "warmup_s=0", "policy.type=reactive_gate"]))
    fa, fg = _first_steps(a), _first_steps(g)
    common = set(fa) & set(fg)
    checks.append(("common random numbers: same (slot,key) -> same exogenous program facts under two policies",
                   len(common) >= 5 and all(fa[k] == fg[k] for k in common)))
    eng = build_engine(load_scenario(base, ["horizon_s=1200", "warmup_s=0"]))
    eng.run()
    ok = True
    for r in eng.res.values():
        try:
            r.check()
        except AssertionError:
            ok = False
    checks.append(("resource accounting: used == sum(holders) for every resource", ok))
    checks.append(("no session keeps holds after ending",
                   all(not p.holds for p in eng.programs.values() if p.status in ("done", "aborted"))))
    fid = sg["fidelity"]
    checks.append(("fidelity: tool tail shares near TraceLab targets",
                   0.55 <= fid["tool_share_calls_lt_1s"] <= 0.85 and 0.02 <= fid["tool_share_calls_gt_60s"] <= 0.08))
    checks.append(("fidelity: output-token median near 252", 150 <= fid["out_median"] <= 350))
    h = simulate(load_scenario(ROOT / "scenarios/hosted_mixed.json", ["horizon_s=1800", "warmup_s=0", "policy.type=clairvoyant"]))[1]
    checks.append(("clairvoyant gate runs and issues no 429 to external providers", h["errors"]["429"] == 0))
    hosted = ROOT / "scenarios/hosted_mixed.json"
    u0, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0"]))
    u1, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "generator.recipe_temperature=1.0"]))
    checks.append(("recipe_temperature=1 leaves every span identical (A7 knob is a no-op at 1)", _digest(u0) == _digest(u1)))
    cold, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "generator.recipe_temperature=0.05"]))
    checks.append(("recipe_temperature=0.05 makes phase transitions near-deterministic (R_1 > 0.8) and changes the digest",
                   _phase_r1(cold) > 0.8 and _digest(cold) != _digest(u0) and _phase_r1(u0) < 0.6))
    try:
        simulate(load_scenario(hosted, ["horizon_s=600", "warmup_s=0", "generator.recipe_temperature=0"]))
        bad_t = False
    except ValueError:
        bad_t = True
    checks.append(("recipe_temperature <= 0 throws", bad_t))
    import sys, tempfile
    sys.path.insert(0, str(ROOT / "rung0"))
    from observer import load as obs_load, sequences as obs_sequences   # noqa: E402
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "t.jsonl"
        write_jsonl(f, u0)
        one, two = obs_sequences(obs_load([f])), obs_sequences(obs_load([f, f]))
        checks.append(("observer: pooling two trace files never merges sessions by colliding trace ids (F15)",
                       len(two) == 2 * len(one) and sorted(len(v) for v in two.values()) == sorted(2 * [len(v) for v in one.values()])))
    from e5_idle_predictability import think_r2   # noqa: E402  (rung0 on sys.path above)
    s0, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "generator.think_snr=0.0"]))
    checks.append(("think_snr=0 leaves every span identical (B7 knob is a no-op at 0)", _digest(s0) == _digest(u0)))
    long0, _ = simulate(load_scenario(hosted, ["horizon_s=7200", "warmup_s=0", "generator.think_snr=0.0"]))
    long9, _ = simulate(load_scenario(hosted, ["horizon_s=7200", "warmup_s=0", "generator.think_snr=0.9"]))
    long9b, _ = simulate(load_scenario(hosted, ["horizon_s=7200", "warmup_s=0", "generator.think_snr=0.9", "seed=2"]))
    lt9 = np.log([sp.duration for sp in long9 + long9b if sp.op == "think"])
    mu, sigma = np.log(84.0), np.log(1236.0 / 84.0) / 1.2815515655446004          # the seeded think_time marginal
    checks.append((f"think_snr=0.9 preserves the log-mean and log-sd of think time (mean {lt9.mean():.2f} vs {mu:.2f}, sd {lt9.std():.2f} vs {sigma:.2f}; n={len(lt9)})",
                   abs(lt9.mean() - mu) <= 0.15 and 0.9 <= lt9.std() / sigma <= 1.1))
    r2_0, r2_9 = think_r2(long0), think_r2(long9 + long9b)             # pooled over two run seeds: one society
    checks.append((f"think time is state-predictable in proportion to think_snr, identically across run seeds (R² {r2_0:.2f} at 0, {r2_9:.2f} at 0.9)",
                   r2_0 < 0.15 and r2_9 > 0.7))
    try:
        simulate(load_scenario(hosted, ["horizon_s=600", "warmup_s=0", "generator.think_snr=1.5"]))
        bad_snr = False
    except ValueError:
        bad_snr = True
    checks.append(("think_snr outside [0, 1] throws", bad_snr))
    cfg = load_scenario(hosted, ["resources.ext.search.rpm=40"])
    checks.append(("set_path resolves dotted resource names greedily (resources.ext.search.rpm)", cfg["resources"]["ext.search"]["rpm"] == 40))
    try:
        load_scenario(hosted, ["resources.ext.nothing.rpm=40"])
        bad_path = False
    except KeyError:
        bad_path = True
    checks.append(("set_path throws on an unknown dotted path", bad_path))
    cfg = load_scenario(hosted, [])
    tag = apply_axes(cfg, ["resources.ext.search.rpm|resources.ext.web.rpm", "policy.type", "recipes"], (15, "reactive_gate", "data/fitted/r.json"))
    checks.append(("grid axes may tie several paths with '|' (search+web rpm set together, tag from the first, file-safe)",
                   cfg["resources"]["ext.search"]["rpm"] == 15 and cfg["resources"]["ext.web"]["rpm"] == 15 and tag == "rpm=15_type=reactive_gate_recipes=data~fitted~r.json"))
    tail1, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "generator.tool_tail_scale=1.0", "generator.p_parallel_scale=1.0"]))
    checks.append(("tool_tail_scale=1 and p_parallel_scale=1 leave every span identical", _digest(tail1) == _digest(u0)))
    w1 = build_engine(load_scenario(hosted, ["generator.tool_tail_scale=1.0"])).wl
    w2 = build_engine(load_scenario(hosted, ["generator.tool_tail_scale=1.6"])).wl
    x1 = np.array([w1.tool_dist["bash"].sample(np.random.default_rng(3)) for _ in range(1)] + [w1.tool_dist["bash"].sample(np.random.default_rng([3, i])) for i in range(4000)])
    x2 = np.array([w2.tool_dist["bash"].sample(np.random.default_rng(3)) for _ in range(1)] + [w2.tool_dist["bash"].sample(np.random.default_rng([3, i])) for i in range(4000)])
    checks.append((f"tool_tail_scale=1.6 widens sampled tool durations with the median kept (bash p99 {np.percentile(x1, 99):.0f} -> {np.percentile(x2, 99):.0f} s, "
                   f"p50 {np.percentile(x1, 50):.2f} -> {np.percentile(x2, 50):.2f} s)",
                   np.percentile(x2, 99) > 1.3 * np.percentile(x1, 99) and 0.7 < np.percentile(x2, 50) / np.percentile(x1, 50) < 1.4))
    with tempfile.TemporaryDirectory() as td:
        n_written = sample_scenarios(hosted, ROOT / "scenarios/knob_ranges.json", 4, 7, Path(td))
        files = sorted(Path(td).glob("*.json"))
        ok_all = n_written == 4 == len(files)
        for f in files:
            c = json.loads(f.read_text(encoding="utf-8"))
            ok_all = ok_all and "sampled_knobs" in c and all(_get(c, k.split(".")[0]) is not None for k in c["sampled_knobs"])
            c["horizon_s"], c["warmup_s"] = 900, 0
            simulate(c)                                              # every sampled scenario must run, not just build
        checks.append(("sample-scenarios writes n scenarios that run, each recording every drawn knob (incl. complements)", ok_all))
        try:
            sample_scenarios(hosted, {"resources.ext.nothing.rpm": {"type": "uniform", "low": 1, "high": 2}}, 1, 0, Path(td))
            bad_range = False
        except KeyError as e:
            bad_range = "ext.nothing" in str(e)
        checks.append(("sample-scenarios throws naming the unknown knob path", bad_range))
    from .fit import fit_all
    src_cfg = load_scenario(hosted, ["horizon_s=3600", "warmup_s=0"])
    src, ssrc = simulate(src_cfg)
    fm, fr = fit_all([src], json.loads((ROOT / "data/marginals.json").read_text(encoding="utf-8")),
                     json.loads((ROOT / "data/recipes.json").read_text(encoding="utf-8")))
    Recipes(fr)
    Marginals(fm)
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m.json").write_text(json.dumps(fm), encoding="utf-8")
        (Path(td) / "r.json").write_text(json.dumps(fr), encoding="utf-8")
        fcfg = load_scenario(hosted, ["horizon_s=3600", "warmup_s=0", f"recipes={Path(td) / 'r.json'}", f"marginals={Path(td) / 'm.json'}"])
        fit_spans, sfit = simulate(fcfg)
    a, b = ssrc["fidelity"], sfit["fidelity"]
    checks.append((f"fit round trip on synthetic traces: fitted recipes+marginals load, run, and reproduce the source "
                   f"(tool time share >60 s {a['tool_share_time_gt_60s']:.2f} vs {b['tool_share_time_gt_60s']:.2f}, out median {a['out_median']:.0f} vs {b['out_median']:.0f}, "
                   f"chats/request {a['chats_per_request']:.1f} vs {b['chats_per_request']:.1f})",
                   abs(a["tool_share_time_gt_60s"] - b["tool_share_time_gt_60s"]) <= 0.15 and 0.7 <= b["out_median"] / a["out_median"] <= 1.3
                   and 0.6 <= b["chats_per_request"] / a["chats_per_request"] <= 1.4))
    g2, sg2 = simulate(load_scenario(hosted, ["horizon_s=3600", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=reactive_gate"]))
    o2, so2 = simulate(load_scenario(hosted, ["horizon_s=3600", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=clairvoyant"]))
    checks.append((f"retention_ttl (B6): KV retained-but-unused token-seconds are measured and the oracle's exact-gap TTL wastes no more than the gate's "
                   f"({so2['waste']['kv_retained_token_s_unused']:.2e} vs {sg2['waste']['kv_retained_token_s_unused']:.2e})",
                   so2["waste"]["kv_retained_token_s_unused"] <= sg2["waste"]["kv_retained_token_s_unused"]))
    from .predict import MarkovNext, QuantileTracker
    qt = QuantileTracker(window=512, min_n=8)
    rq = np.random.default_rng(11)
    for _ in range(5000):
        qt.observe(("coding", "r0"), float(rq.lognormal(4.0, 1.5)))
    q50, q90 = qt.quantile(("coding", "r0"), 0.5), qt.quantile(("coding", "r0"), 0.9)
    checks.append((f"QuantileTracker recovers lognormal quantiles from a stream (p50 {q50:.0f} vs {np.exp(4.0):.0f}, p90 {q90:.0f} vs {np.exp(4.0 + 1.2816 * 1.5):.0f})",
                   0.9 <= q50 / np.exp(4.0) <= 1.1 and 0.85 <= q90 / np.exp(4.0 + 1.2816 * 1.5) <= 1.15))
    checks.append(("QuantileTracker backs off to the parent key, then to the global root; an empty tracker returns None",
                   qt.quantile(("coding", "r8+"), 0.5) == qt.quantile(("coding",), 0.5) and qt.quantile(("research", "r0"), 0.5) == qt.quantile((), 0.5)
                   and QuantileTracker().quantile(("x",), 0.5) is None))
    mk = MarkovNext(k=1)
    rows = {"a": {"a": 0.2, "b": 0.7, "c": 0.1}, "b": {"a": 0.5, "c": 0.5}, "c": {"a": 1.0}}
    state = "a"
    for _ in range(20000):
        nxt = str(rq.choice(list(rows[state]), p=list(rows[state].values())))
        mk.observe((state,), nxt)
        state = nxt
    l1 = max(sum(abs(mk.predict((st,)).get(t, 0.0) - pr) for t, pr in rows[st].items()) for st in rows)
    checks.append((f"MarkovNext recovers a 3-state chain's rows online (max L1 {l1:.3f})", l1 < 0.05))
    checks.append(("MarkovNext backs off to the unigram for an unseen context", abs(sum(mk.predict(("zzz",)).values()) - 1.0) < 1e-9))
    lease_cfg = load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=lease", "generator.think_snr=0.9"])
    leng = build_engine(lease_cfg)
    leng.run()
    lsum = summarize(leng.spans, 1800.0, 0.0, {n: r.capacity for n, r in leng.res.items()})
    checks.append(("lease controller runs on the hosted society: no 429 to providers, no leaked holds, no deadlocks",
                   lsum["errors"]["429"] == 0 and leng.deadlocks_detected == 0
                   and all(not q.holds for q in leng.programs.values() if q.status in ("done", "aborted"))))
    pol = leng.policy
    probe = next(q for q in leng.programs.values() if q.request_idx >= 1)
    honest = all(pol.idle_timeout(probe, a, 300.0) == pol.idle_timeout(probe, b, 300.0) and pol.prewarm_delay(probe, a) == pol.prewarm_delay(probe, b)
                 and pol.retention_ttl(probe, None, leng.now, a, 300.0) == pol.retention_ttl(probe, None, leng.now, b, 300.0)
                 for a, b in ((1.0, 1e6), (0.0, 3600.0)))
    checks.append(("lease controller is honest: contradictory hidden think/gap values change none of its decisions", honest))
    checks.append(("lease controller has learned from observed idle gaps (its idle tracker is non-empty)", pol.idle.n(()) >= 8))
    from .resources import Lease, Ledger
    from .policies import ReactiveGate
    led = Ledger()
    led.set_session("A", [Lease("A", "sandbox.cpu", 4.0, start=10.0, expiry=70.0)])
    cpu = build_engine(load_scenario(hosted, [])).res["sandbox.cpu"]
    cpu.take("A", 1.0)
    checks.append(("ledger: others see capacity minus the holder's unconsumed lease (4 leased, 1 held -> 3 reserved), the holder sees none",
                   abs(led.reserved_for_others("sandbox.cpu", "B", 20.0, cpu.holders) - 3.0) < 1e-9
                   and led.reserved_for_others("sandbox.cpu", "A", 20.0, cpu.holders) == 0.0
                   and led.reserved_for_others("sandbox.cpu", "B", 5.0, cpu.holders) == 0.0))       # not started yet
    n_exp = led.expire(70.0)
    checks.append(("ledger: leases expire unconditionally at their expiry (I3)", n_exp == 1 and led.reserved_for_others("sandbox.cpu", "B", 70.0, cpu.holders) == 0.0))
    try:
        led.set_session("A", [Lease("A", "sandbox.cpu", 100.0, 0.0, 10.0)])
        led.check({"sandbox.cpu": cpu}, 1.0)
        i1 = False
    except AssertionError:
        i1 = True
    try:
        led.set_session("C", [Lease("C", "model.kv", 1.0, 0.0, 10.0)])
        kv_ok = False
    except ValueError:
        kv_ok = True
    checks.append(("ledger: I1 (sum of active leases <= capacity) is asserted; leases on model.kv are rejected", i1 and kv_ok))

    class _ToyLeaser(ReactiveGate):
        name = "toy_lease"

        def leases(self, p, step, now):
            if step is None or step.kind != "chat" or self.ledger.active_sum("sandbox.cpu", now) + 3.0 > self.resources["sandbox.cpu"].capacity:
                return []
            return [Lease(p.sid, "sandbox.cpu", 3.0, now, now + 60.0)]

    tcfg = load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=reactive_gate"])
    teng = build_engine(tcfg)
    teng.policy = _ToyLeaser(tcfg["policy"], np.random.default_rng(1))
    teng.policy.ledger, teng.policy.resources = teng.ledger, teng.res
    teng.run()
    checks.append(("a toy policy that leases CPU ahead runs with accounting and I1-I3 intact, no deadlocks, no leaked holds",
                   teng.deadlocks_detected == 0 and all(not q.holds for q in teng.programs.values() if q.status in ("done", "aborted"))
                   and teng.ledger.expired_total > 0))
    fresh, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0"]))
    checks.append(("with no policy issuing leases the ledger is invisible (digest unchanged)", _digest(fresh) == _digest(u0)))
    gsum = {}
    for pol in ("reactive_gate", "lease"):
        gcfg = load_scenario(hosted, ["horizon_s=3600", "warmup_s=0", "arrivals.rate_per_min=2.0", "generator.p_parallel_scale=3.0", "policy.h=60", f"policy.type={pol}"])
        ge = build_engine(gcfg)
        ge.run()
        gsum[pol] = (summary_of(ge, gcfg), ge)
    sg_, sl_ = gsum["reactive_gate"][0]["steps"], gsum["lease"][0]["steps"]
    le = gsum["lease"][1]
    checks.append((f"gang leases: the lease controller runs more parallel groups in parallel than the gate "
                   f"({sl_['parallel_groups_run_parallel']}/{sl_['parallel_groups']} vs {sg_['parallel_groups_run_parallel']}/{sg_['parallel_groups']}), "
                   f"with no deadlocks, no leaked holds, {le.ledger.issued_total} leases issued",
                   sl_["parallel_groups_run_parallel"] > sg_["parallel_groups_run_parallel"] and le.deadlocks_detected == 0
                   and all(not q.holds for q in le.programs.values() if q.status in ("done", "aborted")) and le.ledger.issued_total > 0))
    lw = gsum["lease"][0]["waste"]
    checks.append((f"gang leases: reserved-but-unused CPU unit-seconds are measured ({lw['lease_unit_s_unused']:.0f} of {lw['lease_unit_s_issued']:.0f} issued)",
                   lw["lease_unit_s_issued"] > 0 and 0 <= lw["lease_unit_s_unused"] <= lw["lease_unit_s_issued"]))
    stress = ROOT / "scenarios/hosted_stress.json"
    ssum = {}
    for pol in ("reactive_gate", "lease"):
        scfg = load_scenario(stress, ["horizon_s=3600", "warmup_s=0", "resources.ext.search.rpm=10", "resources.ext.web.rpm=10", f"policy.type={pol}"])
        se = build_engine(scfg)
        se.run()
        ssum[pol] = (summary_of(se, scfg), se)
    g429, l429 = ssum["reactive_gate"][0]["errors"]["429"], ssum["lease"][0]["errors"]["429"]
    checks.append((f"budget leases: on the stress society at 10 RPM the lease issues no call that 429s (gate {g429}, lease {l429}) "
                   f"with {ssum['lease'][1].ledger.issued_total} leases and no deadlocks",
                   g429 > 0 and l429 == 0 and ssum["lease"][1].deadlocks_detected == 0 and ssum["lease"][1].ledger.issued_total > 0))
    q0, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "policy.queue=fifo"]))
    checks.append(("gate queue=fifo is the gate (digest unchanged)", _digest(q0) == _digest(u0)))
    vcfg = load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=reactive_gate", "policy.queue=vtfq"])
    ve = build_engine(vcfg)
    ve.run()
    vs = summary_of(ve, vcfg)
    fifo_spans, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=reactive_gate", "policy.queue=fifo"]))
    checks.append((f"gate queue=vtfq (realised memory-centric cost, no prediction) runs clean and reorders waiters ({len(ve.policy.tags)} sessions tagged; spans differ from fifo)",
                   ve.deadlocks_detected == 0 and all(not q.holds for q in ve.programs.values() if q.status in ("done", "aborted"))
                   and vs["errors"]["429"] == 0 and len(ve.policy.tags) > 0 and _digest(ve.spans) != _digest(fifo_spans)))
    ccfg = load_scenario(stress, ["horizon_s=3600", "warmup_s=0", "resources.ext.search.rpm=10", "resources.ext.web.rpm=10", "policy.type=lease", "policy.tau=0.5"])
    ce = build_engine(ccfg)
    ce.run()
    p_hat = ce.policy.collision_rate("ext.search")
    csum = summary_of(ce, ccfg)
    checks.append((f"calibrated budget margin: the lease learns the provider's collision rate from observed call costs (p̂ {p_hat:.2f} vs background 0.40) "
                   f"and at tau=0.5 issues with a one-unit margin (429s {csum['errors']['429']}, gate {g429})",
                   abs(p_hat - 0.4) < 0.1 and csum["errors"]["429"] < g429))
    with tempfile.TemporaryDirectory() as td:
        files = sorted((ROOT / "scenarios/sampled").glob("*.json"))[:2]
        variants = {"gate": ["policy.type=reactive_gate"], "oracle": ["policy.type=clairvoyant", "framework.sandbox_idle_timeout_s=0"]}
        n = sweep(files, variants, [1], Path(td) / "sweep.csv", horizon_s=900)
        with (Path(td) / "sweep.csv").open(newline="", encoding="utf-8") as f:
            srows = list(csv.DictReader(f))
        checks.append(("sweep: sampled scenarios x named variants x seeds -> one CSV with knobs and metrics",
                       n == 4 == len(srows) and {r["variant"] for r in srows} == {"gate", "oracle"}
                       and all(k in srows[0] for k in ("scenario", "seed", "failure_rate", "knob:generator.think_snr", "knob:framework.sandbox_cold_start_s"))))
    scfg4 = load_scenario(ROOT / "scenarios/sampled/scenario_0004.json", ["horizon_s=3600", "warmup_s=0", "seed=1"])
    res_q = {}
    for q in ("fifo", "srpt"):
        c = json.loads(json.dumps(scfg4)); set_path(c, "policy.type", "reactive_gate"); set_path(c, "policy.queue", q); set_path(c, "framework.sandbox_idle_timeout_s", 0)
        e = build_engine(c); e.run(); res_q[q] = (summary_of(e, c), e)
    f, sr = res_q["fifo"][0], res_q["srpt"][0]
    checks.append((f"gate queue=srpt (observable service-time estimate: tokens_in/prefill + learned mean tokens_out/decode) on a model-saturated scenario: "
                   f"p50 {f['request_tct_s']['p50']:.0f} -> {sr['request_tct_s']['p50']:.0f} s, throughput {f['throughput_requests_per_hour']:.0f} -> {sr['throughput_requests_per_hour']:.0f}",
                   sr["request_tct_s"]["p50"] < 0.5 * f["request_tct_s"]["p50"] and sr["throughput_requests_per_hour"] > f["throughput_requests_per_hour"]
                   and res_q["srpt"][1].deadlocks_detected == 0))
    pol = res_q["srpt"][1].policy
    probe = next(iter(res_q["srpt"][1].programs.values()))
    from .workload import Step as _Step
    st_a = _Step(kind="chat", name="plan", tokens_in=50000, tokens_out=100, duration=999.0)
    st_b = _Step(kind="chat", name="plan", tokens_in=50000, tokens_out=100000, duration=1.0)
    checks.append(("gate queue=srpt is honest: identical observable requests get identical priority whatever their hidden tokens_out/duration",
                   pol.priority(probe, st_a, 0.0, 0.0) == pol.priority(probe, st_b, 0.0, 0.0)))
    from tracelab_to_spans import convert as tl_convert   # noqa: E402  (rung0 on sys.path)
    import gzip
    with tempfile.TemporaryDirectory() as td:
        rec = lambda idx, first, events, tools, out=40: {"provider": "claude", "session_id": "s1", "round_index": idx, "round_id": f"r{idx}", "model": "m",
                                                        "input_tokens_total": 1000 + idx, "newly_append_tokens": 100, "output_tokens": out, "reasoning_output_tokens": None,
                                                        "first_input_event_type": first, "timing_events": events, "tools": tools, "user": "u1"}
        T = lambda sec: f"2026-06-01T00:{sec // 60:02d}:{sec % 60:02d}.000Z"
        rounds = [rec(0, "user_message", [{"event_type": "user_message", "timestamp": T(0)}, {"event_type": "tool_call", "timestamp": T(2)}],
                      [{"tool_name": "Bash", "tool_call_id": "c1", "emitted_at": T(2), "result_at": T(12)},
                       {"tool_name": "Read", "tool_call_id": "c2", "emitted_at": T(3), "result_at": T(4)}]),          # overlaps c1: parallel
                  rec(1, "tool_result", [{"event_type": "tool_result", "timestamp": T(12)}, {"event_type": "text", "timestamp": T(15)}], []),
                  rec(2, "user_message", [{"event_type": "user_message", "timestamp": T(615)}, {"event_type": "text", "timestamp": T(617)}],
                      [{"tool_name": "AskUserQuestion", "tool_call_id": "c3", "emitted_at": T(617), "result_at": T(900)}])]
        gz = Path(td) / "t.jsonl.gz"
        with gzip.open(gz, "wt", encoding="utf-8") as f:
            for r in rounds:
                f.write(json.dumps(r) + "\n")
        tspans, tstats = tl_convert(gz, "claude")
        ops = Counter(sp.op for sp in tspans)
        think = [sp for sp in tspans if sp.op == "think"]
        par = [sp for sp in tspans if sp.op == "execute_tool"]
        checks.append(("tracelab adapter: sessions -> requests at user messages, chats, one parallel tool group, one think gap; human-wait tools dropped",
                       ops["invoke_agent"] == 1 and ops["invoke_workflow"] == 2 and ops["chat"] == 3 and len(par) == 1 and par[0].name == "parallel"
                       and par[0].attrs["members"] == ["bash", "read"] and len(think) == 1 and abs(think[0].duration - 600.0) < 1e-6
                       and think[0].attrs["phase_end"] == "after:read" and tstats["human_tools_dropped"] == 1))
    from ceilings import hashed_features, SparseLogReg   # noqa: E402  (rung0 on sys.path)
    checks.append(("ceilings: hashed n-gram features are deterministic and order-sensitive",
                   hashed_features("pytest -q tests/") == hashed_features("pytest -q tests/") and hashed_features("a b") != hashed_features("b a")))
    rc = np.random.default_rng(5)
    verbs, args = ["ls", "cat", "grep", "echo", "git", "npm", "pytest", "make"], ["-la", "src/", "foo.py", "|", "wc", "-r", "tests/", "build"]
    X, y = [], []
    for _ in range(3000):
        toks = [str(rc.choice(verbs))] + [str(rc.choice(args)) for _ in range(int(rc.integers(0, 4)))]
        X.append(hashed_features(" ".join(toks)))
        y.append(2 if ("pytest" in toks or ("npm" in toks and "build" in toks)) else (1 if "git" in toks else 0))   # the planted rule
    y = np.array(y)
    clf = SparseLogReg(n_classes=3).fit(X[:2400], y[:2400], epochs=8)
    acc = float(np.mean(clf.predict(X[2400:]) == y[2400:]))
    checks.append((f"ceilings: sparse logistic regression on hashed skeletons recovers a planted rule (held-out accuracy {acc:.3f})", acc > 0.9))
    checks.append(("fit: every fitted entry carries provenance and the fitted recipes use observable phases",
                   all(k in fm["provenance"] for k in ("think_time", "session_requests", "sandbox_mem_gb"))
                   and all(ph == "start" or ph.startswith("after:") for r in fr["recipes"].values() for ph in r["transitions"])))
    # ---- Phase 6: content cues and multi-agent spawn/join --------------------------------------------------
    from .workload import duration_bin, out_bin
    c9, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "generator.content_snr=0.9"]))
    checks.append(("content_snr=0.9 leaves every span identical (cues come from their own stream; the steps are untouched)",
                   _digest(c9) == _digest(u0)))
    def cue_acc(spans):
        hits = tot = 0
        for sp_ in spans:
            if sp_.op == "execute_tool" and sp_.outcome == "ok":
                for cue, d in zip(sp_.attrs["content"], sp_.attrs["durations"]):
                    hits += int(cue.split(":")[-1]) == duration_bin(d)
                    tot += 1
        return hits / tot
    acc0, acc9 = cue_acc(u0), cue_acc(c9)
    checks.append((f"tool content cues reveal the duration class with probability snr + (1 - snr)/4 (acc {acc0:.2f} at 0, {acc9:.2f} at 0.9)",
                   abs(acc0 - 0.25) < 0.05 and abs(acc9 - 0.925) < 0.03))
    plan_ok = all(sp_.attrs["plan"] == "pl:final" for sp_ in c9 if sp_.op == "chat" and sp_.outcome == "ok" and sp_.attrs["n_tools"] == 0 and sp_.name == "final")
    checks.append(("a final chat's plan cue is 'pl:final' (the turn closed without tool calls: observable)", plan_ok))
    try:
        simulate(load_scenario(hosted, ["horizon_s=600", "warmup_s=0", "generator.content_snr=1.5"]))
        bad_c = False
    except ValueError:
        bad_c = True
    checks.append(("content_snr outside [0, 1] throws", bad_c))
    ma = ROOT / "scenarios/multiagent.json"
    mres = {}
    for pol, idle in (("reactive_gate", 300), ("clairvoyant", 0)):
        mcfg = load_scenario(ma, ["horizon_s=3600", "warmup_s=0", f"policy.type={pol}", f"framework.sandbox_idle_timeout_s={idle}"])
        me = build_engine(mcfg)
        me.run()
        mres[pol] = (summary_of(me, mcfg), me)
    ms, me = mres["reactive_gate"]
    kids = [q for q in me.programs.values() if q.parent is not None]
    checks.append((f"multi-agent society (B10): orchestrators spawn sub-agents ({me.spawns} fan-outs, {len(kids)} children, {me.joins} joins) that run "
                   f"their own request and release everything; the parent continues after the join; child requests are counted inside the parent's",
                   me.spawns > 0 and me.joins > 0 and len(kids) >= 2 * me.spawns
                   and all(not q.holds for q in me.programs.values() if q.status in ("done", "aborted"))
                   and all(q.n_requests == 1 and q.trace == me.programs[q.parent].trace for q in kids)
                   and ms["requests"]["started"] == sum(1 for sp_ in me.spans if sp_.op == "invoke_workflow" and not sp_.attrs.get("child"))
                   and ms["steps"]["children"] > 0))
    cyc_gate, cyc_orc = ms["errors"]["deadlock_cycles_seen"], mres["clairvoyant"][0]["errors"]["deadlock_cycles_seen"]
    checks.append((f"join cycles: parents holding sandboxes while their children queue for them are detected as hold-and-wait cycles under the gate "
                   f"({cyc_gate}) and vanish when the orchestrator's sandbox is parked at once ({cyc_orc})", cyc_gate > 0 and cyc_orc == 0))
    ma1, _ = simulate(load_scenario(ma, ["horizon_s=1800", "warmup_s=0"]))
    ma2, _ = simulate(load_scenario(ma, ["horizon_s=1800", "warmup_s=0"]))
    checks.append(("multi-agent runs are deterministic and child span ids are unique", _digest(ma1) == _digest(ma2) and len({sp_.span_id for sp_ in ma1}) == len(ma1)))
    # ---- Phase 6: the Needs controller (reserver.py), its predictor (needs.py), the Observer (features.py), Jev (jev.py) ----
    ncfg = load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=needs", "generator.content_snr=0.6"])
    ne = build_engine(ncfg)
    ne.run()
    nsum = summary_of(ne, ncfg)
    rep_ = nsum["controller"]
    checks.append((f"needs controller runs on the hosted society: no 429, no deadlocks, no leaked holds, I1/I3 intact, "
                   f"{rep_['records']} records, {rep_['predictor']['learned']} online updates, decisions {sum(rep_['decisions'].values())}",
                   nsum["errors"]["429"] == 0 and ne.deadlocks_detected == 0 and rep_["predictor"]["learned"] > 500
                   and all(not q.holds for q in ne.programs.values() if q.status in ("done", "aborted"))))
    ne2 = build_engine(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=needs", "generator.content_snr=0.6"]))
    ne2.run()
    checks.append(("needs controller is deterministic (same seed -> identical spans)", _digest(ne.spans) == _digest(ne2.spans)))
    pol = ne.policy
    probe = next(q for q in ne.programs.values() if q.request_idx >= 1 and q.sid in pol.p_idle)
    from .workload import Step as _Step
    sa = _Step(kind="chat", name="plan", tokens_in=30000, tokens_out=100, duration=999.0, content="pc:o1 pc:t1")
    sb = _Step(kind="chat", name="verify", tokens_in=30000, tokens_out=90000, duration=1.0, content="pc:o1 pc:t1")
    ph = probe.phase
    honest = all(pol.idle_timeout(probe, a_, 300.0) == pol.idle_timeout(probe, b_, 300.0) and pol.prewarm_delay(probe, a_) == pol.prewarm_delay(probe, b_)
                 and pol.retention_ttl(probe, None, ne.now, a_, 300.0) == pol.retention_ttl(probe, None, ne.now, b_, 300.0)
                 for a_, b_ in ((1.0, 1e6), (0.0, 3600.0)))
    honest = honest and pol.priority(probe, sa, 0.0, 10.0) == pol.priority(probe, sb, 0.0, 10.0) and pol.kv_reserve(probe, sa, 4096) == pol.kv_reserve(probe, sb, 4096)
    probe.phase = "zzz"
    honest = honest and pol.leases(probe, sa, ne.now) == pol.leases(probe, sb, ne.now)
    probe.phase = ph
    checks.append(("needs controller is honest: contradictory hidden think/gap/tokens_out/duration/phase values change none of its decisions", honest))
    src_txt = (Path(__file__).parent / "reserver.py").read_text(encoding="utf-8") + (Path(__file__).parent / "features.py").read_text(encoding="utf-8")
    import re as _re
    leaks = [m for m in _re.findall(r"\b(?:p|step|probe|st)\.(phase_end|phase|follow_tools|siblings|duration|tokens_out|spawn)\b", src_txt)]
    checks.append((f"needs controller source never reads a Program/Step hidden field (phase, follow_tools, duration, tokens_out; found {leaks})", not leaks))
    off_all = "+".join(sorted(__import__("agentsim.reserver", fromlist=["SWITCHES"]).SWITCHES))
    g_fifo, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=reactive_gate", "policy.queue=fifo"]))
    n_off, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=needs", f"policy.ablate={off_all}"]))
    checks.append(("needs controller with every switch off reproduces the fifo gate exactly (the switch machinery is sound)", _digest(n_off) == _digest(g_fifo)))
    n_ev, _ = simulate(load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=needs", "policy.ablate=ev"]))
    checks.append(("needs controller: switching the expected-value gate off changes decisions (park always, lease always)", _digest(n_ev) != _digest(ne.spans)))
    jcfg = load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=needs", "policy.needs.jev.enabled=true", "policy.needs.jev.rpm=30"])
    je = build_engine(jcfg)
    je.run()
    jr = je.policy.report()["jev"]
    jd = je.policy.decisions
    checks.append((f"Jev channel: asked at decision points asynchronously within its budget (asked {jd['jev_asked']}, calls {jr['calls']}, dropped {jr['dropped']}, "
                   f"delivered {jd['jev_delivered']}, mean latency {jr['mean_latency_s']} s); views carry the latest record",
                   jd["jev_asked"] > 100 and jr["dropped"] > 0 and jr["calls"] > 0 and jd["jev_delivered"] <= jr["calls"] and 0.05 < jr["mean_latency_s"] < 0.6
                   and any(v.jev is not None for v in je.policy.observer.views.values())))
    mcfg = load_scenario(ma, ["horizon_s=2400", "warmup_s=0", "arrivals.rate_per_min=1.5", "policy.type=needs", "generator.content_snr=0.6", "policy.ablate=ev"])
    me2 = build_engine(mcfg)
    me2.run()
    msum2 = summary_of(me2, mcfg)
    d2 = me2.policy.decisions
    checks.append((f"needs controller on the multi-agent society (EV gate off): spawn gangs issued ({d2['spawn_gang']}; family-owned leases let the children draw on them), "
                   f"{me2.joins} joins, no deadlocks, no leaked holds, I1/I3 intact",
                   d2["spawn_gang"] > 0 and me2.joins > 0 and me2.deadlocks_detected == 0
                   and all(not q.holds for q in me2.programs.values() if q.status in ("done", "aborted"))))
    from .resources import Lease as _Lease, Ledger as _Ledger
    fl = _Ledger()
    fl.set_session("s1", [_Lease("s1", "model.slots", 3.0, 0.0, 100.0)])
    holders = {"s1/1.0": 1.0, "s2": 1.0}
    checks.append(("ledger: a parent's lease is the family's — its children are not kept off it and their holds consume it; strangers see the rest",
                   fl.reserved_for_others("model.slots", "s1/1.1", 10.0, holders) == 0.0 and fl.reserved_for_others("model.slots", "s2", 10.0, holders) == 2.0
                   and fl.own("model.slots", "s1/1.1", 10.0) == 0.0 and fl.active_max("model.slots", 50.0, 150.0) == 3.0 and fl.active_max("model.slots", 100.0, 150.0) == 0.0))
    from .train import train_needs as _train_needs, train_jev as _train_jev, annotate as _annotate
    from .needs import NeedsModel as _NM, NeedsPredictor as _NP
    with tempfile.TemporaryDirectory() as td:
        tcfg = load_scenario(hosted, ["horizon_s=3600", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=reactive_gate", "generator.content_snr=0.9"])
        tsp, tsum = simulate(tcfg)
        _write_run(Path(td) / "r1", tsp, tsum, tcfg)
        rn = _train_needs([str(Path(td) / "r1" / "traces.jsonl")], Path(td) / "needs.npz", epochs=3)
        q = rn["holdout"]["Q_tool"]
        m1 = _NM.load(Path(td) / "needs.npz")
        p1 = _NP(m1.vocab, 0.8, model=m1, learn=False)
        rec0 = ne.policy.observer.featurize(next(iter(ne.policy.observer.views.values())), "tool_ready", 0.0, ["kind:bash", "res:sandbox.cpu"], 1000, ["sk:bash:3"], ("bash",))
        pa = p1.predict(rec0)
        checks.append((f"train-needs: replayed traces train the predictor offline (held-out tool-duration R² {q['r2_median']}, pinball ratio {q['pinball80_ratio']} at "
                       f"content_snr 0.9); the model round-trips through npz",
                       q["r2_median"] > 0.3 and q["pinball80_ratio"] < 0.9 and abs(pa.quantile("Q_tool", 0.8, conformal=False) - _NP(m1.vocab, 0.8, model=_NM.load(Path(td) / "needs.npz"), learn=False).predict(rec0).quantile("Q_tool", 0.8, conformal=False)) < 1e-9))
        rj = _train_jev([str(Path(td) / "r1" / "traces.jsonl")], Path(td) / "jev.npz", epochs=2)
        pq = rj["per_question"]
        checks.append((f"train-jev: the local System One model learns typed answers from content and is calibrated on held-out sessions "
                       f"(phase acc {pq['phase']['accuracy']} vs majority {pq['phase']['majority']}, duration_class acc {pq['duration_class']['accuracy']} vs {pq['duration_class']['majority']}; "
                       f"ECE phase {pq['phase']['ece']}, duration {pq['duration_class']['ece']})",
                       pq["phase"]["accuracy"] > pq["phase"]["majority"] + 0.2 and pq["duration_class"]["accuracy"] > pq["duration_class"]["majority"] + 0.1
                       and pq["phase"]["ece"] < 0.15 and pq["duration_class"]["ece"] < 0.15))       # small hold-out: ECE is noisy; the 0.1 gate is judged on the full grids
        ra = _annotate(Path(td) / "r1" / "traces.jsonl", Path(td) / "jev.npz", Path(td) / "annotated.jsonl")
        checks.append((f"annotate: chat spans gain the System One model's typed phase labels ({ra['annotated']} chats, accuracy vs hidden {ra['phase_accuracy_vs_hidden']})",
                       ra["annotated"] > 100 and ra["phase_accuracy_vs_hidden"] > 0.5))
        pcfg = load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=needs", "generator.content_snr=0.9",
                                      f"policy.needs.model={Path(td) / 'needs.npz'}", f"policy.needs.jev.model={Path(td) / 'jev.npz'}", "policy.needs.jev.enabled=true"])
        pe = build_engine(pcfg)
        pe.run()
        checks.append(("needs controller loads a pre-trained predictor and a trained System One model and runs clean",
                       pe.deadlocks_detected == 0 and pe.policy.pred.n_learned > 0 and pe.policy.jev.calls > 0
                       and all(not q.holds for q in pe.programs.values() if q.status in ("done", "aborted"))))
    # ---- Jev v2: the vendor API shape, batching, the online judge, the veto path ------------------------------------
    from .jev import Question as _Q, RemoteSystemOne as _Remote, Judge as _Judge, JevRecord as _JR, catalogue as _cat, questions_at as _qat, confidence_of as _conf
    cat_ = _cat(["bash", "read"], ["plan", "final"])
    pay = {q.name: q.payload() for q in cat_.values()}
    checks.append(("Jev catalogue: every question is a choice / score / noul with instructions and criteria in the API's payload shape; "
                   f"speculative fan-out asks {len(_qat(cat_, 'chat_end'))} questions at chat_end and {len(_qat(cat_, 'tool_ready'))} at tool_ready",
                   all(v["type"] in ("choice", "score", "noul") and v["instructions"] and "criteria" in v for v in pay.values())
                   and pay["next_chat"]["type"] == "choice" and isinstance(pay["next_chat"]["criteria"], dict)
                   and pay["tail_risk"]["type"] == "score" and 2 <= len(pay["tail_risk"]["criteria"]) <= 10 and pay["fail_soon"]["type"] == "noul"
                   and len(_qat(cat_, "chat_end")) >= 6 and "phase" in _qat(cat_, "chat_end")))
    import os as _os
    _os.environ["JEV_API_KEY"] = "test-key"
    remote = _Remote(cat_)
    canned = {"model": "jev-1.13.0", "usage": {"input_tokens": 321, "output_tokens": 0},
              "answers": {"next_chat": {"choice": "tools", "probabilities": {"final": 0.1, "tools": 0.7, "no_tools": 0.15, "spawn": 0.05}, "confidence": 0.81},
                          "tail_risk": {"score": 1.3, "probabilities": [0.1, 0.55, 0.3, 0.05, 0.0], "confidence": 0.6},
                          "fail_soon": {"noul": 0.07}}}
    remote._post = lambda body: canned
    st_ = {"dp": "chat_end", "recipe": "coding", "child": False, "request_idx": 0, "chats": 2, "tools": 3, "history": ["user", "chat"], "last_tool": "bash",
           "plan": "", "content": [], "revealed": [], "revealed_content": [], "join_width": 0, "tokens_in": 100, "age_s": 1.0, "occupancy": {}}
    jr_ = remote.decide(st_, ("next_chat", "tail_risk", "fail_soon"), 5.0, 0.2)
    checks.append((f"RemoteSystemOne parses the vendor response into the same record the local model produces (choice tools p=0.7 conf 0.81, "
                   f"score {jr_.score('tail_risk'):.2f}, noul {jr_.noul('fail_soon')}, usage {jr_.usage_tokens}, cost {remote.report()['cost_usd']})",
                   jr_.answers["next_chat"] == ("tools", 0.7) and abs(jr_.confidence["next_chat"] - 0.81) < 1e-9 and abs(jr_.score("tail_risk") - 1.3) < 1e-9
                   and abs(jr_.noul("fail_soon") - 0.07) < 1e-9 and jr_.usage_tokens == 321 and jr_.t_ready == 5.2 and jr_.ok("next_chat") and not jr_.ok("tail_risk", 0.7)))
    del _os.environ["JEV_API_KEY"]
    jd = _Judge(cat_, window=200, min_n=50)
    rq3 = np.random.default_rng(4)
    for _ in range(150):                                                   # a well-calibrated answerer at p=0.8 and a broken one at p=0.95
        good = _JR("chat_ready", 0.0, 0.0, schema={"is_final_turn": ("no", "yes"), "fail_soon": ("no", "yes")},
                   dists={"is_final_turn": np.array([0.2, 0.8]), "fail_soon": np.array([0.05, 0.95])})
        good.answers = {"is_final_turn": ("yes", 0.8), "fail_soon": ("yes", 0.95)}
        jd.observe(good, "is_final_turn", "yes" if rq3.random() < 0.8 else "no")
        jd.observe(good, "fail_soon", "yes" if rq3.random() < 0.5 else "no")
    jrep = jd.report()
    checks.append((f"online Judge: ECE per question against realised labels — calibrated question kept (ECE {jrep['is_final_turn']['ece']}), "
                   f"drifted question distrusted (ECE {jrep['fail_soon']['ece']})",
                   jrep["is_final_turn"]["trusted"] and jrep["is_final_turn"]["ece"] < 0.1 and not jrep["fail_soon"]["trusted"] and jrep["fail_soon"]["ece"] > 0.3))
    bcfg = load_scenario(hosted, ["horizon_s=1800", "warmup_s=0", "arrivals.rate_per_min=2.0", "policy.type=needs", "policy.needs.jev.enabled=true",
                                  "policy.needs.jev.batch_window_s=0.05", "policy.needs.jev.model=data/models/jev.hosted.npz"])
    be = build_engine(bcfg)
    be.run()
    br = be.policy.report()
    checks.append((f"Jev batching: decision points inside a 50 ms window share one call ({br['jev']['calls']} calls for {br['jev']['records']} records, "
                   f"delivered {br['decisions']['jev_delivered']}); the online judge scores {len(br['jev_judge'])} questions (e.g. duration_class ECE "
                   f"{br['jev_judge'].get('duration_class', {}).get('ece', 'n/a')})",
                   br["jev"]["calls"] < br["jev"]["records"] and br["decisions"]["jev_delivered"] > 100 and len(br["jev_judge"]) >= 3
                   and be.deadlocks_detected == 0 and all(not q.holds for q in be.programs.values() if q.status in ("done", "aborted"))))
    checks.append(("confidence_of: a point mass is 1, a uniform distribution is 0", abs(_conf(np.array([1.0, 0.0, 0.0])) - 1.0) < 1e-9 and abs(_conf(np.array([0.25] * 4))) < 1e-9))
    # ---- Phase 7: budgets, MCP / GPU resources, the feedback-loop allocation controller -------------------------------
    from .resources import BudgetBucket as _BB
    bb = _BB(10.0, 36.0)
    bb.pay(0.0, 8.0)
    checks.append(("BudgetBucket: pays, refills at per_hour, refuses what it cannot pay, refunds negative amounts",
                   abs(bb.remaining_frac(0.0) - 0.2) < 1e-9 and abs(bb.remaining_frac(100.0) - 0.3) < 1e-9 and not bb.can_pay(100.0, 4.0) and bb.can_pay(100.0, 3.0)
                   and abs(bb.wait_for(100.0, 4.0) - 100.0) < 1e-6 and (bb.pay(100.0, -1.0) or abs(bb.remaining_frac(100.0) - 0.4) < 1e-9)))
    mres = {}
    for pol in ("uncoordinated", "reactive_gate", "needs", "clairvoyant"):
        bcfg2 = load_scenario(ROOT / "scenarios/mcp_budget.json", ["horizon_s=3600", "warmup_s=600", f"policy.type={pol}"])
        be2 = build_engine(bcfg2)
        be2.run()
        mres[pol] = (summary_of(be2, bcfg2), be2)
    su, sg, sn, so = (mres[k][0] for k in ("uncoordinated", "reactive_gate", "needs", "clairvoyant"))
    en = mres["needs"][1]
    span_usd = sum(float(sp_.attrs.get("usd", 0.0)) for sp_ in en.spans if sp_.op in ("chat", "execute_tool") and sp_.outcome == "ok")
    checks.append((f"budgets: tenant spend is accounted (budget spent {en.budgets['usd'].spent:.2f} = sessions' {sum(q.spend_usd for q in en.programs.values()):.2f}; ok spans {span_usd:.2f}); "
                   f"MCP / GPU resources carry calls and costs; uncoordinated agents exhaust the budget and fail (failure {su['failure_rate']:.2f}, {su['errors']['budget']} refusals, "
                   f"{su['spend']['usd_waste_frac']:.0%} of spend wasted)",
                   abs(en.budgets["usd"].spent - sum(q.spend_usd for q in en.programs.values())) < 1e-6 and span_usd <= en.budgets["usd"].spent + 1e-6
                   and su["failure_rate"] > 0.5 and su["errors"]["budget"] > 100 and su["spend"]["usd_waste_frac"] > 0.3
                   and any(sp_.resource == "mcp.github" and sp_.attrs.get("usd", 0) > 0 for sp_ in en.spans if sp_.op == "execute_tool")))
    checks.append((f"budget pacing: the gate's header pause keeps failures low (failure {sg['failure_rate']:.3f}, {sg['errors']['budget']} refusals); the Needs controller's "
                   f"forecast-aware pacing refuses nothing and fails less ({sn['failure_rate']:.3f}, {sn['errors']['budget']} refusals, usd/req {sn['spend']['usd_per_completed_request']:.3f} vs "
                   f"{sg['spend']['usd_per_completed_request']:.3f}); the clairvoyant gate never issues a refused payment ({so['errors']['budget']})",
                   sg["failure_rate"] < 0.2 and sn["errors"]["budget"] == 0 and sn["failure_rate"] <= sg["failure_rate"] and so["errors"]["budget"] == 0
                   and en.deadlocks_detected == 0 and all(not q.holds for q in en.programs.values() if q.status in ("done", "aborted"))))
    fb = en.policy.report()["feedback"]
    checks.append((f"feedback loop: {fb['ticks']} ticks of predicted-vs-realised occupancy per resource (mean |err| model.slots {fb['mean_abs_err'].get('model.slots')}), "
                   f"forecast scales corrected ({fb['scale'].get('model.slots')}), budget floor adapted ({fb['budget_floor'].get('usd')}), fairness weights on {fb['fair_weights']['n']} sessions",
                   fb["ticks"] >= 50 and fb["scale"].get("model.slots") is not None and fb["scale"]["model.slots"] != 1.0 and 0.02 <= fb["budget_floor"]["usd"] <= 0.5))
    pcfg = load_scenario(ROOT / "scenarios/mcp_budget.json", ["horizon_s=1200", "warmup_s=0", "policy.type=needs", "policy.ablate=pacing"])
    pe2 = build_engine(pcfg)
    pe2.run()
    from .policies import ReactiveGate as _RG
    probe2 = next(iter(pe2.programs.values()))
    same_rule = all(pe2.policy.budget_admit(probe2, sa, u, 0.0, pe2.now) == _RG.budget_admit(pe2.policy, probe2, sa, u, 0.0, pe2.now) for u in (0.0, 0.01, 1.0, 100.0))
    checks.append(("needs with pacing off applies the gate's spend-pause rule (same admission verdicts on the same inputs)", same_rule))
    from .jev import cost_class as _cc, answers_from_labels as _afl
    checks.append(("supervisory questions: cost_class bins a step's spend; budget_will_exceed / stuck_in_loop come from realised labels",
                   _cc(0.0) == "free" and _cc(0.004) == "cents" and _cc(0.05) == "dime" and _cc(0.5) == "dollar"
                   and _afl("chat_ready", {"USD": 0.05, "BUDGET_FAIL": 1.0}, 3.0, 600.0) == {"cost_class": "dime", "budget_will_exceed": "yes"}
                   and _afl("chat_end", {"REPEAT": 1.0}, 3.0, 600.0)["stuck_in_loop"] == "yes"))
    from .needs import ConformalLevel as _CL, quantile_at as _qa, knots_from_raw as _kfr
    cl = _CL(0.8)
    kn = _kfr(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    rq2 = np.random.default_rng(3)
    for _ in range(3000):
        cl.update(float(rq2.normal(1.0, 1.0)), _qa(kn, cl.level))     # a fixed, badly centred forecast: the level must climb until coverage ~ 0.8
    checks.append((f"adaptive conformal: a misspecified quantile head is corrected on the stream (level {cl.level:.3f}, coverage {cl.coverage():.3f} for target 0.8)",
                   0.74 <= cl.coverage() <= 0.86 and cl.level > 0.85 and _qa(kn, 0.05) < _qa(kn, 0.5) < _qa(kn, 0.95)))
    for name, passed in checks:
        print(("PASS " if passed else "FAIL ") + name)
    if not all(p for _, p in checks):
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(prog="agentsim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--scenario", required=True); r.add_argument("--out", required=True)
    r.add_argument("--set", action="append", default=[], help="dotted.path=json_value override"); r.set_defaults(fn=cmd_run)
    g = sub.add_parser("grid"); g.add_argument("--grid", required=True); g.add_argument("--out", required=True)
    g.add_argument("--no-traces", action="store_true", help="write summaries only"); g.set_defaults(fn=cmd_grid)
    f = sub.add_parser("fit"); f.add_argument("--traces", nargs="+", required=True); f.add_argument("--out", required=True)
    f.add_argument("--recipes-out", help="also mine observable-phase recipes into this file")
    f.add_argument("--seed-marginals", default="data/marginals.json"); f.add_argument("--seed-recipes", default="data/recipes.json"); f.set_defaults(fn=cmd_fit)
    ss = sub.add_parser("sample-scenarios", help="draw scenarios from declared knob ranges (A8)")
    ss.add_argument("--base", required=True); ss.add_argument("--ranges", required=True); ss.add_argument("--n", type=int, required=True)
    ss.add_argument("--seed", type=int, default=0); ss.add_argument("--out", required=True); ss.set_defaults(fn=cmd_sample_scenarios)
    sw = sub.add_parser("sweep", help="run sampled scenarios x named variants x seeds into one CSV")
    sw.add_argument("--scenarios", nargs="+", required=True); sw.add_argument("--variants", required=True)
    sw.add_argument("--seeds", default="1-5", help="e.g. 1-5 or 1,2,3"); sw.add_argument("--out", required=True)
    sw.add_argument("--axis", action="append", default=[], help="path=v1,v2,... (repeatable): extra grid axes, recorded as columns")
    sw.add_argument("--horizon", type=float, default=None); sw.add_argument("--any-scenario", action="store_true", help="scenarios need not be knob-sampled")
    sw.set_defaults(fn=cmd_sweep)
    tn = sub.add_parser("train-needs", help="pre-train the Needs Predictor on traces (scenario.json beside each)")
    tn.add_argument("--traces", nargs="+", required=True); tn.add_argument("--out", required=True); tn.add_argument("--epochs", type=int, default=3)
    tn.add_argument("--seed", type=int, default=0); tn.add_argument("--no-content", action="store_true")
    tn.add_argument("--scenario", help="one scenario for all traces (real traces have none beside them)"); tn.add_argument("--max-sessions", type=int, default=None)
    tn.set_defaults(fn=cmd_train_needs)
    tj = sub.add_parser("train-jev", help="train + calibrate the local System One model on traces; prints ECE per question")
    tj.add_argument("--traces", nargs="+", required=True); tj.add_argument("--out", required=True); tj.add_argument("--epochs", type=int, default=3)
    tj.add_argument("--seed", type=int, default=0); tj.add_argument("--scenario"); tj.add_argument("--max-sessions", type=int, default=None)
    tj.set_defaults(fn=cmd_train_jev)
    ej = sub.add_parser("evaluate-jev", help="accuracy / ECE per question of a System One model (local file or --remote vendor) on labelled traces")
    ej.add_argument("--traces", nargs="+", required=True); ej.add_argument("--model"); ej.add_argument("--remote", action="store_true")
    ej.add_argument("--max-n", type=int, default=None); ej.add_argument("--scenario"); ej.add_argument("--max-sessions", type=int, default=None)
    ej.set_defaults(fn=cmd_evaluate_jev)
    an = sub.add_parser("annotate", help="label a trace's chats with a System One model's typed answers (phase etc.)")
    an.add_argument("--trace", required=True); an.add_argument("--model", required=True); an.add_argument("--out", required=True); an.set_defaults(fn=cmd_annotate)
    s = sub.add_parser("selftest"); s.set_defaults(fn=cmd_selftest)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
