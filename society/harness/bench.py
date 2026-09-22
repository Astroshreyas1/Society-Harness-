"""With and without the harness: many DAG runs of the society, launched over a window and sharing one world, under each
admission policy in turn — paired by seed (the same arrivals, the same recorded futures per (run, node, attempt)).

  uv run python -m society.harness.bench --policies off,gate,needs,oracle --seeds 1,2,3,4,5 --runs 12 --arrival-s 20
  uv run python rung0/needs_tables.py data/society/bench/bench.csv --axis arrival_s --baseline off --oracle oracle

Columns follow the simulator's sweep CSVs (rung0/compare.py, rung0/needs_tables.py): `variant` is the policy, `seed`
pairs the cells, `arrival_s` is the load axis. failure_rate = escalated runs / runs (a run the society gave up on);
tct = a run's wall-clock from submission to its ship decision; throughput = completed runs per hour of the cell's
makespan; timeouts = calls the framework gave up queueing; budget_errs = payments the platform refused; provider_errs
= attempts burned by 429s / refusals; usd_waste_frac = spend on runs that never completed; jain_all = Jain's index of
the tenants' completed-run shares. Every number is in society seconds (world.py); wall-clock is that / scale.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from society.agents.registry import MODES, ROOT, build_agents, make_workers
from society.gates import RetryPolicy
from society.harness.gateway import POLICIES, Brain, Gateway, OracleBrain, SocietyJev, harness_catalogue
from society.harness.governed import GovernedRunner, govern
from society.harness.world import World, build_world
from society.memory import KVStore, LineageLog
from society.providers import ReplayTable
from society.runner import load_dag

DAGS = {"login": ROOT / "society/examples/login_dag.json", "otp": ROOT / "society/examples/otp_login_dag.json"}
OUT = ROOT / "data" / "society" / "bench"
COLUMNS = ["variant", "seed", "arrival_s", "dag", "mode", "runs", "completed", "escalated", "failure_rate", "tct_p50", "tct_p99",
           "throughput_rph", "makespan_s", "timeouts", "jain_all", "budget_errs", "provider_errs", "refused_429", "usd", "usd_per_req",
           "usd_waste_frac", "wait_p50", "wait_p99", "go_rate", "leases", "lease_unused_frac", "gang", "srpt_orders", "budget_waits", "wall_s"]


def jain(xs: list[float]) -> float:
    xs = [float(x) for x in xs]
    if not xs or sum(xs) <= 0:
        return 1.0
    return (sum(xs) ** 2) / (len(xs) * sum(x * x for x in xs))


def make_gateway(world: World, policy: str, table: ReplayTable, seed: int, mode: str, warm_split: str, jev: str, h: float) -> Gateway:
    providers = MODES[mode]
    if policy == "needs":
        warm = ReplayTable.records(split=warm_split) if warm_split != "none" else None
        brain = Brain(warm)
        s1 = None
        if jev == "remote":
            from agentsim.jev import RemoteSystemOne

            s1 = SocietyJev(RemoteSystemOne(harness_catalogue()), harness_catalogue())
        elif jev == "local":
            from agentsim.jev import LocalSystemOne

            s1 = SocietyJev(LocalSystemOne(harness_catalogue(), seed), harness_catalogue())
        return Gateway(world, "needs", brain=brain, h=h, jev=s1, providers=providers)
    if policy == "oracle":
        return Gateway(world, "oracle", brain=OracleBrain(table, seed, providers), h=h, providers=providers)
    return Gateway(world, policy, providers=providers)


async def run_cell(policy: str, seed: int, *, runs: int, arrival_s: float, dag_name: str, mode: str, world_spec: dict, split: str,
                   jev: str, h: float, out_dir: Path, keep: bool = False, on_event=None, registry: dict | None = None) -> dict:
    """One cell. `on_event(kind, data)` receives every runner and gateway event as it happens (the dashboard streams them);
    `registry`, when given, is filled with the live objects (world, gateway, results) so a watcher can take snapshots."""
    import time

    t_wall = time.monotonic()
    world = build_world(world_spec)
    table = ReplayTable(split=split)
    table.allow_peek = policy == "oracle"                                    # the honesty guard (ReplayTable.peek)
    warm_split = {"even": "odd", "odd": "even", "all": "all"}[split]
    gw = make_gateway(world, policy, table, seed, mode, warm_split, jev, h)
    workers = make_workers(mode, table, seed=seed, scale=world.clock.scale)
    tenant_of = lambda run_id: world.tenant_of(int(run_id[1:]))             # noqa: E731
    agents = govern(build_agents(workers), gw, tenant_of)
    ws = out_dir / f"s{seed}" / policy
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    kv, lineage = KVStore(), LineageLog(ws / "lineage.jsonl")
    observer = None
    if on_event is not None:
        gw.observers.append(lambda kind, now, info: on_event("gw:" + kind, {"t": round(now, 1), **info}))
        observer = lambda event, data: on_event("run:" + event, {"t": round(world.now(), 1), **data})   # noqa: E731
    runner = GovernedRunner(agents, kv, lineage, ws, gw, tenant_of, MODES[mode], policy=RetryPolicy(max_attempts=3, max_run_cost_usd=5.0),
                            observer=observer)
    dag = load_dag(DAGS[dag_name])
    rng = np.random.default_rng([seed, 7])
    arrivals = np.cumsum(rng.exponential(arrival_s, size=runs))
    arrivals -= arrivals[0]
    gw.start()
    results: dict[str, dict] = {}
    if registry is not None:
        registry.update({"world": world, "gateway": gw, "results": results, "arrivals": [round(float(a), 1) for a in arrivals], "agents": agents})

    async def one(i: int) -> None:
        rid = f"r{i:02d}"
        await world.clock.sleep(float(arrivals[i]))
        t0 = world.now()
        res = await runner.run(dag, run_id=rid)
        go = None
        if res.status == "completed":
            d = kv.get_current(rid, "ship", "decision")
            go = bool(d and d.get("go"))
        results[rid] = {"status": res.status, "t0": t0, "t1": world.now(), "tenant": tenant_of(rid), "go": go, "reason": res.reason,
                        "spent": gw.runs[rid].spent}

    try:
        await asyncio.gather(*(one(i) for i in range(runs)))
    finally:
        await gw.close()
    wall = time.monotonic() - t_wall
    done = [r for r in results.values() if r["status"] == "completed"]
    tct = np.array([r["t1"] - r["t0"] for r in done]) if done else np.array([np.nan])
    makespan = max(r["t1"] for r in results.values()) - min(r["t0"] for r in results.values())
    usd = sum(r["spent"] for r in results.values())
    waste = sum(r["spent"] for r in results.values() if r["status"] != "completed")
    waits = [e[2]["wait"] for e in gw.events if e[0] == "step_start" and e[2]["kind"] == "chat"]
    by_tenant = {}
    for r in results.values():
        by_tenant[r["tenant"]] = by_tenant.get(r["tenant"], 0) + (r["status"] == "completed")
    perr = sum(getattr(a.worker, "provider_errors", 0) for a in agents.values())
    rep = gw.report()
    led = rep["ledger"]
    row = {"variant": policy, "seed": seed, "arrival_s": arrival_s, "dag": dag_name, "mode": mode, "runs": runs, "completed": len(done),
           "escalated": sum(r["status"] == "escalated" for r in results.values()), "failure_rate": round(1 - len(done) / runs, 4),
           "tct_p50": round(float(np.nanpercentile(tct, 50)), 1), "tct_p99": round(float(np.nanpercentile(tct, 99)), 1),
           "throughput_rph": round(len(done) / max(1e-9, makespan) * 3600.0, 2), "makespan_s": round(makespan, 1),
           "timeouts": rep["decisions"]["timeouts"], "jain_all": round(jain(list(by_tenant.values())), 4),
           "budget_errs": sum(b["refused"] for b in rep["budgets"].values()), "provider_errs": perr, "refused_429": rep["decisions"]["refused_429"],
           "usd": round(usd, 4), "usd_per_req": round(usd / len(done), 4) if done else float("nan"),
           "usd_waste_frac": round(waste / usd, 4) if usd > 0 else 0.0,
           "wait_p50": round(float(np.percentile(waits, 50)), 1) if waits else 0.0, "wait_p99": round(float(np.percentile(waits, 99)), 1) if waits else 0.0,
           "go_rate": round(sum(1 for r in done if r["go"]) / len(done), 3) if done else float("nan"),
           "leases": led["issued"], "lease_unused_frac": round(led["unit_s_unused"] / led["unit_s_issued"], 3) if led["unit_s_issued"] > 0 else 0.0,
           "gang": rep["decisions"]["gang"], "srpt_orders": rep["decisions"]["srpt_orders"], "budget_waits": rep["decisions"]["budget_waits"],
           "wall_s": round(wall, 1)}
    (ws / "report.json").write_text(json.dumps({"row": row, "gateway": rep, "runs": results}, indent=1, default=str))
    if not keep:
        for p in ws.iterdir():                                               # the built apps are large and identical; keep the logs
            if p.is_dir():
                shutil.rmtree(p)
    return row


def _cell(job: dict) -> dict:
    return asyncio.run(run_cell(**job))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", default="off,gate,needs,oracle")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--arrival-s", default="20", help="mean inter-arrival in society seconds; comma-separated for a load axis")
    ap.add_argument("--dag", default="login", choices=sorted(DAGS))
    ap.add_argument("--mode", default="mixed", choices=sorted(MODES))
    ap.add_argument("--split", default="even", choices=("all", "even", "odd"), help="recorded runs the replay draws from; the predictor is warmed on the other half")
    ap.add_argument("--jev", default="off", choices=("off", "local", "remote"), help="System One as the needs policy's annotator")
    ap.add_argument("--h", type=float, default=600.0, help="forecast horizon (society seconds)")
    ap.add_argument("--world", default="{}", help='JSON overrides of world.DEFAULT_WORLD, e.g. {"budget": {"usd": 0}}')
    ap.add_argument("--out", default=str(OUT / "bench.csv"))
    ap.add_argument("--keep", action="store_true", help="keep the built apps of every run")
    ap.add_argument("--parallel", type=int, default=4, help="cells of one seed run concurrently in separate processes (1 = sequential)")
    args = ap.parse_args(argv)
    policies = [p for p in args.policies.split(",") if p]
    unknown = [p for p in policies if p not in POLICIES]
    if unknown:
        raise SystemExit(f"unknown policies {unknown}; known: {POLICIES}")
    seeds = [int(s) for s in args.seeds.split(",") if s]
    loads = [float(x) for x in args.arrival_s.split(",") if x]
    world_spec = json.loads(args.world)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for load in loads:
        for seed in seeds:
            # the policies of one seed run side by side in their own processes: the same machine load for every member of a pair
            jobs = [dict(policy=policy, seed=seed, runs=args.runs, arrival_s=load, dag_name=args.dag, mode=args.mode, world_spec=world_spec,
                         split=args.split, jev=args.jev, h=args.h, out_dir=out.parent / out.stem, keep=args.keep) for policy in policies]
            if args.parallel > 1 and len(jobs) > 1:
                with ProcessPoolExecutor(max_workers=min(args.parallel, len(jobs))) as pool:
                    batch = list(pool.map(_cell, jobs))
            else:
                batch = [_cell(j) for j in jobs]
            for row in batch:
                rows.append(row)
                print(f"arrival={load:g} seed={seed} {row['variant']:6s} completed {row['completed']}/{row['runs']} fail={row['failure_rate']:.3f} "
                      f"p50={row['tct_p50']:.0f}s thr={row['throughput_rph']:.0f}/h 429s={row['refused_429']} budget_errs={row['budget_errs']} "
                      f"usd={row['usd']:.2f} waste={row['usd_waste_frac']:.2f} leases={row['leases']} wall={row['wall_s']:.0f}s", flush=True)
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                w.writeheader()
                w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
