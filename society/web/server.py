"""The dashboard: watch the society run under the harness — and without it — in the browser.

  uv run python -m society.web.server --port 8020        # then open http://localhost:8020

Panels: the with/without results of the recorded bench (data/society/bench/bench.csv); a live launch of one or more
policies side by side on the same seed (the same arrivals, the same recorded futures) with every DAG run as a card, the
gateway's resources, queues, leases, tenant budgets, forecast and decisions updating as they happen, and a narrated
event feed. Everything is served from this process: FastAPI + one static page, server-sent events for the stream.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from society.harness.bench import DAGS, OUT, run_cell
from society.harness.gateway import POLICIES
from society.harness.world import DEFAULT_WORLD

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Society Harness dashboard")

# ---- live state --------------------------------------------------------------------------------------------------
class Live:
    def __init__(self) -> None:
        self.cells: dict[str, dict] = {}                # policy -> {registry, task, row, started, status}
        self.events: list[dict] = []
        self.clients: list[asyncio.Queue] = []
        self.seq = 0
        self.launch: dict | None = None

    def push(self, kind: str, data: dict) -> None:
        self.seq += 1
        ev = {"seq": self.seq, "kind": kind, **data}
        self.events.append(ev)
        if len(self.events) > 5000:
            del self.events[:1000]
        for q in list(self.clients):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass

    def reset(self) -> None:
        self.cells, self.events, self.seq, self.launch = {}, [], 0, None


live = Live()


class LaunchRequest(BaseModel):
    policies: list[str] = Field(default_factory=lambda: ["off", "needs"])
    seed: int = 1
    runs: int = 8
    arrival_s: float = 20.0
    dag: str = "login"
    mode: str = "mixed"
    scale: float = 20.0
    world: dict[str, Any] = Field(default_factory=dict)


def snapshot(policy: str) -> dict | None:
    cell = live.cells.get(policy)
    if not cell or "gateway" not in cell["registry"]:
        return None
    reg = cell["registry"]
    gw, world = reg["gateway"], reg["world"]
    now = world.now()
    res = {}
    for name, r in gw.res.items():
        active = [{"run": sid, "amt": l.amt, "start": round(l.start - now, 1), "end": round(l.expiry - now, 1)}
                  for sid, l in gw.ledger.by_res.get(name, {}).items()]
        demand = gw.forecast.demand(name, now)[:24].tolist() if gw.forecast is not None else []
        res[name] = {"used": r.used, "capacity": r.capacity, "holders": sorted(r.holders), "waiters": [w.grant.sid for w in gw.waiters[name]],
                     "rpm_frac": round(r.remaining_frac(now), 3), "leases": active, "demand": [round(float(x), 2) for x in demand]}
    tpm = {n: round(b.level, 0) for n, b in world.tpm.items()}
    budgets = {n: {"level": round(b.remaining_frac(now) * b.capacity, 3), "capacity": b.capacity, "spent": round(b.spent, 3), "refused": b.refused,
                   "floor": round(gw.loop.floor.get(n, 0.1), 3)} for n, b in world.budgets.items()}
    runs = {}
    for rid, st in gw.runs.items():
        runs[rid] = {"tenant": st.tenant, "status": st.status, "done": sorted(st.done), "inflight": sorted(st.inflight), "judging": sorted(st.judging),
                     "attempts": st.attempts, "spent": round(st.spent, 3), "age": round(now - st.started, 1)}
    rep = gw.report()
    return {"policy": policy, "t": round(now, 1), "resources": res, "tpm": tpm, "budgets": budgets, "runs": runs, "decisions": rep["decisions"],
            "ledger": rep["ledger"], "loop": rep.get("loop"), "results": reg["results"], "arrivals": reg.get("arrivals", []),
            "status": cell["status"], "row": cell.get("row")}


async def _snapshots() -> None:
    while any(c["status"] == "running" for c in live.cells.values()):
        for policy in list(live.cells):
            snap = snapshot(policy)
            if snap is not None:
                live.push("snapshot", snap)
        await asyncio.sleep(0.5)


async def _cell(policy: str, req: LaunchRequest) -> None:
    cell = live.cells[policy]
    try:
        row = await run_cell(policy, req.seed, runs=req.runs, arrival_s=req.arrival_s, dag_name=req.dag, mode=req.mode,
                             world_spec={**req.world, "scale": req.scale}, split="even", jev="off", h=600.0, out_dir=OUT / "live",
                             on_event=lambda kind, data: live.push(kind, {"policy": policy, **data}), registry=cell["registry"])
        cell["row"], cell["status"] = row, "done"
        live.push("cell_done", {"policy": policy, "row": row})
    except Exception as e:  # noqa: BLE001
        cell["status"] = "error"
        live.push("cell_error", {"policy": policy, "error": f"{type(e).__name__}: {e}"})
        raise


@app.post("/api/launch")
async def launch(req: LaunchRequest) -> dict:
    if any(c["status"] == "running" for c in live.cells.values()):
        raise HTTPException(409, "a launch is still running; stop it first")
    bad = [p for p in req.policies if p not in POLICIES]
    if bad or not req.policies:
        raise HTTPException(400, f"policies must be among {POLICIES}")
    if req.dag not in DAGS:
        raise HTTPException(400, f"dag must be one of {sorted(DAGS)}")
    if not 1 <= req.runs <= 40 or req.arrival_s <= 0 or req.scale <= 0:
        raise HTTPException(400, "runs in 1..40, arrival_s > 0, scale > 0")
    live.reset()
    live.launch = req.model_dump() | {"started": time.time()}
    for policy in req.policies:
        live.cells[policy] = {"registry": {}, "status": "running", "row": None, "task": None}
    live.push("launch", live.launch)
    for policy in req.policies:
        live.cells[policy]["task"] = asyncio.create_task(_cell(policy, req))
    asyncio.create_task(_snapshots())
    return {"ok": True, "policies": req.policies}


@app.post("/api/stop")
async def stop() -> dict:
    n = 0
    for cell in live.cells.values():
        t = cell.get("task")
        if t is not None and not t.done():
            t.cancel()
            n += 1
        if cell["status"] == "running":
            cell["status"] = "stopped"
    live.push("stopped", {"cancelled": n})
    return {"ok": True, "cancelled": n}


@app.get("/api/state")
async def state() -> dict:
    return {"launch": live.launch, "cells": {p: snapshot(p) for p in live.cells}, "seq": live.seq}


@app.get("/api/stream")
async def stream(since: int = 0):
    q: asyncio.Queue = asyncio.Queue(maxsize=10000)
    live.clients.append(q)

    async def gen():
        try:
            for ev in live.events:
                if ev["seq"] > since:
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if q in live.clients:
                live.clients.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---- recorded results ----------------------------------------------------------------------------------------------
NUM = ("failure_rate", "tct_p50", "tct_p99", "throughput_rph", "makespan_s", "timeouts", "jain_all", "budget_errs", "provider_errs", "refused_429",
       "usd", "usd_per_req", "usd_waste_frac", "wait_p50", "wait_p99", "go_rate", "leases", "lease_unused_frac", "gang", "srpt_orders", "budget_waits",
       "wall_s", "completed", "escalated", "runs", "seed", "arrival_s")


@app.get("/api/results")
async def results(path: str = "data/society/bench/bench.csv") -> dict:
    p = ROOT / path
    if not p.exists():
        return {"path": path, "rows": [], "summary": {}, "note": "no bench CSV yet — run society.harness.bench or launch cells from the page"}
    rows = list(csv.DictReader(p.open()))
    for r in rows:
        for k in NUM:
            if k in r:
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = None
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    summary = {}
    for v, rs in by.items():
        summary[v] = {"n": len(rs), "runs": sum(r["runs"] for r in rs), "completed": sum(r["completed"] for r in rs),
                      "escalated": sum(r["escalated"] for r in rs), "refused_429": sum(r["refused_429"] for r in rs),
                      "budget_errs": sum(r["budget_errs"] for r in rs), "provider_errs": sum(r["provider_errs"] for r in rs),
                      "usd": round(sum(r["usd"] for r in rs), 2), "waste_usd": round(sum(r["usd"] * (r["usd_waste_frac"] or 0) for r in rs), 2),
                      "leases": sum(r["leases"] for r in rs), "timeouts": sum(r["timeouts"] for r in rs),
                      "mean": {k: round(float(np.nanmean([r[k] for r in rs if r[k] is not None])), 3) for k in
                               ("failure_rate", "tct_p50", "tct_p99", "throughput_rph", "wait_p50", "wait_p99", "jain_all", "usd_per_req", "usd_waste_frac", "go_rate")}}
    # paired wins against off, per seed
    wins = {}
    if "off" in by:
        base = {(r["seed"], r["arrival_s"]): r for r in by["off"]}
        for v, rs in by.items():
            if v == "off":
                continue
            w = {"failure_rate": 0, "tct_p50": 0, "throughput_rph": 0, "usd_waste_frac": 0, "n": 0}
            for r in rs:
                b = base.get((r["seed"], r["arrival_s"]))
                if b is None:
                    continue
                w["n"] += 1
                w["failure_rate"] += r["failure_rate"] < b["failure_rate"]
                w["tct_p50"] += (r["tct_p50"] or 1e9) < (b["tct_p50"] or 1e9)
                w["throughput_rph"] += r["throughput_rph"] > b["throughput_rph"]
                w["usd_waste_frac"] += r["usd_waste_frac"] < b["usd_waste_frac"]
            wins[v] = w
    return {"path": path, "rows": rows, "summary": summary, "wins": wins}


@app.get("/api/config")
async def config() -> dict:
    return {"policies": list(POLICIES), "dags": sorted(DAGS), "world": DEFAULT_WORLD}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def main(port: int = 8020) -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8020)
    args = ap.parse_args()
    print(f"Society Harness dashboard → http://localhost:{args.port}")
    main(args.port)
