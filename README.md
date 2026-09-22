# Control — runtime service-graph reservation for concurrent agentic workflows

New here? Read [SOLUTION.md](SOLUTION.md) (the idea, the evidence, what is shipped — ten minutes). Then [HANDOFF.md](HANDOFF.md). The Needs Predictor, the Reserver that consumes it and the System One (Jev) integration are designed in [PREDICTOR_DESIGN.md](PREDICTOR_DESIGN.md) and **built** (Phase 6); their evidence is [NEEDS_REPORT.md](NEEDS_REPORT.md). Notes: [SERVICE_GRAPH_RESERVATION.md](SERVICE_GRAPH_RESERVATION.md) (v1: crime, H1–H6, first controller) ·
[DEEP_DIVE_AND_LADDER.md](DEEP_DIVE_AND_LADDER.md) (v2: deep survey, Jev adaptation, synthetic data, limits, ladder) ·
[RUNG0_REPORT.md](RUNG0_REPORT.md) (what the simulator and rung 0 found, after the inflation audit) ·
[ISSUES.md](ISSUES.md) (tracker: what is wrong, missing or unproven).

Code (Python 3.12, numpy only):

```
agentsim/            program-centric discrete-event simulator (AgentServeSim-style), three resource tiers
  schema.py          OTel-GenAI-shaped Span + JSONL I/O — the one trace format for real and synthetic data
  marginals.py       quantity distributions (seeded from published stats; `fit` replaces them from traces)
  workload.py        recipes (phase Markov chains), Program Control Block, lazy closed-loop steps, arrivals
  resources.py       capacities/holders/waiters, KV physics (TTL cache, admission, SIC eviction), token bucket; per-call cost / tokens on tool-tier
                     resources (ext.*, mcp.*, gpu.*), tenant BudgetBuckets (USD / tokens, hourly refill), Lease + Ledger (family-owned leases)
  policies.py        Uncoordinated (SDK-default retries), ReactiveGate (HiveMind primitives), ClairvoyantGate (oracle bound),
                     LeaseController (rung 3: predicted idle → park/prewarm/KV TTL, virtual-time fair queuing, gang + budget leases; knobs tau, h)
  predict.py         online predictors a gateway can run on its own observation stream (streaming quantiles, online Markov)
  features.py        Phase 6 Observer: typed feature records from the observation stream; one code path online and for replayed traces
  needs.py           the Needs Predictor: multi-head model (structure, quantities, timing, idle, risk, spawn width), temperature + adaptive conformal calibration
  jev.py             System One model: typed question schema, local calibrated implementation, vendor seam (JEV_API_URL), `ext.jev` channel
  reserver.py        Reserver v2 (`policy.type=needs`): demand forecast, expected-value-gated leases (gang, budget, spawn), conditional ordering, predicted KV,
                     forecast-aware budget pacing, Jev-judged decisions
  control.py         Phase 7 feedback loop: predicted-vs-realised occupancy error → forecast scale (PI), budget floor (AIMD), fairness weights
  train.py           offline: `train-needs`, `train-jev`, `evaluate-jev [--remote]`, `annotate` on OTel-shaped traces (synthetic or real)
  fit.py             rung 0 proper: marginals (EM lognormal mixtures) and observable-phase recipes mined from OTel-shaped traces
  engine.py          events, causal successor release, hold-and-wait cycle detection, lease ledger (I1–I3), spans
  metrics.py         outcome metrics + fidelity self-check, computed from spans only
  run.py             CLI: run / grid / fit / sample-scenarios / sweep / train-needs / train-jev / evaluate-jev / annotate / selftest (81 checks)
rung0/               the evidence scripts E1–E5 and compare.py (paired statistics); they read JSONL, so they run unchanged on real traces
scenarios/           api_coding, hosted_mixed, hosted_stress (B18), hosted_tracelab (fitted from real traces), multiagent (B10: orchestrators + sub-agents),
                     mcp_budget (Phase 7: metered API model, MCP servers + GPU pool with per-call costs, a tenant budget that binds);
                     19 grids (E1, hosted, pinned, predictability, snr, stress, round trip, rung-3 scoreboards, ablations, ship, tracelab);
                     knob_ranges.json + sampled/ (40, A8) + variants_ship.json / variants_needs*.json (sweep variants)
rung0/               … plus headroom.py (sweep analysis) and tracelab_to_spans.py (real-trace adapter)
data/real/           real traces (git-ignored): tracelab/ (TraceLab gz + span conversions)
data/recipes.json    structure seeds        data/marginals.json  quantity seeds (provenance inside)     data/fitted/  fitted from synthetic traces (C2 round trip)
data/recipes_multiagent.json  the orchestrator recipe (spawn)     data/recipes_mcp.json + marginals_mcp.json  tools on MCP servers / a GPU pool
data/models/         pre-trained predictors and System One models (npz) + training reports (hosted, multiagent, mcp, tracelab)
data/synthetic/      generated runs (regenerate; ~400 MB with traces): e1, hosted, hosted_pinned, pred, e5, hosted_snr, stress, roundtrip, lease_*, stress_lease
```

Generator knobs live in each scenario's `generator` block: `recipe_temperature` (structure entropy, A7), `think_snr` + `think_state_seed` (how much idle time an observable state explains, B7), `tool_tail_scale`, `p_parallel_scale` (A8), `content_snr` (how much a step's content cue reveals: tool duration class, prompt output/tool-count class, the response's next-phase plan; spans are identical at any value). Policy knobs `tau` / `h` are the lease and needs controllers' quantile and horizon; `ablate` switches features off for E6 ablations (`"vtfq+prewarm"` for the lease controller; for `needs`: vtfq park prewarm kv gang budget spawn srpt aging content jev conformal ev forecast learn). `policy.needs` holds the pre-trained model path, online-learning switches and the `ext.jev` channel (enabled, model, rpm, concurrency, latency).

## Run

Dependencies: Python ≥ 3.12 and numpy (`pyproject.toml`). With [uv](https://docs.astral.sh/uv/) installed, prefix every command below with `uv run` (it creates `.venv/` on first use); a plain `python` with numpy works too.

```bash
uv run python -m agentsim selftest
```

```bash
python -m agentsim run --scenario scenarios/api_coding.json --out data/synthetic/demo --set policy.type=reactive_gate
```

```bash
python -m agentsim grid --grid scenarios/grid_e1.json --out data/synthetic/e1
```

```bash
python rung0/e1_failure_vs_concurrency.py data/synthetic/e1/grid_summary.csv --plot data/synthetic/e1/e1.png
```

```bash
python rung0/e2_predictability.py data/synthetic/hosted/*/traces.jsonl --with-phase
```

```bash
python rung0/e3_tails.py data/synthetic/hosted/*/traces.jsonl
```

```bash
python rung0/e4_cycles.py "data/synthetic/hosted/type=uncoordinated_rate_per_min=1.0_hold_model_during_tool=True_seed=1/traces.jsonl"
```

Paired comparison over seeds (A6): mean Δ with bootstrap CI, exact Wilcoxon and paired-t p, Cohen's dz, per-seed wins, Holm across metric families, and the oracle-normalised score `(policy − gate)/(oracle − gate)`. With 5 seeds the exact Wilcoxon cannot reach 0.05 (floor 2/32), so pass `--test t` or run ≥ 6 seeds:

```bash
python rung0/compare.py data/synthetic/hosted/grid_summary.csv --axis rate_per_min --baseline reactive_gate --oracle clairvoyant --test t
```

Idle-time predictability from observable state (E5, B7) and the predictability sweep (A7):

```bash
python rung0/e5_idle_predictability.py data/synthetic/e5/type=uncoordinated_think_snr=0.9_seed=*/traces.jsonl
```

```bash
python -m agentsim grid --grid scenarios/grid_predictability.json --out data/synthetic/pred
```

Domain randomisation (A8): draw scenarios from declared knob ranges, then run any of them:

```bash
python -m agentsim sample-scenarios --base scenarios/hosted_mixed.json --ranges scenarios/knob_ranges.json --n 20 --seed 1 --out scenarios/sampled
```

Rung 3 scoreboards (10 seeds; `--no-traces`), then the oracle-normalised score per cell:

```bash
python -m agentsim grid --grid scenarios/grid_lease_snr.json --out data/synthetic/lease_snr --no-traces
```

```bash
python rung0/compare.py data/synthetic/lease_snr/grid_summary.csv --axis think_snr --filter rate_per_min=2.0 --filter tau=0.8 --baseline reactive_gate --oracle clairvoyant --policies lease
```

Randomised sweep over sampled societies and the "where does prediction pay" analysis:

```bash
python -m agentsim sweep --scenarios scenarios/sampled/*.json --variants scenarios/variants_ship.json --seeds 1-5 --out data/synthetic/sweep/sweep.csv
```

```bash
python rung0/headroom.py data/synthetic/sweep/sweep.csv --policy gate_srpt
```

Phase 6 — the Needs controller, its predictor and the System One model (NEEDS_REPORT.md). Pre-train on traces (a `scenario.json` beside each, or `--scenario`), train + calibrate the local System One model (prints accuracy / ECE per question), annotate a trace with its phase labels, and run the controller:

```bash
python -m agentsim train-needs --traces data/synthetic/train/hosted_s*/traces.jsonl --out data/models/needs.hosted.npz --epochs 3
```

```bash
python -m agentsim train-jev --traces data/synthetic/train/hosted_s*/traces.jsonl --out data/models/jev.hosted.npz --epochs 3
```

```bash
python -m agentsim annotate --trace data/synthetic/train/hosted_s4/traces.jsonl --model data/models/jev.hosted.npz --out data/synthetic/train/hosted_s4/traces.jev.jsonl
```

```bash
python rung0/e2_predictability.py data/synthetic/train/hosted_s4/traces.jev.jsonl --phase-attr jev_phase
```

```bash
python -m agentsim run --scenario scenarios/multiagent.json --out data/synthetic/demo_needs --set policy.type=needs --set policy.needs.model=data/models/needs.multiagent.npz --set policy.needs.jev.enabled=true --set policy.needs.jev.model=data/models/jev.multiagent.npz
```

Named-variant sweeps over any scenario with extra axes, and the paired statistics on them:

```bash
python -m agentsim sweep --scenarios scenarios/hosted_mixed.json --variants scenarios/variants_needs.json --seeds 1-10 --any-scenario --axis generator.content_snr=0.0,0.9 --axis arrivals.rate_per_min=2.0 --out data/synthetic/needs/hosted_content.csv
```

```bash
python rung0/compare.py data/synthetic/needs/hosted_content.csv --axis content_snr --policy-col variant --baseline gate_vtfq0 --oracle oracle0 --policies needs needs_pre needs_jev
```

Phase 7 — budgets and the feedback loop. The budget society (`scenarios/mcp_budget.json`) under the three policies, and the loop's switches:

```bash
python -m agentsim sweep --scenarios scenarios/mcp_budget.json --variants scenarios/variants_needs_mcp.json --seeds 1-10 --any-scenario --axis generator.content_snr=0.6 --horizon 7200 --out data/synthetic/needs/mcp_budget.csv
```

Scoring a System One model — the local file, or the vendor model once `JEV_API_KEY` (and optionally `JEV_API_BASE`, e.g. a LiteLLM proxy's `/typesafe`) is set — on labelled traces (accuracy / ECE per question):

```bash
python -m agentsim evaluate-jev --traces data/synthetic/train/mcp_s1/traces.jsonl --model data/models/jev.mcp.npz
```

```bash
python -m agentsim evaluate-jev --traces data/synthetic/train/mcp_s1/traces.jsonl --remote --max-n 300
```

The predictor on real traces (content = TraceLab's sanitised command skeletons and input sizes; per-head hold-out report):

```bash
python -m agentsim train-needs --traces data/real/tracelab/spans_claude.jsonl --scenario scenarios/hosted_tracelab.json --max-sessions 1500 --out data/models/needs.tracelab.npz --epochs 3
```

Real traces (rung 0 proper): TraceLab's public Claude Code trace (CC BY 4.0, 101 MB) → spans → E2/E3/E5 → fitted seeds → a real-structure society:

```bash
curl -L --fail -o data/real/tracelab/syfi_coding_trace.jsonl.gz https://github.com/uw-syfi/TraceLab/releases/latest/download/syfi_coding_trace.jsonl.gz
```

```bash
python rung0/tracelab_to_spans.py data/real/tracelab/syfi_coding_trace.jsonl.gz data/real/tracelab/spans_claude.jsonl --provider claude
```

```bash
python -m agentsim fit --traces data/real/tracelab/spans_claude.jsonl --out data/fitted/marginals.tracelab.json --recipes-out data/fitted/recipes.tracelab.json
```

When OTel traces from your own society exist, the same three steps apply (write an adapter like `tracelab_to_spans.py` for your exporter's shape):

```bash
python -m agentsim fit --traces real/*.jsonl --out data/marginals.json --recipes-out data/recipes.json
```

## What is modelled (and what is not)

- Sessions → requests → LLM steps → tool calls, sampled lazily from a phase recipe; the next step is released only when its predecessor completes (`r_{k+1} = c_k + g_k`).
- Model tier: API concurrency / TPM bucket (API-like: 429) or server slots + KV admission (self-hosted: queue); prefix cache with TTL; SIC-style least-progressed eviction when generation outgrows the admitted reserve.
- Tool tier: session-lifetime sandbox (memory + CPU slot, cold start), external tools with a provider-side 429 curve and timeouts. Service tier: retrieval with an overload curve.
- Reaction to errors in two layers: SDK retries (jittered exponential backoff, 2 by default; `jitter=false` reproduces the herd), then the agent replans or aborts — a PRIOR to be fitted.
- Client step timeout (600 s) on every step; sandbox idle timeout (300 s) with cold-start resume; provider RPM buckets with headers and background load.
- Parallel tool groups: a chat's tools may run together (gang need); extra CPU units are opportunistic, otherwise the group runs sequentially.
- Hold-and-wait: a framework knob pins the model slot across tool calls (server-side tools); the gate forces release.
- Common random numbers: each program's steps come from an RNG keyed by its exogenous identity, so policies see the same workload.
- Shipped fixed rules (RUNG0_REPORT §12–§13): `framework.sandbox_idle_timeout_s=0`; `policy.queue=vtfq` where the sandbox pool binds, `srpt` where the model tier binds (shortest *predicted* chat first from observable inputs).
- Rung 3 (`policy.type=lease`): leases on sandbox CPU (gang needs of declared parallel groups) and on provider call budgets (`<ext>@rpm`), a ledger with invariants I1 (Σ active leases ≤ capacity) and I3 (nothing past expiry), backfill by construction (only the unconsumed part is kept from others), expiry events; predictions only from the observation stream (`on_event`), enforced by an honesty selftest.
- Multi-agent (B10, Phase 6): a recipe's `spawn` block makes a chat launch 2–6 sub-agents (own recipe, sandbox, model slot, one request each); the orchestrator holds no step while it waits for the join and continues after it; a parent holding a sandbox while its children queue for one is a hold-and-wait cycle the engine detects. Child spans carry the parent's trace id (a request's metrics include its sub-agents' work).
- Content (Phase 6): every step carries an observable cue a gateway proxying bodies would see (tool skeleton class, prompt class, the response's plan), true with probability `content_snr`; the Needs Predictor learns from it; the System One model reads it.
- Phase 6 (`policy.type=needs`): the Needs Predictor (features.py, needs.py) + Reserver v2 (reserver.py) + the System One channel (jev.py; JEV_SURVEY.md) — see NEEDS_REPORT.md for what it buys and where.
- Phase 7: tool-tier resources carry per-call cost and tokens (external APIs, `mcp.<server>` tool servers, `gpu.<pool>` accelerators); the model tier has prices per Mtok; tenant budgets (USD / tokens) refill hourly and refuse what they cannot pay (outcome `budget`); the Needs controller paces admission to the forecast (finish what is started) and closes a feedback loop (control.py) on its own prediction error per resource, on budget refusals and on fairness; Jev's supervisory questions (cost class, budget-will-exceed, stuck-in-loop) guide the pacer and the leases.
- Not modelled: multiple model instances/routing, autoscaling, network, tool-result caching, semantic content beyond the cues above (branching is a Markov chain — see E2 in the report; `recipe_temperature` sets its entropy).

Data volume: `data/synthetic/` is ~500 MB with traces and `data/real/` ~1 GB; the repository keeps only the per-grid summary CSVs (`grid_summary.csv`, `data/synthetic/needs/*.csv`, `sweep.csv`) that the reports quote — regenerate the rest with the commands above. Trained models (`data/models/*.npz`, 48 MB) are not in the repository either; their hold-out reports are, and `train-needs` / `train-jev` rebuild them in minutes from the training traces (`data/synthetic/train/`, generated by `agentsim run` as in NEEDS_REPORT §7).
