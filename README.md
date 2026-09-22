# Society Harness — runtime service-graph reservation for concurrent agentic workflows

A simulator of societies of concurrent agentic workflows contending for shared capacity, and a controller that predicts each workflow's coming tool, model and service needs and reserves capacity for them without over-allocation, starvation or workflow failure. Python 3.12 + numpy, nothing else.

Read [SOLUTION.md](SOLUTION.md) for the idea and what the evidence settled (ten minutes); [RUNG0_REPORT.md](RUNG0_REPORT.md) and [NEEDS_REPORT.md](NEEDS_REPORT.md) are the only places with numbers; [JEV_SURVEY.md](JEV_SURVEY.md) is the System One (Jev) survey; the design notes are [SERVICE_GRAPH_RESERVATION.md](SERVICE_GRAPH_RESERVATION.md) (v1), [DEEP_DIVE_AND_LADDER.md](DEEP_DIVE_AND_LADDER.md) (v2) and [PREDICTOR_DESIGN.md](PREDICTOR_DESIGN.md) (v3).

## How the harness works

**The society.** Each session is a program: `user message → LLM turn → tool calls → LLM turn → … → final turn → human thinks → next request`. The next step is sampled only when the previous one completes, so a controller's decisions change what follows. Three tiers of capacity: **model** (slots, KV memory with a prefix cache and eviction, a TPM bucket; priced per token), **tool** (sandboxes with memory, CPU, cold starts and idle parking; external APIs and MCP servers with concurrency, RPM, rate-limit headers, other tenants' traffic and a cost per call; a GPU pool), **service** (retrieval) — plus a **tenant budget** in dollars or tokens that refills hourly and refuses what it cannot pay. Failures the controller must prevent: 429 storms, client timeouts (600 s per step), hold-and-wait deadlocks, budget refusals; each costs the agent SDK retries and eventually an abort. Orchestrators can spawn 2–6 sub-agents and join them; a parent holding a sandbox while its children queue for one is a real cycle the engine detects.

**The controller** (`policy.type=needs`, [agentsim/reserver.py](agentsim/reserver.py)) sits where a real gateway would (LiteLLM, the Kubernetes inference extension, an MCP proxy) and sees only what a gateway sees. Every gateway-visible event — request start/end, step submitted / started / ended, spawn, join, abort, budget refusal — goes through one loop:

1. **Observe** ([features.py](agentsim/features.py)). A per-session view holds observable state only: recipe, counts, the last eight node tokens, the submitted step's size and content cue, the response's plan cue, the tool calls the response revealed, spawn width, the latest Jev answer with its age, the gateway's own occupancy and queue counters. At four decision points (chat submitted, tool submitted, response arrived, request closed) it emits a typed feature record. The same code replays OTel traces offline, so training and serving never diverge.
2. **Predict** ([needs.py](agentsim/needs.py)). One small multi-head model returns distributions, not points: the next two nodes, tool duration, output tokens, gap to the next chat, idle gap, P(idle ≥ cold start), P(abort within *h*), spawn width. Quantile heads give five monotone knots; categorical heads are temperature-scaled; every quantile the Reserver consumes is conformalised — its level moves until the realised miss rate on the live stream equals 1 − τ. A bad model yields wide honest intervals, never confident wrong ones.
3. **Learn.** When the outcome lands, the record that predicted it gets its label: one gradient step (0.1 ms), the calibrators refit, the conformal levels move. The model starts cold or from one pre-trained on traces (`train-needs`).
4. **Forecast.** Each session contributes a demand curve per resource over the next *h* seconds — current holds until their predicted release, revealed calls at their predicted times, sub-agents from spawn to the predicted join — plus an arrival term. **pressure(resource, window)** = forecast excess over capacity plus the standing queue, per unit of capacity: the price of holding a unit then.
5. **Decide by expected value.** Every reservation, park and prewarm is priced — latency-seconds this session saves versus hold × pressure others lose. Park an idle sandbox iff E[idle] × pressure > (1 − P(safe)) × cold start; prewarm iff P(arrival in the predicted window) × cold start > window × pressure; gang CPU for a revealed parallel group iff P(pay) > pressure × extra units, sized by content-conditioned duration quantiles, with P(pay) learned from whether past gangs ran in parallel; a spawn gang (k slots + k sandboxes, family-owned so the children draw on the parent's lease) iff pressure × k < 1, for the children's start window only; ordering: shortest-predicted-first while the model queue exceeds the slot count, virtual-time fair queuing otherwise; KV admission reserve = q_τ(output) while KV binds, retention = q_τ(gap); budget pacing: a new request is admitted only if, after paying, the in-flight requests' predicted spend-to-completion (net of the refill that arrives while they finish) still fits above a floor — finish what is started.
6. **Act through the ledger** ([resources.py](agentsim/resources.py)). Decisions become leases — *q* units of *r* from *t₀* until *t₀ + h* — in a ledger with invariants asserted every 500 events: Σ active leases ≤ capacity, nothing survives its expiry, only the unconsumed part is kept from others (backfill), acquisition in a fixed global order and all-or-nothing for gangs. Deadlock freedom is by construction, not tuning; prediction quality affects waste and latency, never safety.
7. **Correct** ([control.py](agentsim/control.py)). Every 30 s: realised time-averaged occupancy per resource versus what the forecast predicted → a bounded PI correction on the forecast scale; the budget floor rises multiplicatively on any refused payment and falls slowly on quiet ticks (AIMD); per-session fair-queuing weights follow served-time shares. Every error and correction is reported in the run summary.
8. **Judge with a System One model** ([jev.py](agentsim/jev.py)). At each boundary the controller asks every question that boundary can use, in one call: typed `choice` / `score` / `noul` questions about the next turn, tool duration class, output class, remaining work, cost class — and judgments on its own candidates (*will this reservation be used? is it safe to park? will this request exceed the budget? is the agent looping?*). The call goes through `ext.jev`, a rate-limited, latency-bearing resource; the answer lands asynchronously and enters the next records as a feature with its age, or modulates a decision when its confidence is ≥ 0.55. A `Judge` scores every answer against the realised outcome; a question whose live calibration error drifts past 0.1 stops being asked.

**Why it is honest.** Hidden values (sampled durations, tokens, think times, the provider's draw) exist in hook signatures only for the clairvoyant oracle; the selftest feeds the controller contradictory hidden values and asserts identical decisions, and greps its source for hidden-field reads. Every switch off reproduces the plain FIFO gate bit-for-bit. Common random numbers pair every policy on the same workload; gains are paired 10-seed differences with Holm correction, normalised by the oracle's headroom, and always taken after sweeping the baseline's own knobs — which is how three of the controller's own defaults were caught and changed (NEEDS_REPORT §4).

**Where it pays.** On a saturated model tier it switches to shortest-predicted-first by itself and matches `srpt`; on a binding budget its pacing removes refused payments and cuts p50 by 43 % at +10 % throughput; everywhere else it reproduces the three fixed rules of the earlier phases (park idle sandboxes immediately, fair-queue where the sandbox pool binds, shortest-predicted-first where the model tier binds) — the machinery adds nothing there, and the reports say so.

## Jev: what is TypeSafe's and what is ours

**Jev is TypeSafe AI's model**, not ours: the first "System One" model — typed, calibrated decisions from program state, all questions of a request answered in one parallel pass, 70–500 ms, `POST /v1/systemone` with `choice` / `score` / `noul` questions ([JEV_SURVEY.md](JEV_SURVEY.md), with sources). What this repository contains is the **integration** and a **stand-in**:

- `RemoteSystemOne` is a client for the real TypeSafe API (request/response shape, bearer auth, `JEV_API_BASE` for direct or LiteLLM pass-through use, retries honouring `retry-after`, usage and cost accounting). It has been tested only against a canned response: **no call to TypeSafe was ever made from this project**, because no API key was available. `agentsim evaluate-jev --remote` is the first thing to run once one exists.
- `LocalSystemOne` is ours: one sparse multinomial per question over hashed n-grams of the state, log-loss trained and temperature-calibrated on held-out sessions. It reproduces the *contract* (typed answers, calibrated probabilities, everything answered at once) so the integration could be built and measured, not the model's capability: it reads hashed tokens, the real Jev reads text with frontier understanding. **Every Jev number in the reports is the stand-in's.**
- The question catalogue, the `ext.jev` channel (rate limits, latency, batching), the live calibration judge and the way answers enter the controller are ours and apply to either model unchanged.

## Code



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
