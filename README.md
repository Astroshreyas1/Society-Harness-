# Society Harness — runtime service-graph reservation for concurrent agentic workflows

A simulator of societies of concurrent agentic workflows contending for shared capacity, and a controller that predicts each workflow's coming tool, model and service needs and reserves capacity for them without over-allocation, starvation or workflow failure. Python 3.12 + numpy for the simulator; `society/` (the harness on the Society of LLMs agents) adds pydantic, jsonschema, fastapi and pytest via `uv sync --group society`.

Design notes: [SERVICE_GRAPH_RESERVATION.md](SERVICE_GRAPH_RESERVATION.md) (the problem, hypotheses and literature) and [PREDICTOR_DESIGN.md](PREDICTOR_DESIGN.md) (the predictor and the Reserver that consumes it). The evaluation results referred to below live in the summary CSVs under `data/synthetic/` and the model reports under `data/models/`; `rung0/needs_tables.py` and `rung0/compare.py` turn them into tables and paired statistics.

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

**Why it is honest.** Hidden values (sampled durations, tokens, think times, the provider's draw) exist in hook signatures only for the clairvoyant oracle; the selftest feeds the controller contradictory hidden values and asserts identical decisions, and greps its source for hidden-field reads. Every switch off reproduces the plain FIFO gate bit-for-bit. Common random numbers pair every policy on the same workload; gains are paired 10-seed differences with Holm correction, normalised by the oracle's headroom, and always taken after sweeping the baseline's own knobs — which is how three of the controller's own defaults were caught and changed (the starvation guard, the spawn gang's lifetime, predicted KV reserves).

**Where it pays.** On a saturated model tier it switches to shortest-predicted-first by itself and matches `srpt`; on a binding budget its pacing removes refused payments and cuts p50 by 43 % at +10 % throughput; everywhere else it reproduces the three fixed rules of the earlier phases (park idle sandboxes immediately, fair-queue where the sandbox pool binds, shortest-predicted-first where the model tier binds) — the machinery adds nothing there, and the reports say so.

## Jev: what is TypeSafe's and what is ours

**Jev is TypeSafe AI's model**, not ours: the first "System One" model — typed, calibrated decisions from program state, all questions of a request answered in one parallel pass, 70–500 ms, `POST /v1/systemone` with `choice` / `score` / `noul` questions (TypeSafe's documentation: docs.typesafe.ai). What this repository contains is the **integration** and a **stand-in**:

- `RemoteSystemOne` is a client for the real TypeSafe API (request/response shape, bearer auth, `JEV_API_BASE` for direct or LiteLLM pass-through use, retries honouring `retry-after`, usage and cost accounting). It has been tested only against a canned response: **no call to TypeSafe was ever made from this project**, because no API key was available. `agentsim evaluate-jev --remote` is the first thing to run once one exists.
- `LocalSystemOne` is ours: one sparse multinomial per question over hashed n-grams of the state, log-loss trained and temperature-calibrated on held-out sessions. It reproduces the *contract* (typed answers, calibrated probabilities, everything answered at once) so the integration could be built and measured, not the model's capability: it reads hashed tokens, the real Jev reads text with frontier understanding. **Every Jev number ever measured here is the stand-in's.**
- The question catalogue, the `ext.jev` channel (rate limits, latency, batching), the live calibration judge and the way answers enter the controller are ours and apply to either model unchanged.

## The harness on the Society of LLMs (`society/`)

The controller was built on a simulator. `society/` puts it in front of a **real multi-agent system**: the agents of the user's Society of LLMs project (`society-of-llms`, its `society/` package) — five software-team roles (backend, frontend, integration, ship, planner) that build a login feature into a FastAPI app under a DAG runner with deterministic gates (pytest, schema diff, route presence, decision consistency), versioned KV handoff (`:current` moves only when a gate approves), a retry loop with the rejection fed back, and a System One quality gate. The clone keeps the agents' own code — `society/agents`, `runner`, `gates`, `memory` — and replaces only the LLM providers:

- **`providers/replay.py` — the agents without the LLMs.** A `ReplayWorker` does each role's work by script (the same edits, test runs and reports the project's own e2e stub makes, so every gate runs for real — pytest included) while its *consumption* (seconds on the model tier, input/output tokens, dollars, and whether the attempt comes back in a shape a gate rejects) is drawn from the project's **23 recorded runs** (`data/society/lineage.jsonl`, 90 attempts on Claude Opus 5 and Devstral). Draws are keyed by (seed, run, node, attempt): every policy sees the same futures.
- **`harness/world.py`** — the shared world in *society seconds* (recorded seconds ÷ `scale` = wall-clock): the cloud model tier (concurrency, RPM, TPM, priced per token; api-like → 429), the local coder (one resident model → queue), the sandbox pool (test runs → queue), one budget bucket per tenant. The physics classes are the simulator's (`agentsim/resources.py`).
- **`harness/gateway.py`** — the harness in wall-clock, four policies over identical questions: **`off`** (nobody coordinates: 429s, SDK retries, burned attempts, the platform bills at call end and refuses what the tenant cannot pay), **`gate`** (the project's own `local_slots` semaphore generalised: FIFO queueing, header-based pauses, a fixed 10 % budget floor — no prediction), **`needs`** (the Reserver: per role × provider × retry quantiles learned from the observation stream and warm-started from the deployment's lineage log; a predicted schedule of each run's remaining nodes summed into a demand `Forecast` whose excess is the pressure that prices leases; downstream-tier leases and gang leases under an expected-value gate; SRPT on a model queue deeper than its slots, weighted VTFQ otherwise; **workflow-level budget pacing** — a run is admitted only when the tenant's balance, net of what in-flight runs are predicted to still spend, covers its predicted spend-to-completion: *finish what you started*; the feedback loop's PI forecast correction, AIMD floor and fairness weights; System One as an optional annotator whose answers count only while the online judge finds them calibrated), **`oracle`** (the same rules with the sampled futures — the clairvoyant bound; `ReplayTable.peek` raises for anyone else).
- **`harness/governed.py`** — the three seams: a worker wrapper (model-tier unit + budget around every node), a command-gate wrapper and a tool hook (a sandbox unit around every pytest), a runner subclass (registers each run's service graph; verdicts flow back).
- **`web/server.py`** — the dashboard: the bench's results with paired wins, a live launch of any policies side by side on one seed (run cards per DAG, resources, queues, leases, tenant budgets, the forecast, the feedback loop, a narrated event feed over server-sent events), and the explanation.
- **`gates/jev.py`** — the project's Jev gate re-pointed at this repository's one client: `remote` = TypeSafe's System One through `RemoteSystemOne`; `rule` = a deterministic stand-in that checks the report against the recorded gate evidence and names itself `jev_rule` in every verdict.

```bash
uv sync --group society
uv run python -m society.web.server --port 8020                                # the dashboard: results, live off-vs-needs launches, the explanation
uv run python -m pytest -q -c society/tests/pytest.ini society/tests          # the agents' own tests (33), in their new home
uv run python -m society.harness.bench --policies off,gate,needs,oracle --seeds 1,2,3 --runs 12 --arrival-s 20
uv run python rung0/needs_tables.py data/society/bench/bench.csv --axis arrival_s --baseline off --oracle oracle
```

**With and without** (3 paired seeds, `data/society/bench/bench.csv`). Twelve login-feature runs submitted over ~4 minutes to three tenants ($1.50 each, refilled at $6/h) sharing a cloud tier of 3 concurrent requests, 6 requests and 150 k tokens per minute, one local coder and two sandboxes. Paired by seed; society time at scale 20.

**arrival_s = 20.0** (n = 3 paired seeds; * = significant after Holm vs `off`; wins = per-seed paired wins on failure / p50 / p99 / throughput)

| variant | failure | p50 s | p99 s | req·h⁻¹ | timeouts | Jain | refused | $/req | $ wasted | 429s | burned attempts | leases | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 0.583 | 278 | 395 | 33 | 0 | 0.64 | 6 | 0.817 | 0.403 | 60 | 24 | 0 |  |
| gate | 0.083 | 986 | 1,502 | 26 | 0 | 0.99 | 6 | 0.694 | 0.152 | 0 | 6 | 0 | 3/0/0/0 of 3 |
| needs | 0.000 | 1,189 | 1,531 | 28 | 0 | 1.00 | 0 | 0.566 | 0.000 | 0 | 1 | 4 | 3/0/0/0 of 3 |
| oracle | 0.000 | 517 | 1,376 | 29 | 1 | 1.00 | 1 | 0.575 | 0.000 | 0 | 1 | 1 | 3/0/0/0 of 3 |

Totals over the 3 seeds: **off** 15/36 runs completed, 179 429s, 18 refused payments, $12.03 spent of which 40 % on runs that never completed; **gate** 33/36 runs completed, 0 429s, 17 refused payments, $22.70 spent of which 16 % on runs that never completed; **needs** 36/36 runs completed, 0 429s, 1 refused payments, $20.37 spent of which 0 % on runs that never completed; **oracle** 36/36 runs completed, 0 429s, 2 refused payments, $20.68 spent of which 0 % on runs that never completed.

What the numbers say: uncoordinated agents lose most runs to 429 bursts on the cloud tier — three SDK retries a few seconds apart cannot outwait a saturated concurrency cap, and the executor burns the attempt, exactly the failure the project's own `mixed1` recording shows. The reactive gate removes every 429 but still starts runs it cannot finish: the platform refuses a payment mid-run and the tenant's earlier spend on that run is wasted. The Reserver admits a run only when its predicted spend-to-completion fits, so nothing is refused and nothing is wasted — at the price of deferring runs (their completion time includes the wait). The oracle shows what exact futures would add. Caveats, stated plainly: the predictor is warmed on one half of the recorded runs and the replay draws from the other, so it sees the right distribution family with sampling noise but no drift; the agents' *behaviour* is scripted (the LLMs are not in the loop), only their consumption is real; the Jev gate is the rule stand-in (`jev_rule`) in these runs; leases rarely pay on a chain-shaped DAG, and the expected-value gate mostly skips them; with three seeds nothing can reach significance under the exact Wilcoxon test (floor p = 0.25) — the wins columns are directional, and `--seeds 1,2,3,4,5,6,7,8,9,10` is the run that could settle it.

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
society/             the Society of LLMs agents (cloned) under the harness — see the section above
  agents/            the five roles: prompts, output schemas, gates, KV extractors (make_workers wires replay workers)
  runner/ gates/ memory/   the project's DAG runner, deterministic gates + executor + retry loop, versioned KV + lineage
  providers/         Worker protocol, sandboxed tools, FallbackWorker, replay.py (scripted roles, recorded consumption)
  harness/           world.py (society seconds, shared resources), gateway.py (off / gate / needs / oracle), governed.py (seams), bench.py
  web/               the dashboard (FastAPI + one page): recorded results, live side-by-side launches streamed over SSE, how it works
  examples/          the seeded FastAPI todo app and the login / OTP DAGs
  tests/             the project's own tests (33), adapted to the clone
data/society/        lineage.jsonl (the 23 recorded runs the replay draws from), bench/*.csv (with/without results)
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

Phase 6 — the Needs controller, its predictor and the System One model. Pre-train on traces (a `scenario.json` beside each, or `--scenario`), train + calibrate the local System One model (prints accuracy / ECE per question), annotate a trace with its phase labels, and run the controller:

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
- Shipped fixed rules of the earlier phases: `framework.sandbox_idle_timeout_s=0`; `policy.queue=vtfq` where the sandbox pool binds, `srpt` where the model tier binds (shortest *predicted* chat first from observable inputs).
- Rung 3 (`policy.type=lease`): leases on sandbox CPU (gang needs of declared parallel groups) and on provider call budgets (`<ext>@rpm`), a ledger with invariants I1 (Σ active leases ≤ capacity) and I3 (nothing past expiry), backfill by construction (only the unconsumed part is kept from others), expiry events; predictions only from the observation stream (`on_event`), enforced by an honesty selftest.
- Multi-agent (B10, Phase 6): a recipe's `spawn` block makes a chat launch 2–6 sub-agents (own recipe, sandbox, model slot, one request each); the orchestrator holds no step while it waits for the join and continues after it; a parent holding a sandbox while its children queue for one is a hold-and-wait cycle the engine detects. Child spans carry the parent's trace id (a request's metrics include its sub-agents' work).
- Content (Phase 6): every step carries an observable cue a gateway proxying bodies would see (tool skeleton class, prompt class, the response's plan), true with probability `content_snr`; the Needs Predictor learns from it; the System One model reads it.
- Phase 6 (`policy.type=needs`): the Needs Predictor (features.py, needs.py) + Reserver v2 (reserver.py) + the System One channel (jev.py) — see "Where it pays" above.
- Phase 7: tool-tier resources carry per-call cost and tokens (external APIs, `mcp.<server>` tool servers, `gpu.<pool>` accelerators); the model tier has prices per Mtok; tenant budgets (USD / tokens) refill hourly and refuse what they cannot pay (outcome `budget`); the Needs controller paces admission to the forecast (finish what is started) and closes a feedback loop (control.py) on its own prediction error per resource, on budget refusals and on fairness; Jev's supervisory questions (cost class, budget-will-exceed, stuck-in-loop) guide the pacer and the leases.
- Not modelled: multiple model instances/routing, autoscaling, network, tool-result caching, semantic content beyond the cues above (branching is a Markov chain — see E2 in the report; `recipe_temperature` sets its entropy).

Data volume: `data/synthetic/` is ~500 MB with traces and `data/real/` ~1 GB; the repository keeps only the per-grid summary CSVs (`grid_summary.csv`, `data/synthetic/needs/*.csv`, `sweep.csv`) behind the results — regenerate the rest with the commands above. Trained models (`data/models/*.npz`, 48 MB) are not in the repository either; their hold-out reports are, and `train-needs` / `train-jev` rebuild them in minutes from the training traces (`data/synthetic/train/`, generated by `agentsim run` with `--set generator.content_snr=0.6` on the society in question).
