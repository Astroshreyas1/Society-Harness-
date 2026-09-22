# HANDOFF — Control: runtime service-graph reservation for concurrent agentic workflows

Written 2026-09-22 for whoever picks this up (human or agent); updated the same day after Phases 6–7 (the Needs controller, the System One integration, budgets and the feedback loop). Read this first; it tells you what exists, what was learned, what is wrong, and what to do next. Nothing here is a claim beyond what the referenced files show.

## 1. The problem in one paragraph

N concurrent agentic workflows share models (API concurrency, TPM, KV cache), tools (sandboxes, external APIs with rate limits) and services. Uncoordinated, they over-allocate, starve each other and die (429/502/timeouts/deadlock) even when aggregate capacity is sufficient. The goal is a controller that predicts each workflow's near-future dependencies and reserves capacity without over-allocation, starvation or failure — trained first on synthetic workflows, then fine-tuned on a real society of LLM agents.

## 2. Where things are

| File | What it is | Read it when |
|---|---|---|
| [NEEDS_REPORT.md](NEEDS_REPORT.md) | **Phases 6–7 evidence**: the Needs controller vs the shipped rules, rung 3 and the oracle on six societies (10 seeds, paired), per-switch ablations and the three defaults they overturned, the predictor on real traces, the System One model's calibration, budgets and the feedback loop. The only place with Phase 6–7 numbers | you want numbers for the controller |
| [JEV_SURVEY.md](JEV_SURVEY.md) | What is documented about TypeSafe's Jev / System One models (primitives, API, patterns, limits), how each pattern is used here, what changes with API access, the Phase 7 plan; sources | you are touching `jev.py` or getting the API |
| [PREDICTOR_DESIGN.md](PREDICTOR_DESIGN.md) | v3 notes: the Needs Predictor (multi-head distributional model over structure, quantities, timing, idle, risk; content features from proxied bodies; conformal calibration) and the Reserver logic that consumes it (occupancy forecast, mixture-quantile leases, expected-value gating); the two TraceLab experiments that bound it; the build order — **built** in Phase 6 (`features.py`, `needs.py`, `reserver.py`) | you want the design rationale behind the code |
| [SOLUTION.md](SOLUTION.md) | The reader's guide: problem, idea, ladder, findings, shipped rules, what is unsolved | you are new, or need to explain this to someone |
| [SERVICE_GRAPH_RESERVATION.md](SERVICE_GRAPH_RESERVATION.md) | v1 notes: the "crime", hypotheses H1–H6 with literature evidence, first controller design (Observer → Predictor → Reserver → Gate), evidence plan E1–E6 | you need the *why* and the literature (45 refs) |
| [DEEP_DIVE_AND_LADDER.md](DEEP_DIVE_AND_LADDER.md) | v2 notes: deeper survey by layer, **Jev** adaptation (§2), synthetic-data design (§3), its limitations (§4), realistic path (§5), the 7-rung ladder (§6). §1.9 lists v1 decisions it revised | you are designing rung 3+ or evaluating |
| [RUNG0_REPORT.md](RUNG0_REPORT.md) | What the simulator reproduces and what E1–E5 measured on synthetic data, after the inflation audit; §11 the evening's knobs and corrections; **§12 rung 3's scoreboard and the ablation that deflates it**. The only place with synthetic numbers | you want numbers |
| [tasks/plan.md](tasks/plan.md), [tasks/todo.md](tasks/todo.md) | The 2026-09-22 evening plan (13 tasks, 4 checkpoints) with every outcome and finding recorded per task | you want to know what was done, in what order, and what was found on the way |
| [ISSUES.md](ISSUES.md) | Tracker: 50+ issues with priority/status; §F is the inflation audit | before changing anything |
| [README.md](README.md) | How to run | you want to run it |
| `agentsim/` | The simulator (Python 3.12 + numpy; no other deps). Phase 6–7 modules: `features.py` (Observer), `needs.py` (predictor + calibration), `jev.py` (System One), `reserver.py` (the Needs controller), `control.py` (feedback loop), `train.py` (offline training / evaluation / annotation) | — |
| `data/models/` | Pre-trained predictors and System One models per society (npz) with their hold-out reports (json) | before running `needs_pre` / `needs_jev` variants |
| `rung0/` | E1–E5, `compare.py`, `headroom.py` (sweep analysis), `tracelab_to_spans.py` (real-trace adapter); they read OTel-shaped JSONL | — |
| `data/real/tracelab/` | TraceLab trace (gz) and its span conversions (~1 GB; git-ignored; regenerate with the adapter) | rung 0 on real data |
| `scenarios/` | `api_coding`, `hosted_mixed`, `hosted_stress`, `hosted_tracelab` (fitted from real traces); 19 grids incl. the rung-3 scoreboards, ablations and the TraceLab grid; `knob_ranges.json` + `sampled/` (40) + `variants_ship.json` | — |
| `data/recipes.json`, `data/marginals.json` | Structure and quantity seeds with provenance per entry | before trusting any number |
| `data/synthetic/` | generated runs (~400 MB with traces; the rung-3 grids are summaries only). Regenerate; do not sync | — |

Terminology: **"JEV" = Jev**, TypeSafe AI's System One model (typed, calibrated decisions; 70–500 ms; early access) — *not* JEPA. An earlier draft assumed JEPA; that is retired (v2 §2, last paragraph).

## 3. What is true, as far as the synthetic evidence goes

1. **Coordination is the win; prediction is not (on the API tier).** A reactive gate with SDK-style jittered retries removes essentially all failures uncoordinated agents suffer past ~1.25× the slot count (98% → 0.5% at 2.5×), and a clairvoyant oracle — knowing every duration, token count, think time and provider state — adds nothing on top of it there. Remaining failures at 50 agents / 8 slots are client timeouts: raw capacity.
2. **On the hosted mixed society, uncoordinated == gate.** Nothing API-like binds at 0.5–2 sessions/min; the gate's primitives are idle. The oracle's only gain (4.1% → 2.4% failures, −23% p99, +18% throughput at top load) comes from parking idle sandboxes the moment a session goes idle and pre-warming before the next request, plus SRPT. That is **idle-time prediction**, not tool/model demand prediction.
3. **Pinning the model slot across tool calls (server-side tools) creates real hold-and-wait cycles** (82–267 per 2 h) that the gate's forced release eliminates — but with SDK retries and client timeouts the cost is +141% p50 latency, −38% throughput and Jain 0.37 vs 0.68, not failures.
4. **The seeded synthetic society is only R≈0.30 / 0.50 one-step predictable** — below the 0.7 gate that funds prediction rungs. This is a property of hand-written transition tables, not of real agents (PBKV reports 0.94 on real agent roles).
5. **Tails are as published:** 85% of tool time in the 3.8% of calls > 1 min; reservations must be on quantiles.

Consequently the honest ladder state in the morning was: rung 1 (reactive gate) is the deliverable; rung 3's case must be made on idle-time / sandbox / KV reservation.

6. **(Evening.) Rung 3 was built and ablated (RUNG0_REPORT §12).** The `LeaseController` — predictions from the observation stream only, leases with invariants, fair queuing, gang and budget reservations — beats the §5 gate on every family at 10 seeds. The ablation attributes ~90% of that to the gate's own idle-timeout PRIOR (300 s → 0 s), the rest to virtual-time fair queuing (a fixed rule), and **nothing measurable to the idle-time prediction** at any snr or cold start (3/30/90 s); gang leases raise the parallel share but waste 72–77% of reservations on a saturated pool; budget leases eliminate 429s but, with an honest margin, buy no failure or throughput over SDK retries. Corrections on the way: E2 understated predictability (F15: R₃ 0.40 hidden / 0.58 exposed); KV admission (F16); the 5-seed significance floor (F18).

7. **(Later still — Phase 5, RUNG0_REPORT §13.)** The shipped rules were tested head-on and hold. A randomised sweep of 40 knob-sampled societies found the oracle's headroom over the shipped gate concentrated where the **model tier saturates** — and there a gate that sorts chats by *predicted* service time from observable inputs (`queue=srpt`: known prefill + learned mean output) captures most of it (0.99 of the failure headroom, throughput within 2% of the oracle on the model-bound scenarios), at a small failure cost where the tier is not saturated. **Rung 0 on real data happened:** TraceLab's 5,312 Claude Code sessions give one-step structure predictability 0.72 (over the 0.7 gate; the seeds had 0.54), idle-time predictability from a gateway's view R² 0.05, tails as published; a society fitted from them shows the same ranking of policies as the synthetic ones.

The honest ladder state after Phase 5: **ship rung 1 with three fixed rules — park idle sandboxes immediately; fair-queue where the sandbox pool binds; shortest-predicted-chat-first where the model tier binds. The only prediction that pays is the trivial one (prefill size + mean output per recipe); idle-time, KV and budget prediction do not, on synthetic or real-structured societies.** What real traces could not answer: cold-start costs, pool slack, provider budgets, and multi-agent societies (B10).

8. **(Second session — Phases 6–7, NEEDS_REPORT.md.)** The full design was built without the ladder's gating: content cues and a multi-agent recipe in the generator; the Observer / Needs Predictor / Reserver v2 (`policy.type=needs`); a System One integration built to Jev's real API (JEV_SURVEY.md) with a local calibrated stand-in, an online per-question calibration judge and decision judging; tool-tier costs, MCP / GPU resources, tenant budgets; a feedback loop on the controller's own prediction error. What the evidence says, in order of size: (i) **where a tenant budget binds, the controller's forecast-aware pacing is the largest gain in the project** (budget society, 10 seeds: p50 −43 % and throughput +10 % significant after Holm, zero refused payments in every seed, spend wasted on abandoned work 2.9 % → 1.6 %, 9 % cheaper per completed request, failure 0.035 → 0.019 at 7/10 wins but not significant; the loss column is client timeouts +26 per run; the clairvoyant gate 0.017 / +30 % throughput — NEEDS_REPORT §5.6); (ii) where the model tier saturates it chooses shortest-predicted-first by itself and gets `srpt`'s outcome (failure 0.001 / 0.054, throughput 463 / 294 req·h⁻¹ vs the shipped gate's 0.046 / 0.192 and 117 / 133; oracle 488 / 332) — the tier-conditional rule of RUNG0 §13.6 without an operator choosing it; (iii) on the hosted, multi-agent, TraceLab-fitted and stress societies it **equals the shipped rules** — nothing measurable from content, the predictor, Jev features or the leases, at any `content_snr`; (iv) the predictor is real on real traces (tool-duration log R² 0.34 from command skeletons; gap-to-next-chat R² 0.21) and the local System One model is calibrated (ECE ≤ 0.05) — but neither moves an outcome metric on these societies; (v) three defaults were wrong and were found only by per-switch ablation (ISSUES F26): the starvation guard, the spawn gang's lifetime, predicted KV reserves.

## 4. What was wrong and got fixed today (so you don't re-introduce it)

The first results were inflated on both sides (ISSUES §F, all closed): full-context TPM instead of fresh tokens; agents aborting on the first 429 without SDK retries; pinned-slot config as the default; session-kill on deadlock; a gate with infinite patience (no client timeout); no sandbox idle timeout in the baseline; a 50%-per-call external 429 curve instead of provider buckets; survivor-only fairness; cycle events counted as deadlocks; a Hill index on a capped mixture; a shared RNG (unpaired comparisons). Also: parallel tool groups deadlocked among themselves by waiting for extra CPU (B17) — now sequential fallback. And tracker items had been marked done before verification — corrected.

Every one of these changed magnitudes a lot (gate "0%" → 2.6%; uncoordinated 57% → 15%). Before reporting any gain, run this checklist (it lives here, not in any memory file): weak baseline? strong policy? timeouts modelled? metrics include failures? paired seeds with CIs — and enough of them (5 seeds cannot reach α = 0.05 on an exact test, ISSUES F18)? oracle bound? honest inputs (does the policy read anything a gateway cannot observe)? **per-switch ablation of the new controller's own defaults** (Phases 6–7 found three that hurt, F26)?

Found and fixed later the same day (ISSUES F15–F18): E2 pooled seeds by colliding session ids (predictability was understated); KV admission under-counted a reused idle prefix; the `parallel` span attribute was never set; the 5-seed significance floor.

## 5. How to run (5 minutes)

```bash
uv run python -m agentsim selftest
```
40 checks (≈2 s): determinism, common random numbers, resource accounting, no leaked holds, fidelity bounds, oracle sanity, every generator knob's identity at its default (bit-identical spans), the F15 observer guard, structured think time (log-moments preserved, R² ≈ snr), predictor convergence, the lease controller's honesty (contradictory hidden values → identical decisions), ledger invariants I1/I3, gang and budget leases, and the `fit` round trip. Must pass before and after any change. No `python` on PATH here: use `uv run python` (pyproject.toml).

```bash
python -m agentsim grid --grid scenarios/grid_e1.json --out data/synthetic/e1 --no-traces
```
```bash
python -m agentsim grid --grid scenarios/grid_hosted.json --out data/synthetic/hosted
```
```bash
python rung0/e1_failure_vs_concurrency.py data/synthetic/e1/grid_summary.csv --plot data/synthetic/e1/e1.png
```
```bash
python rung0/e2_predictability.py data/synthetic/hosted/*/traces.jsonl --with-phase
```
E3 / E4 likewise (README). Aggregation over seeds: `rung0/compare.py` (paired statistics, Holm, oracle-normalised score; `--filter` for multi-axis grids, `--test t` for 5-seed grids). Rung-3 scoreboards: `scenarios/grid_lease_snr.json`, `grid_lease_parallel.json`, `grid_stress_lease.json` (10 seeds each; README).

## 6. How the simulator is built (so you can change it safely)

- **Engine** (`engine.py`): heap of events `(t, seq, kind, sid, (epoch, uid, attempt), payload)`. A program's step is a `Pending`; anything that cancels a step bumps `attempt`, anything that replaces it makes a new `uid`, ending a session bumps `epoch`; stale events are dropped in `run()`. Steps acquire resources in `_try_run` (global order: model.slots, sandbox.mem, sandbox.cpu, ext.*, svc.*), then KV admission, then TPM, then provider budgets (headers checked for all members before anything is consumed), then `_start`. `_finish_step_holds` is the one place a step's own holds go back. `_react` implements SDK retries then agent replan/abort. Client timeouts: `WAIT_TIMEOUT`; deadlock: `DEADLOCK_BREAK` → same path. Sandboxes park on `SANDBOX_PARK` and can be pre-warmed.
- **Resources** (`resources.py`): capacity/holders/waiters; `CallBucket` (RPM with headers, background load); `TokenBucket` (TPM on fresh tokens); `ModelPhysics` (TTL prefix cache, admission on *active* KV, least-progressed eviction, shared decode capacity). `Resource.check()` is the accounting invariant.
- **Workload** (`workload.py`): recipes = phase Markov chains; per-program RNG streams (`rng_steps` keyed by exogenous identity → common random numbers; `rng_env` for lotteries/reactions); parallel groups; partly-open arrivals (MMPP) or closed population.
- **Policies** (`policies.py`): one interface (`cap`, `on_unavailable`, `priority`, `sdk_retries`, `retry_delay`, `release_model_for_tools`, `tpm_fraction`, `can_issue_external(res, now, u, p, n_calls)`, `external_retry_after`, `idle_timeout`, `prewarm_delay`, `prewarm_hold`, `retention_ttl`, `kv_evict_key`, `on_event`, `leases`, AIMD hooks). Hidden values (`think`, `gap`, `u`, `Step.duration`/`tokens_out`) are passed to some hooks for the oracle's sake; a non-clairvoyant policy must not read them — the selftest feeds the `LeaseController` contradictory values and asserts identical decisions. Keep that check when adding hooks.
- **Leases** (`resources.Lease/Ledger`, engine `_free`, `_replace_leases`, `LEASE_EXPIRE`): a policy returns its whole forecast for a session from `leases(p, step, now)` after every step/request end; the engine keeps others off the unconsumed part (`_free = free − reserved_for_others`), expires lazily and by event, consumes on take, and asserts I1/I3 in `check_accounting`. Budget leases are `"<ext>@rpm"`. Gang waits (`Pending.gang_wait`) are bounded by the lease's expiry and excluded from the deadlock probe.
- **Predictors** (`predict.py`): `QuantileTracker` (bounded windows per key with prefix back-off) and `MarkovNext`; fed only through `on_event`.
- **Generator knobs** (`workload.py`): `recipe_temperature` (tempers every structure lottery; identity at 1), `think_snr`/`think_state_seed` (offsets standardised per (recipe, bucket) group under the recipe's analytic end-phase probabilities), `tool_tail_scale`, `p_parallel_scale`; fitted recipes use observable phases with `"*"` transitions (`fit.py`).
- **Metrics** (`metrics.py`): computed from spans only; window = after `warmup_s`; requests are `invoke_workflow` spans; failure = aborted ÷ (completed + aborted); fairness = Jain of service-ratio over all sessions incl. failed.
- **Schema** (`schema.py`): OTel-GenAI-shaped `Span`; `invoke_agent` (session) → `invoke_workflow` (request) → `chat` / `execute_tool` / `retrieval`, plus `wait` and `think`.

Rules that kept this honest: every precondition throws; no fallback branches; one place per responsibility; selftest after every change; long Python patches go through files, not shell heredocs (the shell layer here mangles long/escaped heredocs — two hours were lost to that).

## 7. Open work, in the order it should be done

Items 2–5 of the original list were done on 2026-09-22 (tasks/plan.md, tasks/todo.md; results in RUNG0_REPORT §11–§12). What remains:

| Order | Item (ISSUES id) | Why first |
|---|---|---|
Phase 5 (same day) did the real-trace step with TraceLab, the ship test, the sweep, the gain-vs-R curve and a calibrated budget margin (RUNG0_REPORT §13). What remains:

| Order | Item (ISSUES id) | Why first |
|---|---|---|
| 1 | **Shadow traces from the user's own society** (E3) and the **Codex half of TraceLab** (its timing convention differs; `rung0/tracelab_to_spans.py` skips 144 k rounds) | TraceLab is one coding-agent population; every §13.5 number is conditional on it |
| 2 | **Queue-length-conditional SRPT** (srpt when the model queue is deep, aging/fair otherwise) and a per-tier rule switch in the gate | §13.3–13.4: the shipped rule is tier-conditional; the 3-of-40 failure losses of pure srpt are the cost to remove |
| 3 | **Multi-agent spawn/join** (B10) and a society with several model instances (B11) | The trace and the generator are single-agent; the title says multi-agent |
| 4 | ~~Jev integration (D1–D3)~~ built (Phases 6–7); **vendor validation** (`evaluate-jev --remote`, D4) the day the API key exists | the local stand-in cannot say what frontier reading of real text buys |
| 5 | Fidelity upgrades (B12–B14), performance (A9), per-kind reactions (B15) | When the above needs them |

After Phases 6–7 the order is: (1) **shadow traces from the user's society** (tool arguments, response text, provider usage and rate-limit headers, MCP server names, sub-agent spans) — the only data that can test costs, MCP servers, concurrency and Jev on real text; (2) the vendor model through `RemoteSystemOne` (D4) and the `needs_jev` grids re-run; (3) G1–G5 (ISSUES §G): real prices, per-user budgets, the paced-session forecast bias, the starvation-guard sweep per society; (4) the items above.

## 8. Decisions already taken (don't relitigate without new evidence)

- Only the Predictor is learned; the reservation rule is a fixed algorithm with two knobs (τ, h). No fallback branches: uncertainty lives in the distribution the lease consumes.
- Deadlock freedom by construction (ordering + all-or-nothing + expiry); starvation bound via virtual-time fair queuing (Justitia), DRF only as a cross-tier cap.
- The oracle is a clairvoyant gate (Hermes-Oracle style), not a hindsight optimum; headroom = oracle − reactive gate.
- The Gate lives in an existing flow-control seat (K8s Gateway API Inference Extension / LiteLLM), not a new proxy.
- Jev is an async step-boundary featurizer, offline annotator and reaction/verification oracle — never on the ≤5 ms hot path; it is reserved like any tool.
- Success is judged on outcomes (failure, p99, waste, fairness, throughput, task success), never on predictor accuracy.
- (Evening.) **No gain is attributed to prediction without an E6 ablation over the baseline's knobs** (idle timeout, header pause, retries) and the controller's own switches (`policy.ablate`); gains are reported as curves over those knobs, over load, over `think_snr` and over `recipe_temperature`. Learned rungs use 10 seeds.
- (Evening.) A policy's inputs are the observation stream (`on_event`) only; hidden values in hook signatures exist for the oracle and are guarded by the honesty selftest.
- (Phases 6–7.) The Needs controller's defaults are what the per-switch ablations support: starvation guard **off** (opt-in), SRPT only when the model queue exceeds the slots, spawn gangs priced by pressure (never issued on a saturated tier), predicted KV reserves only when KV binds, gang leases sized by content-conditioned quantiles and gated by a learned pay rate. A System One question is used only while its live ECE ≤ 0.1 and its confidence ≥ 0.55; noul answers are probabilities, not decisions. Budgets are paced by forecast (finish what is started) with an AIMD floor on refusals.

## 9. Things that need the user

- ~~Approval to download public datasets~~ — TraceLab (CC BY 4.0, 101 MB) was downloaded on 2026-09-22 under the user's blanket permission and lives in `data/real/tracelab/` (git-ignored). Exgentic/agent-llm-traces-v2 (10 k OTel sessions across frameworks, 236 MB, licence not stated on the page) is the next candidate. Shadow traces from the user's own society remain the better source.
- Whether to `git init` and ignore `data/synthetic/` (E1) — asked 2026-09-22, answer was "not yet"; `.gitignore` is ready.
- **Jev API credentials** (JEV_API_KEY; JEV_API_BASE if via LiteLLM): the integration is built and tested against a canned response; `agentsim evaluate-jev --remote` and the `needs_jev` grids are the first two things to run (JEV_SURVEY.md §5; ~cents per evaluation).
- **The dataset question (asked 2026-09-22):** the data is enough for mechanism-level results and relative comparisons under stated priors, not for real-world claims about costs, MCP servers, sub-agents or Jev on text. Shadow OTel traces from the target society are the input that changes that; a text-bearing public trajectory corpus (SWE-agent / OpenHands) is the stop-gap for the text questions.
