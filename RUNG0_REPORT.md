# Rung 0 report — simulator, synthetic data, E1–E4, and the inflation audit

Date: 2026-09-22 (v2; v1 of this report was withdrawn after the audit in [ISSUES.md §F](ISSUES.md)). §0–§10 were written in the morning; §11–§12 the same evening, after which the tables of §4–§6 were regenerated on the corrected engine (F16, and the oracle's KV retention of B6) — the prose of §0–§10 is kept as written where it still holds and annotated where it does not. Everything here is *synthetic*: it shows what the simulator reproduces and what the rung-0 scripts measure. Real-trace numbers replace these once shadow-mode traces exist (`python -m agentsim fit --recipes-out …`, then rerun the scripts).

## 0. Verdict in one paragraph

After removing eleven sources of inflation, the picture is: **coordination (a reactive gate with SDK-style retries) removes essentially all failures that uncoordinated agents suffer under overload, and a clairvoyant oracle adds little on top of it in failures.** On the API-concurrency tier the oracle equals the gate in failure rate; once it may keep KV prefixes across tool gaps (B6, evening) it gains throughput there (+25% at N ≥ 20 — a provider-dependent lever, §11.7). On the self-hosted mixed society the gate equals *uncoordinated* (nothing API-like binds at these loads) and the oracle's gain — 3.8% → 2.5% request failures, +18% throughput, Jain 0.43 → 0.61 at the highest load; its −23% p99 is not significant (§11.1) — comes from parking idle sandboxes and SRPT ordering, i.e. from **idle-time prediction**, not from predicting tool/model demand. Pinning the model slot across tool calls (server-side tools) creates real hold-and-wait cycles (82–267 per 2 h) but, with SDK retries and client timeouts, costs latency and throughput rather than failures. Predictability of the seeded recipes is R₃ ≈ 0.40 (one-step 0.54; §11.1 corrects §7's 0.30 / 0.50), below the 0.7 gate. **State of the ladder after the evening's work: rung 1 (reactive gate) is the deliverable on the API tier; rung 3's controller exists and is scored in §12 on the idle-time, gang and budget targets; whether its gains are real for *your* society depends on the society's idle predictability and binding tiers, which only real traces can establish.**

## 1. What was built

| Piece | File | Faithful to |
|---|---|---|
| Program-centric event engine, causal successor release `r_{k+1} = c_k + g_k`, PCB, events with (epoch, uid, attempt) staleness | `agentsim/engine.py` | AgentServeSim |
| Three tiers as named capacities; API-like (429) vs OS-like (queue) semantics per resource; provider **RPM buckets with headers and background load** | `agentsim/resources.py` | v1 §1.1, HiveMind, provider docs |
| KV physics: TTL prefix cache, admission on *active* KV (vLLM semantics), least-progressed eviction when generation outgrows the admitted reserve; **TPM debited on fresh tokens** | `agentsim/resources.py`, `engine.py` | Service-Induced Congestion; provider TPM semantics |
| Partly-open arrivals: MMPP sessions × closed-loop requests with think time; closed population with staggered start for E1 | `agentsim/workload.py` | Schroeder et al. |
| **Per-program RNG streams** keyed by exogenous identity (common random numbers across policies) | `agentsim/workload.py` | evaluation protocol §6.0 |
| Phase recipes, heavy-tailed quantities with provenance, **parallel tool groups** (gang need, opportunistic extra CPU, sequential fallback), **two-layer reaction** (SDK retries with jittered backoff, then agent replan/abort) | `data/recipes.json`, `data/marginals.json`, `workload.py` | WfChef/WfGen, TraceLab, SDK defaults |
| **Client step timeout** (600 s) on every step, **sandbox idle timeout** (300 s) with cold-start resume, hold-and-wait cycle detection, deadlock break = step timeout | `engine.py` | Coffman, DPBench, SDK/cloud-sandbox defaults |
| Policies: `Uncoordinated` (SDK defaults; `jitter=false` = herd), `ReactiveGate` (HiveMind primitives + header-based pause + priority waiters), **`ClairvoyantGate`** (oracle: true durations/tokens/think/provider state; SRPT; park-at-idle; prewarm) | `agentsim/policies.py` | HiveMind; Hermes-Oracle |
| OTel-GenAI-shaped spans incl. **`invoke_workflow` per request**; request-level metrics with warm-up; fairness as service-ratio Jain over all sessions | `schema.py`, `metrics.py` | OTel GenAI semconv; A2/A3 |
| `python -m agentsim selftest`: determinism, CRN, accounting invariants, no leaked holds, fidelity bounds, oracle sanity (7 checks in the morning; 40 by the evening — knob identities, F15 guard, B7 moments, predictors, lease honesty, ledger invariants, fit round trip) | `run.py` | A10/B9 |
| E1–E4 scripts on JSONL (E5 and `compare.py` added in the evening, §11) | `rung0/` | v1 §8 |
| **Evening additions (§11–§12):** generator knobs (`recipe_temperature`, `think_snr`, `tool_tail_scale`, `p_parallel_scale`), `sample-scenarios`, `fit --recipes-out`, `LeaseController` + lease ledger, ablation switches | `workload.py`, `fit.py`, `predict.py`, `policies.py`, `resources.py` | v2 §3.5, §6.1 rung 3 |

Generated (morning): `data/synthetic/e1` (60 runs × 1 h, summaries only), `data/synthetic/hosted` (45 runs × 2 h, traces) and `data/synthetic/hosted_pinned` (15 runs × 2 h, traces); 5 seeds per cell, warm-up 600 s excluded. Regenerated in the evening on the corrected engine (F16, B6); ≈2,500 further runs for §11–§12 as summaries; ≈420 MB in all.

## 2. The inflation audit — before vs after

| What changed (ISSUES F#) | Uncoordinated, E1 N=10 | Reactive gate, E1 N=50 | Hosted pinned, uncoordinated |
|---|---|---|---|
| v1 report (session-level, shared RNG, full-context TPM, abort on first 429, no timeouts, no idle timeout, session-kill on deadlock) | **57% sessions failed** at 51% utilisation | **0% failures**, p50 latency 769 s | **36–49% failures**, 40 sessions killed by deadlock |
| v2 (F1–F11 fixed; request-level; 5 seeds) | **14.8% ± 14.3% requests failed** (bimodal: 0% in 3 seeds, 11%/40% in 2) | **2.6% ± 0.9%** (client timeouts), p50 191 s | **2.1% ± 1.6%** failures; cost is **+154% p50 latency, −37% throughput, Jain 0.37 vs 0.68** (after F16) |

Every headline in v1 was too optimistic for the gate and too pessimistic for the baseline. The direction of the findings survived; the magnitudes did not.

## 3. Fidelity of the generator (level 1 — marginals)

| Statistic | Synthetic (hosted, gate, 5 seeds) | Target (TraceLab, Claude Code) |
|---|---|---|
| tool calls < 1 s: share of calls / of tool time | 67.8% / 1.7% | 70% / <1% |
| tool calls > 60 s: share of calls / of tool time | 3.8% / 85% | 4.9% / 92% |
| tool duration mean | 11.0 s | 16.8 s |
| output tokens median / p99 | 201 / 5,906 | 252 / 6,571 |
| think time median / p90 | 82 s / 1,211 s | 84 s / 1,236 s |
| LLM steps / tool calls per request (coding, completed sessions — F12) | 8.0 / 9.7 | 8.8 / 10.8 |

Levels 2–4 (structure, dynamics, policy ordering against the real society) need real traces.

## 4. E1 — failure vs concurrency (H1), `api_coding`, 8 API slots, mean ± 95% CI over 5 seeds

| policy | N | request failure | model util | 429s/run | timeouts/run | p50 / p99 s | Jain (all) | req·h⁻¹ |
|---|---|---|---|---|---|---|---|---|
| uncoordinated | 5 | 0.014 ± 0.016 | 0.18 | 3 | 2 | 172 / 1,249 | 0.77 | 38 |
| uncoordinated | 10 | 0.148 ± 0.143 | 0.40 | 175 | 5 | 154 / 1,279 | 0.66 | 76 |
| uncoordinated | 20 | 0.983 ± 0.002 | 0.93 | 36,295 | 2 | 35 / 400 | 0.01 | 80 |
| uncoordinated | 50 | 0.998 | 0.93 | 225,184 | 4 | 26 / 266 | 0.00 | 53 |
| reactive gate | 5 | 0.000 | 0.16 | 0 | 5 | 167 / 1,224 | 0.79 | 34 |
| reactive gate | 10 | 0.006 ± 0.012 | 0.32 | 0 | 11 | 151 / 1,302 | 0.74 | 63 |
| reactive gate | 20 | 0.005 ± 0.006 | 0.54 | 0 | 30 | 149 / 1,420 | 0.72 | 103 |
| reactive gate | 50 | 0.025 ± 0.012 | 0.68 | 0 | 138 | 180 / 1,468 | 0.48 | 112 |
| clairvoyant | 5 | 0.000 | 0.19 | 0 | 2 | 172 / 1,299 | 0.81 | 40 |
| clairvoyant | 10 | 0.000 | 0.38 | 0 | 6 | 154 / 1,237 | 0.81 | 77 |
| clairvoyant | 20 | 0.000 | 0.67 | 0 | 19 | 164 / 1,418 | 0.77 | 132 |
| clairvoyant | 50 | 0.022 ± 0.010 | 0.77 | 0 | 145 | 191 / 1,435 | 0.55 | 129 |

- **H1 holds, but only past ~1.25× the slot count.** With SDK-default jittered retries, 5 agents on 8 slots are fine; 10 agents are bimodal (the herd forms in 2 of 5 seeds); 20 agents (2.5×) collapse to 98% failure with 36k 429s. The 14.8% at N=10 occurs at 40% utilisation — capacity was sufficient; retry storms did it.
- **The oracle equals the gate in failure rate** at every N (differences inside the CI). Residual failures at N=50 are client timeouts on an 8-slot endpoint serving 50 agents: raw capacity, which no prediction creates. **No prediction headroom in failures on this tier.** (Evening, B6: with KV retention across tool gaps the oracle's throughput is +25% over the gate at N ≥ 20 — prefix reuse, §11.7.)
- Survivorship: uncoordinated p50/p99 *fall* at high N because only trivial requests survive (F14). Read failure and latency together.

## 5. Hosted mixed society (self-hosted 16-slot model, partly-open arrivals, no pinning), 5 seeds

| policy | sessions·min⁻¹ | request failure | timeouts/run | p50 / p99 s | Jain (all) | req·h⁻¹ | max blocking wait |
|---|---|---|---|---|---|---|---|
| uncoordinated | 0.5 | 0.004 ± 0.001 | 17 | 78 / 1,167 | 0.88 | 144 | 600 s |
| reactive gate | 0.5 | 0.004 ± 0.001 | 17 | 78 / 1,167 | 0.88 | 144 | 600 s |
| clairvoyant | 0.5 | 0.004 ± 0.001 | 16 | 72 / 1,194 | 0.91 | 147 | 600 s |
| uncoordinated | 1.0 | 0.010 ± 0.005 | 83 | 104 / 1,995 | 0.68 | 219 | 600 s |
| reactive gate | 1.0 | 0.010 ± 0.005 | 83 | 104 / 1,995 | 0.68 | 219 | 600 s |
| clairvoyant | 1.0 | 0.006 ± 0.003 | 46 | 99 / 1,648 | 0.82 | 243 | 600 s |
| uncoordinated | 2.0 | 0.038 ± 0.020 | 554 | 111 / 2,792 | 0.43 | 310 | 600 s |
| reactive gate | 2.0 | 0.038 ± 0.020 | 555 | 110 / 2,835 | 0.43 | 310 | 600 s |
| clairvoyant | 2.0 | 0.025 ± 0.020 | 402 | 112 / 2,252 | 0.60 | 363 | 600 s |

- **Uncoordinated and the gate are identical** here: no 429s occur at these loads (provider buckets never empty, model slots queue), so every primitive of the gate is idle. The residual failures are client timeouts while waiting for a *sandbox* (max blocking wait = the 600 s timeout) — sessions hold sandboxes across think time until the 300 s idle timeout.
- **The oracle's gain is the only measured headroom for prediction**: at 2 sessions·min⁻¹ it cuts failures 3.8% → 2.5%, raises throughput 18% and Jain 0.43 → 0.61 (its p99 reduction, 2,835 → 2,190 s, is not significant on 5 seeds — §11.1), entirely by parking sandboxes the instant a session idles, re-acquiring one cold start before the next request, and SRPT ordering. That is *idle-time* prediction (Copilot traces: 86–90% of idle time is predictable) — not tool/model demand prediction.

## 6. Pinned configuration (server-side tools: model slot held across tool calls), 1 session·min⁻¹, 5 seeds

| policy | request failure | cycles seen per run | timeouts/run | p50 / p99 s | Jain (all) | req·h⁻¹ |
|---|---|---|---|---|---|---|
| uncoordinated (pinned) | 0.021 ± 0.016 | 82–267 | 143 | 264 / 2,476 | 0.37 | 138 |
| reactive gate | 0.010 ± 0.005 | 0–0 | 83 | 104 / 1,995 | 0.68 | 219 |
| clairvoyant | 0.006 ± 0.003 | 0–0 | 46 | 99 / 1,648 | 0.82 | 243 |

- **H4 holds mechanically:** the pin creates hold-and-wait cycles on (model.slots, sandbox.mem) and (model.slots, sandbox.cpu); the gate's forced release removes every one of them.
- **Its cost is latency, throughput and fairness, not failures:** the 600 s client timeout breaks the cycle before the deadlock timer; the SDK retries; the request survives. The v1 claim of "40 sessions dead of deadlock" was the session-kill artefact (F4). (Table regenerated after F16: p50 264 vs 104 s, throughput 138 vs 219 req·h⁻¹.)
- Found on the way (ISSUES B17): parallel tool groups that *wait* for extra CPU units deadlock among themselves (incremental acquisition of one resource); real frameworks degrade to sequential execution, which the engine now does. A lease controller could instead reserve the units — a measurable rung-3 target.

## 7. E2 — predictability (H5), hosted uncoordinated, 5 seeds (≈16 k nodes)

| context k | phases hidden: R_k / acc@1 / acc@3 | phases exposed: R_k / acc@1 / acc@3 |
|---|---|---|
| 1 | 0.17 / 0.45 / 0.44 | 0.38 / 0.41 / 0.18 |
| 2 | 0.26 / 0.50 / 0.38 | 0.45 / 0.43 / 0.17 |
| 3 | 0.30 / 0.50 / 0.36 | 0.52 / 0.41 / 0.17 |

The seeded recipes are R≈0.30 predictable (0.50 one-step) — **below the 0.7 gate** (ISSUES C1). This is a property of the hand-written transition tables (v2 §4.11), not evidence about real agents (PBKV: 0.94 on real agent roles). Prediction rungs are not funded until E2 runs on real traces.

## 8. E3 — tails (H2/H3)

85% of tool time sits in the 3.8% of calls longer than a minute (target 92% / 4.9%); output tokens median 201, p99 5,906; think time median 82 s, p90 1,211 s. Reservations must be on quantiles. (The Hill index printed by the script is not a Pareto index on a capped mixture — F10.)

## 9. What the numbers say about the ladder

1. **Rung 1 (reactive gate) is the deliverable** for API-tier contention: it captures the whole failure-rate gain and everything the oracle can do there.
2. **Rung 3's case must be made on idle-time and sandbox/KV reservation**, where the oracle shows 30–40% relative improvements at high load — and only if the real society (a) shares sandboxes/KV across sessions and (b) has predictable idle time (Copilot says yes for coding agents). The synthetic think-time model is i.i.d. (ISSUES B7), so even that headroom cannot be *earned* by a learned predictor in this simulator yet.
3. **Rungs 4–4b have no measured target yet.** They require a society whose external tier actually binds (search/web-heavy recipes, lower RPM, higher background load) — legitimate scenario knobs (ISSUES A8) — but the conclusion must be reported as "headroom exists only where the environment has predictable slack".
4. The pinned configuration is the one place a gate *mechanism* (release across the tool boundary) matters independently of load.

## 10. Known limitations of this synthetic data (subset of v2 §4 that bites here)

- Recipes are hand-seeded (E2); reaction probabilities, provider buckets (120–300 RPM, 10–20% background load), idle timeout 300 s, client timeout 600 s, `p_parallel` are PRIORS; all set the numbers directly.
- One model instance, no routing/autoscaling; decode rate fixed at decode start; no semantic content; think time i.i.d.
- Coding-agent monoculture; the research recipe is a prior.
- Closed-population runs replace aborted sessions instantly; read requests·h⁻¹ *completed*, never sessions started (F13).

## 11. Phase 1 (2026-09-22, later the same day): the instrument, the knobs, and two corrections

Everything in §11–§12 is synthetic; all grids were regenerated on the final code after F21 (means moved by ≤ 0.005 failure; the §13.2 table was re-derived). New scripts: `rung0/compare.py` (paired statistics, oracle-normalised score), `rung0/e5_idle_predictability.py`; new knobs in the `generator` block of every scenario: `recipe_temperature` (A7), `think_snr` + `think_state_seed` (B7), `tool_tail_scale`, `p_parallel_scale` (A8); `agentsim sample-scenarios` with `scenarios/knob_ranges.json` (A8); `scenarios/hosted_stress.json` (B18); `fit --recipes-out` mines observable-phase recipes (C2). Selftest: 24 checks.

### 11.1 Two corrections to §4–§7

- **Statistics (A6).** With 5 seeds the exact two-sided Wilcoxon signed-rank test cannot go below p = 2/32 = 0.0625, so *nothing* in §4–§6 is distribution-free significant at α = 0.05, however large the gap. `compare.py` prints that floor, a paired-t p, bootstrap CIs, Cohen's dz and per-seed wins, and applies Holm across nine metric families on the chosen test (`--test t` for the 5-seed grids). Re-read with it: on E1 no gate-vs-oracle difference at any N; the uncoordinated deficit at N ≥ 20 is significant on every family (t, Holm). On the hosted society at 2 sessions·min⁻¹ the oracle's gains in failure rate, throughput, Jain and timeouts survive Holm (p_t 0.005 / 0.002 / 0.0001 / 0.001, 5/0/0 wins) — **but its −23% p99 does not** (p_t 0.067, 4 wins / 1 loss). §5's p99 claim is downgraded to "suggestive". Tables: `data/synthetic/compare/{e1,hosted}.md`.
- **E2 pooled runs by colliding session ids (F15).** `rung0/observer.load` concatenated files and grouped by `trace_id`, which restarts at `s000001` in every run, so §7 interleaved sessions from five seeds into 200 chimeric sequences (783 real ones). Fixed (ids namespaced by file; selftest guard). Corrected §7, hosted uncoordinated 1 session·min⁻¹, 5 seeds, 36 k nodes:

| context k | phases hidden: R_k / acc@1 / acc@3 | phases exposed: R_k / acc@1 / acc@3 |
|---|---|---|
| 1 | 0.23 / 0.48 / 0.34 (was 0.17 / 0.45 / 0.44) | 0.51 / 0.46 / 0.19 (was 0.38 / 0.41 / 0.18) |
| 2 | 0.36 / 0.53 / 0.37 (was 0.26 / 0.50 / 0.38) | 0.56 / 0.49 / 0.19 (was 0.45 / 0.43 / 0.17) |
| 3 | **0.40 / 0.54 / 0.37** (was 0.30 / 0.50 / 0.36) | **0.58 / 0.49 / 0.19** (was 0.52 / 0.41 / 0.17) |

  The seeded society is more predictable than reported, and still below the 0.7 one-step gate. The verdict of §7 stands; its numbers did not.

### 11.2 Predictability is now a knob (A7): E2 vs `recipe_temperature`

Hosted, uncoordinated, 5 seeds each, T sharpens (< 1) or flattens (> 1) every transition, tools-per-chat, tool-kind, retrieval and parallel lottery as p^(1/T) renormalised; T = 1 is bit-identical to the seeded rows.

| T | hidden R₁ / acc@1 | hidden R₃ / acc@1 | exposed R₃ / acc@1 / acc@3 |
|---|---|---|---|
| 0.1 | 0.42 / 0.69 | **0.81 / 0.89** | 0.87 / 0.90 / 0.76 |
| 0.3 | 0.31 / 0.56 | 0.63 / **0.74** | 0.76 / 0.73 / 0.40 |
| 0.5 | 0.27 / 0.52 | 0.51 / 0.65 | 0.67 / 0.63 / 0.30 |
| 1.0 | 0.23 / 0.48 | 0.40 / 0.54 | 0.58 / 0.49 / 0.19 |
| 2.0 | 0.22 / 0.46 | 0.35 / 0.50 | 0.54 / 0.41 / 0.13 |
| 4.0 | 0.22 / 0.48 | 0.34 / 0.51 | 0.54 / 0.37 / 0.13 |

The 0.7 one-step gate is crossed at T ≈ 0.3. Gains of prediction rungs are to be reported along this axis (v2 §4.11). Data: `data/synthetic/pred/e2_vs_temperature.tsv`.

### 11.3 Idle time can now be learned (B7): E5 vs `think_snr`

`log think = μ + σ·(√snr·z_state + √(1−snr)·ε)`, state = (recipe, last phase before the final chat, request-index bucket), offsets standardised within each (recipe, bucket) group under the recipe's analytic end-phase probabilities so the log-mean and log-sd of think time are preserved at every snr (selftest: 4.43 vs 4.43, 2.08 vs 2.10). The *quantiles* reshape — a few-state location mixture is not a lognormal: realised median / p90 over 6 seeds = 82 / 1,277 s at snr 0, 90 / 1,192 s at 0.5, 103 / 1,002 s at 0.9 — so cross-snr comparisons of absolute numbers carry that caveat; within an snr level every policy sees identical think times (CRN). The offsets are keyed by `think_state_seed` (the society's structure), not the run seed.

E5 (hosted uncoordinated, 5 seeds, ≈1,660 think spans, 24 states, trace-level 70/30 split):

| snr | R² of log think from state | pinball ratio state/global at τ = 0.5 / 0.8 / 0.9 (lower quantile) |
|---|---|---|
| 0.0 | −0.02 | 1.02 / 1.02 / 1.02 |
| 0.5 | 0.53 | 0.72 / 0.69 / 0.68 |
| 0.9 | 0.89 | 0.37 / 0.33 / 0.30 |

### 11.4 The headroom rung 3 must earn: oracle − gate vs `think_snr` (hosted, 5 seeds, paired-t, Holm across 9 families)

| load | snr | failure gate → oracle | p99 s | req·h⁻¹ | Jain | timeouts | significant after Holm |
|---|---|---|---|---|---|---|---|
| 1/min | 0.0 | 0.010 → 0.006 | 1,965 → 1,715 | 221 → 243 | 0.68 → 0.83 | 82 → 47 | Jain |
| 1/min | 0.5 | 0.011 → 0.006 | 2,058 → 1,629 | 220 → 250 | 0.64 → 0.82 | 94 → 46 | Jain |
| 1/min | 0.9 | 0.014 → 0.008 | 2,145 → 1,668 | 213 → 249 | 0.62 → 0.79 | 110 → 58 | all five |
| 2/min | 0.0 | 0.041 → 0.024 | 2,896 → 2,226 | 311 → 366 | 0.42 → 0.61 | 556 → 388 | failure, throughput, Jain, timeouts |
| 2/min | 0.5 | 0.037 → 0.025 | 2,742 → 2,410 | 309 → 364 | 0.41 → 0.60 | 570 → 416 | throughput, Jain, timeouts |
| 2/min | 0.9 | 0.041 → 0.028 | 2,887 → 2,428 | 305 → 369 | 0.42 → 0.59 | 577 → 420 | throughput, Jain, timeouts |

The oracle is clairvoyant, so its headroom is roughly flat in snr; what snr changes is how much of it a *learned* controller can capture — that is the rung-3 measurement (§12, pending). Uncoordinated equals the gate at every cell (no API-like tier binds). Tables: `data/synthetic/hosted_snr/compare_rate={1.0,2.0}.md`.

### 11.5 A society where the external tier binds (B18): `hosted_stress.json`

Search/web-heavy mix (0.3 / 0.3 / 0.4), `ext.search` and `ext.web` RPM tied and swept with 40% background load, `ext.api` 120 RPM, sandbox pool 64 GB / 12 CPU, 2 sessions·min⁻¹, 5 seeds. All PRIORS.

| RPM | uncoordinated: failure / 429s | gate: failure / 429s | oracle: failure | gate → oracle req·h⁻¹ | uncoordinated vs gate (t, Holm) |
|---|---|---|---|---|---|
| 10 | 0.121 / 682 | 0.087 / 319 | 0.024 | 225 → 333 | failure worse; p99, throughput, Jain, timeouts "better" (survivorship, F14) |
| 20 | 0.085 / 330 | 0.052 / 0 | 0.026 | 289 → 378 | failure worse |
| 40 | 0.054 / 1 | 0.053 / 0 | 0.026 | 296 → 380 | none (identical: the external tier no longer binds) |

At 10 RPM the gate itself takes 319 429s: header-based pausing at 10% remaining cannot see other tenants' calls (background load). The oracle's headroom here is both the external budget (its 429s are 0) and the sandbox pool — the target of rung-3 slice 3. Table: `data/synthetic/stress/compare.md`. `sample-scenarios` draws 18 knobs from `scenarios/knob_ranges.json`; 20 samples in `scenarios/sampled/` all run.

### 11.6 `fit` round trip on synthetic traces (C2)

Fitted from the 5-seed hosted uncoordinated traces: all 21 marginal entries (tool durations as 1–3-component lognormal mixtures by EM/BIC) and three observable-phase recipes (coding 8 phases, function_calling 4, research 5; replan chats folded into their predecessor so reactions are not double-counted). Regenerated with the fitted files, 5 seeds: hidden E2 R₃ 0.40 → 0.35, one-step acc 0.54 → 0.52; E3 share of tool time in calls > 60 s 85.1% → 80.9%, calls < 1 s 67.8% → 63.8%, mean 11.0 → 9.6 s; nodes 36 k → 24 k because `session_requests` is fitted on *completed* sessions only (censored ones are the long ones — the bias v2 §4.3 predicts). Files: `data/fitted/*.hosted_synthetic.json`, `data/synthetic/roundtrip/`.

## 12. Rung 3 — the `LeaseController`, its scoreboard, and what the ablation says it is worth

Built the same evening (tasks/plan.md, Phase 2–3): `policy.type=lease` = the reactive gate plus, from its own observation stream only (`on_event`; `predict.py`), (a) parking a sandbox when the predicted idle lower quantile exceeds a cold start and prewarming it at that quantile if the forecast interval is narrower than `h` (slice 1), (b) KV retention set to the predicted gap with Belady eviction (B6), (c) virtual-time fair queuing on a memory-centric cost (B8), (d) gang leases for declared parallel tool groups (slice 2, B17's target), (e) budget leases on provider call budgets with a worst-case-cost margin (slice 3). Two knobs, τ and h. Leases live in a ledger with I1 (Σ active ≤ capacity) and I3 (nothing past expiry) asserted; only the unconsumed part is kept from others (backfill by construction). The honesty selftest feeds the controller contradictory hidden values and asserts identical decisions. All grids below: 10 seeds (exact Wilcoxon floor 0.002), paired, Holm across six families; "score" = (lease − gate)/(oracle − gate).

### 12.1 Slice 1 scoreboard — hosted society, lease vs gate vs oracle (`data/synthetic/lease_snr`, τ = 0.8)

| load | snr | failure gate → lease (oracle) | p99 s | req·h⁻¹ | Jain | score: failure / p99 / thr / Jain | significant after Holm |
|---|---|---|---|---|---|---|---|
| 1/min | 0.0 | 0.008 → 0.004 (0.005) | 2,023 → 1,708 (1,660) | 222 → 239 (244) | 0.67 → 0.80 (0.82) | 1.00 / 1.04 / 0.80 / 0.83 | all four, 9–10 / 10 wins |
| 1/min | 0.5 | 0.010 → 0.004 (0.005) | 2,107 → 1,790 (1,655) | 222 → 248 (254) | 0.64 → 0.78 (0.81) | 1.02 / 0.71 / 0.80 / 0.79 | all four |
| 1/min | 0.9 | 0.012 → 0.005 (0.005) | 2,270 → 1,764 (1,794) | 219 → 250 (257) | 0.61 → 0.75 (0.79) | 1.00 / 0.98 / 0.81 / 0.77 | all four |
| 2/min | 0.0 | 0.042 → 0.013 (0.025) | 3,002 → 2,347 (2,438) | 302 → 347 (362) | 0.41 → 0.50 (0.59) | 1.63 / 0.93 / 0.79 / 0.47 | all four, 10 / 10 wins |
| 2/min | 0.5 | 0.042 → 0.013 (0.027) | 2,996 → 2,449 (2,544) | 306 → 348 (364) | 0.41 → 0.49 (0.59) | 2.00 / 0.92 / 0.67 / 0.49 | all four |
| 2/min | 0.9 | 0.045 → 0.015 (0.029) | 3,096 → 2,423 (2,565) | 300 → 342 (363) | 0.41 → 0.49 (0.58) | 1.72 / 1.12 / 0.71 / 0.49 | all four |

τ = 0.5 and 0.9 give the same picture within ±0.05 of score. Read naively this says "rung 3 captures 70–100% of the headroom and beats the oracle on failures" (the oracle's SRPT starves long requests into client timeouts; fair queuing does not). Two things in the table itself say otherwise: the gains are **the same at snr 0 as at 0.9** — they cannot come from idle-time *prediction* (E5: R² −0.02 at snr 0) — and p50 latency, absent from the table, is **+95%** (119 → 230 s at 2/min). Hence §12.2.

### 12.2 Ablation (E6) — what the gain is made of (`data/synthetic/lease_ablation`, 2/min, 10 seeds)

Gate at three sandbox idle timeouts; lease with features switched off (`policy.ablate`). "everything off" reproduces the gate at the same idle timeout exactly (rows identical) — the switch machinery is sound.

| snr | idle timeout | policy | failure | p50 s | p99 s | req·h⁻¹ | Jain | prewarm idle s |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 300 (the §5 baseline) | gate | 0.042 | 119 | 3,002 | 302 | 0.41 | — |
| 0.0 | 60 | gate | 0.023 | 168 | 2,586 | 333 | 0.48 | — |
| 0.0 | **0** | **gate** | **0.016** | 232 | 2,343 | 346 | 0.51 | — |
| 0.0 | any | lease, full | 0.013 | 230–236 | 2,347–2,442 | 345–347 | 0.50 | 3,077–3,181 |
| 0.0 | any | lease, park + prewarm off | 0.013–0.039 (= the gate's row) | | | | | 0 |
| 0.0 | any | lease, fair queuing off | 0.016–0.017 | 229–234 | 2,408–2,443 | 344–347 | 0.51 | 3,026–3,256 |
| 0.9 | 0 | gate | 0.019 | 228 | 2,454 | 344 | 0.49 | — |
| 0.9 | any | lease, full | 0.015–0.016 | 237–242 | 2,414–2,434 | 342–347 | 0.49 | 2,933–3,185 |

- **~90% of the lease's gain over the §5 gate is the gate's own idle-timeout knob.** Parking a sandbox the moment a session idles (idle timeout 0, cold start 3 s) takes the gate from 0.042 to 0.016 failure, 3,002 → 2,343 s p99, 302 → 346 req·h⁻¹ — and costs the same p50 (119 → 232 s) the lease pays. The 300 s default was a PRIOR (B3/F6); on this society the sandbox *pool* binds, so freeing it beats avoiding cold starts at every cold start tried (3, 30, 90 s; `data/synthetic/lease_coldstart`).
- **The rest is fair queuing, a fixed rule:** with VTFQ off the lease is the gate-at-0 (0.016–0.017); with it on, 0.013 — at no p50 cost.
- **The learned part buys nothing measurable.** Prewarm — the only feature that consumes the idle-time forecast — changes no metric at snr 0.9 versus 0, or versus the lease without it, at any cold start; it holds ≈3,000 sandbox-seconds per run warm for 18–21 hits at 3 s cold start and is used 0–1 times at 30–90 s (the forecast interval exceeds `h`). Predicted KV retention: KV does not bind here (≈10 evictions per run).

**Verdict for slice 1 (ladder §6.1, rung 3):** not shipped as a prediction rung. What the ablation supports shipping is rung 1 with the idle timeout at 0 (or the lease's "park when the predicted idle exceeds a cold start" rule, which reduces to it here) plus virtual-time fair queuing — both fixed rules. This is the rung-2 kill criterion firing after the fact: the "headroom" the oracle showed in §5/§11.4 was mostly a baseline knob, which the oracle used by knowing the think time and which a zero timeout uses by not needing to. The weak-baseline check in HANDOFF §4 was the right question; it should have been asked of the idle timeout before §5 was written.

### 12.3 Slice 2 — gang leases (`data/synthetic/lease_parallel`, 2/min, 10 seeds)

| p_parallel × | h | policy | groups run in parallel | failure | p50 s | leased CPU unit·s unused |
|---|---|---|---|---|---|---|
| 1 | 60 | gate / lease / oracle | 6.4% / **12.4%** / 10.9% | 0.042 / 0.013 / 0.025 | 119 / 241 / 116 | — / 74% / — |
| 3 | 60 | gate / lease / oracle | 5.8% / **10.3%** / 9.2% | 0.044 / 0.013 / 0.024 | 123 / 232 / 116 | — / 77% / — |
| 3 | 600 | gate / lease / oracle | 5.8% / 8.5% / 9.2% | 0.044 / 0.013 / 0.024 | 123 / 233 / 116 | — / 72% / — |

The mechanism works (parallel share +4.5 to +6 pp, 10/10 wins, p = 0.002, above the clairvoyant gate), but on this society fan-outs rarely find free units — the CPU pool is saturated by session units — so 72–77% of reserved unit-seconds are waited out, far above the ≤ 10% criterion. h = 60 beats 600. **Not shipped**; a real target only where the CPU pool has slack.

### 12.4 Slice 3 — budget leases on the stress society (`data/synthetic/stress_lease`, `stress_ablation`, 10 seeds)

Scoreboard against the §11.5 gate (idle timeout 300 s): at 10 RPM the lease issues **0 calls that 429** (gate 304) and cuts failure 0.080 → 0.051 (oracle 0.027; score 0.63); at 20 / 40 RPM failure 0.055 → 0.019 / 0.053 → 0.014, p99 −27% / −31%, throughput +21% / +20% (all 10/10 wins), p50 **+95% / +129%** (0/10), budget waste 2.7% / 1.2%. The ablation says what that is:

| RPM | idle timeout | gate | lease full | lease, budget off | lease, budget only (all else off) |
|---|---|---|---|---|---|
| 10 | 300 | 0.080 failure, 304 429s, 243 req·h⁻¹ | 0.051, 0, 256 | 0.052, 258, 255 | 0.087, 0, 220 |
| 10 | 0 | 0.047, 365, 288 | 0.049, 0, 253 | 0.046, 255, 259 | 0.052, 0, 256 |
| 20 | 300 | 0.055, 0, 290 | 0.019, 0, 352 | 0.016, 0, 353 | 0.055, 0, 293 |
| 20 | 0 | 0.019, 0, 355 | 0.018, 0, 352 | 0.016, 0, 353 | 0.020, 0, 356 |

- At 20 RPM the external tier does not bind for anyone; the whole gain is the sandbox idle-timeout knob (budget-only ≡ gate; budget-off ≡ full).
- At 10 RPM the budget mechanism does exactly what it promises — no 429s — and it **buys nothing**: failure 0.047 → 0.049–0.052 and throughput 288 → 253–256 versus the gate at idle 0, whose 365 429s are absorbed by SDK-default jittered retries. The worst-case-cost margin (2 units per call, because another tenant's call may land first) idles budget that retries would have used; the oracle gets 0.027 / 332 by knowing the provider's draw, which no gateway can. A margin calibrated to the *observed* collision rate (≈ the 0.4 background load) is the obvious next knob; it is not built.

**Verdict for slice 3:** not shipped. Correct mechanism, wrong price on this society; the honest comparison is against retries, not against 429 counts.

### 12.5 What rung 3 established, in one paragraph

A gateway that learns only from what it can see can be built, kept deadlock-free and invariant-checked, and made to look excellent on a scoreboard against the gate as configured in §5 — and an ablation against the gate's own knobs shows that on this synthetic society the learned predictions are not what pays. What pays is (i) parking sandboxes immediately (a knob), (ii) fair queuing (a fixed rule), and, where the CPU pool has slack, (iii) gang reservations for declared fan-outs. Idle-time prediction, predicted KV retention and budget reservations with an honest margin do not move any outcome metric beyond noise here. This is the rung-2 kill criterion, reached one rung late; the discipline that caught it (paired seeds, an oracle, and an ablation over the baseline's knobs) is the thing to keep. Whether a real society differs — costlier cold starts than 3–90 s, a CPU pool with slack, provider budgets that bind without background noise, idle gaps more predictable than the state a gateway can key on — is what rung 0 on real traces decides; the scripts and the knob ranges to ask it are in place.

Losses to carry into any real deployment of what does pay: p50 request latency roughly doubles when sandboxes are parked immediately (every request pays a cold start and re-acquisition queue), and 72–77% of gang reservations are waited out on a saturated pool.

## 13. Phase 5 (2026-09-22, later still): the shipped rules, a randomised sweep, and rung 0 on real traces

### 13.1 The shipped configuration, tested head-on (`data/synthetic/ship`, hosted, 10 seeds)

`ReactiveGate` gained a `queue` knob: `fifo`, `vtfq` (virtual-time fair queuing on the *realised* memory-centric cost of each step — no prediction) and `srpt` (§13.4). Shipped = gate + `vtfq` + sandbox idle timeout 0.

| load | variant | failure | p50 s | p99 s | req·h⁻¹ | Jain |
|---|---|---|---|---|---|---|
| 1/min | gate fifo, idle 300 (§5 baseline) | 0.008 | 122 | 2,023 | 222 | 0.67 |
| 1/min | **gate vtfq, idle 0 (shipped)** | **0.003** | 136 | **1,583** | 239 | 0.79 |
| 1/min | lease (full) | 0.004 | 138 | 1,708 | 239 | 0.80 |
| 1/min | clairvoyant | 0.005 | 112 | 1,660 | 244 | 0.82 |
| 2/min | gate fifo, idle 300 | 0.042 | 119 | 3,002 | 302 | 0.41 |
| 2/min | **gate vtfq, idle 0 (shipped)** | **0.013** | 244 | 2,408 | 346 | 0.50 |
| 2/min | lease (full) | 0.013 | 230 | 2,347 | 347 | 0.50 |
| 2/min | clairvoyant | 0.025 | 116 | 2,438 | 362 | 0.59 |

Shipped vs lease: no family differs at p < 0.05 on either load except p50 at 2/min (lease 230 vs 244 s, p 0.049 — its prewarm saves a few cold starts). The ablation of §12.2, tested directly, holds.

### 13.2 Slice 3 with a learned margin (`data/synthetic/stress_margin`, stress society at 10 RPM, everyone at idle 0 + vtfq, 10 seeds)

The gateway observes each call's cost from the provider's headers; the lease learns the collision rate p̂ (0.45 learned vs 0.40 background) and requires n + Binomial(n, p̂)-τ-quantile units instead of the worst case 2n.

| variant | failure | 429s | p50 s | p99 s | req·h⁻¹ | Jain | budget waste |
|---|---|---|---|---|---|---|---|
| gate vtfq, idle 0 | 0.049 | 252 | 34 | 3,336 | 255 | 0.34 | — |
| lease τ 0.5 | **0.037** (9/10, p 0.006) | **16** | 52 | 3,037 | **277** (+9%, 9/10, p 0.004) | 0.38 | 6% |
| lease τ 0.8 (2-unit margin) | 0.043 (n.s.) | 0 | 44 | 3,316 | 263 (n.s.) | 0.36 | 6% |
| clairvoyant | 0.027 | 0 | 59 | 2,905 | 332 | 0.47 | — |

−94% 429s, −25% failures and +9% throughput at τ = 0.5 (≈ 30% of the oracle's throughput headroom, ≈ 55% of its failure headroom); at τ = 0.8 the margin is still worst-case and the gains are inside noise. p50 is the loss column (34 → 52 s).

### 13.3 Where does prediction pay? A randomised sweep (`agentsim sweep`, 40 scenarios from `knob_ranges.json` × 5 variants × 5 seeds = 1,000 runs, `data/synthetic/sweep`)

Variants: the gate at each scenario's own sampled idle timeout (fifo); the shipped gate; the shipped gate with `srpt`; the lease; the oracle — the last three at idle 0. `rung0/headroom.py`:

| metric | scenarios with significant oracle headroom over the shipped gate (paired t, 5 seeds) | median relative headroom | lease's median share | `srpt`'s median share |
|---|---|---|---|---|
| failure | 3 / 40 | +16% | 0.08 | **0.99** |
| p99 | 4 / 40 | −1% | 0.06 | 0.15 |
| throughput | 19 / 40 | +4% | −0.14 | **0.50** |

Headroom rises with load (Spearman ρ = 0.66 with arrival rate, 0.35 with burstiness) and lives where the **model tier saturates**: in the two extreme scenarios (8 slots, 2–3.7 sessions·min⁻¹, model utilisation ≈ 1) the oracle's throughput is 4–5× the shipped gate's (118 → 586, 137 → 395 req·h⁻¹; failure 0.32 → 0.06, 0.47 → 0.15) and its p50 is 19 s against 388 — shortest-job-first on the model queue. The shipped VTFQ is *worse* than the sampled gate there (0.32 vs 0.16 failure): fair queuing helps where the sandbox pool binds and hurts where the model tier does. The recommendation is tier-conditional.

### 13.4 The one prediction that earns its keep: `queue = srpt`

Shortest *predicted* chat service first, from observable inputs only: `tokens_in / prefill_rate` (known exactly at request time) plus the running mean of *observed* `tokens_out` per recipe over `decode_base`; tools stay FIFO. On the model-saturated scenario it takes p50 530 → 21 s and throughput 132 → 385 in a 1 h run; across the sweep it matches the oracle on the model-bound scenarios (0004: 578 vs 586 req·h⁻¹, failure 0.057 vs 0.063; 0013: 358 vs 395; 0029: 623 vs 622) — Hermes's "a decent demand model gets within ~10% of an oracle", with a trivial demand model. Its losses: across the 40 scenarios it is never significantly worse than the shipped gate on p50, Jain or timeouts (13 / 12 / 13 better), better on throughput in 10 (worse in 1), and **worse on failure in 3** (model tier *not* saturated; long-context requests starve into aborts; median +3% relative). A queue-length-conditional SRPT / aging hybrid (HexAGenT-style) is the obvious refinement; not built. Gain vs generator entropy (`data/synthetic/pred_policies`, hosted 2/min, all at idle 0, 10 seeds): at T ≤ 0.3 (sharper recipes → longer requests → the model tier saturates even here) `srpt` matches the oracle (T = 0.1: failure 0.133 / p50 39 s / 178 req·h⁻¹ vs 0.133 / 39 / 181; shipped gate 0.163 / 247 / 131); at T ≥ 1 all gates are equivalent and the oracle's own SRPT costs it failures (0.026 vs 0.013).

### 13.5 Rung 0 on real traces — TraceLab (Zhu et al. 2026, CC BY 4.0), 5,312 Claude Code sessions

`rung0/tracelab_to_spans.py` converts the public trace (101 MB gzipped JSONL) into this repo's spans: 305,445 chat rounds, 302,011 tool calls (8,989 overlapping groups), 43,827 requests, 38,482 human gaps. Codex rounds (the other half of the trace) fail the timestamp sanity check under this mapping and are left out. Two "tools" that wait for the human (`AskUserQuestion`, `ExitPlanMode`) are dropped.

| Rung-0 script on the real trace | Result | The paper / the seeds |
|---|---|---|
| E3 tails | calls < 1 s: 74.6% of calls, 0.48% of tool time; calls > 60 s: 3.5% of calls, **88.8%** of tool time; mean 17.3 s | 70% / <1%; 4.9% / 92%; 16.8 s |
| E3 output tokens (incl. reasoning) | median 355, p99 7,159 | 252 / 6,571 (output only) |
| E3 think time (gap to the next user message) | median 221 s, p90 1,150 s | 84 s / 1,236 s |
| **E2 predictability, phases hidden (real)** | **k=3: R₃ 0.47, one-step 0.72, three-step 0.67**; k=1: 0.39 / 0.70 | seeds: 0.40 / 0.54 / 0.37 |
| **E5 idle predictability from gateway-visible state** (last tool kind, request bucket) | **R² 0.045**; pinball ratios 0.95–0.96 | synthetic snr knob 0–0.9 |

The real society crosses the 0.7 one-step gate — barely, and half of that is the tool→chat alternation (the majority class alone gives 0.48). Idle time is essentially unpredictable from what a gateway sees, which is what the synthetic ablation concluded from the other direction.

**Fitted society** (`fit --recipes-out` on the real spans → `data/fitted/*.tracelab.json`; `scenarios/hosted_tracelab.json`): 9 observable phases, p_parallel 0.33, tool durations as 3-component mixtures; it regenerates E2 at R₃ 0.48 / one-step 0.71 (real 0.47 / 0.72) with a lighter tool tail (67% vs 89% of time in calls > 60 s — the EM caps a tail whose maximum is days). `data/synthetic/tracelab`, 16 model slots, 10 seeds:

| load | gate fifo idle 300 | gate fifo idle 0 | gate vtfq idle 0 | gate srpt idle 0 | lease | clairvoyant |
|---|---|---|---|---|---|---|
| 1/min | 0.011 / 145 s | 0.010 / 111 | 0.010 / 110 | 0.010 / 111 | 0.010 / 115 | 0.010 / 105 |
| 2/min | 0.037 / 418 | 0.017 / 323 | 0.016 / 343 | 0.017 / 320 | 0.016 / 348 | 0.020 / 193 |
| 4/min | 0.137 / 766 | 0.088 / 702 | 0.069 / 1,060 | 0.089 / 689 | 0.070 / 1,021 | **0.103** / 316 |

(failure / p50 s.) Same picture on real structure: the idle timeout is the lever; fair queuing buys −20% failures at 4/min for +50% p50; `srpt` adds nothing because the model tier never saturates at these loads (utilisation ≤ 0.85); the lease equals fair queuing; the clairvoyant gate has the *worst* failure rate at 4/min (its SRPT starves long requests) and the best throughput — it is a bound on information, not on any one metric.

### 13.6 What Phase 5 changes in the verdict

Nothing in §12.5 is reversed; two things are added. (1) The shipped rule is tier-conditional: park immediately everywhere; fair-queue where the sandbox pool binds; **shortest-predicted-first where the model tier binds** — the one place a (trivial, observable) prediction captures most of an oracle's headroom, at a small failure cost where the tier is not saturated. (2) Real Claude Code traces are more structurally predictable than the seeds (one-step 0.72 vs 0.54) and their idle time is not predictable from a gateway's view (R² 0.05); the fitted society reproduces both and shows the same ranking of policies as the synthetic one. The remaining unknowns are the ones the trace cannot answer: real cold-start costs, real pool slack, real provider budgets, and whether real *multi-agent* societies (B10) look like single coding agents at all.

## 14. Phase 6, Task 20 — ceilings for the content heads, measured on TraceLab before training anything

`rung0/ceilings.py` (PREDICTOR_DESIGN §7, experiments 1–2). Claude Code half of TraceLab; session-level splits; numpy-only models (hashed n-grams → sparse multinomial logistic regression / ridge).

### 14.1 Head Q — tool duration from the command (120,000 Bash calls with a sanitised `command_skeleton`, 3,987 sessions, 70/30)

The skeleton keeps binary names and pipes, not arguments (`echo ; which ; python | head`), so this is a **lower bound** on what a gateway that sees the full command can do.

| What | Result | Baseline |
|---|---|---|
| duration class {<1 s, 1–10 s, 10–60 s, >60 s}, accuracy | 0.623 | majority 0.554 |
| log duration, ridge R² | **0.31** | 0 |
| tail (>60 s) by argmax: recall / precision | 0.17 / 0.53 — flagged calls hold 5% of tool time (true tail: 91%) | — |
| tail by **ranking** P(>60 s): top decile of calls | **30% long** (base rate 5%: 6× lift); captures 27% of all tool time; top half captures 84% | 10% / 50% |
| lease quantile τ = 0.9 per predicted-class (argmax) vs global | pinball ratio 0.96 | 1.00 |
| lease quantile τ = 0.9 / 0.95 per **P(long) decile** vs global | pinball ratio **0.92 / 0.92**, coverage 0.875 / 0.932 kept; reserves 146 / 290 s for the riskiest decile and 3.5 / 6.6 s for the safest | 1.00; one number for every call |

**Decision: build Head Q — as a risk ranker feeding per-bin quantiles, not as a classifier.** The body of the distribution is predictable from content (R² 0.31); the tail is not classifiable but is *rankable* (6× lift at the top decile), and a lease keyed on the predicted-risk bin is 8% better in pinball loss while allocating reservation where the risk is. Real commands should raise this; the trace cannot show by how much.

### 14.2 Head I — idle gap from who / when / what (38,482 gaps in 1,385 sessions, 70/30 by session, users shared)

| key set (per-key mean, log idle) | R² | pinball ratio, lower quantile, τ = 0.8 / 0.9 | states |
|---|---|---|---|
| last tool kind (E5 as before) | 0.004 | 1.01 / 0.92 | 8 |
| user | −0.07 | 1.15 / 0.94 | 38 |
| user, hour | −0.08 | 1.17 / 1.09 | 443 |
| user, hour, weekday, last tool, bucket | −0.05 | 1.18 / 1.21 | 1,348 |
| **previous gap** (log bucket) | **0.105** | **0.95 / 0.93** | 8 |
| user-message length (log bucket) | 0.001 | 1.01 / 1.00 | 6 |
| ridge on all of the above + user×hour | 0.068 | — | — |

**Decision: do not build Head I as a content/identity head.** Who and when carry no out-of-sample signal on this trace (per-key means overfit; the pooled ridge reaches 0.07); the only signal is autocorrelation with the previous gap (R² 0.10), which the existing tracker gets by adding `prev_think` to its key. TraceLab is sanitised, so the "what was said" axis is untested — the one reason to revisit with shadow traces. Consistent with §12.2 and §13.5: idle-time prediction is not where this problem's leverage is.

Not measured here (needs prompt/response text the trace does not carry): output-length ranking for `srpt` (§13.4) and P(final) — the next ceiling to take on shadow traces.
