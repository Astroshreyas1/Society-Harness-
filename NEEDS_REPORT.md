# NEEDS_REPORT — the Needs controller, the System One integration, budgets and the feedback loop (Phases 6–7)

*2026-09-22, second session. The only place with Phase 6–7 numbers. Every synthetic number is a mean over 10 paired seeds (common random numbers) unless marked "3 seeds" (exploratory ablations); "wins" are per-seed paired wins against the named baseline; statistics via `rung0/compare.py --policy-col variant` on the CSVs in `data/synthetic/needs/`. Real-trace numbers are session-level hold-outs. The design is PREDICTOR_DESIGN.md; the System One survey is JEV_SURVEY.md; the code map is HANDOFF §2.*

---

## 0. Verdict in one paragraph

The whole design was built, without the ladder's kill gate between rungs: a content-aware, calibrated multi-head predictor; a Reserver that forecasts every session's demand over a horizon and prices every reservation, park and prewarm by that forecast; family-owned gang leases for sub-agents; a System One (Jev) integration to the vendor's real API with a local calibrated stand-in, a live per-question calibration judge and decision judging; costed tools, MCP servers and GPU pools with tenant budgets; and a feedback loop that corrects the controller's own prediction error. On the societies the ladder had already settled — hosted, multi-agent, TraceLab-fitted, stress — the controller **equals the three shipped fixed rules** on every outcome metric (10 paired seeds, nothing significant after Holm), at every level of content signal, with or without the System One model: nothing measurable from the predictor, from content, from Jev features or from the leases. It differs in two regimes. Where the **model tier saturates** it chooses shortest-predicted-first by itself (the queue-length-conditional switch) and gets `srpt`'s outcome — failure 0.001 / 0.054 and throughput 463 / 294 req·h⁻¹ against the shipped gate's 0.046 / 0.192 and 117 / 133 (oracle 488 / 332) — so the tier-conditional recommendation of RUNG0 §13.6 no longer needs an operator to pick the rule. Where a **tenant budget binds** — the regime the earlier phases could not model — its forecast-aware pacing is the largest gain in the project: p50 −43 % and throughput +10 % (significant), zero refused payments in every seed, spend wasted on abandoned work 2.9 % → 1.6 %, 9 % cheaper completed requests, failure 0.035 → 0.019 (7/10 wins, not significant at n = 10); the loss column is client timeouts while paced work waits for the refill (the clairvoyant gate: 0.017 failure, +30 % throughput). Three of the controller's own defaults hurt and were found only by per-switch ablation. The predictor is real on real traces (tool-duration log R² 0.34 from command skeletons) and the local System One model is calibrated (ECE ≤ 0.05); neither moves an outcome metric on these societies.

## 1. What was built (Phases 6–7)

| Piece | Where | What |
|---|---|---|
| Content cues | `workload.py` (`content_snr`) | every tool call carries a skeleton class cue, every chat a prompt cue and a plan cue, true with probability `content_snr`; spans identical at any value (regression against every stored run) |
| Multi-agent | `workload.py`, `engine.py` (`spawn` / `join`), `data/recipes_multiagent.json`, `scenarios/multiagent.json` | orchestrator turns launch 2–6 sub-agents (own recipe, sandbox, model slot, one request); the parent joins; the join is a hold-and-wait edge the cycle probe sees (102 cycles / h under the gate at idle 300; 0 when the parent's sandbox is parked) |
| Observer | `features.py` | one code path online (`on_event`) and offline (`replay_events`, same same-instant order as the engine); typed records at chat/tool submission, chat end, request end, spawn; labels emitted when outcomes realise |
| Predictor | `needs.py` | heads S2/S3 (next nodes), Q_tool, Q_out, T_gap, I_gap (5 monotone quantile knots), I_cold, R, G; temperature scaling per categorical head; adaptive conformal per (head, key); online (0.1 ms/update) and offline (`train-needs`) |
| Reserver v2 | `reserver.py` (`policy.type=needs`) | demand forecast per resource over *h*; expected-value gate (latency-seconds saved vs hold × pressure) for park / prewarm / gang / spawn / predicted-spawn leases; SRPT only when the model queue exceeds the slots, VTFQ otherwise; predicted KV reserve and retention; forecast-aware budget pacing |
| System One | `jev.py` (JEV_SURVEY.md) | catalogue of 21 typed questions (choice / score / noul) with instructions and criteria to the vendor's rules; speculative fan-out at every boundary; `LocalSystemOne` (trained, temperature-calibrated), `RemoteSystemOne` (exact API, retries, cost), `Judge` (live ECE per question; kill rule at 0.1), `JevChannel` (`ext.jev`: concurrency, RPM, latency, batching) |
| Budgets and costs | `resources.py`, `engine.py`, `scenarios/mcp_budget.json` | per-call cost / tokens on `ext.*`, `mcp.*`, `gpu.*`; model prices per Mtok; tenant `BudgetBucket`s refilling hourly; the provider refuses what a budget cannot pay (outcome `budget`); spend reconciles to the cent |
| Feedback loop | `control.py` | every 30 s: realised vs forecast occupancy per resource → PI forecast scale; budget floor by AIMD on refusals; per-session fairness weights on VTFQ tags |
| Offline | `train.py` | `train-needs`, `train-jev`, `evaluate-jev [--remote]`, `annotate` (Jev phase labels onto chat spans; `rung0/e2_predictability.py --phase-attr`) |
| Tests | `agentsim selftest` | 81 checks; honesty (contradictory hidden values → identical decisions; a source grep for hidden-field reads); every switch off ≡ the fifo gate bit-for-bit; I1/I3; determinism; calibration; the vendor payload against a canned response |

## 2. The predictor, offline

`train-needs` on replayed traces (session-level hold-out). Synthetic: three 2-hour hosted runs at `content_snr` 0.6 (45 k records). Real: 1,500 TraceLab Claude Code sessions (287 k records, 300 sessions held out), content = the sanitised command skeleton and input-size bucket the trace carries.

| Head | Synthetic hosted (snr 0.6) | Real TraceLab | Reads |
|---|---|---|---|
| Q_tool — log tool duration | R² 0.58; pinball at τ 0.8: 0.70 of the constant quantile; coverage 0.85 | **R² 0.34; pinball 0.81; coverage 0.88** | the skeleton, the kind, the context: real commands predict a third of log-duration variance (the ceilings experiment's ridge got 0.31 on Bash alone) |
| T_gap — log seconds to the next chat | R² 0.51; pinball 0.70 | **R² 0.21; pinball 0.78** | how long the revealed tool calls will keep the model idle — the KV-retention quantity |
| Q_out — log output tokens | R² 0.44 | R² 0.05 | on real traces only the user-message size bucket is available; no response text |
| S2 / S3 — the two nodes after the revealed ones | S2 ≈ majority (0.73); S3 0.40 vs 0.17 | S2 ≈ majority (0.88); S3 ≈ majority (0.45) | horizon-2 structure is not learnable from these features on real data |
| I_gap / I_cold — idle | none (R² < 0) | none | as every earlier phase found |
| G — spawn width | 1.0 (multi-agent society: 0.98 vs 0.98 majority) | — | spawning is not predictable from the cues at chat submission |

Label efficiency on the real trace (PREDICTOR_DESIGN §7: 300 / 1,000 / 3,000 sessions, the first N of the file, 20 % of sessions held out): Q_tool log R² 0.32 / 0.28 / 0.45 (pinball at τ 0.8: 0.63 / 0.74 / 0.66), T_gap 0.26 / 0.09 / 0.24, Q_out 0.20 / −0.34 / 0.20. The hold-out variance (60 / 200 / 600 sessions, heavy-tailed users) dominates any trend: the content head is learnable from a few hundred sessions and does not improve monotonically with more, on this trace. A session-stratified, repeated split would be the right instrument; not run.

## 3. The System One model, offline and live

`train-jev`: one sparse multinomial per question over hashed n-grams of the state, log-loss trained, temperature-calibrated on held-out sessions. Accuracy vs majority and ECE per question:

| Question (type) | hosted (snr 0.6) | multi-agent | MCP / budget |
|---|---|---|---|
| phase (choice, offline annotation) | 0.75 vs 0.19, ECE 0.030 | 0.71 vs 0.27, 0.021 | 0.69 vs 0.35, 0.046 |
| duration_class (choice) / tail_risk (score) | 0.86 vs 0.69, 0.021 | 0.86 vs 0.77, 0.008 | 0.84 vs 0.68, 0.021 |
| out_class (choice) | 0.71 vs 0.36, 0.036 | 0.72 vs 0.39, 0.026 | 0.64 vs 0.46, 0.065 |
| next_chat (choice) | 0.79 vs 0.70, 0.019 | 0.82 vs 0.76, 0.024 | 0.86 vs 0.83, 0.047 |
| next_tool (choice) | 0.57 vs 0.30, 0.025 | 0.52 vs 0.24, 0.020 | 0.30 vs 0.18, 0.063 |
| tier_next (choice) | 0.93 vs 0.76, 0.014 | 0.97 vs 0.88, 0.004 | 1.00 (trivial) |
| remaining_work (score) | 0.52 vs 0.38, 0.027 (MAE 0.60 levels) | 0.49 vs 0.38, 0.025 | 0.45 vs 0.39, 0.049 |
| is_final_turn (noul) | 0.94 vs 0.87, 0.004 | 0.92 vs 0.87, 0.007 | 0.92 vs 0.88, 0.024 |
| cost_class (choice) | trivial (no costs) | trivial | **0.79 vs 0.41, 0.030** |
| idle_class, safe_to_park (request end) | no signal (T ≈ 4: flat, honest) | no signal | no signal |
| fail_soon, risk, budget_will_exceed, stuck_in_loop | base rates ≈ 0 in gate traces: trivially right | | |

Every question with signal is calibrated below the 0.1 gate; the ones without signal say so through a flat distribution rather than a confident wrong answer. Offline annotation (point A): on a held-out hosted trace the Jev phases recover R₃ 0.57 of the 0.61 the true phases carry (0.41 with phases hidden), at 72 % phase accuracy.

**Live judging.** The `Judge` scores every delivered answer against the label the Observer later emits. A model trained where content is informative (snr 0.6) and run where it is noise (snr 0) shows `duration_class` ECE 0.27 within minutes and is dropped — the design's kill rule, per question, online. On the hosted grid with matched content the live ECEs stay under 0.1 and every question remains asked.

## 4. Per-switch ablations — what the controller's defaults got wrong

Three-seed ablations on three societies (`policy.ablate=<switch>`), run before the grids. Each row is the controller with one switch off.

**Hosted, 2/min (first build).** gate vtfq/0: failure 0.0097. needs: **0.0156**. needs−aging: **0.0097**. Every other switch: 0.012–0.014. The starvation guard aged waiters by the *retry's* ready time: a request that had timed out once re-entered the queue "young", starved behind aged waiters and exhausted its retries (fewer timeouts, more aborts). Aged by first submission: needs 0.0096. Then: needs−forecast 0.0120, needs−ev 0.0109, needs−gang 0.0103, needs−kv 0.0091 — the predicted KV reserve added evictions (7 vs 1) where KV never binds; now used only while KV admission has waiters.

**Multi-agent, 1.5/min.** gate vtfq/0: 0.020. needs (first build): **0.108**, throughput 120 vs 207. The spawn gang reserved k model slots for the whole join; a child holds a slot only while it chats, so the lease re-blocked everyone at every release for ten minutes. With the lease limited to the children's *start* window: 0.028 / 173; with the gang priced like the parallel gang (issue iff pressure × k < 1): never issued here, 0.021 / 194 = the gate. Gang priority for sub-agents pays only where the tier has slack; on this society it does not.

**Model-saturated sampled society 0013, 3 seeds.** gate srpt/0: failure 0.067, throughput 299. needs: 0.074 / 198 with the guard, **0.075 / 280** without it — the guard prevented no failure and cost 30 % throughput. It is now opt-in (`policy.needs.starvation_guard`). The SRPT switch depth (queue > 0 / 0.25 / 0.5 / 1.0 × slots) made no difference (198–210).

The lesson of RUNG0_REPORT §12 one level up: a controller's own defaults are baseline knobs too.

## 5. Grids (10 seeds, paired; `rung0/needs_tables.py`)

### 5.1 Hosted society, 2/min, content_snr 0 and 0.9 (`hosted_content.csv`)

`scenarios/hosted_mixed.json` at 2 sessions·min⁻¹, 2 h, 10 seeds; the content axis is the generator's `content_snr` (0: every cue is noise; 0.9: nine in ten are the truth). The first run of this grid (13 variants including the per-switch ablations, `hosted_content_v1.csv`, starvation guard on) gave the same picture within noise.

**content_snr = 0.0** (n = 10 paired seeds; * = significant after Holm vs `gate_vtfq0`; wins = per-seed paired wins on failure / p50 / p99 / throughput)

| variant | failure | p50 s | p99 s | req·h⁻¹ | timeouts | Jain | wins |
|---|---|---|---|---|---|---|---|
| gate_fifo300 | 0.042* | 119* | 3,002* | 302* | 596* | 0.41* | 0/10/0/0 of 10 |
| gate_vtfq0 | 0.013 | 241 | 2,342 | 347 | 381 | 0.50 |  |
| gate_srpt0 | 0.016 | 214* | 2,459 | 350 | 321* | 0.54* | 2/8/6/7 of 10 |
| lease0 | 0.013 | 235 | 2,383 | 348 | 381 | 0.50 | 5/6/4/5 of 10 |
| needs | 0.014 | 241 | 2,418 | 344 | 373 | 0.50 | 5/4/4/5 of 10 |
| needs_pre | 0.013 | 241 | 2,408 | 348 | 380 | 0.51 | 7/6/6/7 of 10 |
| needs_jev | 0.012 | 235 | 2,394 | 347 | 387 | 0.51 | 7/7/4/6 of 10 |
| oracle0 | 0.026* | 116* | 2,494 | 360* | 416* | 0.59* | 2/10/6/9 of 10 |

**content_snr = 0.9** (n = 10 paired seeds; * = significant after Holm vs `gate_vtfq0`; wins = per-seed paired wins on failure / p50 / p99 / throughput)

| variant | failure | p50 s | p99 s | req·h⁻¹ | timeouts | Jain | wins |
|---|---|---|---|---|---|---|---|
| gate_fifo300 | 0.042* | 119* | 3,002* | 302* | 596* | 0.41* | 0/10/0/0 of 10 |
| gate_vtfq0 | 0.013 | 241 | 2,342 | 347 | 381 | 0.50 |  |
| gate_srpt0 | 0.016 | 214* | 2,459 | 350 | 321* | 0.54* | 2/8/6/7 of 10 |
| lease0 | 0.013 | 235 | 2,383 | 348 | 381 | 0.50 | 5/6/4/5 of 10 |
| needs | 0.013 | 238 | 2,477* | 346 | 386 | 0.51 | 6/6/1/6 of 10 |
| needs_pre | 0.013 | 239 | 2,356 | 346 | 384 | 0.51 | 7/5/5/5 of 10 |
| needs_jev | 0.013 | 239 | 2,363 | 346 | 396 | 0.51 | 6/4/5/5 of 10 |
| oracle0 | 0.026* | 116* | 2,494 | 360* | 416* | 0.59* | 2/10/6/9 of 10 |

At either content level the controller is the shipped gate: failure 0.013–0.014 (gate 0.013; rung 3 0.013), p50 238–241 s, p99 within noise, throughput 344–348 (gate 347); no family differs after Holm. Content changes nothing, because nothing here is limited by knowing a tool's duration or an output's length: the sandbox pool binds, and parking at once plus fair queuing is the whole answer (RUNG0 §12–13). `srpt` (unconditional) buys p50 and fairness here at a failure cost (0.016) the conditional switch avoids. The oracle's p50 and throughput come with a higher failure rate (its SRPT).

### 5.2 Multi-agent society, 1.0 and 1.5/min (`multiagent.csv`)

`scenarios/multiagent.json`: the hosted society with 30 % orchestrators whose `delegate` turns launch 2–6 coding sub-agents (each a session with its own sandbox and model slot) and join them. `content_snr` 0.6, 2 h, 10 seeds. Models for `needs_pre` / `needs_jev` trained on this society's own gate traces.

**rate_per_min = 1.0** (n = 10 paired seeds; * = significant after Holm vs `gate_vtfq0`; wins = per-seed paired wins on failure / p50 / p99 / throughput)

| variant | failure | p50 s | p99 s | req·h⁻¹ | timeouts | Jain | children aborted | join p99 s | wins |
|---|---|---|---|---|---|---|---|---|---|
| gate_fifo300 | 0.018* | 127* | 3,032 | 162* | 270* | 0.47* | 6.9* | 3,720 | 0/10/6/2 of 10 |
| gate_vtfq0 | 0.007 | 204 | 3,059 | 178 | 180 | 0.57 | 3.5 | 3,565 |  |
| gate_srpt0 | 0.008 | 192 | 2,993 | 179 | 163 | 0.57 | 3.9 | 3,653 | 5/7/6/5 of 10 |
| lease0 | 0.007 | 184 | 3,109 | 176 | 183 | 0.56 | 3.3 | 3,659* | 3/8/4/2 of 10 |
| needs | 0.006 | 190 | 3,079 | 178 | 184 | 0.56 | 3.5 | 3,618 | 3/6/5/3 of 10 |
| needs_pre | 0.008 | 202 | 2,977 | 179 | 179 | 0.57 | 3.7 | 3,609 | 2/5/6/4 of 10 |
| needs_jev | 0.008 | 197 | 3,047 | 177 | 185 | 0.57 | 3.4 | 3,584 | 3/7/5/4 of 10 |
| needs-content | 0.007 | 192 | 3,008 | 178 | 172 | 0.57 | 3.6 | 3,640 | 4/6/6/4 of 10 |
| needs-ev | 0.007 | 215 | 2,919 | 161* | 207 | 0.50* | 2.3 | 3,500 | 4/3/7/0 of 10 |
| needs-forecast | 0.007 | 231* | 3,009 | 169* | 198 | 0.53 | 2.6 | 3,604 | 3/0/5/1 of 10 |
| needs-conformal | 0.007 | 198 | 3,033 | 178 | 182 | 0.57 | 3.7 | 3,645 | 3/6/5/5 of 10 |
| needs-learn_pre | 0.007 | 202 | 3,027 | 178 | 184 | 0.57 | 3.5 | 3,589 | 4/4/3/5 of 10 |
| oracle0 | 0.012 | 112* | 2,824 | 186* | 200 | 0.62* | 5.1 | 3,620 | 4/10/5/9 of 10 |

**rate_per_min = 1.5** (n = 10 paired seeds; * = significant after Holm vs `gate_vtfq0`; wins = per-seed paired wins on failure / p50 / p99 / throughput)

| variant | failure | p50 s | p99 s | req·h⁻¹ | timeouts | Jain | children aborted | join p99 s | wins |
|---|---|---|---|---|---|---|---|---|---|
| gate_fifo300 | 0.039* | 97* | 3,462 | 192* | 662* | 0.34* | 13.4* | 4,330* | 0/10/5/1 of 10 |
| gate_vtfq0 | 0.021 | 240 | 3,517 | 207 | 528 | 0.41 | 7.3 | 3,733 |  |
| gate_srpt0 | 0.025 | 241 | 3,542 | 211 | 444* | 0.42* | 8.0 | 3,739 | 1/5/6/9 of 10 |
| lease0 | 0.022 | 255 | 3,640 | 204 | 534 | 0.41 | 7.6 | 3,991 | 4/1/5/4 of 10 |
| needs | 0.019 | 252 | 3,507 | 206 | 529 | 0.41 | 6.6 | 3,776 | 7/2/2/5 of 10 |
| needs_pre | 0.020 | 252 | 3,475 | 205 | 541 | 0.40 | 7.1 | 3,908 | 6/5/4/4 of 10 |
| needs_jev | 0.022 | 257 | 3,555 | 206 | 545 | 0.41 | 6.6 | 4,091 | 3/3/4/4 of 10 |
| needs-content | 0.020 | 246 | 3,439 | 207 | 535 | 0.41 | 7.4 | 3,979 | 5/4/5/5 of 10 |
| needs-ev | 0.028 | 233 | 3,661 | 181* | 573* | 0.35* | 6.1 | 3,989 | 2/8/3/0 of 10 |
| needs-forecast | 0.022 | 313* | 3,528 | 199* | 566 | 0.38* | 6.9 | 3,877 | 3/1/3/1 of 10 |
| needs-conformal | 0.020 | 250 | 3,374 | 206 | 540 | 0.41 | 6.8 | 3,709 | 6/4/6/4 of 10 |
| needs-learn_pre | 0.022 | 253 | 3,410 | 205 | 537 | 0.41 | 7.1 | 3,675 | 5/4/6/4 of 10 |
| oracle0 | 0.031* | 93* | 3,431 | 227* | 534 | 0.48* | 13.2* | 4,139 | 1/10/6/10 of 10 |

The controller equals the shipped gate on failure, p50, p99, throughput, fairness and the sub-agent columns at both loads (no family differs after Holm; the join p99 is within noise). The two switches that matter are *protective*: with the demand forecast off, p50 +41 s / +73 s and throughput −5 % / −4 % (significant) — leases and parks priced by the standing queue alone are issued when they should not be; with the expected-value gate off, throughput −10 % / −13 % and Jain −0.07 (significant). Content, the pre-trained predictor, Jev features, conformal calibration and online learning change nothing. The spawn gang (family-owned k slots + k sandboxes) is priced by pressure and is never issued at these loads: on a saturated model tier a priority for sub-agents is zero-sum (§4). The clairvoyant gate's SRPT buys p50 (112 / 93 s) and throughput (+4 % / +10 %) at a higher failure rate (0.012 / 0.031) — the same bound-on-information, not on any one metric, as RUNG0_REPORT §13.5.

### 5.3 Model-saturated societies 0004 and 0013 (`saturated.csv`) — the tier-conditional rule, chosen automatically

The two knob-sampled societies of RUNG0_REPORT §13.3 whose model tier saturates (8 slots, 2–3.7 sessions·min⁻¹). 1 h, `content_snr` 0.6, 10 seeds. Models for `needs_pre` / `needs_jev` are the hosted ones (no society-specific training).

**scenario = scenario_0004** (n = 10 paired seeds; * = significant after Holm vs `gate_vtfq0`; wins = per-seed paired wins on failure / p50 / p99 / throughput)

| variant | failure | p50 s | p99 s | req·h⁻¹ | timeouts | Jain | wins |
|---|---|---|---|---|---|---|---|
| gate_fifo300 | 0.006* | 680* | 2,217 | 129 | 184* | 0.07 | 9/0/4/7 of 10 |
| gate_vtfq0 | 0.046 | 314 | 2,037 | 117 | 398 | 0.07 |  |
| gate_srpt0 | 0.002* | 18* | 1,595* | 471* | 269* | 0.29* | 9/10/8/10 of 10 |
| lease0 | 0.033 | 323 | 2,287 | 128 | 361* | 0.08 | 7/4/3/8 of 10 |
| needs | 0.001* | 18* | 1,933 | 463* | 286* | 0.28* | 9/10/5/10 of 10 |
| needs_pre | 0.002* | 19* | 1,757 | 467* | 282* | 0.29* | 9/10/7/10 of 10 |
| needs_jev | 0.003* | 18* | 1,961 | 464* | 289* | 0.28* | 9/10/6/10 of 10 |
| needs-content | 0.002* | 18* | 1,834 | 462* | 287* | 0.28* | 9/10/6/10 of 10 |
| needs-ev | 0.002* | 19* | 1,795* | 463* | 282* | 0.28* | 9/10/8/10 of 10 |
| needs-forecast | 0.002* | 18* | 1,721 | 459* | 291* | 0.28* | 9/10/7/10 of 10 |
| needs-conformal | 0.002* | 19* | 1,604 | 463* | 288* | 0.28* | 9/10/8/10 of 10 |
| needs-learn_pre | 0.006* | 18* | 1,727 | 477* | 287* | 0.30* | 9/10/8/10 of 10 |
| oracle0 | 0.004* | 18* | 1,409* | 488* | 290* | 0.32* | 9/10/9/10 of 10 |

**scenario = scenario_0013** (n = 10 paired seeds; * = significant after Holm vs `gate_vtfq0`; wins = per-seed paired wins on failure / p50 / p99 / throughput)

| variant | failure | p50 s | p99 s | req·h⁻¹ | timeouts | Jain | wins |
|---|---|---|---|---|---|---|---|
| gate_fifo300 | 0.097* | 538* | 2,120* | 139 | 311* | 0.16 | 10/0/0/6 of 10 |
| gate_vtfq0 | 0.192 | 298 | 1,756 | 133 | 686 | 0.14 |  |
| gate_srpt0 | 0.057* | 22* | 1,592 | 301* | 498* | 0.39* | 10/10/7/10 of 10 |
| lease0 | 0.157* | 330 | 1,966 | 144 | 621* | 0.16 | 9/3/3/7 of 10 |
| needs | 0.054* | 23* | 1,863 | 294* | 524* | 0.36* | 10/10/3/10 of 10 |
| needs_pre | 0.052* | 23* | 1,776 | 299* | 513* | 0.36* | 10/10/4/10 of 10 |
| needs_jev | 0.054* | 23* | 1,747 | 300* | 517* | 0.37* | 10/10/5/10 of 10 |
| needs-content | 0.059* | 24* | 1,846 | 294* | 525* | 0.36* | 10/10/4/10 of 10 |
| needs-ev | 0.059* | 23* | 1,791 | 294* | 522* | 0.36* | 10/10/4/10 of 10 |
| needs-forecast | 0.053* | 24* | 1,758 | 292* | 516* | 0.36* | 10/10/3/10 of 10 |
| needs-conformal | 0.061* | 23* | 1,867 | 295* | 522* | 0.36* | 10/10/3/10 of 10 |
| needs-learn_pre | 0.061* | 24* | 1,847 | 312* | 503* | 0.38* | 10/10/3/10 of 10 |
| oracle0 | 0.052* | 26* | 1,833 | 332* | 471* | 0.43* | 10/10/2/10 of 10 |

Here the shipped `vtfq` gate is the wrong rule (RUNG0 §13.3: fair queuing hurts where the model tier binds) and `srpt` the right one. The controller is not told which society it is on: its ordering switches to shortest-predicted-first when the model queue exceeds the slot count and back to fair queuing when it does not, and the outcome is `srpt`'s — failure 0.001 / 0.054 (srpt 0.002 / 0.057; oracle 0.004 / 0.052), p50 18 / 23 s, throughput 463 / 294 req·h⁻¹ (srpt 471 / 301; oracle 488 / 332), every family significant against the shipped gate at 10/10 or 9/10 wins. Its predicted service time is the conformal median of the output head instead of the gate's running mean; on these societies that changes nothing (`needs-conformal`, `needs-content`, `needs-learn_pre` are within noise), and neither do the forecast, the EV gate or Jev. With the starvation guard on (the first build) the same controller had 30 % less throughput here (§4); RUNG0 §13.4's queue-length-conditional switch removes `srpt`'s off-saturation failure cost on the hosted society (§5.1) without giving up its gain here.

### 5.4 TraceLab-fitted society, 2 and 4/min (`tracelab.csv`) — real structure

`scenarios/hosted_tracelab.json`: recipes and marginals fitted from 5,312 real Claude Code sessions (RUNG0 §13.5), 16 model slots, 2 h, `content_snr` 0.6, 10 seeds. The System One model is trained on this society's own traces (its phases are the fitted recipes' observable ones).

**rate_per_min = 2.0** (n = 10 paired seeds; * = significant after Holm vs `gate_vtfq0`; wins = per-seed paired wins on failure / p50 / p99 / throughput)

| variant | failure | p50 s | p99 s | req·h⁻¹ | timeouts | Jain | wins |
|---|---|---|---|---|---|---|---|
| gate_fifo300 | 0.037* | 418* | 2,408* | 204* | 366* | 0.44* | 1/1/1/0 of 10 |
| gate_vtfq0 | 0.016 | 343 | 1,838 | 243 | 198 | 0.61 |  |
| gate_srpt0 | 0.017 | 320* | 1,851 | 242 | 184 | 0.61 | 3/9/7/3 of 10 |
| lease0 | 0.016 | 348 | 1,861 | 240* | 208 | 0.59* | 3/3/3/0 of 10 |
| needs | 0.016 | 347 | 1,821 | 241 | 203 | 0.60 | 2/5/4/2 of 10 |
| needs_pre | 0.017 | 357 | 1,821 | 242 | 206 | 0.60* | 1/3/3/3 of 10 |
| needs_jev | 0.017 | 344 | 1,859 | 242 | 201 | 0.60 | 1/5/4/2 of 10 |
| oracle0 | 0.020 | 193* | 1,866 | 251 | 231* | 0.67* | 3/9/5/7 of 10 |

**rate_per_min = 4.0** (n = 10 paired seeds; * = significant after Holm vs `gate_vtfq0`; wins = per-seed paired wins on failure / p50 / p99 / throughput)

| variant | failure | p50 s | p99 s | req·h⁻¹ | timeouts | Jain | wins |
|---|---|---|---|---|---|---|---|
| gate_fifo300 | 0.137* | 766* | 3,598* | 249* | 1,472* | 0.20* | 0/8/0/0 of 10 |
| gate_vtfq0 | 0.069 | 1,060 | 2,600 | 280 | 1,210 | 0.27 |  |
| gate_srpt0 | 0.089* | 689* | 2,809 | 286 | 1,068* | 0.29* | 0/10/3/6 of 10 |
| lease0 | 0.070 | 1,021 | 2,626 | 279 | 1,211 | 0.27 | 4/7/4/3 of 10 |
| needs | 0.070 | 1,066 | 2,588 | 277 | 1,236 | 0.27 | 3/5/5/2 of 10 |
| needs_pre | 0.071 | 1,047 | 2,667 | 278 | 1,239 | 0.27 | 4/5/6/3 of 10 |
| needs_jev | 0.071 | 1,050 | 2,705 | 277 | 1,230 | 0.27 | 4/4/3/4 of 10 |
| oracle0 | 0.103* | 316* | 3,105 | 297* | 1,245 | 0.37* | 0/10/2/9 of 10 |

On real structure the controller is the shipped gate at both loads (failure 0.016 / 0.070 vs 0.016 / 0.069; every family within noise). The model tier does not saturate here (utilisation ≤ 0.85 at 4/min), so the conditional switch never fires, and `srpt` costs failures (0.089) for its p50; the oracle's SRPT costs even more (0.103). Content, the pre-trained predictor and the System One model change nothing.

### 5.5 Stress society, 10 RPM (`stress.csv`) — the provider-budget-bound society

`scenarios/hosted_stress.json` with the search / web providers at 10 RPM (RUNG0 §11.5, §12.4, §13.2). 2 h, `content_snr` 0.6, 10 seeds; τ = 0.8 (the scenario default). The first run of this grid crashed on the controller's budget leases (ISSUES F24, fixed).

**rpm = 10** (n = 10 paired seeds; * = significant after Holm vs `gate_vtfq0`; wins = per-seed paired wins on failure / p50 / p99 / throughput)

| variant | failure | p50 s | p99 s | req·h⁻¹ | timeouts | Jain | wins |
|---|---|---|---|---|---|---|---|
| gate_fifo300 | 0.080* | 32 | 3,590 | 244 | 706 | 0.33 | 0/2/3/2 of 10 |
| gate_vtfq0 | 0.049 | 34 | 3,336 | 255 | 658 | 0.34 |  |
| gate_srpt0 | 0.049 | 54* | 3,023 | 282* | 501* | 0.39* | 7/0/7/10 of 10 |
| lease0 | 0.043 | 44 | 3,316 | 263 | 670 | 0.36 | 8/2/6/7 of 10 |
| needs | 0.047 | 46* | 3,217 | 266 | 687 | 0.36 | 5/1/5/8 of 10 |
| needs_pre | 0.045 | 44 | 3,392 | 265 | 705 | 0.36 | 6/2/5/6 of 10 |
| needs_jev | 0.044 | 45 | 3,232 | 262 | 695 | 0.36 | 6/2/6/6 of 10 |
| oracle0 | 0.027* | 59* | 2,905* | 332* | 423* | 0.47* | 9/1/9/10 of 10 |

The controller equals the shipped gate (failure 0.047 vs 0.049; nothing significant except a +12 s p50). Its provider-budget leases are rung 3's (calibrated collision margin) and at τ 0.8 they carry the worst-case margin RUNG0 §13.2 showed to buy nothing; at τ 0.5 rung 3 reached 0.037 there — the setting is the scenario's, not the controller's, and was not re-swept. `srpt` buys throughput (+11 %) at a p50 cost; the oracle's headroom (0.027, +30 % throughput) is knowing the provider's draw, which no gateway can.

### 5.6 The budget society (`mcp_budget.csv`) — where forecasting pays

`scenarios/mcp_budget.json`: 10 closed-loop coding agents on an API-tier model ($3 / $15 per Mtok), tools on two MCP servers and a GPU pool with per-call costs, a tenant budget of $10 refilling at $25/hour against ~$46/hour of unmetered demand. 2 h, `content_snr` 0.6, 10 seeds.

| variant | failure | p50 s | p99 s | req·h⁻¹ | refused payments | client timeouts | $ per completed request | $ wasted on abandoned work | Jain |
|---|---|---|---|---|---|---|---|---|---|
| uncoordinated | 0.982 | 52 | 454 | 42 | 40,370 | 8 | 0.650 | 75.5 % | 0.02 |
| gate fifo, idle 300 (spend pause at 10 %) | 0.038 | 333 | 2,455 | 47 | 27 | 20 | 0.567 | 3.0 % | 0.71 |
| gate vtfq, idle 0 (baseline) | 0.035 | 327 | 2,483 | 46 | 23 | 21 | 0.565 | 2.9 % | 0.71 |
| **needs** | **0.019** | **186** | 2,508 | **51** | **0** | 48 | **0.516** | **1.6 %** | 0.73 |
| needs + Jev (judged pacing) | 0.018 | 182 | 2,663 | 52 | 0 | 47 | 0.511 | 1.3 % | 0.73 |
| needs, pacing off (the gate's pause) | 0.032 | 316 | 2,335 | 48 | 26 | 21 | 0.542 | 2.7 % | 0.73 |
| needs, feedback off (fixed 10 % floor) | 0.022 | 239 | 2,683 | 48 | 0 | 44 | 0.533 | 1.8 % | 0.68 |
| clairvoyant (knows every cost and the level) | 0.017 | 178 | 2,489 | 60 | 0 | 41 | 0.439 | 1.4 % | 0.77 |

Paired against the vtfq gate (Holm across six families, exact Wilcoxon, n = 10): **p50 −141 s [−188, −96], 10/10 wins, p 0.002** and **throughput +4.8 req·h⁻¹ [+2.0, +7.7], 8/10** (needs + Jev: +5.5, significant after Holm) are the significant gains; failure −0.015 [−0.030, +0.001], 7/10 wins, p 0.10 — the direction is consistent, the 10-seed test is not conclusive; refused payments 23 → 0 in every seed; the loss column is **client timeouts +26 [+21, +32], 0/10**: when the budget is the bottleneck someone must wait, and the pacer makes in-flight chats wait for the refill (600 s client timeout) where the gate lets them be refused. Oracle-normalised: failure 0.94 of the clairvoyant's headroom, p50 0.95, throughput 0.20–0.31 — the oracle uses the last cent because it knows every cost; the pacer keeps a reserve.

Pacing is the mechanism (pacing off = the gate); the feedback loop's floor adaptation adds the rest (feedback off: failure 0.022, p50 239). Jev's supervisory questions (`budget_will_exceed`, `cost_class`, `remaining_work` in the spend forecast) add nothing significant on top: the trackers already know the costs here.

## 6. What the design got wrong, and what was learned

- **The starvation guard** (PREDICTOR_DESIGN §5.4, "aging") is not a free safety net: aged by the wrong clock it *causes* aborts; aged correctly it costs throughput where SRPT is doing its job. Off by default; the client timeout and VTFQ-when-shallow bound starvation on every society tested.
- **Gang leases for sub-agents** (§5.2) must cover the start, not the lifetime, and must be priced: a lease is a priority, and a priority on a saturated tier is zero-sum at best.
- **A smaller predicted KV reserve** helps only where KV admission binds; elsewhere it is eviction risk for nothing.
- **Content, conformal calibration, the demand forecast and Jev features** change no outcome on the settled societies, at any `content_snr`. The mechanisms that pay — SRPT on a saturated model tier, pacing on a binding budget, parking idle sandboxes — need one number each, and the predictor supplies that number; the rest of the distribution is unused there.
- **Budgets are where forecasting earns its keep**: a pacer must know what in-flight work still needs, and what the refill will bring while it finishes; the header pause knows neither and pays in refusals and aborted work.
- **The data**: enough for these mechanism-level results and relative comparisons under stated priors; not for real-world claims about costs, MCP servers, sub-agents or a System One model reading text. Shadow OTel traces from the target society are the next input (HANDOFF §9).

## 7. Reproduction

Models: `data/models/{needs,jev}.{hosted,multiagent,mcp}.npz` (+ `.report.json`), `data/models/needs.tracelab.npz`. Grids: the `sweep` commands in README (variants in `scenarios/variants_needs*.json`); statistics: `python rung0/compare.py data/synthetic/needs/<grid>.csv --axis <axis> --policy-col variant --baseline gate_vtfq0 --oracle oracle0`. Ablations: `policy.ablate=` with any of vtfq park prewarm kv gang budget spawn srpt aging content jev conformal ev forecast learn feedback pacing.
