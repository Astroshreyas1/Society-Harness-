# The Needs Predictor — design (v3 notes)

*Runtime service-graph reservation for multi-agent systems: an AI controller that predicts the future tool, model and service dependencies of concurrent agentic workflows and dynamically reserves resources without over-allocation, starvation or workflow failure.*

*Status 2026-09-22 (second session): built — `agentsim/features.py` (§2), `agentsim/needs.py` (§3–§4), `agentsim/reserver.py` (§5), `agentsim/jev.py` (§2 D, to the vendor's real API: JEV_SURVEY.md), `agentsim/train.py` (§7–§9); evidence in NEEDS_REPORT.md. Deviations from this design that the evidence forced are listed there (§ "what the design got wrong").*

This document designs the **Predictor** — the learned half of the controller — and the **Reserver logic that consumes it**. It replaces the placeholder in `agentsim/predict.py` (per-key streaming quantiles and a first-order Markov chain), which was built to test whether prediction could pay at all. It could, in one place (RUNG0_REPORT §13.4); everywhere else the placeholder was too crude to know. This design is what a serious attempt looks like, what it must beat, and how it is killed if it does not.

Companion notes: [SERVICE_GRAPH_RESERVATION.md](SERVICE_GRAPH_RESERVATION.md) §6 (the controller's components and invariants), [DEEP_DIVE_AND_LADDER.md](DEEP_DIVE_AND_LADDER.md) §1.4 (what has been predicted, from what, how well), §2 (Jev), §6 (the ladder). Evidence: [RUNG0_REPORT.md](RUNG0_REPORT.md).

---

## 0. What the evidence constrains

Five facts from this repo's measurements shape the design; a predictor that ignores them will be measured against them and lose.

| Fact | Where | Consequence for the design |
|---|---|---|
| The *next* step's structure is largely **revealed, not predicted**: the LLM response lists its tool calls; the framework declares fan-out; the gateway proxies both | engine `on_event`, RUNG0_REPORT §12.3 | Horizon-1 structure needs no model. The model's job at horizon 1 is **quantity from content**; structure prediction begins at horizon 2 |
| Tool time is in the tail: 3.5% of calls hold 89% of it, and *which* calls are long is decided by the call's content (`npm test` vs `ls`) | §13.5 (real) | Per-kind marginals cannot see the tail. Duration must be predicted from the call's **content** |
| Idle time is unpredictable from structural state (R² 0.05 real) but the trace carries user id and clock time, and the last exchange carries intent | §13.5; TraceLab fields `user`, timestamps | Idle prediction needs **who / when / what was said**, not "last tool kind" |
| Where the model tier saturates, ordering chats by predicted service time captures ~all of an oracle's headroom; the mean-per-recipe estimate leaves ~2% throughput and all of the p50 gap | §13.3–13.4 | Output-length prediction from **content** is the highest-value quantity head |
| Real next-step predictability is 0.72 one-step / 0.67 three-step on structural tokens alone; half of that is the tool→chat alternation | §13.5 | Structure at horizon ≥ 2 is learnable but modest; the model must **carry calibrated uncertainty** into the lease, not a point guess |

And one architectural fact: the gateway sits on the request/response path, so it sees prompts, responses (including tool-use blocks and the model's stated plan), tool arguments and tool outputs. "Structured program state + unstructured context" is exactly what is available at the seat — no new data path is needed.

---

## 1. The prediction problem, stated precisely

At every **decision point** of workflow *w* (a step boundary: chat end, tool end, request end, request start) at time *t*, produce for the next *K* steps and the next *H* seconds:

1. **Structure** `S_{w,k}`: a distribution over the node type of step *k* ∈ {1..K} — `chat`, `tool:<kind>`, `retrieval`, `spawn(n)`, `final`, `idle` — including fan-out size for tool groups and spawn width for sub-agents. At *k* = 1 this is a point mass on what was revealed.
2. **Quantity** `D_{w,k,res}`: for each candidate node type, the demand on each (tier, resource) as a *distribution*: tool duration; chat prefill tokens (exact), output tokens, KV footprint; sandbox memory; provider call-budget units; retrieval size.
3. **Timing** `T_{w,k}`: when step *k* starts (cumulative), as a distribution.
4. **Idle** `I_w`: at request end, `P(idle ≥ cold start)` and the idle-gap distribution.
5. **Risk** `R_{w,H}`: `P(429 / timeout / abort within H)` under the *forecast* occupancy — the failure-within-horizon head (v2 §1.9).

Every output is a distribution with a **calibration guarantee** (coverage of the τ-quantile ≈ τ on the live stream). The Reserver never receives a point estimate.

---

## 2. Features — what the gateway can legitimately see

Three feature groups, all computable in the Observer from spans and proxied bodies; nothing hidden (the honesty selftest extends to the model: its inputs are a typed feature record, never a `Step`).

**A. Structural (from spans; already in `on_event`)**
node-type history (last *k* tokens), counts (chats/tools so far in this request; requests in this session), request index bucket, session age, agent/recipe type, loop signatures (same tool kind × *n* in a row; edit→test cycles), the revealed next-step list at chat end, occupancy and queue length on each resource at *t*, provider headers (remaining budget), time of day / weekday, tenant/user pseudonym.

**B. Content (from proxied bodies) — the new signal**
- **Tool-call arguments**: for shell calls the *command skeleton* (binary names, flags, `test`/`build`/`install`/`pytest`/`npm`/`cargo`/`docker` tokens, redirections, pipes, argument count); for file tools the path depth and extension; for web/search tools the query length. Represented as **hashed n-grams over the skeleton** (cheap, deterministic, no vocabulary drift) plus a handful of explicit indicators.
- **Model response text**: the plan statement that precedes tool calls ("let me run the test suite", "I'll search for…", "that completes the task") — hashed n-grams over the first/last 200 tokens; the number and kinds of tool-use blocks; the `stop_reason`.
- **Tool outputs**: error indicators (non-zero exit, traceback, "not found"), output size — predictors of *retry* and *replan* (the reaction model, v2 §4.1).
- **User message**: length, question marks, closing phrases, task-type indicators — predictors of idle and of request length.

**C. Population priors (from the crude trackers)**
The current per-key streaming quantiles (duration by kind, output by recipe, idle by state) are *fed in as features*, so the model can learn when to defer to them. They are also the conformal residual pool (§4). The placeholder is not thrown away; it becomes the prior.

**Optional D. Typed annotations from a System One model (Jev)**
At step boundaries, asynchronously, a typed question per decision point (v2 §2.3): `{phase ∈ {plan, explore, edit, verify, finalize}, next_op_class, duration_class ∈ {ms, s, min, long}, p_final, p_idle_long, p_retry}` with calibrated probabilities. These enter as categorical features with their confidences; the model learns how much to trust them; the ECE gate (§7) decides whether they stay. Jev is never on the admission path; its 70–500 ms latency is reserved like any tool.

---

## 3. The model — a small multi-head sequence model with distributional outputs

Size is a design goal, not a constraint to fight: PBKV's structure predictor is ~350 K parameters; Decima's scheduler 12.7 K. The admission path budget is ≤ 5 ms; inference happens at step boundaries (tens per second per gateway), not per token.

```
   per-step record r_i = [ structural one-hots ⊕ hashed content n-grams (2^14 dims, sparse)
                           ⊕ env features ⊕ prior-quantile features ⊕ (Jev typed answers ⊕ confidences) ]
                                          │
                 embedding (linear, 128) → GRU over the last k = 8 steps (hidden 128)  ┐
                 revealed next-step list  → set encoder (mean of embeddings)           ├─ concat → MLP(256) = z
                 occupancy forecast (§5)  → 8 scalars                                  ┘
                                          │
      ┌──────────────┬────────────────────┼─────────────────────┬──────────────────┬──────────────┐
   Head S            Head Q               Head T                Head I             Head R         Head G
   structure k=2..K  quantities           timing                idle               risk           gang / spawn
   softmax per k     quantile regression  quantile regression   P(idle≥cold),      P(fail in H)   fan-out size,
   over node types   τ∈{.5,.8,.9,.95}     of step-k start       idle quantiles     (BCE)          spawn width
   (cross-entropy)   per node type        (pinball)             (BCE + pinball)                   (softmax)
                     (pinball)
```

Design choices that matter:

- **Quantile heads, not means.** Reservations are on quantiles (H2/H3); the pinball loss trains the quantiles the lease will actually use. Monotonicity across τ is enforced by predicting the median and non-negative increments.
- **Mixture semantics for uncertain structure.** For *k* ≥ 2 the demand on a resource is the mixture Σ_j P(type_j)·D_j. The Reserver takes the τ-quantile of the *mixture* (§5) — if the model is unsure whether a shell call or a file read comes next, the lease covers both at level τ, and uncertainty shows up as a wider reservation, not a branch.
- **Content enters through hashing.** Hashed n-grams over command skeletons and short text windows need no tokenizer, no vocabulary maintenance, and no PII beyond what the gateway already proxies; they train on TraceLab's `command_skeleton` today.
- **The revealed next-step list is an input, not a target**, at horizon 1. The model predicts *quantities* for revealed steps and *structure* only beyond them.
- **Spawn/join.** Head G predicts sub-agent spawn width from content ("launch three research agents in parallel") and the join wait from the children's predicted completion — the gang need of the multi-agent case (B10), which the generator must emit before this head can be trained (§8).

Training: synthetic spans with labels free (the generator knows every future); real spans (TraceLab: structure, durations, tokens, idle gaps, `command_skeleton`, `user`, timestamps); synthetic pre-training → real fine-tuning; label-efficiency curve at 300 / 1 k / 3 k real sessions (v2 §6.1 rung 4). Implementation in numpy (a two-layer MLP + GRU with Adam is ~300 lines and keeps "numpy only"), with an optional PyTorch path for the same architecture if training on the full trace is slow.

---

## 4. Calibration — the guarantee that makes τ mean something

A learned quantile is a guess until it is calibrated on the live stream. Two layers:

1. **Adaptive conformal inference per (resource, node type)**: keep the last *n* residuals of the model's τ-quantile against realised values; adjust the quantile level online so that the empirical miss rate tracks 1 − τ (step size γ; non-stationarity is expected — model versions, prompt updates, compaction change the society in weeks, v2 §4.8). The Reserver reserves the *conformalised* quantile. Over-allocation is thereby bounded by the coverage target, whatever the model's quality: a bad model produces wide, honest intervals, not confident wrong ones.
2. **Calibration of the categorical heads** (structure, idle, risk): temperature scaling on a rolling window; expected calibration error reported per head; the drift monitor raises a retrain when ECE > 0.05 or coverage drifts > 3 points.

This is also how the *crude* trackers remain in the loop: they are the residual pools. The model narrows the intervals where it has learned something; where it has not, the intervals are the trackers'.

---

## 5. The Reserver logic that consumes the predictions

The reservation rule stays fixed and small; what changes is that it now runs on distributions over a horizon and on a forecast of everyone else's demand.

**5.1 Occupancy forecast.** At each decision point the Reserver sums, per resource, every live workflow's predicted demand curve over the next *H* seconds: for workflow *w*, step *k*, resource *r*: demand `q_τ(D_{w,k,r})` over the interval `[q_lo(T_{w,k}), q_hi(T_{w,k}) + q_τ(dur)]`, weighted by `P(S_{w,k} = type)`. The result is a **service-graph demand horizon** — the thing the problem statement asks for: the future tool, model and service dependencies of all concurrent workflows, as a time-indexed distribution per resource.

**5.2 Lease sizing (two knobs, τ and h).** `lease(w, k, r) = (q_τ of the mixture demand, start = q_lo(T_k), expiry = start + h)`, with `h = min(h_steps, h_time)`; DRF-capped at the workflow's fair share plus slack; acquired in global tier order; gang for multi-component steps. Unchanged from v1 §6.2 except that the quantile is conformalised and the structure is a mixture.

**5.3 Expected-value gating — when to reserve at all.** A lease is issued only when its expected benefit exceeds its expected cost, both computed from the same distributions:

```
benefit(w,k,r) = P(step k needs r) · E[ queueing delay avoided | forecast occupancy of r at T_k ]
cost(w,k,r)    = E[ (q_τ − actual)⁺ ] · h  ·  price(r)          (the backfill-adjusted waste)
issue iff benefit > cost
```

`price(r)` is the resource's scarcity — the forecast utilisation of *r* over the horizon (free resources cost nothing to over-reserve; saturated ones cost a lot). This is what removes the failure mode the ablation exposed (RUNG0_REPORT §12.2: prewarming held sandboxes warm for nothing): a prewarm is issued only when `P(idle ≥ cold) · cold_start_saved > E[warm hold] · price(sandbox)`. No new knob — τ and h remain the only ones; the prices are measured.

**5.4 Model-tier ordering.** Chats are ordered by the conformalised median of predicted service time (`tokens_in / prefill + q_0.5(output) / decode`) **when the model queue exceeds its slots**, and by virtual finish time (fair queuing) otherwise — the queue-length-conditional switch that removes SRPT's off-saturation failure cost (§13.4). KV admission uses `q_τ(output)` instead of a fixed output reserve (SIC-safe: admission never exceeds forecast active KV).

**5.5 Predictive shaping.** When Head R says `P(429 within H)` on a provider exceeds the risk level, the gate spreads issuance over the horizon (a token-bucket shaped to the forecast) instead of waiting for headers to hit 10% — HiveMind's pause made predictive and budget-aware (§13.2's learned collision rate is the degenerate case).

**5.6 Guarantees, unchanged.** I1–I6 of v1 §6.6 hold as before: Σ leases ≤ capacity (the forecast is *advisory*, the ledger is the constraint), unconditional expiry, global order + all-or-nothing gang acquisition (no hold-and-wait), virtual-time fair queuing (bounded starvation), DRF cap (no workflow reserves beyond its share). Prediction quality affects *waste and latency*, never safety.

---

## 6. Why this beats "a separate model that predicts the proposed plan and re-samples"

It *is* that, with three differences that the evidence forced:

- **Plans are revealed one step at a time**, and the first step is already known at each boundary. A plan-level predictor that guesses step 1 is throwing away free information; this design uses the revealed step and predicts from step 2 with calibrated uncertainty.
- **Re-sampling is causal successor release.** Every completed step is an observation; the ledger's replace-on-event semantics re-issue the workflow's whole forecast at each boundary. That *is* dynamic re-planning — without a separate planning loop that could disagree with the ledger.
- **Distributions, not sampled plans.** Sampling *N* plans and reserving for the union over-allocates by construction; reserving for one sampled plan fails when it is wrong. The τ-quantile of the mixture reserves exactly the amount whose coverage the operator chose, and conformal calibration makes that choice true on the live stream.

Jev fits as the featurizer of §2 D: fast, typed, calibrated answers to the questions that need semantics (phase, intent, duration class, "is this the end?"), consumed as features with confidences. It is not the predictor, because the predictor's outputs are quantities on resources over time — a regression the gateway can train on its own spans — and because a vendor model on the admission path would violate the latency budget and the "no path bypasses the gate" rule.

---

## 7. Evaluation, and how it dies

The machinery exists: paired 10-seed grids, the clairvoyant oracle, `compare.py`, `headroom.py`, the ablation switches, and a real-trace society.

**Ablation ladder (E6):** none → crude trackers (today) → model, structural features only → + content → + Jev annotations → oracle. Per metric, oracle-normalised score on the 40-society sweep, the TraceLab-fitted society, and the stress society; gain-vs-load, gain-vs-`recipe_temperature`, gain-vs-`content_snr` (§8) curves.

**Kill criteria (v2 §6.1 rungs 4/4b, unchanged):** ship a head only if its Δ over the crude tracker on outcome metrics is significant at 10 seeds and > 2× simulator error; kill Jev features if ECE > 0.1 on real traces or no outcome gain at equal data. Predictor accuracy is logged, never optimised as an objective (H3).

**Two experiments available now, before any model is trained**, that bound what content can buy — *run 2026-09-22, RUNG0_REPORT §14: Head Q is worth building as a risk ranker (log-duration R² 0.31, 6× tail lift in the top decile, 8% better lease quantiles on sanitised skeletons); Head I is not (who/when R² ≤ 0.07; only the previous gap carries signal, R² 0.10)*:
1. TraceLab `command_skeleton` → duration class (≈150 k timed shell calls): how much of the tail is predictable from the command? This is the ceiling for Head Q on tools.
2. TraceLab `(user, hour, weekday, last exchange length, request count)` → idle gap: how far above R² 0.05 does who/when take idle prediction? This is the ceiling for Head I.

---

## 8. What the generator must grow first

The synthetic society cannot train content heads it does not emit. Two additions, both knobs in the A8 sense:

- **Content proxies with a `content_snr` knob**: each tool call carries a latent *task class* (e.g. `run-tests`, `install`, `list`, `edit-small`) that determines its duration distribution, and emits a noisy skeleton token the way `think_snr` emits a state; `content_snr` sets how much of the duration variance the skeleton explains. Same for chats: a latent "remaining-work" class that sets output length and P(final), with a noisy plan token. Gain is then reported as a curve over `content_snr`, with the real society's value (from experiment 7.1) marked.
- **Multi-agent spawn/join recipes (B10)**: a `spawn(n)` node that creates *n* child programs sharing the parent's session and a join wait — so Head G has something to predict and the gang reservation has its hardest case.

---

## 9. Build order (proposed Phase 6)

1. Experiments 7.1 and 7.2 on TraceLab — one day; they decide whether Heads Q and I are worth building.
2. Generator content proxies + `content_snr`; spawn/join recipes.
3. `agentsim/features.py` (typed feature records from the observation stream) and the numpy model with heads S, Q, T, I, R; conformal layer; drift monitors.
4. Reserver v2: occupancy forecast, mixture-quantile leases, expected-value gating, queue-length-conditional ordering, predictive shaping — behind `policy.type = needs`.
5. E6 ablation on the three societies; label-efficiency curve; ship/kill per §7.
6. Jev annotator behind a flag, ECE-gated.
