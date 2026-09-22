# Implementation Plan: Control — from rung 0/1 evidence to a measurable rung 3 (synthetic)

> **Status 2026-09-22 (after Phase 5):** all 19 tasks done; Phase 5 added the shipped rules, a randomised sweep (`srpt` is the one prediction that pays, where the model tier binds) and rung 0 on real TraceLab traces (§13). Earlier status: all 13 tasks done; outcomes and findings per task in [todo.md](todo.md); numbers in RUNG0_REPORT §11–§12. Headline: rung 3 was built and beats the configured gate everywhere, and the ablation shows the gain is ~90% the gate's idle-timeout PRIOR plus fair queuing — the prediction part does not pay on the synthetic society (ISSUES F20). Six further defects found and fixed on the way (F15–F20). The two decisions that were the user's remain theirs: real traces (C1/E3) and git (E1).

Written 2026-09-22 after reading README, HANDOFF, ISSUES, RUNG0_REPORT, DEEP_DIVE_AND_LADDER §1.9/§3–§6 and SERVICE_GRAPH_RESERVATION §6, and reading `agentsim/` + `rung0/` in full. Task list: [tasks/todo.md](todo.md). Issue ids refer to [ISSUES.md](../ISSUES.md).

## Diagnosis — what is completed (verified, not just claimed)

| Claim in the docs | Verified how | Result |
|---|---|---|
| Simulator exists, 7-check selftest passes | `uv run --python 3.12 --with numpy python -m agentsim selftest` | 7/7 PASS, 1.5 s |
| 120 synthetic runs, 5 seeds/cell | `ls data/synthetic/{e1,hosted,hosted_pinned}` | 60 + 45 + 15 run dirs, `grid_summary.csv` in each |
| RUNG0_REPORT §4 (E1) and §5 (hosted) numbers | aggregated the CSVs over seeds (mean ± 95% CI) in the scratchpad | every cell matches the report to the printed digit |
| All P0 ISSUES closed except C1/E3 (blocked on real data) | read ISSUES §A–§F | consistent; F1–F11 closed, F12–F14 open P1 |
| Policies: Uncoordinated / ReactiveGate / ClairvoyantGate; interface has `priority`, `idle_timeout`, `prewarm_lead`, AIMD hooks (B2) | `agentsim/policies.py` | as documented |
| Structural mechanisms B3 (idle timeout), B4 (provider buckets + headers), B5 (parallel groups), B17 (sequential fallback), F5 (client timeout) | `agentsim/engine.py`, `resources.py` | present, one place each |

**Environment facts that differ from the docs**
- README says "Python 3.12, numpy only" and uses bare `python`. On this machine there is no `python`; `/usr/bin/python3` is 3.9 without numpy; a uv-managed 3.12 exists. `uv run --python 3.12 --with numpy python -m agentsim …` works. There is no `pyproject.toml`.
- `agentsim/`, `rung0/`, `scenarios/`, `data/` directories are `r-x` (no write bit): existing files are editable but **no new file can be created** in them until `chmod u+w`. This blocks `rung0/compare.py`, `agentsim/predict.py`, new scenarios.
- The previous session ran on Windows (`data\synthetic\hosted\grid_summary.csv` in the logs); nothing depends on it.
- HANDOFF §4 says "run the checklist in memory/feedback" — no such memory exists in this project's memory dir. The checklist itself is in HANDOFF §4; the pointer should be fixed.
- No git repository (ISSUES E1, needs the user's go-ahead). No `tasks/` existed before this plan.

## Diagnosis — what is lacking, and why it is the next work

The ladder's honest state (HANDOFF §3, RUNG0_REPORT §9): rung 1 (reactive gate) is the deliverable on the API tier; the *only measured headroom* for prediction is the oracle's idle-sandbox parking/prewarm + SRPT on the hosted society (4.1% → 2.4% failures, −23% p99, +18% throughput at 2 sessions·min⁻¹). Rung 3 must earn that headroom with a real (non-clairvoyant) controller. Three things stand in the way, all in the simulator, none needing real data:

1. **Think time is i.i.d.** (B7) — the headroom exists but *cannot be learned* by any predictor in this simulator. Without B7, a LeaseController cannot be evaluated at all.
2. **No paired statistics / oracle-normalised score** (A6) — every "gain" so far is eyeballed from means ± CI; rung 3's exit criterion ("captures ≥ 50% of headroom with waste ≤ 10%") is literally undefined without it.
3. **The hosted society never binds an API-like tier** (B18) and knobs are never randomised (A8); the predictability of the generator is fixed at R≈0.30 (A7) — so gains cannot be reported as curves over load / R / knobs, which §6.0 makes mandatory.

Everything in HANDOFF §7 item 1 (real traces, C1/E3) is blocked on a user decision and is *not* in this plan's critical path; item 6 (Jev) is conditional on item 1. Items 2–5 are exactly phases 1–3 below, reordered by dependency: A6 first because it is the instrument every later task reports through.

## Architecture decisions (kept from HANDOFF §8; new ones marked NEW)

- Only the Predictor is learned; the reservation rule is fixed with knobs τ (quantile) and h (horizon). No fallback branches.
- The oracle is the clairvoyant gate; headroom = oracle − reactive gate; rung 3 is scored as `(lease − reactive)/(oracle − reactive)` per metric, per seed (paired).
- Success is judged on outcomes, never on predictor accuracy.
- NEW: **The LeaseController learns only from what a gateway can observe**: request start/end times (→ realised think time), step start/end, step kind, recipe name, request index. It never reads `Step.duration`, `tokens_out`, the sampled `think`, or provider internals. Enforced by a selftest that feeds the policy contradictory hidden values and asserts identical decisions.
- NEW: **B7 preserves the fitted marginal.** Structured think time decomposes the lognormal variance into a state-determined share `think_snr` and an i.i.d. share `1 − think_snr`; `think_snr = 0` reproduces today's runs bit-for-bit (CRN preserved: same number of draws from `rng_steps`).
- NEW: **Rung 3 is sliced by reservation target, not by component**: slice 1 = idle-time (sandbox park/prewarm, KV retention TTL, fair queuing) — the target the oracle actually showed; slice 2 = gang leases for parallel groups' extra CPU (the B17 target); slice 3 = reserve-ahead on external RPM budgets, only if the stress society (B18) shows a binding API-like tier.
- NEW: numpy-only statistics (exact Wilcoxon signed-rank by enumeration for n ≤ 20 seeds, bootstrap CIs, Cohen's dz, Holm–Bonferroni) — no scipy, keeping "numpy only" true.
- Repository hygiene: add `pyproject.toml` so `uv run python -m agentsim …` works; do not `git init` without the user's answer (E1).

## Dependency graph

```
T1 env (chmod, pyproject)
 └─ T2 compare.py (A6)  ──────────────────────────────────────────┐
 ├─ T3 recipe_temperature (A7)  ─ E2 sweep                        │
 ├─ T4 structured think time (B7) + E5 idle predictability ───┐   │  all report through T2
 ├─ T5 stress scenario + sample-scenarios (B18, A8)            │   │
 ├─ T6 extend fit (C2), round-trip on synthetic                │   │
 │                                                             ▼   │
 ├─ T7 retention_ttl hook (B6)  ─┐                                 │
 ├─ T8 predict.py (online quantile + Markov) ──┐                   │
 │                                             ▼                   │
 └─ T9 LeaseController slice 1 (idle-time, VTFQ B8)  ◄── T4, T7, T8, T2
      └─ T10 lease ledger + expiry + invariants I1–I3 (no behaviour change)
           └─ T11 slice 2: gang leases for parallel groups (B17 target)
           └─ T12 slice 3: reserve-ahead on ext.* budgets  ◄── T5 (only if the stress tier binds)
T13 docs/tracker pass (after each checkpoint; final at the end)
```

Safe to parallelise: T3, T4, T5, T6 (independent files/knobs) after T1–T2. Sequential: T7 → T8 → T9 → T10 → T11/T12.

## Task list

### Phase 0 — Environment and the measurement instrument
- [x] Task 1: make the tree writable and runnable (`chmod u+w`, `pyproject.toml`, README run note)
- [x] Task 2: `rung0/compare.py` — paired stats + oracle-normalised score (A6)

### Checkpoint 0
- [x] `uv run python -m agentsim selftest` passes with no `--with numpy`
- [x] `compare.py` on the existing e1/hosted CSVs reproduces "gate == oracle on E1" (no significant difference at any N) and "oracle > gate at hosted 2/min" with p-values and per-seed win counts
- [x] Review with human

### Phase 1 — A synthetic society on which rung 3 can be measured (all synthetic, no real data)
- [x] Task 3: `recipe_temperature` knob + predictability grid (A7)
- [x] Task 4: structured think time with `think_snr` knob + `rung0/e5_idle_predictability.py` (B7)
- [x] Task 5: stress scenario + `agentsim sample-scenarios` (B18, A8)
- [x] Task 6: extend `fit` to transitions and the remaining marginals; round-trip test on synthetic traces (C2)

### Checkpoint 1
- [x] selftest passes; `think_snr=0`, `recipe_temperature=1` reproduce the existing grid CSVs exactly
- [x] Hosted grid re-run at `think_snr ∈ {0, 0.5, 0.9}`; oracle headroom vs snr tabulated by compare.py (this is the headroom rung 3 must earn)
- [x] Stress grid shows a binding API-like tier (uncoordinated ≠ gate, 429 > 0)
- [x] RUNG0_REPORT gets §11 (headroom vs snr, stress, R vs temperature); ISSUES A7/A8/B7/B18/C2 → done
- [x] Review with human

### Phase 2 — Rung 3, slice 1: a LeaseController that earns the idle-time headroom
- [x] Task 7: `retention_ttl` policy hook, per-session KV TTL (B6)
- [x] Task 8: `agentsim/predict.py` — online quantile tracker + online Markov, observable inputs only
- [x] Task 9: `LeaseController` (policy type `lease`, knobs τ, h): predicted idle → park/prewarm/KV TTL; virtual-time fair queuing (B8); honesty selftest

### Checkpoint 2
- [x] selftest (now ≥ 10 checks) passes; deadlocks = 0; no leaked holds; no 429 to providers under `lease`
- [x] Grid: hosted × `think_snr {0, 0.5, 0.9}` × load {1, 2}/min × τ {0.5, 0.8, 0.9} × 5 seeds × {gate, lease, oracle}; compare.py oracle-normalised score per metric
- [x] Ladder §6.1 rung-3 decision recorded: captures ≥ 50% of headroom with waste ≤ 10% at snr ≥ 0.5 → ship; else the τ/h sweep result and why
- [x] Review with human

### Phase 3 — Rung 3, slices 2–3: leases proper (ledger, expiry, gang, reserve-ahead)
- [x] Task 10: lease ledger in the engine with expiry and invariants I1–I3; no policy uses it yet (behaviour-preserving)
- [x] Task 11: slice 2 — gang leases for parallel tool groups' extra CPU (B17's measurable target)
- [x] Task 12: slice 3 — reserve-ahead leases on `ext.*` RPM budgets, on the stress society (only if Checkpoint 1 showed binding)

### Checkpoint 3
- [x] Curves: lease vs gate vs oracle over load, over R (T3), over snr (T4); loss columns (waste, added queueing latency, throughput)
- [x] Every gain reported with p-value, effect size, per-seed wins, oracle-normalised score
- [x] Review with human

### Phase 4 — Tracker and notes
- [x] Task 13: ISSUES statuses, RUNG0_REPORT (only place with numbers), HANDOFF §4 pointer + §7 order, README commands

### Phase 5 — Next steps after the ablation (added on the user's "continue", 2026-09-22)
- [x] Task 14: notes consistency pass
- [x] Task 15: ship the fixed rules (`ReactiveGate.queue=vtfq`, idle timeout 0) and test the ablation's claim head-on
- [x] Task 16: calibrated budget margin (slice 3)
- [x] Task 17: randomised sweep — where does prediction pay? (`agentsim sweep`, A8 in use)
- [x] Task 18: gain-vs-R curve for the lease
- [x] Task 19: real traces — TraceLab downloaded (CC BY 4.0, 101 MB), converted, fitted; rung 0 on real data done

### Checkpoint 4
- [x] RUNG0_REPORT §13; HANDOFF §3/§7 — review with human pending

### Phase 6 — The Needs Predictor (designed 2026-09-22 in PREDICTOR_DESIGN.md; not started)
- [x] Task 20: TraceLab ceilings (RUNG0_REPORT §14): Head Q worth building as a risk ranker; Head I not worth building
- [ ] Task 21: generator content proxies + `content_snr`; spawn/join recipes (B10) (§8)
- [ ] Task 22: `agentsim/features.py` + numpy multi-head model (S, Q, T, I, R) + conformal layer + drift monitors (§3–§4)
- [ ] Task 23: Reserver v2 behind `policy.type=needs`: occupancy forecast, mixture-quantile leases, expected-value gating, queue-length-conditional ordering, predictive shaping (§5)
- [ ] Task 24: E6 ablation on sweep / TraceLab / stress societies; label-efficiency curve; ship or kill per §7
- [ ] Task 25: Jev annotator behind a flag, ECE-gated (§2 D, §7)

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| B7's structured think time silently changes the marginal or the CRN draw order, invalidating paired comparisons | High | `think_snr=0` must reproduce today's spans byte-for-byte (selftest digest); fidelity think median/p90 within ±10% at every snr |
| The LeaseController cheats by reading hidden `Step`/`think` values through the existing hook signatures (`idle_timeout(p, think, default)`) | High — inflated rung 3 | Honesty selftest (contradictory hidden values → identical decisions); the policy's observable state is a separate object built only from `on_event` |
| Headroom on hosted at 2/min is small in absolute terms (failure 4.1% → 2.4%, n = 5 seeds) — rung 3 differences may be inside noise | Med | 10 seeds for the lease grid (ladder §6.0 for learned rungs); report effect sizes; only claim gains > 2× simulator noise |
| Lease ledger (T10) touches `_try_run`/`_wake`, the engine's most delicate code | Med | T10 is behaviour-preserving by construction (no policy issues leases) and gated by the span-digest determinism check; invariants I1–I3 asserted every 500 events like `check_accounting` |
| Stress scenario knobs are PRIORS; a "binding" tier can be manufactured | Med | Report gains as curves over the knobs (A8) and say so; no single-point headline |
| Directory permissions / OneDrive-style sync mangling | Low | T1 first; keep files small; no heredocs for long Python (HANDOFF §6) |
| Runtime: the phase-2 grid is 3 × 2 × 3 × 5 × 3 = 270 runs × ~2 s ≈ 10 min; phase-3 curves more | Low | `--no-traces` for grids that only need summaries; A9 (running totals) only if a grid exceeds ~30 min |

## Open questions (need the user)

1. **Real traces (C1/E3)** — approve downloading public datasets (TraceLab, Exgentic/agent-llm-traces, nebius/nvidia trajectories) or provide shadow-mode OTel traces from the society? Not on this plan's critical path; T6 makes `fit` ready for the day they arrive.
2. **`git init` + `.gitignore data/synthetic/ .venv/`** (E1)? Recommended; not done without a yes.
3. **`chmod u+w agentsim rung0 scenarios data`** — required for any new file; done in T1 unless you object.
4. Whether to drop `hour_of_day` from B7's state (a 2 h horizon makes it a constant per run) — plan assumes yes, state = (recipe, last phase before `final`, request-index bucket).

### Phase 6 build note (2026-09-22, second session — user: "omit the ladder")
Tasks 21–25 are built in one pass, in dependency order, without the ship/kill gate between them: generator (content cues + spawn/join) → `features.py` + `needs.py` (multi-head model, conformal) → `jev.py` (typed System One model: local calibrated implementation, remote seam, `ext.jev` channel) → `reserver.py` (`policy.type=needs`: forecast, mixture-quantile leases, expected-value gating, queue-length-conditional ordering) → grids + `NEEDS_REPORT.md`. The evidence discipline stays (paired seeds, oracle bound, ablation over the baseline's knobs and the controller's switches); only the *gating* is dropped.
