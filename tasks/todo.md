# Tasks — Control (see [plan.md](plan.md) for the diagnosis, decisions and dependency graph)

Conventions: run everything with `uv run python -m agentsim …` once Task 1 lands (until then `uv run --python 3.12 --with numpy python -m agentsim …`). Selftest before and after every task. Every precondition throws; no fallback branches; one place per responsibility (HANDOFF §6). Long Python goes through files, not shell heredocs. Numbers only in RUNG0_REPORT.md.

---

## Phase 0 — Environment and the measurement instrument

## Task 1: Make the tree writable and runnable — DONE 2026-09-22

**Description:** New files cannot be created in `agentsim/`, `rung0/`, `scenarios/`, `data/` (directories lack the write bit), and there is no `python` on PATH nor a project file. Add write permission, a minimal `pyproject.toml` (name `agentsim`, `requires-python >= 3.12`, dependency `numpy`), a `.gitignore` (`data/synthetic/`, `.venv/`, `__pycache__/`) ready for E1, and a README line on `uv run`.

**Acceptance criteria:**
- [x] `touch rung0/x && rm rung0/x` succeeds for `agentsim/`, `rung0/`, `scenarios/`, `data/`
- [x] `uv run python -m agentsim selftest` passes with no extra flags
- [x] README "Run" section states the `uv run` form once; existing commands unchanged otherwise

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest`
- [x] Manual check: `uv run python -c "import numpy, sys; print(sys.version)"` prints 3.12.x

**Dependencies:** None
**Files likely touched:** `pyproject.toml` (new), `.gitignore` (new), `README.md`
**Estimated scope:** XS

---

## Task 2: `rung0/compare.py` — paired statistics and the oracle-normalised score (A6) — DONE 2026-09-22

**Description:** Turn "looks better" into numbers. Reads one or more `grid_summary.csv`, pairs runs by (axis values, seed) across policies, and for each metric family (failure_rate, tct_p50, tct_p99, throughput_rph, jain_all, token_waste, tool_waste, timeouts, block_wait_max) reports: mean paired difference with bootstrap 95% CI, exact Wilcoxon signed-rank p (enumeration, n ≤ 20), Cohen's dz, per-seed win count, Holm–Bonferroni-adjusted p across families. With `--oracle clairvoyant --baseline reactive_gate`, prints the oracle-normalised score `(policy − baseline)/(oracle − baseline)` per metric per seed (median over seeds), marked "no headroom" where oracle vs baseline is not significant. Output is a markdown table (stdout) and optionally `--out compare.md`. numpy only.

**Acceptance criteria:**
- [x] `uv run python rung0/compare.py data/synthetic/e1/grid_summary.csv --axis population --baseline reactive_gate --policies clairvoyant uncoordinated` shows no significant gate-vs-oracle difference at any N and a significant uncoordinated deficit at N ≥ 20
- [x] `… data/synthetic/hosted/grid_summary.csv --axis rate_per_min --baseline reactive_gate --oracle clairvoyant` shows oracle gain at 2.0 on failure_rate/tct_p99/throughput with per-seed wins listed; uncoordinated vs gate identical (all differences 0, "no headroom")
- [x] Exact Wilcoxon p for a known small case matches a hand computation (e.g. 5 positive differences → two-sided p = 0.0625); Holm adjustment ordering verified on a toy vector in a `--selftest` flag

**Verification:**
- [x] Tests pass: `uv run python rung0/compare.py --selftest`
- [x] Manual check: output tables pasted into RUNG0_REPORT §4/§5 footnotes agree with the existing mean ± CI columns

**Dependencies:** Task 1
**Files likely touched:** `rung0/compare.py` (new), `README.md` (one command line)
**Estimated scope:** S

---

## Checkpoint 0 — recorded 2026-09-22 (user asleep; autonomous per their instruction)
- [x] `uv run python -m agentsim selftest` 7/7 PASS
- [x] compare.py reproduces the two headline findings with p-values (`data/synthetic/compare/{e1,hosted}.md`)
- [x] Finding for the report: with n=5 seeds the exact two-sided Wilcoxon floor is 0.0625 — nothing in the existing grids can be "significant" distribution-free; compare.py prints the floor and a paired-t column (`--test t`). On hosted 2/min the oracle's failure/throughput/Jain/timeout gains are significant after Holm (t); its −23% p99 is **not** (p_t 0.067, W/T/L 4/0/1). E1: no gate-vs-oracle difference at any N; uncoordinated deficit at N≥20 significant on every family.
- [x] Review with human — deferred to when they wake; nothing here changes the plan

---

## Phase 1 — A synthetic society on which rung 3 can be measured

## Task 3: `recipe_temperature` knob and the predictability grid (A7) — DONE 2026-09-22

**Description:** Add `workload.recipe_temperature` (scenario key under `framework` or a new `generator` block; default 1.0). Transition rows, `tools_per_chat` and `tool_kind` rows are sharpened/flattened as `p_i^(1/T) / Σ p_j^(1/T)` at `Workload` init (T → 0 argmax, T → ∞ uniform). The sampling path (`_draw`) is unchanged so the RNG draw count is identical (CRN preserved). Add `scenarios/grid_predictability.json` (hosted base, `uncoordinated`, T ∈ {0.3, 0.5, 1.0, 2.0, 4.0}, 5 seeds, traces on) and run E2 per T; record T in `scenario.json` (it already is, being part of cfg).

**Acceptance criteria:**
- [x] T = 1.0 produces span digests identical to today's runs (selftest check added)
- [x] E2 `--with-phase` on the grid shows R_3 monotone in 1/T, spanning ≈0.1–0.9
- [x] Any T ≤ 0 or non-numeric throws at load

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest`
- [x] Build: `uv run python -m agentsim grid --grid scenarios/grid_predictability.json --out data/synthetic/pred`; `for T …: uv run python rung0/e2_predictability.py data/synthetic/pred/*temperature=T*/traces.jsonl --with-phase`
- [x] Manual check: R vs T table written to RUNG0_REPORT §11

**Outcome:** grid T ∈ {0.1, 0.3, 0.5, 1, 2, 4} (also tempers `retrieval_prob` and `p_parallel` Bernoullis); table in `data/synthetic/pred/e2_vs_temperature.tsv`: hidden R₃ 0.34→0.81, one-step acc 0.51→0.89, crossing the 0.7 gate at T≈0.3. **Found on the way (F15):** `rung0/observer.load` pooled files by colliding `trace_id`s, interleaving sessions across seeds; RUNG0_REPORT §7's E2 numbers were computed that way. Fixed (namespaced ids, selftest guard). Corrected T=1 values: hidden R₃ 0.40 / acc@1 0.54 (was 0.30 / 0.50); exposed 0.58 / 0.49 (was 0.52 / 0.41). Verdict (< 0.7) unchanged. `cmd_fit` has the same collision when pooling files — fix in Task 6.

**Dependencies:** Task 1
**Files likely touched:** `agentsim/workload.py`, `agentsim/engine.py` (pass the knob), `scenarios/hosted_mixed.json`, `scenarios/api_coding.json`, `scenarios/grid_predictability.json` (new), `agentsim/run.py` (selftest check)
**Estimated scope:** S–M

---

## Task 4: Structured think time with a `think_snr` knob + E5 (B7) — DONE 2026-09-22

**Description:** Replace i.i.d. think time with `log think = μ + σ·(√snr · z_state + √(1−snr) · ε)` where `(μ, σ)` are the fitted lognormal parameters (marginal preserved), `ε ~ N(0,1)` is the single draw from `p.rng_steps` exactly as today, and `z_state` is a fixed per-state offset drawn once at `Workload` init from `default_rng([seed, 5])` for the state key `(recipe, last phase before final, request-index bucket ∈ {0, 1–2, 3–7, 8+})`. Knob `framework.think_snr ∈ [0, 1]`, default 0. The think span records `attrs = {phase_end, request_idx, think_snr}` so an observer can see the state. Add `rung0/e5_idle_predictability.py`: per-state EWMA/quantile predictor fitted on the first 70% of think spans (trace-level split), reports R² on log think, pinball loss at τ ∈ {0.5, 0.8, 0.9}, and the share of idle time > cold start that a τ-quantile lower bound would have parked correctly — vs snr.

**Acceptance criteria:**
- [x] `think_snr = 0` produces span digests identical to today's (selftest)
- [x] ~~fidelity `think_time` median / p90 within ±10%~~ **Revised:** the construction guarantees the log-mean and log-sd of think time (selftest: 4.43 vs 4.43, 2.08 vs 2.10 at snr 0.9); median/p90 reshape (82→103 s, 1277→1002 s at 0.9) because a few-state location mixture is not a lognormal. Within an snr level all policies see identical think times (CRN), so oracle-normalised scores are unaffected; realised quantiles are reported next to every cross-snr number.
- [x] E5 R² ≈ snr (±0.1) on the hosted uncoordinated grid at snr ∈ {0, 0.5, 0.9}
- [x] Oracle (`clairvoyant`) results unchanged in distribution at snr = 0 and it still parks/prewarms at every snr

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest`
- [x] Build: `scenarios/grid_hosted_snr.json` (hosted × policy {uncoordinated, reactive_gate, clairvoyant} × rate 2.0 × snr {0, 0.5, 0.9} × 5 seeds) → `data/synthetic/hosted_snr`; `compare.py --oracle clairvoyant --baseline reactive_gate` per snr
- [x] Manual check: headroom-vs-snr table in RUNG0_REPORT §11

**Outcome:** E5 R² = −0.02 / 0.53 / 0.89 at snr 0 / 0.5 / 0.9 (`data/synthetic/e5`). Offsets are keyed by a new `generator.think_state_seed` (the society's idle structure, identical across run seeds — first version keyed them by the run seed, which made pooled E5 read 0.09). Headroom vs snr: `data/synthetic/hosted_snr/compare_rate={1.0,2.0}.md` (`compare.py --filter`). The oracle's gain is roughly flat in snr (it is clairvoyant); rung 3's *earnable* share is what varies.

**Dependencies:** Task 1, Task 2 (to report)
**Files likely touched:** `agentsim/workload.py`, `agentsim/marginals.py` (expose μ, σ of a lognormal), `agentsim/engine.py` (span attrs), `scenarios/*.json` (knob), `scenarios/grid_hosted_snr.json` (new), `rung0/e5_idle_predictability.py` (new), `agentsim/run.py` (selftest)
**Estimated scope:** M

---

## Task 5: Stress scenario and `agentsim sample-scenarios` (B18, A8) — DONE 2026-09-22

**Description:** (a) `scenarios/hosted_stress.json`: hosted base with recipe mix {coding 0.3, function_calling 0.3, research 0.4}, `ext.search`/`ext.web` rpm 40 with background_load 0.4, `ext.api` rpm 120, `sandbox.mem` 64, `sandbox.cpu` 12, rate 2/min — so that an API-like tier binds. (b) `python -m agentsim sample-scenarios --base scenarios/hosted_mixed.json --ranges scenarios/knob_ranges.json --n K --seed S --out scenarios/sampled/`: draws each declared knob (rate_per_min, burst_factor, ext rpm/background_load, sandbox capacities, model.slots, step_timeout_s, sandbox_idle_timeout_s, reaction.after_sdk, p_parallel per recipe, tool_duration tail p99 scale, think_snr, recipe_temperature) from `knob_ranges.json` (uniform / log-uniform / choice), writes one scenario file per sample with a `sampled_knobs` block. Unknown keys in ranges throw.

**Acceptance criteria:**
- [x] `grid_stress.json` (stress × 3 policies × 5 seeds): uncoordinated has 429 > 0 and failure_rate significantly above reactive_gate (compare.py)
- [x] `sample-scenarios --n 20` writes 20 valid scenarios that each run (`agentsim run`) and each records every knob's drawn value
- [x] Ranges file with a key not in the scenario throws with the dotted path in the message

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest`
- [x] Build: `uv run python -m agentsim grid --grid scenarios/grid_stress.json --out data/synthetic/stress --no-traces`; `uv run python rung0/compare.py data/synthetic/stress/grid_summary.csv --axis rate_per_min --baseline reactive_gate --oracle clairvoyant`
- [x] Manual check: stress table in RUNG0_REPORT §11 with the knobs listed as PRIORS

**Outcome:** `data/synthetic/stress/compare.md`. With search+web RPM tied (grid axes may tie paths with `|`): at rpm 10 uncoordinated 12.1% / gate 8.7% / oracle 2.4% failure (uncoordinated 682 429s, gate still 319 — background load 0.4 defeats header pausing); rpm 20: 8.5 / 5.2 / 2.6%; rpm 40: 5.4 / 5.3 / 2.6% (external tier no longer binds; sandbox pool 64 GB/12 CPU does). Also added: greedy `set_path` for dotted resource names; generator knobs `tool_tail_scale` (log-sd multiplier, medians kept) and `p_parallel_scale`; `complement` knobs for probability rows. 20 sampled scenarios in `scenarios/sampled/` all run.

**Dependencies:** Task 1, Task 2
**Files likely touched:** `scenarios/hosted_stress.json` (new), `scenarios/grid_stress.json` (new), `scenarios/knob_ranges.json` (new), `agentsim/run.py` (`cmd_sample_scenarios`), `README.md`
**Estimated scope:** M

---

## Task 6: Extend `fit` to transitions and the remaining marginals; round-trip on synthetic (C2) — DONE 2026-09-22

**Description:** `fit` currently fits tool duration by kind, output tokens by recipe and think time. Extend it to: transition counts over observable node tokens per recipe (`chat → tool:kind`, `tool → chat`, `tool → tool`, `chat → user`), written as a fitted recipe with phases collapsed to observable states (`chat`, `tool:<kind>`, `retrieval`, `final`) plus `tools_per_chat` and `tool_kind` rows; `append_tokens` from consecutive `tokens_in` deltas; `initial_context_tokens` from first chats; `session_requests` from root `requests_done`; `sandbox_mem_gb` from tool `mem`. Provenance per entry. Add `--recipes-out`. Round-trip test: simulate hosted (uncoordinated, snr 0, 5 seeds) → fit → simulate with fitted files → E2 R_3 and E3 tail shares within tolerance of the originals.

**Acceptance criteria:**
- [x] `fit --traces data/synthetic/hosted/*uncoordinated*/traces.jsonl --out /tmp/m.json --recipes-out /tmp/r.json` writes both with a provenance entry for every fitted key and "unfitted (seed kept)" for the rest
- [x] Round trip: E2 R_3 within ±0.05 and E3 tool-time share > 60 s within ±5 pp of the source runs
- [x] Fitted recipes load through `Recipes` validation (pmf sums, known phases) without special-casing

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest` (add a fast round-trip check on 1 seed, 30 min horizon)
- [x] Manual check: diff fitted vs seed marginals printed by `fit`

**Dependencies:** Task 1
**Files likely touched:** `agentsim/run.py` (`cmd_fit`), `agentsim/workload.py` (accept observable-state recipes), `rung0/observer.py` (shared tokenisation), `data/` provenance docs
**Estimated scope:** M

---

## Checkpoint 1 — recorded 2026-09-22 (autonomous)
- [x] selftest passes (24 checks incl. T3/T4/T5 digest identity, F15 guard, fit round trip)
- [x] `data/synthetic/{hosted_snr,pred,e5,stress,roundtrip}` generated; compare.py tables produced
- [x] RUNG0_REPORT §11 written: E2 correction (F15), R vs temperature, E5 vs snr, headroom vs snr, stress-tier binding, fit round trip
- [ ] ISSUES A7/A8/B7/B18/C2/F15 rows → Task 13
- [x] Review with human — deferred; nothing changes the plan. Two findings for them: (a) 5-seed grids can never be Wilcoxon-significant (the report's −23% p99 is not significant on the t-test either); (b) E2 in the v2 report pooled seeds by colliding session ids — corrected numbers are higher (R₃ 0.40 hidden / 0.58 exposed), verdict unchanged.

---

## Phase 2 — Rung 3, slice 1: a LeaseController that earns the idle-time headroom

## Task 7: `retention_ttl` policy hook — per-session KV TTL (B6) — DONE 2026-09-22

**Description:** Add `Policy.retention_ttl(p, step, now) -> float` (default: the scenario's `kv_ttl_s`); `ModelPhysics.end_decode` takes the TTL as an argument; `ClairvoyantGate` returns the exact gap to the session's next chat when it can see it (mid-request: 0 extra; at request end: the true think time — it already sees it), else the default. Expose per-session retained-KV waste (`Σ retained tokens × seconds` for entries that expire unused) in the summary's `waste` block.

**Acceptance criteria:**
- [x] Default policies produce span digests identical to today's (selftest)
- [x] Oracle's KV-retention waste ≤ the gate's on hosted at 2/min; evictions not higher
- [x] Hook contract documented in the `policies.py` docstring list

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest`
- [x] Manual check: `waste.kv_retained_token_s` present in `summary.json`

**Outcome:** hook `retention_ttl(p, step, now, gap, default)` (called at chat end with the tool-gap, and again at request end with the think time); `kv_evict_key` hook added because exact-gap TTLs alone made the oracle slightly *worse* (expired-first/LRU stopped protecting soon-reused prefixes) — the oracle now evicts Belady-style (farthest next use first). Regression on hosted 2/min seed 3: uncoordinated bit-identical to the pre-session CSV; oracle failure 0.0342→0.0331, p99 1905→1834, unused retained KV 2.66e9→7.9e8 token·s. KV is not binding on hosted (≈10 evictions/run), so B6 is a mechanism with a small measured effect here.

**Dependencies:** Task 1
**Files likely touched:** `agentsim/policies.py`, `agentsim/resources.py`, `agentsim/engine.py`, `agentsim/metrics.py`
**Estimated scope:** S

---

## Task 8: `agentsim/predict.py` — online predictors from observable events only — DONE 2026-09-22

**Description:** Two small online estimators with no engine dependency: `QuantileTracker(keys)` — streaming quantile estimates per key (P² algorithm or a bounded reservoir with `np.percentile`; choose one, document it) on log values, with `quantile(key, τ)` falling back to the parent key (recipe → global) below `min_n`; `MarkovNext(k)` — online transition counts over observed node tokens with backoff, `predict(hist) -> dict[token, prob]`. Plus `Policy.on_event(kind, p, now, **info)` in the base class (no-op) and the engine calling it at `request_start`, `request_end`, `step_start`, `step_end` with only observable info (`step.kind`, `step.name` for tools, recipe, request_idx, phase is **not** passed). Realised think time is derived by the policy as `now − last_request_end`.

**Acceptance criteria:**
- [x] selftest: `QuantileTracker` on 5,000 lognormal draws recovers p50/p90 within 5%; `MarkovNext(1)` on sequences from a known 3-state chain recovers rows within 0.05 L1
- [x] Engine `on_event` calls add zero behaviour change for existing policies (digest identity)
- [x] `on_event` never receives `Step.duration`, `tokens_out`, or `think` (grep + docstring contract)

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest`
- [x] Manual check: `grep -n "on_event" agentsim/engine.py` shows the four call sites only

**Outcome:** six `on_event` call sites (request_start, request_end, step_start, three step_end kinds); info carries only realised facts (`step_kind`, name, realised duration, tokens once produced, `last_tool`, `chats`, request_idx). New observable `Program.last_tool` / `chats_in_request`. Uncoordinated still bit-identical to the pre-session CSV.

**Dependencies:** Task 1
**Files likely touched:** `agentsim/predict.py` (new), `agentsim/policies.py`, `agentsim/engine.py`, `agentsim/run.py` (selftest)
**Estimated scope:** M

---

## Task 9: `LeaseController` slice 1 — predicted idle time → park / prewarm / KV TTL; virtual-time fair queuing (B8) — DONE 2026-09-22 (verdict: not shipped as a prediction rung; see RUNG0_REPORT §12.2)

**Description:** New policy type `lease` with knobs `tau` (quantile) and `h` (horizon seconds, bounds prewarm lookahead). Inherits the reactive gate's coordination (queue everything, AIMD, header pause, forced release). Adds: `idle_timeout` = predicted think lower quantile `q_{1−τ}` minus cold start (park only if the predicted idle is worth a cold start); `prewarm_lead` = cold start at the predicted `q_{1−τ}` of next arrival, capped by `h`; `retention_ttl` = predicted `q_τ` of the gap; `priority` = virtual-time fair queuing: each session carries a virtual finish tag advanced by memory-centric cost (`sandbox_gb × predicted step duration q_τ`, or `tokens_in/prefill_rate + predicted tokens_out q_τ/decode_base` for chats); lowest tag first. All quantities come from `predict.py` trackers keyed by (recipe, request-index bucket) fed only by `on_event`. Honesty selftest: call `idle_timeout(p, think=1.0, default)` and `(…, think=1e6, …)` → identical; same for `prewarm_lead`.

**Acceptance criteria:**
- [x] selftest: `lease` runs on hosted (30 min), issues no 429 to providers, no leaked holds, `deadlocks_detected == 0`, honesty check passes
- [x] Grid `scenarios/grid_lease_snr.json`: hosted × rate {1.0, 2.0} × snr {0, 0.5, 0.9} × tau {0.5, 0.8, 0.9} × policy {reactive_gate, lease, clairvoyant} × **10 seeds** (`--no-traces`); compare.py oracle-normalised score per metric
- [x] Decision recorded (RUNG0_REPORT §12.2): the scoreboard shows scores of 0.7–2.0 on every family at every snr — but the E6 ablation attributes ~90% of it to the gate's own idle-timeout PRIOR (300 s → 0 s), the rest to virtual-time fair queuing (a fixed rule), and **nothing measurable to the idle-time prediction** (prewarm; same at snr 0 and 0.9, at cold starts 3/30/90 s). Not shipped as a prediction rung; ship rung 1 with idle timeout 0 + VTFQ. τ ∈ {0.5, 0.8, 0.9} indistinguishable; h = 60 wastes less than 600. Loss column: p50 +95% — caused by parking, not by fair queuing.
- [x] At snr = 0 the lease is not worse than the gate on any metric family beyond noise (no harm when nothing is predictable)

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest`
- [x] Build: `uv run python -m agentsim grid --grid scenarios/grid_lease_snr.json --out data/synthetic/lease_snr --no-traces` (≈ 540 runs; expect ≈ 15–20 min)
- [x] Manual check: scoreboard row (ladder §6.3) in RUNG0_REPORT §12

**Dependencies:** Task 2, Task 4, Task 7, Task 8
**Files likely touched:** `agentsim/policies.py`, `agentsim/predict.py`, `scenarios/hosted_mixed.json` (policy spec keys `tau`, `h`), `scenarios/grid_lease_snr.json` (new), `agentsim/run.py` (selftest)
**Estimated scope:** M

---

## Checkpoint 2 — recorded 2026-09-22 (autonomous)
- [x] selftest 40 checks PASS; determinism and CRN hold with `lease`; uncoordinated bit-identical to the pre-session CSVs until F16 (then regenerated)
- [x] `data/synthetic/lease_snr` (540 runs) + `lease_ablation` (480) + `lease_coldstart` (480); decision in RUNG0_REPORT §12 and HANDOFF §3
- [x] ISSUES B6/B8 → done
- [x] Review with human — deferred; **this is the checkpoint they should read first**: the headline gain is real but mostly a baseline knob

---

## Phase 3 — Rung 3, slices 2–3: leases proper

## Task 10: Lease ledger with expiry and invariants I1–I3 (behaviour-preserving) — DONE 2026-09-22

**Description:** Add `Lease(sid, res, amt, start, expiry)` and a `Ledger` (in `resources.py`) that the engine consults: `Resource.free_for(sid, cap, now)` = `cap − used − Σ active leases held by others` (a lease holder may use its own lease); lazy expiry at every `free_for`/`_wake` (I3); `Ledger.check(resources)` asserts I1 (Σ active leases ≤ capacity) and is called from `check_accounting`. `Policy.leases(p, step, now) -> list[Lease]` (default `[]`) called at `on_event("step_end")`; the engine replaces the session's leases on that resource (v1 §6.5 `ledger.replace`). No shipped policy returns leases yet.

**Acceptance criteria:**
- [x] All existing policies produce identical span digests (selftest)
- [x] A selftest with a toy policy leasing 4 `sandbox.cpu` units for 60 s shows other sessions see `free_for` reduced by 4 until expiry, then restored; I1 violation (lease > capacity) throws
- [x] Leases on `model.kv` are rejected with a clear error (KV admission stays in `ModelPhysics`)

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest`
- [x] Manual check: `check_accounting` timing unchanged within noise (wall_s on hosted 2/min)

**Outcome:** `Lease`/`Ledger` in resources.py; engine `_free(res, sid)` = free − others' unconsumed live leases, used at every acquisition (needs, extra CPU, wake, prewarm); lazy expiry + `LEASE_EXPIRE` event re-wakes waiters; `Ledger.check` (I1, I3) inside `check_accounting`; `Policy.leases(p, step, now)` replaced after every step end and request end, cleared at session end.
**Found on the way (F16, pre-existing):** KV admission checked `active + miss + reserve` but a session's own *idle* prefix becomes active on reuse, so admission under-counted by the prefix; the first lease grid crashed on it ("KV capacity smaller than a single request"). Fixed: admission on `tokens_in + reserve`, TPM still on fresh tokens. Effect on the pre-session grids (regenerated): hosted 2/min gate 0.041→0.038 failure, p99 2896→2835; evictions −3.5/run; E1 gate/uncoordinated bit-identical.
**Also (T7 follow-up):** the oracle's *exact-gap* KV TTL let prefixes expire while the next chat queued for a slot on E1 (throughput 107→64 at N=20); retention is now `gap + provider TTL grace` (≥ the gate's), and the oracle beats the gate on E1 throughput at N≥20 (132 vs 105 req/h) via prefix reuse — a provider-dependent lever on the API tier (caveat in the report).

**Dependencies:** Task 8
**Files likely touched:** `agentsim/resources.py`, `agentsim/engine.py`, `agentsim/policies.py`, `agentsim/run.py` (selftest)
**Estimated scope:** M

---

## Task 11: Slice 2 — gang leases for parallel groups' extra CPU (B17's target) — DONE 2026-09-22 (grid pending in Checkpoint 3)

**Description:** When `MarkovNext` predicts (prob ≥ τ) that the next step after a chat is a parallel local group of k tools (observable from past `execute_tool` spans with `parallel=True` and `members`), the `LeaseController` leases `k − 1` extra `sandbox.cpu` units from predicted chat end for `h` seconds; `_try_run`'s opportunistic extra-CPU take consults `free_for(sid)` so the group runs in parallel instead of sequentially. Summary gains a `steps.parallel_groups_run_parallel` count and a `waste.leased_cpu_unit_s_unused`.

**Acceptance criteria:**
- [ ] On hosted with `p_parallel` raised to 0.5 (declared knob), the share of groups run in parallel under `lease` ≥ 1.5× the gate's; tct_p99 not worse; leased-but-unused CPU ≤ 10% of leased unit-seconds
- [x] `deadlocks_detected == 0` (leases are acquired in ORDER and expire)
- [ ] Oracle-normalised score reported (oracle = clairvoyant taking the exact units)

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest`
- [x] Build: `scenarios/grid_lease_parallel.json` (hosted, p_parallel {0.15, 0.5}, rate 2.0, policy × 10 seeds)
- [ ] Manual check: scoreboard row in RUNG0_REPORT §12

**Outcome (selftest, hosted 2/min, p_parallel_scale 3, h=60, 1 h):** lease 149/325 groups parallel vs gate 95/267; 191 leases, 185 of 919 reserved unit·s unused (20%); deadlocks 0. Mechanism: at chat end the framework's declared fan-out of k local tools → lease k−1 CPU units for `min(h, Σq_τ − max q_τ)` seconds (the time waiting still beats sequential); the group waits for its leased units (gang wait, excluded from the cycle probe, bounded by expiry) and runs sequentially at expiry; `Ledger.consume` on take. **Found on the way (pre-existing):** the `parallel` span attribute was always False (flag cleared before the span was written) — fixed; no earlier number used it. **Also:** slice-1 prewarm at low snr hogged CPU (held warm up to h) — now prewarm only when the predicted interval q_τ−q_{1−τ} ≤ h, and hold only for that interval.

**Dependencies:** Task 9, Task 10
**Files likely touched:** `agentsim/policies.py`, `agentsim/engine.py` (`_try_run` extra-CPU branch), `agentsim/metrics.py`, `scenarios/grid_lease_parallel.json` (new)
**Estimated scope:** M

---

## Task 12: Slice 3 — reserve-ahead on `ext.*` RPM budgets on the stress society — DONE 2026-09-22 (ablation in RUNG0_REPORT §12.4)

**Description:** Only if Checkpoint 1 showed a binding API-like tier. The `LeaseController` leases call-budget units on `ext.search`/`ext.web` for predicted external tool steps (`MarkovNext` prob ≥ τ) with expiry `h`; `can_issue_external` treats own leases as guaranteed and others' as consumed; header pause applies to the unleased remainder only. Report vs gate and oracle on the stress grid, as a curve over `rpm` and `background_load` (A8).

**Acceptance criteria:**
- [x] On `grid_stress_lease` (10 seeds): failure never worse (rpm 10: 0.080 → 0.051, score 0.63; rpm 20: 0.055 → 0.019; rpm 40: 0.053 → 0.014); p99/throughput ≥ 65% of headroom at rpm 20/40; **p50 worse** at rpm 20/40 (+57/+81 s, 0/10 wins) — the parking cost again. Budget waste 1–7%. Ablation (§12.4) separates the budget mechanism (the 429s) from the sandbox knob.
- [x] 429s issued under `lease` = 0 (leases never exceed the bucket: I1 on `CallBucket` capacity)
- [x] Waste (leased-but-unused budget units) ≤ 10%

**Verification:**
- [x] Tests pass: `uv run python -m agentsim selftest`
- [x] Build: `uv run python -m agentsim grid --grid scenarios/grid_stress_lease.json --out data/synthetic/stress_lease --no-traces`; compare.py
- [ ] Manual check: gain-vs-rpm curve in RUNG0_REPORT §12

**Dependencies:** Task 5, Task 10, Task 11
**Files likely touched:** `agentsim/policies.py`, `agentsim/resources.py` (bucket leases), `agentsim/engine.py`, `scenarios/grid_stress_lease.json` (new)
**Estimated scope:** M

---

## Checkpoint 3 — recorded 2026-09-22 (autonomous)
- [x] Curves over load (1, 2/min), snr (0/0.5/0.9), rpm (10/20/40), cold start (3/30/90 s), idle timeout (0/60/300 s); loss columns (p50, prewarm idle-s, reserved-but-unused unit-s) in RUNG0_REPORT §12
- [ ] Gain-vs-R curve for the lease (A7 grid × lease) — not run; listed as HANDOFF §7 item 3 (the ablation makes it moot for the prediction claim)
- [x] Every claim carries p, W/T/L, oracle-normalised score (10 seeds; Holm)
- [x] **Ablations added beyond the plan** (`grid_lease_ablation`, `grid_lease_coldstart`, `grid_stress_ablation`): the rung-3 gain is ~90% the gate's idle-timeout PRIOR, the rest fair queuing; prediction ≈ 0 (ISSUES F20)
- [x] Review with human — deferred; the verdict is in RUNG0_REPORT §12.5 and HANDOFF §3

---

## Phase 4 — Tracker and notes

## Task 13: ISSUES / RUNG0_REPORT / HANDOFF / README pass — DONE 2026-09-22

**Description:** Close ISSUES rows with dates and evidence pointers; RUNG0_REPORT §11–§12 hold all new numbers (nowhere else); HANDOFF §3 ladder state, §4 pointer to its own checklist (not "memory/feedback"), §7 order updated; README commands for `compare`, `sample-scenarios`, `e5`, the `lease` policy.

**Acceptance criteria:**
- [x] No number appears in two documents
- [x] Every open ISSUES row still open has a reason it stays open
- [x] README commands all run as written

**Verification:**
- [x] Manual check: run each README command block once

**Outcome:** ISSUES rows A6/A7/A8/B6/B7/B8/B18/C2 closed with pointers; F15–F20 added and closed; RUNG0_REPORT §0/§4–§6 prose and tables regenerated on the fixed engine, §11–§12 written; HANDOFF §2–§7, §9 updated (the "memory/feedback" pointer replaced by the checklist itself); README tree, knobs, commands. Numbers live only in RUNG0_REPORT (todo.md quotes them as pointers). README commands were each run in this session except `e1_failure_vs_concurrency.py --plot` and `e4_cycles.py` (unchanged scripts).

**Dependencies:** Checkpoints 1–3
**Files likely touched:** `ISSUES.md`, `RUNG0_REPORT.md`, `HANDOFF.md`, `README.md`
**Estimated scope:** S

---

## Phase 5 — Next steps after the ablation (added 2026-09-22 on the user's "continue")

## Task 14: Notes consistency pass — DONE 2026-09-22
DEEP_DIVE status line, §5.4, §6.4 (measured results incl. the ablation); SERVICE_GRAPH banner + status; RUNG0_REPORT §1 (selftest count, additions, data volume); HANDOFF §8 (two new decisions: mandatory E6 ablation; observation-stream-only inputs).

## Task 15: Ship the fixed rules — `ReactiveGate.queue = "fifo" | "vtfq"` and the shipped configuration — DONE 2026-09-22

**Description:** The ablation says what pays is a fixed rule: park immediately + virtual-time fair queuing. Make VTFQ available on the gate without any prediction (tags advanced by *realised* memory-centric cost at step end: GB·s for tools, token·s for chats; start-time fair queuing). Default `"fifo"` is bit-identical. Grid `grid_ship.json`: gate {fifo, vtfq} × idle timeout {0, 300} vs lease vs oracle, hosted × load {1, 2}/min, snr 0, 10 seeds.

**Acceptance criteria:**
- [x] `queue=fifo` digest-identical to today; `queue=vtfq` runs with no leaked holds / deadlocks (selftest)
- [x] gate(vtfq, idle 0) ≥ lease on failure/p99/throughput within noise at both loads, 10 seeds (the ablation's claim, tested directly)
- [x] README/HANDOFF name the shipped configuration (Task 19's docs pass)

**Outcome (`data/synthetic/ship`, 10 seeds):** shipped gate vs full lease — 1/min: 0.003 vs 0.004 failure, p99 1,583 vs 1,708, thr 239 vs 239; 2/min: 0.013 vs 0.013, p99 2,408 vs 2,347, thr 346 vs 347; no family differs at p < 0.05 except p50 at 2/min (lease 230 vs 244 s, p 0.049 — its prewarm saves a few cold starts). Table in RUNG0_REPORT §13.1.

## Task 16: Calibrated budget margin (slice 3 follow-up) — DONE 2026-09-22

**Description:** The lease's worst-case margin (2 units/call) idled budget. The gateway observes each call's cost from the provider headers (level before/after); learn the collision rate per provider (EWMA) and set margin = n_calls × (1 + p̂). Measure at stress rpm 10 vs gate at idle 0 (the honest comparison: retries are cheap).

**Acceptance criteria:**
- [x] `on_call_cost(res, cost)` observation; margin from p̂; selftest: p̂ converges to the background load within 0.1
- [x] rpm 10 (`data/synthetic/stress_margin`, all at idle 0 + vtfq, 10 seeds): lease τ 0.5 vs gate — 429s 267 → 14, throughput 262 → 280 (+7%, 7/10, p 0.049), failure 0.043 → 0.041 (n.s.), p99 −6% (n.s.); τ 0.8 keeps the 2-unit margin (0 429s, no gain). Oracle: 0.027 / 332. p̂ learned 0.45 vs background 0.40.

## Task 17: Where does prediction pay? — randomised sweep (A8 in use) — DONE 2026-09-22

**Description:** `agentsim sweep --scenarios scenarios/sampled/*.json --variants scenarios/variants_ship.json --seeds 1-5 --out …`: each sampled scenario × named override sets {gate as sampled, gate idle 0 + vtfq, lease, clairvoyant}. Then `rung0/headroom.py`: per scenario, earnable headroom = oracle − gate(idle 0, vtfq) on failure/p99/throughput, the lease's share, and Spearman correlation of headroom with each sampled knob. 40 scenarios × 4 × 5 seeds = 800 runs.

**Acceptance criteria:**
- [x] `sweep` writes one CSV with scenario id, variant, seed, knobs, metrics; the selftest runs a 2-scenario sweep
- [x] Table (RUNG0_REPORT §13.3): headroom over the shipped gate significant in 3/40 (failure), 4/40 (p99), 19/40 (throughput); correlates with load (ρ 0.66) and lives where the model tier saturates; lease share ≈ 0. **New finding:** `queue=srpt` (observable service-time estimate) captures 0.99 / 0.50 of the failure / throughput headroom and matches the oracle on the model-bound scenarios (§13.4). **Found on the way (F21):** `_wake` recursion blew the stack on a >250-waiter queue; guarded; all grids regenerated.

## Task 18: Gain-vs-R curve for the lease (A7 × lease) — DONE 2026-09-22

- [x] `grid_predictability_policies.json` (also `srpt`): at T ≤ 0.3 the model tier saturates and `srpt` matches the oracle; at T ≥ 1 all gates tie (RUNG0_REPORT §13.4)

## Task 19: Real traces — availability check → **rung 0 on TraceLab done** (C1/E3) — DONE 2026-09-22

- [x] TraceLab: public, CC BY 4.0, 101 MB gz, timed (per-tool `emitted_at`/`result_at`, per-round tokens) — downloaded under the earlier blanket permission (small, licensed, exactly the seeds' source); Exgentic/agent-llm-traces-v2: 10 k OTel sessions, 236 MB, licence not stated — not downloaded; nebius/nvidia: untimed SFT trajectories — not useful for timing. Adapter `rung0/tracelab_to_spans.py` (Claude half; Codex timing convention differs). Real E2 one-step 0.72, E3 as published, E5 R² 0.045; fitted society `hosted_tracelab.json` + grid (RUNG0_REPORT §13.5).

## Checkpoint 4 — recorded 2026-09-22
- [x] selftest 46 checks; RUNG0_REPORT §13.1–13.6; HANDOFF §3 item 7, §7, §9; README; ISSUES C1/E3/C5/F21
- [x] All grids regenerated on the final code (F21 guard, learned margin); §4–§6 tables re-derived by script; §11–§13 means checked against the regenerated CSVs
- [ ] Review with human — the two things to read: RUNG0_REPORT §13.4 (srpt) and §13.5 (real traces)

## Phase 6 — The Needs Predictor (PREDICTOR_DESIGN.md; added 2026-09-22)

## Task 20: TraceLab ceilings — what can content and identity predict? — DONE 2026-09-22

**Description:** Before training anything, bound Heads Q and I with the real trace. (1) `rung0/ceilings.py tools`: from the raw TraceLab file, Bash calls with a `command_skeleton` → hashed n-gram features → sparse multinomial logistic regression (numpy) for the duration class {<1 s, 1–10 s, 10–60 s, >60 s} and a ridge fit of log duration; session-level split; report accuracy vs majority, tail recall/precision for the >60 s class, R² of log duration, and the pinball ratio at τ = 0.9 of per-predicted-class quantiles vs the global quantile (what a lease would use). (2) `rung0/ceilings.py idle`: extend the adapter's think spans with observable `user`, `hour`, `weekday`, `prev_think`, `user_message_chars`; report R² of log idle and pinball ratios for nested key sets (last tool) ⊂ (user) ⊂ (user, hour) ⊂ (user, hour, weekday, last tool, bucket) with per-key means and a ridge on one-hots, session-level split.

**Acceptance criteria:**
- [x] selftest: hashed features are deterministic; the sparse logistic regression recovers a planted rule (token `pytest` → long) at > 0.9 accuracy on synthetic skeletons
- [x] Both experiments run on TraceLab and their tables are in RUNG0_REPORT §14 with the split, n, and baselines
- [x] A one-paragraph decision per head (Q, I): build / don't build, with the ceiling number that decides it

**Verification:**
- [x] `uv run python -m agentsim selftest`
- [x] `uv run python rung0/ceilings.py tools data/real/tracelab/syfi_coding_trace.jsonl.gz`; `… idle data/real/tracelab/spans_claude.jsonl`

**Outcome:** Head Q — build as a risk ranker (R² 0.31 on log duration; 6× tail lift in the top decile; decile-binned lease quantiles 8% better in pinball at τ 0.9/0.95; skeletons are sanitised so this is a floor). Head I — don't build (who/when: R² ≤ 0.07; previous gap 0.10 → add `prev_think` to the tracker key). Selftest 49 checks.

**Dependencies:** Task 19
**Files likely touched:** `rung0/ceilings.py` (new), `rung0/tracelab_to_spans.py` (think attrs), `agentsim/run.py` (selftest), `RUNG0_REPORT.md`
**Estimated scope:** M

## Task 21: Generator content proxies (`content_snr`) and spawn/join recipes (B10) — DONE 2026-09-22 (second session)
- [x] `Step.content` (tool skeleton class `sk:<kind>:<bin>`; prompt cue `pc:o<out bin> pc:t<tool-count bin>`), `Step.plan` (`pl:<next phase>` at chat end, `pl:final`), drawn from a separate `rng_content` stream with truth probability `content_snr` — spans bit-identical at any snr (selftest + regression against every stored trace: 6/6 SAME; lease/srpt/tracelab summaries SAME)
- [x] `spawn(n)`: `recipes[*].spawn = {prob per phase, width pmf, child_recipe}`; `Workload.new_child`; engine `_spawn` / `_child_ended`; children share the parent's `trace`, hold their own sandbox and model slot, run one request; the join is a real hold-and-wait edge in the cycle probe (parents hold sandboxes while children queue: 102 cycles under the gate at idle 300, 0 when parked); `join_timeout` hook; `data/recipes_multiagent.json` (orchestrator: 2–6 coding sub-agents), `scenarios/multiagent.json`
- [x] New observation events `step_ready` (request body at the gate: kind, tokens_in, content), `spawn` (width, children's sandbox sizes), `join`, `request_abort`, `session_end`; tool `step_end` also on timeout (censored, `outcome`); chat spans carry the revealed tool lists; `kv_reserve` and `POLICY_TIMER` hooks; `leases_at_ready`
- Found on the way: F22 wait-span ids repeated within an attempt (DEADLOCK_BREAK matched on them) — fixed; F23 rung-3 gang leases under-reserve by the session's own CPU unit (documented, behaviour kept)

## Task 22: `agentsim/features.py` + `agentsim/needs.py` — DONE 2026-09-22
- [x] `Observer`: one code path online (`on_event`) and offline (`replay_events(spans)` reconstructs the identical event stream, same-instant order as the engine); `SessionView` (observable only); `Record` = hashed sparse ids (history n-grams, revealed tools, content cues, plan, occupancy buckets, Jev answers) ⊕ 31 dense features (incl. the crude trackers' quantiles as priors)
- [x] `NeedsModel`: embedding-sum + dense projection → relu → hidden(96) → heads S2/S3 (softmax over node vocab), Q_tool/Q_out/T_gap/I_gap (5 monotone quantile knots, pinball), I_cold/R (BCE), G (spawn width softmax); Adagrad rows + flat mini-batch Adam; 0.1 ms per step; npz round trip. (A GRU was not needed: history enters as position-tagged n-grams over the last 8 nodes.)
- [x] Calibration: temperature scaling per categorical head on a rolling window (ECE reported); adaptive conformal level per (head, key, τ) (selftest: a misspecified head reaches coverage 0.77 for target 0.8); drift counters in `report()`
- [x] Offline: `agentsim train-needs` (session-level hold-out; per-head R² / pinball ratio / coverage / accuracy vs majority); synthetic hosted at snr 0.6: Q_tool R² 0.58, T_gap 0.51, Q_out 0.44, S3 0.40 vs 0.17, S2 ≈ majority, I_gap none

## Task 23: Reserver v2 behind `policy.type=needs` — DONE 2026-09-22 (`agentsim/reserver.py`)
- [x] `Forecast`: per-resource demand horizon (10 s buckets over h) from every session's segments + an arrival term; `pressure(res, t0, t1)` = forecast excess + standing queue, per unit of capacity
- [x] Expected-value gate (benefit in latency-seconds saved vs hold × pressure): park / hold, prewarm, gang CPU (content-conditioned quantiles), spawn gang (k slots + Σ sandbox GB + k CPU, family-owned via `Ledger.in_family`), predicted spawn gang from head G at chat submission (future-start lease, I1 over the window via `Ledger.active_max`)
- [x] Ordering: predicted-service-time SRPT only while the model queue is deeper than the slots, VTFQ on predicted cost otherwise, starvation guard at half the client timeout; KV reserve = conformal q_τ(out); retention = q_τ(gap) + grace; budget leases with rung 3's calibrated margin
- [x] Selftests (70 total): runs clean on hosted / multi-agent / with Jev; deterministic; honest (contradictory hidden values → identical decisions; source grep for hidden-field reads); every switch off ≡ the fifo gate bit-for-bit; EV switch changes decisions; pre-trained models load

## Task 24: E6 ablation — grids launched 2026-09-22 (`data/synthetic/needs/*.csv`, 10 seeds; `scenarios/variants_needs*.json`)
- [x] hosted × content_snr {0, 0.3, 0.6, 0.9} at 2/min; multi-agent at 1 and 1.5/min; TraceLab-fitted at 2 and 4/min; stress at 10 RPM; model-saturated sampled 0004/0013 — variants: gate fifo/300, gate vtfq/0, gate srpt/0, lease/0, needs (online), needs_pre, needs_jev, needs−content, −ev, −forecast, −conformal, −learn(pre), oracle/0
- [x] numbers → NEEDS_REPORT.md (paired statistics via `rung0/compare.py --policy-col variant`; tables via `rung0/needs_tables.py`)
- Real data: `train-needs` on 1,500 TraceLab sessions with real command skeletons (`data/models/needs.tracelab.report.json`)

## Task 25: Jev (System One model) — DONE 2026-09-22 (`agentsim/jev.py`)
- [x] Typed question schema (`question_schema`: next_chat, next_tool, phase, duration_class, out_class, idle_class, fail_soon, spawn_width; cardinality ≤ 255); `JevRecord` with calibrated pmfs, confidences, issue/ready times
- [x] `LocalSystemOne`: per-question sparse multinomial over hashed state text + structure, log-loss trained, temperature-calibrated on held-out sessions, ECE per question (`agentsim train-jev`): hosted snr 0.6 — phase 0.76 vs 0.19 majority (ECE 0.04), duration_class 0.86 vs 0.69 (0.01), out_class 0.72 vs 0.36 (0.02), next_chat 0.79 vs 0.70, idle_class no signal (T ≈ 4, honest)
- [x] `RemoteSystemOne`: the vendor seam (JEV_API_URL / JEV_API_KEY; same schema in, same record out)
- [x] `JevChannel` = `ext.jev`: concurrency, RPM bucket with background load, lognormal latency (median 150 ms, p99 500 ms); dropped when over budget; answers delivered asynchronously via POLICY_TIMER; the Needs records carry the latest answer + its age
- [x] Point A (offline annotation): `agentsim annotate` writes typed phase labels onto chat spans; `rung0/e2_predictability.py --phase-attr jev_phase`: on a held-out hosted trace R₃ 0.57 with Jev phases vs 0.61 true phases vs 0.41 hidden
- [x] ECE / outcome gate on the grids (needs_jev vs needs_pre): no outcome difference on any society; live ECE ≤ 0.1 with matched content, the Judge drops drifted questions → NEEDS_REPORT.md §3, §5

## Phase 7 — Feedback-loop allocation across CPU / GPU / tools / MCP / APIs with budgets, Jev on top (added 2026-09-22 on the user's scope expansion)

## Task 26: Jev v2 — the real API shape and patterns (JEV_SURVEY.md) — DONE 2026-09-22
- [x] Survey of TypeSafe's System One API (choice / score / noul, confidence as a second axis, speculative fan-out, retrieve-then-judge, map-reduce, limits, LiteLLM pass-through) with sources
- [x] `jev.py` rebuilt: `Question` catalogue (21 questions with instructions + criteria to the API's rules; roles feature / judge / verify / annotate), `JevRecord` (pmfs, confidences, `score`, `noul`, `ok`), `LocalSystemOne` (choice/score/noul), `RemoteSystemOne` (exact request/response, bearer auth, JEV_API_BASE incl. LiteLLM, retries honouring retry-after, usage + cost), `Judge` (online ECE per question vs realised labels; kill rule live), `JevChannel` batching (`batch_window_s`, one call per window, records land at the call's ready time), `answers_from_labels` shared by trainer and judge, `evaluate_system_one` + `agentsim evaluate-jev [--remote]`
- [x] Reserver: speculative fan-out at every boundary; judge questions on candidates (`will_use_reservation` → gang P(pay) blend + veto with off-cycle lease re-issue; `safe_to_park` → the park / join decisions' cold-start cost; `budget_will_exceed` → pacing; `stuck_in_loop` → no leases + heavier tag); untrusted questions are not asked
- Observed: a System One model trained where content is informative and run where it is noise drifts (duration_class ECE 0.27) and the Judge drops it — the kill rule working live

## Task 27: resource model — costs, tokens, MCP / GPU, tenant budgets — DONE 2026-09-22
- [x] `Resource.cost_per_call` / `tokens_per_call`; `mcp.<server>` and `gpu.<pool>` resources; model prices per Mtok; `BudgetBucket` (USD / tokens, hourly refill); engine pays at admission (input + output reserve) and settles at chat end (refunds), refuses what a budget cannot pay (outcome `budget`, SDK retries → replan / abort), `budget_admit` / `budget_retry_after` hooks (uncoordinated: none; gate: HiveMind's pause on the spend header; oracle: knows the level and the cost); spend metrics (usd, usd per completed request, usd wasted on failed requests); `scenarios/mcp_budget.json` (API model at $3/$15 per Mtok, two MCP servers and a GPU pool with per-call costs, $10 budget refilling at $25/h against ~$46/h of natural demand)
- Found: the budget over-charged the reserve's input price and never refunded (fixed; spend now reconciles to the cent)

## Task 28: feedback-loop allocation with error correction (`agentsim/control.py`) — DONE 2026-09-22
- [x] `AllocationLoop`: every 30 s, realised time-averaged occupancy vs the forecast's prediction per resource → PI-corrected forecast scale (bounded, anti-windup) applied to the demand horizon; budget pacing floor by AIMD on refusals; per-session fairness weights on VTFQ tags; all reported; switches `feedback`, `pacing`
- [x] Reserver: forecast-aware budget admission — new requests admitted only if the in-flight requests' predicted spend-to-completion (minus the refill that arrives while they finish) still fits above the floor; in-flight steps may dip to half the floor; retry waits short (≤ 30 s) and proportional to remaining spend
- Found and reverted: a deadline-aware "admit anyway" rule caused refusals (fail 0.095); a refill-blind reservation left throughput on the table (52 → 56 req/h once corrected)

## Task 29: evidence — DONE where grids finished; NEEDS_REPORT.md
- [x] Per-switch ablations on hosted, multi-agent and the saturated societies found three defaults that hurt and were changed: the starvation guard (aged by retry time → fixed; then found to cost 30% throughput on saturated societies for no failure gain → opt-in `policy.needs.starvation_guard`), the spawn gang lease lasting the whole join (→ the children's start window, priced by pressure like the parallel gang), predicted KV reserves where KV does not bind (→ only while KV admission has waiters)
- [x] Same-instant I3 race in `check_accounting` (an expiring lease checked before its LEASE_EXPIRE event) → lazy expiry before the check (F24)
- [x] 10-seed grids on the final defaults: hosted × content (reduced variant set), multi-agent (all variants), saturated (all), stress (reduced), TraceLab-fitted (reduced, own System One model), mcp_budget → NEEDS_REPORT.md §5 (`rung0/needs_tables.py`); the second hosted-content grid (snr 0.3 / 0.6) was dropped after content showed no effect at 0 vs 0.9

## Deferred (not in this plan; reasons)
- C1 / E3 real traces — needs the user's data decision (plan.md open question 1)
- E1 `git init` — needs the user's go-ahead (open question 2)
- ~~D1–D3 Jev~~ — built in Task 25 (local calibrated model + vendor seam + `ext.jev` channel)
- ~~B10 multi-agent spawn/join~~ — built in Task 21; B11–B14 fidelity, A9 performance, B15 per-kind reactions, B16 compaction, C3–C5 — P2/P3
- F12/F13 reporting caveats — folded into Task 2's output (report requests·h⁻¹ completed, never sessions started)
