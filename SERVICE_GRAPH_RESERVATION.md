# Runtime Service-Graph Reservation for Multi-Agent Systems

**Investigation notes: literature survey, how existing controllers are written, and the synthetic → real training ladder.**

> **Audit (2026-09-22):** the first synthetic results were inflated on both sides (ISSUES.md §F); RUNG0_REPORT.md v2 holds the corrected numbers. H1 and H4 survive with smaller magnitudes; the oracle showed no prediction headroom on the API tier.
>
> **v2 (2026-09-22):** the deeper survey, the JEPA adaptation, synthetic-dataset design, its limitations, the realistic path and the development ladder are in [DEEP_DIVE_AND_LADDER.md](DEEP_DIVE_AND_LADDER.md). Its §1.9 lists the decisions in this file that it revises (starvation mechanism, admission stability, a third Predictor head, the oracle definition, where the Gate lives).
>
> **Rung 3 built and ablated (2026-09-22 evening):** the controller of §6 exists as `agentsim/policies.py::LeaseController` (leases with invariants I1/I3 of §6.6, gang acquisition, virtual-time fair queuing, predictions only from observable events). Scored against the reactive gate and a clairvoyant oracle on synthetic societies, its gain is ~90% the gate's idle-timeout knob and fair queuing; the learned part does not pay (RUNG0_REPORT.md §12, ISSUES.md F20). H1 and H4 stand; H5 is a property of the generator (`recipe_temperature`); the "solved" criteria of §5.3 in v2 are met on synthetic data by fixed rules alone.

- Date: 2026-09-21
- Status (2026-09-21): ideation, no code yet. See the banner above for what exists now.
- Method: detective. Theory of the crime → evidence → only then the fix. Every claim below carries its evidence; anything unverified is marked.
- Design principles in force: simple > complex, one way, no fallbacks, fail fast (throw), single responsibility, surgical changes, evidence-based.

---

## 0. One-screen summary

**The crime.** N concurrent agentic workflows share a pool of *models* (KV cache, tokens/min, request slots), *tools* (concurrency slots, external API quotas, sandboxes) and *services* (retrieval, DBs, queues). Uncoordinated, they over-allocate (hold what they don't use), starve each other, and die (429 / 502 / connection reset / timeout / deadlock) — **even when aggregate capacity is sufficient**.

**Theory of the crime, after evidence (§2).**

| # | Claim | Verdict | Key evidence |
|---|---|---|---|
| H1 | Failures come from *uncoordinated bursts*, not from missing capacity | Confirmed | HiveMind incident: 3 of 11 parallel agents died (27%) although the API could serve all 11 sequentially; uncoordinated agents fail 72–100% under contention, a proxy with retry+admission brings it to 0–18% |
| H2 | Future demand is revealed *incrementally* and is *heavy-tailed*; "declare your max claim up front" (Banker's) is impossible | Confirmed | HexAGenT ("dependencies revealed incrementally"), PBKV ("practical workflows are dynamic"), TraceLab: tool calls >1 min are 4.9% of calls but 92% of tool time |
| H3 | Point predictions are the wrong artifact; scheduling quality is *non-monotone* in prediction accuracy | Confirmed | Tsafrir–Feitelson (inaccurate runtime estimates can beat accurate ones), learning-to-rank (rank suffices), TIE (log-t + CVaR), PBKV Thm 5.1 (regret linear in error only if predictions are used conservatively) |
| H4 | Deadlock is real; LLM agents cannot self-resolve it; the *protocol* decides | Confirmed | DPBench: 25–90% deadlock under simultaneous action; 0% when a resource-ordering primitive is imposed |
| H5 | Next-step structure *is* predictable at short horizon and decays fast | Confirmed | PBKV: 0.94 accuracy at 1 step → 0.77 at 3 steps; CacheScout: gains vanish when routing is random; Speculative Actions: ≤55% next-action accuracy |
| H6 | The three tiers have different reservation currencies and the bottleneck *moves* between them | Confirmed | AgentSysBench: non-LLM components dominate latency in 5 of 10 apps; task latencies diverge 32×; bottlenecks shift across requests |

**The fix (§6).** Four single-purpose components. Observer (online-revealed service graph) → Predictor (K-step demand *distributions* per tier) → Reserver (τ-quantile *leases* with horizon h; backfillable; all-or-nothing for multi-tier steps; globally ordered) → Gate (the single admission path; violations throw). Over-allocation is bounded by lease horizon + backfill; starvation is prevented by aging + dominant-share fairness; deadlock is impossible by construction (three of the four Coffman conditions are broken).

**The ladder (§7).** Yes, train on synthetic first, then fine-tune on a real society — with three conditions. (a) "Synthetic" must mean *sampled from a generator fitted to real trace marginals, domain-randomized over the knobs that differ between deployments* (the Decima recipe), not hand-drawn DAGs. (b) What climbs the ladder is the **Predictor + its calibration** (supervised, cheap, stable), not the reservation rule (a fixed algorithm with two knobs). (c) Success is measured by scheduling outcomes (failure rate, p99 completion, waste, fairness) — never by predictor accuracy. Conformal recalibration on a small real set is what closes the sim-to-real gap for the quantile the lease depends on.

---

## 1. Problem statement, precisely

### 1.1 Resources: three tiers, three currencies

| Tier | Unit that gets reserved | Failure when exhausted | Lead time to add capacity | Evidence |
|---|---|---|---|---|
| **Model** | KV-cache blocks; tokens/min and requests/min per endpoint or key; decode slots | 429 (TPM/RPM); queueing → TTFT blow-up; KV eviction → recompute | new replica 2–10 min (weights load); model/adapter prewarm seconds–minutes | Predictive K8s autoscaling [R31]; Hermes prewarm [R3]; WarmServe [R32] |
| **Tool** | concurrency slots; external API quota; CPU-bound sandboxes; sandbox memory (peaks 28 GB/session) | 429 / 502 / connection reset; timeouts; OOM | container start seconds; durations heavy-tailed (mean 16.8 s; >1 min calls are 4.9% of calls and 92% of tool time) | HiveMind [R12]; TraceLab [R21]; AgentSysBench [R23] |
| **Service** | retrieval / DB / queue capacity along the call graph | latency cascades; hotspot services | call graphs tree-like, heavy-tailed, with hotspots | Alibaba microservice traces [R25] |

### 1.2 Failure modes (definitions used throughout)

- **Over-allocation**: capacity-time that is reserved/held but not used, and that blocks someone else. Metric: Σ (reserved − used) × duration, per tier.
- **Starvation**: a workflow whose wait grows without bound or that is never admitted. Metric: max queue wait; Jain fairness of dominant shares.
- **Workflow failure**: 429 / 502 / reset / timeout / killed-by-budget / deadlock / livelock. Metric: fraction of workflows that do not complete; wasted compute of failed workflows.
- **Latency**: p50 / p99 workflow completion time (the user experiences the *workflow*, not the call — HexAGenT [R6]).

### 1.3 What "service-graph reservation" means here

A workflow's **service graph** is the DAG — revealed online, one step at a time — of `(agent step → model | tool | service)` edges it touches. **Reservation** is holding a *lease* on capacity along the *predicted* future sub-graph within a horizon h, so that when demand materialises the step is admitted immediately. A lease is **not** an exclusive lock: capacity under a lease is backfillable by anything that will finish before the lease's start (§6.2). This is the resolution of the over-allocation-vs-failure tension.

"Society of LLMs" in this doc = a deployed multi-agent system of heterogeneous LLM agents with real tools and services (AutoGen / CrewAI / LangGraph / MetaGPT-style), as used in the PBKV and CacheScout evaluations.

---

## 2. Theory of the crime — hypotheses and evidence

### H1. The failures are caused by uncoordinated bursts, not by insufficient capacity

*Supporting.* HiveMind [R12]: in a real incident (15 Apr 2026), 3 of 11 parallel coding agents died from connection resets and HTTP 502 "despite the API having sufficient aggregate capacity to serve all 11 sequentially". Across 5–50 concurrent agents, uncoordinated execution fails at 72–100%; a transparent proxy with admission control, rate-limit tracking, AIMD backpressure, token budgets and priority queuing reduces failures to 0–18% and removes 48–100% of wasted compute. Ablation: **transparent retry with backoff is the single most important primitive** — not admission control and not prediction. Practitioner reports [R13] describe the same mechanism: 20 agents sharing one key hit 429s in seconds; naive per-agent retry produces a thundering herd that re-triggers the limit.

*Contrary.* None found. Note HiveMind's own limits: single machine, mock API, 4-chars-per-token estimate, static priorities.

*Verdict.* Confirmed. *Implication:* the first job of the controller is coordination (one gate, coordinated retry). Prediction is what lets coordination be *proactive* rather than reactive, but a controller whose prediction is switched off must still remove H1 failures.

### H2. Demand is revealed incrementally and is heavy-tailed; up-front maximum claims are impossible

*Supporting.* HexAGenT [R6] models each request "as an online-revealed DAG" because "workflow dependencies are revealed incrementally at runtime". PBKV [R8]: "practical workflows are typically dynamic" (retry loops, conditional refinement). Hermes [R3] profiles each application 1000× and still concludes profiling "can only mitigate but not eliminate demand uncertainty". TraceLab [R21] (4,265 Claude Code / Codex sessions, 432,510 tool calls): tool calls under 1 s are 70% of calls but <1% of tool time; calls over 1 min are 4.9% of calls but 92% of tool time; mean tool-call latency 16.8 s (Codex's *residual* overhead beyond tool execution averages 1.11 s, p99 10.0 s — Table 10). Output length is heavy-tailed and log-t distributed (TIE [R17]). Copilot production traces [R22] (13 M sessions, 761 M LLM calls): "variable and long-tailed token consumption, time span, and tool calls".

*Contrary.* Static workflows exist (FinanceBench + CrewAI in PBKV is static) — for those, Hermes-style profiling nearly suffices (Hermes is within 10% of an oracle).

*Verdict.* Confirmed for the general case. *Implication:* Banker's algorithm cannot be applied as-is (it needs `max[i]` at admission); any reservation must be probabilistic, short-horizon and self-expiring.

### H3. Point predictions are the wrong artifact; scheduling quality is non-monotone in prediction accuracy

*Supporting.* Tsafrir & Feitelson [R14]: under EASY backfilling, *inaccurate* user runtime estimates produced *better* schedules than accurate ones ("heel-and-toe" dynamics make backfilling approximate SJF); real users cluster 90% of jobs on ~20 estimate values. Learning-to-Rank for LLM scheduling [R16]: exact length prediction "is difficult yet unnecessary"; relative rank suffices. TIE [R17]: fit a log-t distribution and rank by CVaR instead of a point estimate — 2.31× lower per-token latency. PBKV [R8] Theorem 5.1: eviction regret is 0 for perfect prediction and grows *linearly* in prediction error — but only because predictions are used conservatively (retired-cache-first eviction; prefetch only into idle space on decode-only batches). CacheScout [R9]: a multiplicative blend that "naturally falls back toward LRU" when the transition graph is uninformative. Learning-augmented algorithms [R18]: the consistency–robustness trade-off is the formal statement of this.

*Contrary.* None. But note that "distribution" must be *calibrated* — a confidently wrong distribution is worse than a point estimate. Conformal prediction [R19] gives distribution-free coverage from a small calibration set.

*Verdict.* Confirmed. *Implication:* the Predictor must emit *distributions* (quantiles); the Reserver must consume a *quantile*, and the damage of a wrong quantile must be bounded by construction (lease horizon, backfill), not by a second code path.

### H4. Deadlock is real; LLM agents cannot self-resolve it; the protocol decides

*Supporting.* DPBench [R28] (Dining Philosophers with GPT-5.2, Claude Opus 4.5, Grok 4.1, Gemini 2.5 Flash, Llama 4 Maverick): under simultaneous action at N=5, deadlock ranges from 25.0% (GPT-5.2) to 90.0% (Gemini 2.5 Flash). Holding the model fixed, a prompt encoding a classical primitive (resource ordering or symmetry breaking) drives deadlock to 0.0% versus 100% for the minimal prompt; three rounds of pre-commitment communication also give 0.0% vs 86.7% for a single round. "Whether the same model coordinates or deadlocks is determined by the protocol, not by the model's capability." MAST [R27] (1,600+ annotated multi-agent traces, 14 failure modes) finds most failures are system-design and coordination failures, not model failures.

*Contrary.* Sequential (one-at-a-time) execution removes deadlock in DPBench but is exactly the throughput we're trying not to give up.

*Verdict.* Confirmed. *Implication:* deadlock freedom must be a property of the *controller's* acquisition protocol (ordering + all-or-nothing + expiry), never delegated to agents' judgement.

### H5. Next-step structure is predictable at short horizon and decays fast

*Supporting.* PBKV [R8]: GraphSAGE + prefix attention + last-prefill hidden state, trained on 1 K traces, reaches 0.935 (1-step), 0.848 (2-step), 0.771 (3-step) next-agent accuracy; a Markov baseline is worse but usable. CacheScout [R9]: an *online* first-order Markov chain with no offline training lifts KV hit rate 10–18 pp; "gains diminish when execution approaches random routing". KVFlow [R7]: a static "agent step graph" with steps-to-execution distance already yields 1.8–2.2× when the workflow is known. Speculative Actions [R11]: a fast model predicts the next action with up to 55% accuracy → up to 20% latency reduction. Copilot [R22]: a lightweight idle-time predictor captures 86–90% of total idle time.

*Contrary.* PBKV's own limitation: "the predictor needs to be trained on a specific workload" — cross-workload transfer is not demonstrated. Accuracy at 3 steps (0.77) is already too low to *hold* anything expensive on.

*Verdict.* Confirmed with a horizon caveat. *Implication:* lease horizon h ≈ 1–2 predicted steps. Longer-horizon predictions may *prewarm* (cheap, Hermes-style) but must not *reserve*.

### H6. The three tiers have different currencies and the bottleneck moves between them

*Supporting.* AgentSysBench [R23] (10 agentic apps + production traces): non-LLM components dominate latency in 5 of 10 apps; components have heterogeneous affinity (GPU inference, memory-bound retrieval, CPU-bound sandboxes) with task latencies diverging up to 32×; "bottlenecks shift across requests, models, and deployments"; sessions hold state idle for minutes to hours; a "control-plane tax" of auxiliary LLM calls crowds out productive compute; tool-result caching removes 35.2% of redundant search calls. Model-tier capacity has minutes of lead time [R31]; tool-tier quotas are opaque and dynamic [R12][R13]; service graphs are tree-like with hotspots [R25].

*Verdict.* Confirmed. *Implication:* one controller, one lease primitive, but per-tier capacity vectors and per-tier demand distributions. A single-tier controller (every existing system, §5) is always wrong somewhere.

---

## 3. Literature survey

Grouped by sub-problem. Each entry: what it does, the number that matters, and the gap relative to our problem.

### 3.1 Workflow-aware LLM serving (program/DAG-level scheduling)

- **Parrot** (OSDI '24) [R1] — *Semantic Variable* exposes the application-level DAG and prompt structure to a public LLM service so it can optimise across requests instead of blindly per request. *Gap:* scheduling of LLM calls only; no tools/services; no reservation.
- **Autellix** (2025) [R2] — treats programs as first-class; PLAS/ATLAS are *non-clairvoyant* multi-level-feedback policies that prioritise calls by the program's attained service. 4–15× program throughput at equal latency vs vLLM/SGLang. *Gap:* deliberately no prediction; single tier; no admission or reservation.
- **Hermes** (TACO 2025/26) [R3] — Probabilistic Demand Graph (PDGraph): per functional unit, a distribution of backend demand and branch probabilities; Gittins-index queueing; LSTF for deadlines; probabilistic backend *prewarming*. >70% improvement; within 10% of an oracle. *Gap:* prewarm ≠ reserve; per-application offline profiling (1000 runs); static workflow templates.
- **HexAGenT** (2026) [R6] — online-revealed DAG; running estimate of each workflow's standalone completion horizon; prioritises ready calls by *projected risk of missing that horizon*; joint prefill/decode placement on heterogeneous GPUs. SLO scale reduced 20.1% (p95) / 33.0% (p99). *Gap:* model tier only; no tools; no reservation semantics.
- **Murakkab** (OSDI '26) [R4] — declarative workflow spec decoupled from execution config; profile-guided optimiser + adaptive runtime chooses models and hardware to meet SLOs. 2.8× less GPU, 4.3× less cost. *Gap:* placement/configuration, not runtime reservation under contention.
- **Helium** (2026) [R5] — models agentic workloads as *query plans* with LLM invocations as operators; proactive caching + cache-aware scheduling. 1.56×. *Gap:* caching/reuse, not capacity control.
- **Dyserve** (2026) [R10] — physical-plan compiler + adaptive runtime selects model / verification policy / backend per node; under bursts, variant switching raises on-time correct completions 18.1% → 67.2%. *Gap:* degrades quality to survive bursts; no cross-workflow reservation.
- **Chimera** (2026) [R33] — heterogeneous-LLM multi-agent serving: semantic router for per-model confidence, a CPU regressor predicting *remaining output tokens of the whole workflow*, and an activity monitor tracking in-flight predicted token volume per engine. *Gap:* model tier only, but its "in-flight predicted volume per engine" is a direct ancestor of our lease ledger.
- **SMetric** (2026) [R34] — session-centric routing for agentic serving (spread first request for balance, then cache-affinity). *Gap:* routing, not reservation.
- **Continuum** (2025) [R35] — KV cache time-to-live so tool gaps don't evict live agents' state; >8× JCT on SWE-bench/BFCL/OpenHands. *Gap:* memory tier only; TTL is a fixed-horizon *hold* — the primitive we generalise.

### 3.2 Predicting future dependencies of a workflow

- **KVFlow** (NeurIPS '25) [R7] — *Agent Step Graph*; each agent gets a steps-to-execution distance that drives KV eviction and overlapped prefetch. 1.83× / 2.19×. Assumes the workflow is known.
- **PBKV** (2026) [R8] — GraphSAGE over an offline-estimated transition graph + attention over the executed prefix + the last prefill hidden state → MLP → logits over agents for each of the next K steps; ~350 K parameters; 1.56 ms per 1,024 requests; 0.94 (1-step) / 0.77 (3-step). Conservative use: retired-cache-first eviction; prefetch only into idle GPU space. Regret bound linear in prediction error. Workload-specific.
- **CacheScout** (2026) [R9] — online first-order Markov transition counts on prompt-prefix fingerprints; survival probability by BFS over the transition graph; score = survival × recency × size; prefetch gated by an entropy-reduction metric R ≥ R_min. No training at all. +10–18 pp hit rate.
- **Speculative Actions** (ICLR '26) [R11] — a fast Speculator predicts the next action and pre-launches safe/reversible calls while the slow Actor deliberates; commit only on match. ≤55% accuracy → ≤20% latency.
- **Hermes** PDGraph [R3] — branch-taking probability = historical jumping frequency; demand conditioned on upstream observations via Pearson correlation (ρ > 0.5) + Monte Carlo.
- **Chimera** [R33] — remaining-workflow-token regressor (CPU).

Pattern: the *structure* (which agent/tool next) is predicted from the executed prefix; the *quantity* (tokens, duration) is predicted as a distribution conditioned on the same prefix. Nobody predicts *service*-tier demand.

### 3.3 Tool-aware serving and coordination proxies

- **InferCept** (ICML '24) [R36] — tool calls are *interceptions*, not terminations; decides per intercept whether to keep / swap / discard KV state using duration estimates.
- **Conveyor** (2024) [R37] — tool *partial execution* overlapped with decoding; ≤38.8% latency.
- **HiveMind** (2026) [R12] — transparent HTTP proxy with five OS primitives (admission via condition variable `A < Cmax`, provider-aware rate-limit tracking, AIMD backpressure + circuit breaker, per-agent token budgets from a global pool, priority queue ordered by (priority, estimated tokens, FIFO)). Zero agent changes. Retry is the most important primitive. <3 ms overhead.
- Practitioner guidance [R13] — centralise per-key token buckets (~80% of tier), a concurrency semaphore, and a priority queue; adaptive quota (−20% on >5% 429s, +10% after 10 clean minutes).

### 3.4 Classical reservation, fairness and deadlock theory

- **Banker's algorithm** (Dijkstra) [R38] — safe-state check against declared maximum claims; conservative (rejects some deadlock-free states); no time dimension. Inapplicable as-is (H2) but its *invariant* — never grant a request that leaves the system unsafe — is the right shape.
- **Coffman conditions** [R39] — deadlock needs all four: mutual exclusion, hold-and-wait, no preemption, circular wait. Break any one.
- **DRF** (NSDI '11) [R15] — max-min over *dominant shares* across heterogeneous resources; strategy-proof, envy-free, Pareto-efficient. Our fairness baseline across tiers.
- **EASY backfilling + Tsafrir–Feitelson** [R14] — reserve for the head-of-queue job at the earliest feasible time; let shorter jobs backfill gaps that don't delay the reservation. Over-estimation is *harmless* under backfilling. This is the mechanism that makes conservative quantile reservation affordable.
- **Gang scheduling / placement groups** (Ray, Kubernetes coscheduling) [R40] — all-or-nothing binding for a group; prevents partial allocations that hold expensive resources while waiting for the rest.
- **Learning-augmented online algorithms** [R18] — consistency (good when predictions are good) vs robustness (bounded when they are bad); optimal trade-offs for ski-rental and non-clairvoyant scheduling.

### 3.5 Prediction under uncertainty for schedulers

- **Learning to Rank** (NeurIPS '24) [R16] — rank, not length.
- **TIE** (ICML '26) [R17] — log-t output-length distribution + CVaR ranking; 2.31× per-token latency, 1.42× throughput.
- **Beyond Prediction: Tail-Aware Scheduling** (2026) [R41], **Robust Length Prediction** (2026) [R42] — heavy-tailed, prompt-conditioned distributions; tail awareness beats accuracy.
- **Conformal prediction** [R19] — distribution-free coverage for quantiles from a small calibration set; the tool for recalibrating a synthetic-trained predictor on real data.

### 3.6 Learned schedulers and sim-to-real

- **Decima** (SIGCOMM '19) [R20] — GNN over job DAGs (per-node, per-DAG summary node, global summary) + policy gradient. Simulator built from *real* profiled task durations and an Alibaba production trace (~20 K jobs); models warm-up, JVM start-up, parallelism degradation. Training tricks for continuous arrivals: curriculum on episode length with *exponentially random* termination; *input-dependent baselines* (fix the arrival sequence across episodes). Deployed to a 25-node Spark cluster with **no fine-tuning**; ≥21% lower JCT, up to 2× under load. Generalisation: a policy trained on a narrow arrival distribution "generalizes poorly"; training on a *mixed* range with the knob exposed as an input feature restores it (16% over best heuristic).
- **When Simulation Lies** (2026) [R30] — sim-to-real gap for *tool-use agents* as POMDP perturbations (observation / action / reward-metadata / transition): observation noise costs <5% accuracy, reward-metadata ~40%, transition dynamics ~30%; scale alone does not close it. Domain-randomised RL on static perturbations transfers to unseen runtime failures (closes ~27% of the transition gap never seen in training).
- **AgentServeSim** (2026) [R24] — simulator whose unit of execution is the *agent program* (Program Control Block, Orchestrator releasing successor turns causally, Retention Plane for KV across tool gaps, Dispatch Plane); validated against real vLLM in 20 paired cells: mean JCT error ≤ 5.5%. Used as a CPU fitness evaluator for policy search.

### 3.7 Workload characterisation and public traces

- **TraceLab** (2026) [R21] — 4,265 sessions / 357,161 LLM steps / 432,510 tool calls from real Claude Code and Codex use; per session 9.2 requests and 73.6 tool-initiated steps; tool-call mean 16.8 s (Codex residual overhead 1.11 s avg, p99 10.0 s — Table 10); output median 252 / p99 6,571 tokens; think time median 1.4 min, p90 20.6 min; heavy tail as above; 95.7% token-weighted prefix-cache hit rate. Dataset and pipeline public.
- **GitHub Copilot traces** (2026) [R22] — 3.2 M users, 13 M sessions, 761 M LLM calls, 95 T tokens; KV hit 90% within a turn, 55% across turns, invalidated on model switch / compaction; idle-time predictor captures 86–90% of idle time.
- **AgentSysBench** (2026) [R23] — the six properties in H6.
- **Exgentic/agent-llm-traces** (HF) [R26] — 1,781 replayable traces across 6 benchmarks with per-call token counts, timestamps, tool calls and tool schemas; DAG per session recoverable.
- **Alibaba microservice traces 2021/2022** [R25] — ~20 K microservices, >10 K nodes, 13 days; call graphs heavy-tailed, tree-like, hotspot services. The best public *service-tier* graph data.
- Gap noted in several 2026 papers: no public trace exists that spans model + tool + service tiers with capacity/rate-limit events. We will have to record our own (§7.2).

### 3.8 Synthetic workflow and trace generation

- **SyntheticAgentTraceQA** (2026) [R43] — Profiler extracts tool metadata → Template Generator makes abstract DAG workflows → DFS instantiates executable traces under data-flow constraints → execution validates.
- **AgentSim** (2026) [R44] — visual workflow design + CLI for scalable trace creation; corpus-aware seeding; active validation.
- **ESAT** (2026) [R45] — environment-free: an LLM simulates tool feedback from API specs.
- These generate *semantic* traces for training agents. For a *controller* we need *systems* traces (durations, tokens, arrivals, rate-limit events) — hence "generator fitted to real marginals" in §7.3.

### 3.9 Failure taxonomies and deadlock in agent systems

- **MAST** (NeurIPS '25) [R27] — 14 failure modes in 3 categories (specification/system design, inter-agent misalignment, verification) from 1,600+ traces across 7 frameworks.
- **DPBench** (2026) [R28] — see H4.
- Runtime deadlock/livelock detectors for tool-using agents (happens-before graphs over message traces) exist as open-source tools [R29]; useful as *test oracles* for §8.

### 3.10 Comparison table

| System | Predicts | How | Uses prediction for | Damage bound when wrong | Tiers | Reservation? |
|---|---|---|---|---|---|---|
| Hermes | unit demand dist. + branch prob. | offline profiling ×1000; Pearson-conditioned MC | Gittins queueing; prewarm | prewarm is cheap; <10% to oracle | model (+containers) | no (prewarm) |
| PBKV | next-K agents | GraphSAGE+attn+prefill state, CE | KV eviction/prefetch | regret linear in error (conservative use) | model memory | no |
| CacheScout | next agent | online Markov counts | KV eviction/prefetch | falls back to LRU | model memory | no |
| KVFlow | steps-to-execution | static step graph | KV eviction/prefetch | n/a (assumes known) | model memory | no |
| HexAGenT | completion horizon | running estimate | priority + placement | re-estimated each event | model | no |
| Chimera | remaining tokens | CPU regressor | routing + load estimate | activity monitor | model | ledger of in-flight demand |
| Autellix | nothing | attained service | preemption/priority | n/a | model | no |
| HiveMind | nothing | — | admission/AIMD/budgets | retry | model API | no |
| Continuum | TTL | fixed heuristic | KV hold | TTL expiry | model memory | hold with expiry |
| Decima | learned policy | GNN + RL in sim | node choice + parallelism | none (learned) | cluster CPU | no |
| Banker's | declared max | given | safe-state gate | conservative | any | yes (static) |
| EASY backfill | runtime estimates | user/ML | head-of-queue reservation + backfill | over-estimates harmless | HPC nodes | yes (time-based) |
| **Ours (§6)** | next-K structure + per-tier demand *quantiles* | small seq model, CE + quantile loss; conformal calibration | short leases, gang acquisition, aging/DRF | lease horizon + backfill; three Coffman conditions broken | model + tool + service | **yes (leases)** |

---

## 4. How the controllers' logic is actually written (Q2)

Faithful sketches of the algorithms, so the common shape is visible.

```text
# Banker's algorithm (Dijkstra)
admit(i, max[i]):             assert max[i] <= total
request(i, r):
  if r > need[i]: throw       # exceeded declared claim
  if r > avail:   wait
  tentatively grant; if safe(): commit else rollback; wait
safe():
  work = avail; finish = [false]*n
  loop: pick i with !finish[i] and need[i] <= work  →  work += alloc[i]; finish[i] = true
  return all(finish)
```

```text
# DRF (Ghodsi et al.)
dominant_share(i) = max_r alloc[i][r] / capacity[r]
schedule(): i = argmin dominant_share over users with pending demand; grant next task of i if it fits
```

```text
# EASY backfilling (+ Tsafrir–Feitelson)
head = queue[0]
T = earliest time head fits, using predicted end times of running jobs      # the reservation
for j in queue[1:]:
  if j fits now and (predicted_end(j) <= T or j avoids head's reserved nodes): start j   # backfill
# over-estimated durations → later T → more backfill; harmless
```

```text
# Hermes (PDGraph)
offline: run app 1000×; per functional unit store raw (backend, consumption, next_unit); FIFO cap 1000
online, on unit completion:
  downstream demand ← filter stored tuples by observed upstream values where Pearson ρ > 0.5; Monte Carlo sample
  app rank = Gittins  G(D,a) = inf_Δ  E[min{X−a, Δ} | X > a] / P{X−a ≤ Δ | X > a}
  prewarm backend b for downstream unit when  p_s · P(t_c > t_s + t_p)  reaches K     # p_s branch prob, t_p prewarm time
```

```text
# PBKV (predictor)
h_cur  = GraphSAGE_2layer(agent, transition graph estimated from offline traces)
h_path = Attention(query = current agent, keys = executed-prefix agents)
h_txt  = last-token post-norm hidden state from prefill (free)
logits[k] = MLP([h_cur; h_path; h_txt])   for k = 1..K      # ~350K params, CE loss on offline traces
# use: reuse score per cache entry from predicted invocations
evict: retired-workflow cache first; then lowest score          # deterministic guardrail before probabilistic policy
prefetch: only into idle GPU space, only on decode-only batches  # cannot backfire
```

```text
# CacheScout (online Markov)
on dispatch(prev → cur): C[prev][cur] += 1;  P = (C + ε) / rowsum(C + ε)
p_surv(a) = P(reach a within K steps | cur)  via BFS on P
score(block b) = (p_surv(a_b) + δ) · (e^{−λ·age(b)} + δ) · |b|;  evict argmin
prefetch predicted next agents only if predictability R >= R_min   # entropy reduction
```

```text
# HexAGenT
per workflow w: DAG_w revealed online; H_w = running estimate of standalone completion horizon
for each ready call c in w: priority(c) = projected risk that w misses H_w if c is delayed
choose (prefill placement, decode placement, local queue priority) jointly, s.t. KV capacity and transfer latency
```

```text
# HiveMind (proxy)
admit(req): while A >= Cmax: wait(condvar);  A += 1                      # Cmax resized by AIMD on 429/5xx
rate: pause all agents when provider-reported remaining < 10%;  sliding-window counters as proactive check
budget: per-agent token ceiling from global pool; warn at 85%; checkpoint+stop at 100%
queue: order by (priority, est_tokens ASC, created_at)                   # SJF inside a priority class
retry: exponential backoff + jitter, transparently, before the agent sees the error
```

```text
# Autellix PLAS (non-clairvoyant)
priority(call) = attained service of its program so far (sum of completed calls' tokens)  # lower → higher priority
multi-level feedback queue; preempt long programs; no prediction anywhere
```

```text
# TIE (uncertainty-aware length)
predict (μ, σ, ν) of log-t output-length distribution from the prompt
rank requests by CVaR_α (tail-inflated expectation), not by E[length]
```

```text
# Decima
e_v = g[ Σ_{u ∈ children(v)} f(e_u) ] + x_v          # per-node embedding by message passing
per-DAG summary node over all nodes; global summary over all DAGs
policy π(node to schedule, parallelism limit | embeddings); REINFORCE
train: fix arrival sequence across episodes → baseline per sequence; episode length curriculum with Exp-random termination
sim: real profiled task durations + Alibaba trace; deploy to 25-node Spark; no fine-tune
```

**The common shape** (every prediction-driven system above):

```text
OBSERVE   the revealed structure (graph / prefix / session)
MODEL     future demand as a *distribution conditioned on the prefix*
DECIDE    with a risk-sensitive rule (Gittins, CVaR, risk-of-missing-horizon, survival×recency)
ACT       with a bounded blast radius (prewarm-only, idle-space prefetch, admission gate, TTL)
REFRESH   on every completion event
```

**What is absent from all of them:** a reservation primitive that (i) spans model + tool + service tiers, (ii) expires, (iii) is backfillable, (iv) is acquired all-or-nothing in a global order, and (v) comes with starvation and deadlock arguments. That is the gap this project fills.

---

## 5. The gap, stated

1. Every existing system optimises *one* tier (KV memory, GPU placement, or one API key). H6 says the bottleneck moves; a one-tier controller is always wrong somewhere.
2. Every existing system *schedules* or *caches*; none *reserves* (Continuum's TTL and Chimera's in-flight ledger are the closest fragments).
3. No existing system states, let alone proves, freedom from deadlock and starvation for agent workflows — H4 says this is exactly where agents die.
4. No public dataset spans the three tiers with capacity events (429s, cold starts, sandbox OOMs). Stage 0 of the ladder (§7.2) must produce one.

---

## 6. Proposed controller (evidence-derived)

### 6.1 Components — one responsibility each

| Component | Input | Output | Never does |
|---|---|---|---|
| **Observer** | events: `workflow_start`, `step_start`, `step_end`, `model_call`, `tool_call`, `error(429/502/timeout)` | per-workflow online-revealed graph `G_w`; global occupancy per (tier, resource) | decide anything |
| **Predictor** | `G_w` prefix (+ last step's embedding if available) | for k = 1..K: distribution over next node type; per tier a demand distribution `D_{w,k,tier}` (as quantiles) | decide anything; touch capacity |
| **Reserver** | predictions; capacity vectors `C[tier][res]`; ledger | leases; expirations; gang acquisition order | admit steps |
| **Gate** | a step request from a workflow runtime | `admit` or `throw` | predict; hold state beyond the ledger |

The workflow runtime (the society's orchestrator) talks **only** to the Gate. Agents never call providers directly (HiveMind's proxy pattern — otherwise agent-local retries reintroduce H1).

### 6.2 The one primitive: a lease

```text
lease(w, k, tier, res):
  q  = quantile_τ( D_{w,k,tier,res} )                 # knob τ (e.g. 0.9), conformally calibrated on real data
  t0 = predicted start of step k                        # from predicted durations of intervening steps
  h  = min(h_steps predicted steps, h_time)             # knob h; H5 says ≈ 1–2 steps
  return Lease(w, k, tier, res, q, start=t0, expiry=t0+h)
```

Properties, each tied to a failure mode:

- **Backfillable** (EASY rule [R14]). Capacity under a lease may be used by any step whose *upper-quantile* end time ≤ `t0`. Over-reservation therefore costs only what cannot be backfilled — which is why a conservative τ is affordable (Tsafrir–Feitelson).
- **Expires unconditionally** at `t0+h`. Renewal happens only through a *new* prediction on the next event. Nothing is held on a stale forecast.
- **Fair-share capped**: `q ≤ dominant_fair_share(w, tier) + slack` (DRF [R15]). A workflow cannot lease more than its share when others are waiting.
- **Gang-acquired** for multi-tier steps (model slot + tool slot + sandbox memory): acquire components in the **global tier order** `model < tool < service < memory`; if any component is unavailable, release all and requeue. No hold-and-wait, no partial allocation (gang scheduling [R40]).

### 6.3 Why the failure modes cannot occur (by construction)

- **Over-allocation** — bounded per lease by `(q − actual) × h` minus what backfill recovers; total bounded by Σ over live leases; nothing survives its expiry (I3 below).
- **Starvation** — admission order is `(age(w) ascending priority boost, dominant_share(w))`. Age increases strictly while queued (I5), so every workflow eventually has top priority and a lease at its fair share; standard aging argument.
- **Deadlock** — Coffman [R39]: hold-and-wait is broken by gang acquisition; no-preemption is broken by lease expiry; circular wait is broken by global ordering. Any one suffices; three are broken. DPBench [R28] shows imposed resource ordering takes LLM agents from 90–100% deadlock to 0%.
- **Workflow failure** — a step that would exceed capacity *waits* (wait bounded by aging) instead of receiving a 429/502; retries are coordinated at the Gate (HiveMind's #1 primitive), so no thundering herd.

### 6.4 What is learned, what is fixed (per the design principles)

- **Learned**: the Predictor only — next-node distribution (cross-entropy) and per-tier demand quantiles (pinball/quantile loss). Small (PBKV is ~350 K parameters; Decima 12.7 K).
- **Fixed**: lease rule, tier ordering, gang acquisition, DRF cap, aging. Two knobs: τ and h.
- **Not a fallback**: there is no "if the predictor is unsure, use policy B" branch. Uncertainty widens the distribution, which moves the τ-quantile; backfill and expiry bound the cost. One path.
- **Why not end-to-end RL** (Decima-style)? Hermes measures <10% gap between a fixed rule with a decent demand model and an oracle — the headroom RL would chase is small; RL cannot give deadlock/starvation guarantees by construction; Decima generalises poorly off-distribution and needed a hand-built high-fidelity simulator. RL stays a *later* option, only if E6 (§8) measures real headroom.

### 6.5 Pseudo-code

```text
on event e:                                            # Observer
  G[e.w].append(e);  occupancy.apply(e)
  if e.kind in {workflow_start, step_end}:
    preds = Predictor(G[e.w])                          # K-step structure + per-tier demand quantiles
    for (k, tier, res) in preds.needs():               # Reserver
      ledger.replace(e.w, k, tier, res, lease(e.w, k, tier, res))
  ledger.expire(now)                                   # I3

on step_request(w, step):                              # Gate — the ONLY admission path
  L = ledger.get(w, step.k, step.tier, step.res)
  if L is None:              raise NoLease(w, step)                # precondition: bug in the runtime or Observer
  if step.demand > L.q:      raise LeaseExceeded(w, step, L)       # precondition: runtime re-requests → new prediction → new lease
  if not gang_acquire(step.components_in_global_order()):
       requeue(w, key=(−age(w), dominant_share(w)));  return       # no hold-and-wait
  admit(step);  ledger.consume(L, step.demand)

on step_end(w, step):
  release(step.components);  Observer(step_end)
```

`LeaseExceeded` is thrown, not absorbed: the runtime's re-request re-enters the same path with a fresh prediction. That is the whole retry story.

### 6.6 Invariants (assert; throw on violation; log with the minimum context to reproduce)

- **I1** ∀ tier, res: Σ active leases ≤ `C[tier][res]`.
- **I2** every admitted step has a lease with `q ≥ demand`.
- **I3** no lease exists past its expiry.
- **I4** components of any gang acquisition are acquired in non-decreasing global order.
- **I5** `age(w)` is strictly increasing while `w` is queued.
- **I6** the Gate is the only code path that mutates `occupancy` for admissions (enforced by construction: providers are reachable only through it).

### 6.7 Metrics (the only success criteria)

`failure_rate`, `p50/p99 workflow completion`, `waste = Σ(reserved − used)·Δt` per tier, `max_wait`, `Jain(dominant shares)`, `throughput (workflows/min)`, `gate_overhead_ms`. Predictor accuracy is logged for diagnosis, never used as an objective (H3).

---

## 7. The ladder: synthetic → society of LLMs (Q3)

### 7.1 Verdict

Yes, with three conditions.

1. **"Synthetic" must be fitted, not invented.** Decima's simulator used real profiled durations and a real production trace; its policy transferred with no fine-tuning. A generator that draws durations, token counts, fan-out and tail shape from *fitted real marginals* (§7.2) and randomises the knobs that vary between deployments is a faithful simulator. Hand-drawn DAGs with uniform durations are not; they would miss the 92%-of-time tail and teach the Predictor the wrong distribution.
2. **Only the Predictor and its calibration climb the ladder.** The reservation rule is fixed (§6.4), so sim-to-real risk is confined to the demand model — and PBKV shows 1 K real traces are enough to fine-tune a ~350 K-parameter model to 0.94.
3. **Evaluate by outcomes.** Because scheduling quality is non-monotone in accuracy (H3), the sim's fitness function must be §6.7's metrics under the *full* controller, never predictor accuracy.

Why synthetic first is right for *scalability*: the real society is expensive per step (LLM API cost, minutes-long tools) and cannot be run at 100+ concurrent workflows just to generate training states. The simulator can, on CPU (AgentServeSim runs as a CPU fitness evaluator).

### 7.2 Stage 0 — collect real traces and fit marginals

Sources: TraceLab [R21], Copilot [R22], Exgentic [R26], Alibaba microservices [R25] for the service tier, and **our own society in shadow mode**. *(2026-09-22: the seeded generator, simulator and E1–E4 scripts now exist — see README.md and RUNG0_REPORT.md.)* (Observer + Gate recording, not enforcing). Fit:

- structure: step-count distribution; next-node transition matrix per template; loop/retry probabilities; fan-out
- quantity: tool duration (heavy-tailed; fit log-t / log-normal with tail check), output tokens (log-t), input tokens, sandbox memory
- timing: workflow inter-arrival; idle gaps between turns (minutes–hours [R22][R23])
- capacity events: 429/502/reset rates as a function of concurrency; cold-start delays; provider quota ceilings (learned by AIMD probing [R12])

Deliverable: `marginals.json` and a template library. This is the artifact §3.7 says nobody has published.

### 7.3 Stage 1 — synthetic training

```text
generator(knobs):
  template ← sample(template_library)                       # DAG grammar with branch/loop probabilities (Hermes PDGraph / SyntheticAgentTraceQA style)
  instantiate durations, tokens, memory from fitted marginals
  arrivals ← Poisson/bursty process(rate ∈ knobs)
  perturb per 'When Simulation Lies' classes: observation noise, tool errors, timeouts, transition dynamics (429 bursts, cold starts)
simulator: AgentServeSim-style discrete-event model (program control block, causal successor release, per-tier capacity vectors, backfill, leases)
```

Domain-randomised knobs (each also *exposed to the Predictor as an input feature* — the Decima lesson): arrival rate, fan-out, tail index of durations, per-tier capacity, rate-limit ceilings, cold-start delay, error-injection rate, fraction of static vs dynamic templates.

Train the Predictor with cross-entropy on next-K node types and quantile loss on per-tier demand. Use input-dependent baselines only if RL is ever introduced; for supervised training, fix seeds per arrival sequence for reproducible evaluation. Sweep τ and h in the simulator; choose by §6.7 metrics.

Exit criterion: in sim, the controller beats (a) uncoordinated, (b) HiveMind-style reactive proxy, and (c) an oracle-prediction controller by a *measured* gap that tells us how much prediction is worth (Hermes-style oracle comparison).

### 7.4 Stage 2 — fine-tune on the real society

1. **Shadow mode** on the real society: Observer + Gate record, Reserver computes leases, nothing is enforced. Collect real traces (the PBKV benchmarks are a template: HoVer + LangChain, SWE-bench + AutoGen, FinanceBench + CrewAI).
2. **Fine-tune** the Predictor on real traces with the same losses, small learning rate. Expect the *structure* head to adapt fast (CacheScout's online counts converge in-session; PBKV needs ~1 K traces).
3. **Recalibrate τ by conformal prediction** [R19] on real residuals — this is the step that repairs any sim-to-real miscalibration of the quantiles, with a distribution-free guarantee. Never carry a sim-calibrated τ to production.
4. **Enforce**, A/B against uncoordinated and against a reactive proxy, on §6.7 metrics. Use the deadlock/livelock trace detectors [R29] as test oracles.
5. Keep the online Markov counts (CacheScout) running as *features* for the Predictor so drift within a deployment is absorbed without retraining.

### 7.5 What transfers and what does not (evidence)

| Transfers | Does not transfer |
|---|---|
| Topology-level structure prediction (templates, loops) — PBKV/CacheScout need little data | Exact transition probabilities — "predictor needs to be trained on a specific workload" (PBKV) |
| The reservation rule, ordering, DRF, aging — no learning involved | Dynamics: 429 behaviour, cold-start times, tool latency tails — the 30–40% accuracy loss in "When Simulation Lies" is exactly the transition/reward-metadata classes |
| Tail *awareness* — if the generator was fitted | Prompt-conditioned tails (Robust Length Prediction) — need real prompts |
| Knob-conditioned behaviour — if knobs were randomised and exposed (Decima) | Anything trained on a single narrow knob setting (Decima's anti-skew failure) |

---

## 8. Evidence-collection plan — run *before* writing the controller

Each experiment confirms or refutes a hypothesis on *our* society; if one refutes, the design changes before code is written.

| ID | Experiment | Confirms | Refutes if |
|---|---|---|---|
| E1 | N ∈ {5, 10, 20, 50} concurrent workflows, uncoordinated, against shared endpoints with real tier limits. Measure failure rate vs aggregate capacity. | H1 | failures scale with true capacity shortfall, not concurrency bursts |
| E2 | From shadow traces: entropy of next node given last-k nodes (CacheScout's R); 1/2/3-step accuracy of a Markov baseline and of a small sequence model | H5; decides Predictor size and h | 1-step accuracy < ~0.7 → reservations must be near-zero-horizon; controller degrades to HiveMind-style reactive gate |
| E3 | Fit tool-duration and token distributions; compare tail share to TraceLab's 4.9%→92% | H2, H3; sets τ | tails are light → point estimates suffice, simplify Predictor |
| E4 | Force multi-tier steps (model + tool + sandbox) under contention with agent-local acquisition; detect circular waits with a happens-before detector | H4 | no cycles ever form → drop gang acquisition (keep ordering, it is free) |
| E5 | Replay traces in the simulator with leases at τ ∈ {0.5, 0.9, 0.99} × backfill on/off; measure waste vs failure | lease + backfill premise | backfill recovers nothing → shorten h, lower τ |
| E6 | Oracle-prediction controller vs learned-Predictor controller vs no-prediction gate, in sim | how much prediction is worth | oracle ≈ no-prediction → ship the gate only, defer the Predictor (simple > complex) |

Minimal logging for all of the above: per event `(t, w, step_k, tier, res, demand, admitted?, wait, error)`. Nothing more until a question needs it.

---

## 9. Risks and open questions

- **Cold start for unseen workflows.** Predictor priors come from the template class; with h ≈ 1–2 steps the damage of a wrong prior is one lease. Verify in E2.
- **Backfill needs duration predictions too.** Heavy tails mean the *upper* quantile of a backfill candidate's duration must be used; a wrong one delays the reserving workflow by at most h.
- **Opaque external quotas.** Provider limits change without notice; treat `C[tool][key]` as a *learned* capacity via AIMD probing (HiveMind) with an uncertainty margin (the K8s autoscaling decomposition [R31]).
- **Fairness granularity.** DRF per workflow vs per user vs per tenant is a policy choice; start per workflow, expose the key as configuration.
- **Distributed ledger.** HiveMind was single-machine. Start with a single-node ledger; the invariants I1–I6 are what a distributed version must preserve.
- **Bypass.** Any agent that can reach a provider without the Gate reintroduces H1. The Gate must be the only network path (proxy / egress rule), not a library convention.
- **Static workflows.** If E2 shows the society is mostly static (FinanceBench-like), Hermes-style per-template profiling replaces the learned Predictor — simpler, and within 10% of oracle.

---

## 10. References

Serving / scheduling
- [R1] Lin et al., *Parrot: Efficient Serving of LLM-based Applications with Semantic Variable*, OSDI 2024. https://www.usenix.org/conference/osdi24/presentation/lin-chaofan
- [R2] Luo et al., *Autellix: An Efficient Serving Engine for LLM Agents as General Programs*, 2025. https://arxiv.org/abs/2502.13965
- [R3] *Hermes: Efficient Serving of LLM Applications with Probabilistic Demand Modeling*, ACM TACO. https://arxiv.org/abs/2506.14851 · https://doi.org/10.1145/3803390
- [R4] Chaudhry et al., *Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms*, OSDI 2026. https://arxiv.org/abs/2508.18298
- [R5] Wadlom, Shen, Lu, *Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective* (Helium), 2026. https://arxiv.org/abs/2603.16104
- [R6] Peng et al., *HexAGenT: Efficient Agentic LLM Serving via Workflow- and Heterogeneity-Aware Scheduling*, 2026. https://arxiv.org/abs/2605.16637
- [R7] Pan et al., *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows*, NeurIPS 2025. https://arxiv.org/abs/2507.07400
- [R8] *Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management* (PBKV), 2026. https://arxiv.org/abs/2605.06472
- [R9] *Learning Agent Execution for KV-Cache Management in Agentic Serving* (CacheScout), 2026. https://arxiv.org/abs/2608.14624
- [R10] Qian et al., *Serving Agentic Workflows with a Physical-Plan Compiler and Adaptive Runtime* (Dyserve), 2026. https://arxiv.org/abs/2607.02942
- [R11] Ye et al., *Speculative Actions: A Lossless Framework for Faster Agentic Systems*, ICLR 2026. https://arxiv.org/abs/2510.04371
- [R12] *HiveMind: OS-Inspired Scheduling for Concurrent LLM Agent Workloads*, 2026. https://arxiv.org/abs/2604.17111
- [R13] Practitioner notes on multi-agent rate limiting (token bucket + semaphore + priority queue; thundering herd). https://www.tamirdresher.com/blog/2026/03/21/rate-limiting-multi-agent · https://zuplo.com/learning-center/token-based-rate-limiting-ai-agents
- [R33] *Chimera: Latency- and Performance-Aware Multi-agent Serving for Heterogeneous LLMs*, 2026. https://arxiv.org/abs/2603.22206
- [R34] *SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling*, 2026. https://arxiv.org/abs/2607.08565
- [R35] Li et al., *Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live*, 2025. https://arxiv.org/abs/2511.02230
- [R36] *InferCept: Efficient Intercept Support for Augmented Large Language Model Inference*, ICML 2024. https://arxiv.org/abs/2402.01869
- [R37] *Conveyor: Efficient Tool-aware LLM Serving with Tool Partial Execution*, 2024. https://arxiv.org/abs/2406.00059
- Also seen, not read in depth: GraphFlow (https://arxiv.org/abs/2605.22566), TokenCake (https://arxiv.org/abs/2510.18586), TokenDance (https://arxiv.org/abs/2604.03143), Leyline (https://arxiv.org/abs/2606.01065), HeraSys (https://arxiv.org/abs/2607.22578), IdleSpec (https://arxiv.org/abs/2605.22154), SPORK (https://arxiv.org/abs/2607.03333), *A Policy-Driven Runtime Layer for Agentic LLM Serving* (https://arxiv.org/abs/2605.27744).

Classical theory
- [R14] Tsafrir & Feitelson, *The dynamics of backfilling: solving the mystery of why increased inaccuracy may help* (IISWC 2006); Tsafrir, Etsion, Feitelson, *Modeling User Runtime Estimates* (JSSPP 2005) https://www.cs.huji.ac.il/labs/parallel/workload/m_tsafrir05/Est05JSSPP.pdf ; Tsafrir, *Using Inaccurate Estimates Accurately* (JSSPP 2010) https://dants.github.io/papers/Keynote10JSSPP.pdf
- [R15] Ghodsi et al., *Dominant Resource Fairness*, NSDI 2011. https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types
- [R18] Lykouris & Vassilvitskii, *Competitive Caching with Machine Learned Advice* (ICML 2018 / J.ACM 2021); Wei & Zhang, *Optimal Robustness-Consistency Trade-offs for Learning-Augmented Online Algorithms*, NeurIPS 2020. https://arxiv.org/abs/2010.11443
- [R19] Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, 2021. https://arxiv.org/abs/2107.07511 (not fetched this session; classical)
- [R38] Dijkstra, Banker's algorithm (1965). https://en.wikipedia.org/wiki/Banker%27s_algorithm
- [R39] Coffman, Elphick, Shoshani, *System Deadlocks*, ACM Computing Surveys 1971 (not fetched this session; classical)
- [R40] Ray placement groups / Kubernetes coscheduling (gang scheduling). https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html · https://www.kubernetes.io/docs/concepts/scheduling-eviction/gang-scheduling/

Prediction under uncertainty
- [R16] Fu et al., *Efficient LLM Scheduling by Learning to Rank*, NeurIPS 2024. https://arxiv.org/abs/2408.15792
- [R17] *Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions* (TIE), ICML 2026. https://arxiv.org/abs/2604.00499
- [R41] *Beyond Prediction: Tail-Aware Scheduling for LLM Inference*, 2026. https://arxiv.org/abs/2606.18431
- [R42] *Robust Length Prediction: A Perspective from Heavy-Tailed Prompt-Conditioned Distributions*, 2026. https://arxiv.org/abs/2604.07931

Learned schedulers / sim-to-real
- [R20] Mao et al., *Learning Scheduling Algorithms for Data Processing Clusters* (Decima), SIGCOMM 2019. https://arxiv.org/abs/1810.01963
- [R30] Zhou et al., *When Simulation Lies: A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents*, 2026. https://arxiv.org/abs/2605.11928
- [R24] Rajib, Zheng, Lou, *AgentServeSim: Serving-System Simulation and Policy Search for LLM Agent Programs*, 2026. https://arxiv.org/abs/2606.09613
- [R31] *Decomposing Predictive Kubernetes Autoscaling for LLM Serving Under Long Startup Delays*, 2026. https://arxiv.org/abs/2609.20874
- [R32] *WarmServe: Enabling One-for-Many GPU Prewarming for Multi-LLM Serving*, 2025. https://arxiv.org/abs/2512.09472

Traces / characterisation
- [R21] Zhu et al., *TraceLab: Characterizing Coding Agent Workloads for LLM Serving*, 2026. https://arxiv.org/abs/2606.30560
- [R22] Liu et al., *Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale*, 2026. https://arxiv.org/abs/2608.00101
- [R23] Chang et al., *From LLM Inference to Agentic Workloads: Characterization and Implications for Serving Systems* (AgentSysBench), 2026. https://arxiv.org/abs/2608.15127
- [R25] Luo et al., *Characterizing Microservice Dependency and Performance: Alibaba Trace Analysis*, SoCC 2021; trace v2022. https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022
- [R26] Exgentic/agent-llm-traces (1,781 traces). https://huggingface.co/datasets/Exgentic/agent-llm-traces

Synthetic generation
- [R43] *Execution-First Synthetic Tool-Use Trace Generation for LLM Agents* (SyntheticAgentTraceQA), 2026. https://arxiv.org/abs/2607.29175
- [R44] *AgentSim: A Platform for Verifiable Agent-Trace Simulation*, 2026. https://arxiv.org/abs/2604.26653
- [R45] *Environment-free Synthetic Data Generation for API-Calling Agents* (ESAT), 2026. https://arxiv.org/abs/2607.16900

Failures / deadlock
- [R27] Cemri, Pan, Yang et al., *Why Do Multi-Agent LLM Systems Fail?* (MAST), NeurIPS 2025. https://arxiv.org/abs/2503.13657
- [R28] Hasan & BusiReddyGari, *DPBench: Structural Determinants of Multi-Agent LLM Coordination Under Simultaneous Resource Contention*, 2026. https://arxiv.org/abs/2602.13255
- [R29] Open-source agent deadlock/livelock detectors (happens-before over message traces). https://github.com/rishika1099/Agent-Deadlock-Detector · https://github.com/shanyukollipara/multi-agent-deadlock-detector

Verification note: all numbers above were read from the papers' own abstracts/text during this session unless marked "not fetched". One summariser hallucination was caught and discarded (a fabricated "open problems" list attributed to [R5]); [R5] is Helium and says nothing of the sort.
