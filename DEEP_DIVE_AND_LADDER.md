# Deep dive: survey, Jev adaptation, synthetic data, its limits, a realistic path, and the ladder

Companion to [SERVICE_GRAPH_RESERVATION.md](SERVICE_GRAPH_RESERVATION.md) (v1: crime, hypotheses H1–H6 with evidence, first-cut controller). This document goes deeper and does not repeat v1; where it changes a v1 decision it says so (§1.9).

- Date: 2026-09-22
- Status: ideation → simulator, seeded generator, rung-0 scripts, the generator knobs of §3.5, and a rung-3 controller (`LeaseController`) exist as of 2026-09-22 (README.md, RUNG0_REPORT.md §11–§12). The controller was scored and **ablated**: on the synthetic society its gain over the reactive gate is ~90% the gate's idle-timeout knob plus fair queuing; the learned predictions do not pay (ISSUES F20). §6.4 records what that changes.
- Every number below was read from the paper's own abstract/text in this session unless marked *(reported by a secondary source)* or *(classical, not fetched)*.

---

## 0. Scope and one assumption

**"JEV" = Jev, TypeSafe AI's first "System One" model** (typesafe.ai, announced 15 Sep 2026, early access). §2 is written for it. An earlier draft of §2 assumed JEPA (Joint-Embedding Predictive Architecture); that reading was wrong and is kept only as a two-line note at the end of §2.

**What this document answers.**
1. A deeper survey, organised by the *layer of the problem* each paper touches (§1).
2. How Jev (a calibrated, typed-decision "System One" model) would be adapted, honestly including how it could be very wrong (§2).
3. How to build the synthetic dataset the controller is tested on (§3).
4. The limitations of that dataset / simulated log, and which ones cannot be mitigated (§4).
5. How to realistically get this solved with things that already exist (§5).
6. A sketched ladder to ideate → develop → tune the controller and establish gains/losses at every rung (§6).

---

## 1. Deep survey, by layer

The problem has five layers: **(A)** the schedulable unit and its dependency structure; **(B)** fairness and starvation; **(C)** admission and stability of the physical resource; **(D)** prediction of what a workflow will need next; **(E)** learning a controller and transferring it from simulation. Each subsection ends with what it *settles* for us.

### 1.1 Layer A — workflow as the schedulable unit

| System | Unit | Structure it exploits | Key mechanism | Headline | Cost it admits |
|---|---|---|---|---|---|
| Parrot (OSDI'24) [S1] | app DAG | Semantic Variables expose DAG + prompt structure | cross-request optimisation | — | requires app cooperation |
| Teola/Ayo (ASPLOS'25) [S2] | primitive-level dataflow graph | query parsed into primitives; Graph Optimizer passes | fine-grained orchestration | up to 2.09× | needs developer-declared workflow |
| Autellix (2025) [S3] | program | attained service of program | PLAS/ATLAS, non-clairvoyant | 4–15× throughput | no prediction by design |
| HexAGenT (2026) [S4] | online-revealed DAG | running completion-horizon estimate | risk-of-missing-horizon priority + placement | 20.1%/33.0% SLO-scale reduction (p95/p99) | model tier only |
| **SAGA (2026)** [S5] | **whole agent workflow, atomic** | Agent Execution Graphs predict KV reuse across tool-call boundaries (within 1.31× of Bélády) | session-affinity batching + work stealing; **Agent Fair Share** (task-completion-time fairness, provable bounded deviation) | 1.64× TCT (geomean, p<0.001) on 64 GPUs; 99.2% SLO attainment under multi-tenant interference | **~30% lower peak throughput** than throughput-optimal |
| SGH (2026, position) [S6] | static DAG per version | explicit plan; separated plan/execute/recover | scheduler-theoretic framing; 70 systems surveyed | none (no implementation) | loses expressiveness |
| Murakkab (OSDI'26), Helium, Dyserve, Chimera, SMetric, Continuum | see v1 §3.1 | | | | |

*Settles:* program-level scheduling is now mainstream and SAGA shows it can be **atomic** (workflow-as-gang) with a **provable fairness metric** — but it pays ~30% peak throughput. Our lease must be cheaper than a gang: atomic only for the *components of one step*, not the whole workflow (v1 §6.2).

### 1.2 Layer B — fairness with guarantees

| Work | Fairness notion | Guarantee | Prediction needed? |
|---|---|---|---|
| DRF (NSDI'11) *(classical)* | max-min over dominant shares, multi-resource | strategy-proof, envy-free, Pareto | no |
| VTC (OSDI'24) [S7] | token-cost service, continuous batching | **2× tight bound** on service difference between backlogged clients; work-conserving | no |
| Equinox (2025) [S8] | dual counters: user (weighted tokens + latency) and operator (throughput + GPU util) | holistic score with tunable weights | yes — MoPE predicts latency, output tokens, throughput, util; 1.3× throughput, −60% TTFT, +13% fairness vs VTC |
| **Justitia (2025)** [S9] | task-parallel *agents*; memory-centric true cost; "selective pampering" by completion order under idealised fair sharing | **virtual-time fair queuing with guaranteed worst-case delay** | yes — lightweight agent-cost predictor |
| **SAGA Agent Fair Share** [S5] | task-completion-time deviation per agent workflow | provable bounded deviation | uses execution graph |

*Settles:* fairness for agents is defined at the **workflow/agent** granularity, cost is **memory-centric** on the model tier, and the mechanism with a worst-case delay bound is **virtual-time fair queuing** (Justitia) — a strictly stronger primitive than v1's "aging". **Change to v1:** starvation freedom will be delivered by virtual-time fair queuing over a memory-centric cost per workflow, with DRF only to *cap* leases across tiers (§1.9).

### 1.3 Layer C — admission and physical stability

- **Service-Induced Congestion (Ao, Dong, Luo, Simchi-Levi, 2026)** [S10]. A discrete-time dynamical model of memory-constrained LLM inference (admission → KV growth → eviction under continuous batching). In the saturated-input regime the system has eviction-free fixed points *and* limit cycles with evictions; for homogeneous workloads the eviction-free equilibrium is **unstable** and the system converges to a worst-case limit cycle with **throughput losses up to 50%**; heterogeneity (coprime decode lengths) *stabilises* it. This is the plant model of the model tier: the naive "admit while there is free memory" rule oscillates. Any admission gate must respect its stability criterion.
- **QLM (SoCC'24)** [S11]. Request-waiting-time estimator + LP over virtual queues, mixing batch and interactive SLOs; +40–90% SLO attainment, +20–400% throughput.
- **K8s Gateway API Inference Extension / llm-d** [S12]. Endpoint Picker with KV-cache- and request-cost-aware routing; a **Flow Control layer that "governs physical capacity (KV cache, queue saturation), not HTTP connection counts"**, holding excess load in policy-aware queues *at the gateway*; `InferenceObjective` carries criticality. This is the production home of our Gate for the model tier.
- **LiteLLM priority-based rate limiting** [S13]. Reserves TPM/RPM capacity per priority level; below a saturation threshold any key may use idle capacity, above it each level is held to its reserved share; Redis-shared counters across instances. This *is* a crude reservation-with-backfill for the API-key tier, already deployed widely.
- **HiveMind** (v1) — admission `A < Cmax`, AIMD, budgets, coordinated retry.
- **Predictive K8s autoscaling decomposition (2026)** [S14]. Splits predictive scaling into token-aware demand tracking, startup-delay lookahead, a bounded uncertainty margin and plant-state observation, and finds that **EWMA + delay-aware lookahead + UCB margin captures most of the benefit**. Simple beats complex, again.

*Settles:* the physical model tier has its own instability that no predictor fixes; the Gate needs a stability-respecting admission rule (SIC) *and* the flow-control seat already exists in the gateway. For the API-key tier, LiteLLM's priority reservation is the baseline to beat.

### 1.4 Layer D — what has been predicted, from what, how well

| Target | System | Input | Model | Accuracy / effect | Use |
|---|---|---|---|---|---|
| next agent (1..K steps) | PBKV [v1] | transition graph + prefix + prefill hidden state | GraphSAGE+attn+MLP, 350K params | 0.94 @1, 0.77 @3 (HoVer) | KV eviction/prefetch |
| next agent | CacheScout [v1] | dispatch stream | online Markov, BFS survival | +10–18 pp hit rate; gains vanish at R≈0.12 | eviction/prefetch |
| **concrete future tool invocation** | **PASTE (2026)** [S15] | recurring agent patterns | pattern matcher | −43.5% task completion; 1.8× lower observed tool latency; results isolated until LLM confirms; joint scheduling so the bottleneck does not shift to the GPU | speculative tool execution |
| next action | Speculative Actions [v1] | fast model | LLM speculator | ≤55% | speculative env calls |
| remaining workflow tokens | Chimera [v1] | request | CPU regressor | — | routing + load |
| output length (dist.) | TIE [v1] | prompt | log-t + CVaR | 2.31× per-token latency | ranking |
| output length (rank) | LTR [v1] | prompt | learning to rank | rank suffices | SRPT-like |
| user idle time | Copilot traces [v1] | session | lightweight predictor | captures 86–90% of idle time | proactive orchestration |
| **failure within warning horizon** | **PrefixGuard (2026)** [S16] | trace prefix (typed steps via StepView) | supervised monitor | AUPRC 0.900 WebArena, 0.710 τ²-Bench, 0.533 SkillsBench, 0.557 TerminalBench; +0.137 vs text baselines | online warning |
| QoS violation, ahead of time | Seer (ASPLOS'19) [S17] | streaming distributed traces | DNN | anticipates 91%, avoids 84% | microservice debugging |
| tail-latency impact of an allocation | Sinan (ASPLOS'21) [S18] | microservice graph state | validated ML models | — | resource manager |
| function invocations + resources | Aquatope (ASPLOS'23) [S19] | history + external features (time of day) | Bayesian, noise-aware, uncertainty | 5× fewer QoS violations; −34% cost avg, −52% max | pre-warm + allocate |
| tool-call latency | TraceLab [v1] | tool type | — | tool type alone is insufficient; needs semantics of the requested operation + recent latency history *(reported)* | — |
| MCP cost structure | ProMCP (ACL'26 Findings) [S20] | MCP deployments | profiling | customised clients spend 56–72% of tokens and 60–67% of latency on planning + schema injection; off-the-shelf clients >85% of latency in final synthesis | — |

*Settles:* (i) short-horizon *structure* prediction is mature and cheap; (ii) *quantity* must be a distribution; (iii) **failure-within-horizon** is a learnable target (PrefixGuard, Seer) and belongs in the Predictor as a third head; (iv) tool latency needs semantics + recent history, not type alone; (v) on the tool tier the dominant cost may be the *planning/schema* tokens, not the tool — so "tool reservation" is partly a *model-tier token* reservation.

### 1.5 The microservice/serverless lineage (the original "service graph" controllers)

Seer → Sinan → FIRM (OSDI'20) → Aquatope → Ditto is a decade of learning controllers over *service graphs* with backpressure and cascading QoS violations. Three transferable lessons:

1. **Predict the violation, not the load** (Seer): the target is "SLO violation within Δ", from streaming traces. Our analogue: "step k of workflow w cannot be admitted within h".
2. **Model the *impact of an allocation*** (Sinan): the controller needs `P(tail latency | allocation, graph state)`, i.e. an *action-conditioned* model — the impact/reaction question Jev is asked at point C of §2.3.
3. **Clone, don't share** (Ditto, ASPLOS'23) [S21]: capture the dependency graph from distributed tracing, recreate control/data flow per tier, then generate syscalls/assembly that mimic the CPU/memory behaviour — *without revealing application logic*. This is the template for turning a private agent society into a shareable, systems-faithful synthetic clone (§3).

### 1.6 Layer E — learning to control with exogenous inputs

- **Decima** (v1): GNN + policy gradient, trained in a simulator built from real profiled durations + Alibaba trace; transferred to a 25-node Spark cluster with no fine-tuning; generalises only if the workload knobs were randomised *and* exposed as features.
- **Input-driven variance reduction (Mao et al., ICLR'19)** [S22]: in input-driven environments (exogenous stochastic arrivals), standard baselines have high variance; fix the input sequence across episodes and learn input-dependent baselines. *Our arrivals and demands are exogenous → this applies to any RL rung.*
- **Hindsight Learning for Exo-MDPs (Sinclair et al., ICML'23)** [S23]: when the only uncertainty is exogenous, past decisions can be revisited in hindsight with the realised inputs to infer counterfactual consequences; scaled to VM-to-physical-machine allocation with real public-cloud datasets, **outperforming domain heuristics and state-of-the-art RL**. *This gives us the oracle:* with a recorded trace, a hindsight planner that knows the future computes the best reservation schedule — an upper bound for gains, and an imitation target.
- **MPC autoscaling** [S14, S24]: receding-horizon control with a plant model; SageServe (forecast-aware), TokenScale (token velocity). Our fixed-rule lease is a one-step certainty-equivalent controller; a K-step receding-horizon version is a later option, not on the ladder.

*Settles:* the evaluation upper bound is a **hindsight planner**, not a "perfect predictor" bolted onto our rule. Rung 2 of the ladder computes it before any learning happens.

### 1.7 Simulators and the sim-to-real question

- **AgentServeSim (2026)** [v1 R24]: unit of execution = agent program; Program Control Block; Orchestrator "causally releases successor turns from simulated predecessor completions"; Retention Plane; Dispatch Plane; validated in 20 paired sim/real cells against vLLM on two GPU platforms and two model sizes, **mean JCT error ≤ 5.5%**. Its stated reason to exist: request-stream simulators "cannot jointly represent the cross-turn state and policy-dependent successor releases needed to evaluate counterfactual agent-serving trajectories."
- **Open vs Closed (Schroeder, Wierman, Harchol-Balter, NSDI'06)** [S25]: closed (arrivals triggered by completions + think time) vs open (arrivals independent of completions) workload models produce "a vast difference in behavior", and scheduling policies are impacted differently. Agentic workloads are **closed within a workflow** (the next call waits for the previous one) and **open across workflows** (users arrive independently) — a *partly-open* model. A replayed trace is neither.
- **When Simulation Lies (2026)** [v1 R30]: for tool-use agents, transition-dynamics and reward-metadata perturbations cost 30–40% accuracy; observation noise <5%; domain-randomised RL on static perturbations closes ~27% of an unseen transition gap.
- **WfCommons** [S26]: WfChef "analyzes real instances to discover recurring dependency patterns and statistical distributions, producing 'recipes'"; WfGen "generates realistic synthetic workflow instances from recipes — preserving structure, runtime, and I/O distributions at any scale"; WfBench adds CPU/memory/I/O pressure; WfFormat is the JSON schema; 318+ curated real instances (Pegasus 135, Nextflow 116, Snakemake 34, Makeflow 30, …). The site now explicitly targets agentic workflows: "Test LLM/agent planners that adapt DAGs on the fly — replanning, retries, provenance-aware decisions — using realistic workflow 'digital twins'."

*Settles:* the simulator must be **program-centric with causal successor release** (AgentServeSim) driven by a **partly-open** arrival model (Schroeder), fed by **recipe-based generators fitted to real instances** (WfChef/WfGen), with **transition-dynamics randomisation** (When Simulation Lies). Anything else produces conclusions that flip on contact with reality. *Built on 2026-09-22 as `agentsim/` (README.md); results in RUNG0_REPORT.md.*

### 1.8 Inventory of usable traces

| Source | Scale | Timing | Tokens | Tool calls | Errors/429 | Concurrency | Tiers | Notes |
|---|---|---|---|---|---|---|---|---|
| TraceLab [v1] | 4,265 sessions; 357,161 LLM steps; 432,510 tool calls (Claude Code, Codex) | yes | yes | yes, typed | partial | no (single-user sessions) | model+tool | tool-call mean 16.8 s (Codex residual overhead 1.11 s avg, p99 10.0 s — Table 10); output tokens median 252 / p99 6,571; append 857 / 232,206; prefix 126 k median; 95.7% prefix hit; median session = 1 request (mean 9.2, p99 137); 92.3% of session wall-clock is human think time; 10.8 tool calls per request |
| GitHub Copilot traces [v1] | 3.2M users, 13M sessions, 761M LLM calls, 95T tokens | yes | yes | yes | — | yes (production) | model+tool | KV hit 90% within turn, 55% across; idle predictor 86–90%; **data not public** |
| AgentSysBench [v1] | 10 apps + production traces (3 apps) | yes | yes | yes | yes | controlled | model+tool+service | non-LLM dominates 5/10; 28 GB sandbox peaks; 32× latency divergence |
| Exgentic/agent-llm-traces [v1] | 1,781 traces, 6 benchmarks | yes | yes | yes + schemas | — | no | model+tool | replayable; DAG recoverable |
| nebius/SWE-agent-trajectories [S27] | 80,036 | **no** | partial | yes | task-level | no | — | structure only |
| nebius/SWE-rebench-openhands-trajectories [S27] | (Qwen3-Coder-480B, OpenHands 0.54) | **no** | partial | yes | task-level | no | — | structure only |
| nvidia/Open-SWE-Traces, SWE-Zero [S27] | 200k+, 318k | **no** | partial | yes | task-level | no | — | structure only; SFT-curated (biased toward successes) |
| BurstGPT (KDD'25) [S28] | 10.31M requests, Azure OpenAI, 213 days | yes | yes | no | — | **yes** (burstiness) | model | arrival + conversation-interval + response-length model |
| Alibaba microservices v2022 [v1] | ~20K services, >10K nodes, 13 days | yes | n/a | n/a | — | yes | service | tree-like heavy-tailed call graphs; hotspots |
| WfCommons WfInstances [S26] | 318+ | yes | n/a | n/a | — | n/a | compute | scientific DAGs, recipe machinery |
| Own society, shadow mode | whatever we run | yes | yes | yes | **yes** | **yes** | **all three** | the only source with 429/cold-start/gang events |

*Settles:* nothing public spans all three tiers with capacity events; public *structure* is abundant (hundreds of thousands of trajectories) but **timing-free**; public *timing* is scarce (thousands of sessions) and single-tier. §3 is built around that asymmetry.

### 1.9 What the deep survey changes relative to v1

| v1 decision | Change | Because |
|---|---|---|
| starvation freedom via aging + DRF | **virtual-time fair queuing** over a memory-centric per-workflow cost (Justitia/VTC), DRF only as the cross-tier cap on lease size | worst-case delay bound instead of "eventually" (§1.2) |
| model-tier admission = "lease fits in capacity" | admission must also satisfy the **SIC stability criterion** (no oscillating admit/evict) | limit cycles lose up to 50% throughput (§1.3) |
| Predictor heads: structure + quantity | add **failure-within-horizon** head (PrefixGuard/Seer target) | learnable, and it is the quantity the lease actually protects (§1.4) |
| "tool reservation" = tool concurrency slots | tool steps also reserve **planning/schema tokens** on the model tier (ProMCP) | that is where the tool step's cost may be (§1.4) |
| oracle = perfect predictor + our rule | oracle = **hindsight planner** on recorded exogenous inputs (Sinclair) | it bounds *all* rules, not just ours (§1.6) |
| Gate is a new proxy | Gate = **plugin in an existing flow-control seat** (GAIE EPP / LiteLLM / Envoy ext_proc) | the seat exists; do not rebuild it (§1.3, §5) |
| sim = "AgentServeSim-style" | sim = program-centric **+ partly-open arrivals + transition randomisation + recipe generator** | Schroeder; When Simulation Lies; WfChef (§1.7) |

---

## 2. Adapting Jev — a System One model (experimental)

*2026-09-22: superseded in detail by [JEV_SURVEY.md](JEV_SURVEY.md) (the documented API — choice / score / noul, confidence, fan-out, limits — and how `agentsim/jev.py` implements each pattern) and by NEEDS_REPORT.md §3 (measured calibration). The reasoning below stands; the question schema of §2.2–2.4 is now the catalogue in `jev.py`.*

### 2.1 What Jev is, in TypeSafe's own words [S35]

"System One Models" are "a new class of frontier models built to make fast, structured decisions that software can use directly." Jev is "a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out." It "gives up string generation," is "optimized for structured outputs and *can't* hallucinate" in the sense that "schema matching is guaranteed," and "all answers are accompanied with calibrated probabilities and confidence scores," trained with "Reinforcement Learning for Calibrated Decisions (RLCD)" which "optimizes for calibrated decisions: answers with epistemically honest probabilities." Inputs are "unstructured data (e.g. text) with an emphasis on *structured program state*"; sampling is "parallel — generates all outputs in a single query." Claimed cost and speed: "end-to-end response time is 70ms–500ms … 40x–200x faster for the same levels of frontier intelligence," "$0.042 / MTok" input, output "FREE." Intended uses: "smart if-statements … classify, route, score, extract, or branch," "map-reducing over big data," "real-time applications," "verify everything." Stated limits: "cardinality up to 255," no images yet, evaluation workflows "made by individuals on our model capabilities team, so some bias could exist," early access.

### 2.2 Why this is a better fit than a world model for *this* problem

H3 (v1) says the artifact the lease needs is a **calibrated distribution**, not a point prediction. Jev's whole design goal is calibrated typed decisions from program state — which is what the Observer holds (OTel spans, occupancy). Three consequences:

1. **The Predictor's four heads are Jev questions.** Next node type (enum ≤ 255 ✓), demand bucket (log-binned enum), "step needing tier X starts within h" (bool), "429/timeout/loop within h" (bool), "reaction if this call fails" (enum retry/replan/abort). Each answer comes with a probability that RLCD claims is calibrated; the lease consumes the bucket probabilities directly as the τ-quantile.
2. **It reads the text the generator cannot see.** RUNG0_REPORT.md E2 shows the seeded synthetic society is only R≈0.34 predictable from node types alone, and that exposing *phases* changes what is predictable. Real branching is decided by tool outputs and model text; Jev is a labeler that turns that text into typed phase / intent / remaining-work features — the semantic signal PBKV gets from the prefill hidden state, without touching the serving engine.
3. **It is cheap enough to annotate everything offline.** At $0.042/MTok, labelling 10⁵ public trajectories × ~50 k tokens ≈ 5 B tokens ≈ $210: phase labels and next-tool intent for the entire nebius/nvidia corpora, which is how recipes with *real* transition structure get mined (v2 §3.1, §4.2, §4.3).

### 2.3 Where Jev sits — and where it must not

**Not on the per-event hot path.** 70–500 ms per call versus the Gate's ≤5 ms budget (Hermes decides in <3 ms; PBKV 1.56 ms per 1,024). Jev is invoked **once per LLM step boundary, asynchronously**; an LLM step itself takes seconds (TraceLab decode ~47 tok/s, TTFT ~3 s), so a 500 ms decision about the *next* steps lands before it is needed. The Gate reads the latest Jev decision record *with its timestamp*; staleness is an input feature of the small on-path model, not a fallback branch.

Three integration points, one interface (`decide(state, questions) -> typed answers + probabilities`):

| Point | When | Questions | Consumer |
|---|---|---|---|
| **A. Offline annotation** | once, batch | phase, intent, next tool kind, remaining-steps bucket, task success | recipe miner (§3.1), training labels for the on-path predictor, `fit` |
| **B. Step-boundary features** | after each `chat` span, async | next_op, demand buckets, will-need-external-tool-within-h, will-fail-within-h, will-go-idle (think) | Reserver via the on-path predictor; idle prediction is what lets sandbox leases *expire* during human think time (RUNG0_REPORT: 1,300–5,900 s blocking waits) |
| **C. Reaction and verification oracle** | on error / at request end | reaction_if_429 ∈ {retry, replan, abort}; request succeeded? | simulator's reaction model (§3.4) fitted from real error episodes; task-success metric (§6.7) |

Jev is itself an external, rate-limited, early-access service — i.e. a **tool-tier resource the controller must reserve like any other** (its own `ext.jev` entry with a 429 curve). That is not a joke; it is the first integration test.

### 2.4 Design of the on-path predictor with Jev features

```
features_t = [ online Markov counts (CacheScout) , EWMA quantiles per tier , occupancy per tier ,
               jev_t' = latest Jev record (phase, next_op probs, buckets, p_fail, p_idle) , age(t - t') ]
heads      = next-node dist (K steps) | per-tier demand quantiles | survival CDF over h | failure hazard
model      = small (10^5 params), supervised; conformal calibration on real residuals (v2 §7.4)
```
Everything else (lease rule, ordering, gang acquisition, fairness cap, invariants I1–I6) is unchanged from v1 §6.

### 2.5 How it could be very wrong (and what would tell us)

| Risk | Why | Tell-tale | Mitigation |
|---|---|---|---|
| **Calibration does not transfer to our domain** | RLCD calibration was measured on TypeSafe's workflows ("some bias could exist") | expected calibration error > 0.1 on held-out real traces | use answers as *features* (conformalised by us), never as raw probabilities, until ECE is measured |
| **Latency in practice** | 70–500 ms is a claim under their load; early access | p99 decision latency > the median LLM step | keep it async; the on-path model works from stale features; measure staleness → outcome sensitivity |
| **Availability / lock-in** | early access; single vendor; cardinality 255 caps some enums | 429s from `ext.jev`; schema churn | reserve it like any tool; keep Markov/EWMA features so the controller is whole without it (they are inputs, not a fallback path) |
| **Text-only input** | no images/screens (browser agents) | browser-agent recipes unlabeled | label from DOM/text observations only |
| **"Can't hallucinate" ≠ correct** | schema guarantee only | confident wrong phases | measure phase-label accuracy against human labels on 200 traces before mining recipes |
| **Confounded gains (R6 of the old §2)** | more features vs better model | Jev-features win only where they also add data | fixed-data ablation: Markov / +EWMA / +Jev features / +Jev calibrated probs |
| **Non-monotonicity (H3)** | better predictions ≠ better reservations | prediction metrics up, outcomes flat | outcomes decide (§6.0) |

**Decision gate (rung 4b).** Adopt Jev features only if, at equal data, outcome metrics (failure, p99, waste, max wait) improve over rung 3 with p<0.05 across ≥5 seeds under Holm–Bonferroni **and** ECE on real traces ≤ 0.1 **and** the async integration adds no gate latency. Adopt Jev *annotation* (point A) earlier and separately: its test is whether recipes mined from Jev-labelled real trajectories raise the simulator's policy-ordering fidelity (§3.8-4) — measurable before any controller exists.

**Minimal first experiment (days, not weeks).** Label 200 real (or Exgentic) traces with Jev phase/next-op questions; measure label accuracy vs. a human pass and ECE of `next_op`; rerun `rung0/e2_predictability.py --with-phase` on the labelled traces. If R rises past the 0.7 one-step gate, the prediction rungs are funded; if not, rung 1 ships.

*Note on JEPA.* The earlier reading of "JEV" as JEPA (latent world models; LeJEPA/V-JEPA 2/TD-JEPA/HEPA) is retired. If a self-supervised trace encoder is ever wanted, HEPA's recipe (freeze encoder, fine-tune a horizon-conditioned survival head) is the one to copy; it is not part of the ladder.

## 3. Building the synthetic dataset

Principle (Ditto, WfChef): **clone the systems behaviour of real traces, not their semantics.** The generator has six layers; each is *fitted*, and each has a fidelity test.

### 3.1 Layer 1 — structure (which node next)

- **Mine recipes** (WfChef-style) from real traces: recurring sub-graphs — `plan → act → observe` loops, retry loops on error, fan-out/fan-in (parallel tool calls), verifier loops, hand-offs between agents. Sources: TraceLab, Exgentic (timed); nebius/nvidia trajectories (untimed, 10⁵ scale); own shadow traces.
- **Fit a structure model per recipe class**: a k-th-order Markov chain (or a tiny autoregressive model) over node types with loop-length and step-count distributions. **Match the entropy**: measure CacheScout's predictability R on real data and require the generator's R to match within tolerance — otherwise the controller's gains are an artefact of an over-regular generator (§4.11).
- **Dynamic vs static mix**: PBKV's benchmarks show both exist (HoVer/SWE-bench dynamic; FinanceBench static). Sample the mix as a knob.

### 3.2 Layer 2 — quantity (how much)

Per node type, conditional distributions of `tokens_in`, `tokens_out`, `tool_duration`, `sandbox_mem`, `retrieval_size`:
- families: log-t (TIE), log-normal body + Pareto tail, fitted with explicit tail-index estimation; validate the tail share against TraceLab (calls >1 min = 4.9% of calls, 92% of tool time; mean 16.8 s) and Copilot's long-tailed token consumption;
- **correlations, not independence**: context length grows with step index; `tokens_out` depends on op; upstream/downstream demands correlate (Hermes conditions on ρ > 0.5). Use conditional sampling on the prefix features or a copula; never independent marginals;
- memory: AgentSysBench sandbox peaks (28 GB/session) and the 32× cross-component latency divergence set the ranges.

### 3.3 Layer 3 — timing and arrivals (partly-open)

- **Across workflows (open):** arrivals from a BurstGPT-fitted process (Azure OpenAI burstiness over 213 days; conversation counts and intervals) with diurnal modulation; user think/idle gaps from Copilot (minutes-long idle at turn boundaries) and AgentSysBench (idle minutes–hours).
- **Within a workflow (closed):** the next step is released only on completion of its predecessor plus a sampled think time — AgentServeSim's "causal successor release." **Never replay timestamps.**

### 3.4 Layer 4 — the environment's dynamics (the part trace replay cannot give)

- Model tier: KV growth/eviction per the SIC dynamical model (so limit cycles can occur); TPM/RPM token buckets with provider-style headers; replica cold start 2–10 min; adapter/prewarm seconds.
- Tool tier: per-tool concurrency limits; external 429/502/reset probability as a rising function of concurrency (calibrated to E1 on the real society; HiveMind's 72–100% uncoordinated failure at 5–50 agents is the shape); MCP-style timeouts (~7–10 s → failed-dependency errors *(practitioner report)*); sandbox CPU contention slowing tool durations.
- Service tier: Alibaba-shaped call trees with hotspot services and queueing.
- **Agent reaction model** (critical, §4.1): on error/timeout/delay, `retry same node` / `replan branch` / `abort` with probabilities fitted from real error episodes (MAST categories; DPBench's protocol sensitivity; When Simulation Lies' finding that retry policies emerge).

### 3.5 Layer 5 — perturbations (domain randomisation)

Knobs, each randomised across scenarios **and exposed to the Predictor as features** (Decima's lesson): arrival rate, fan-out, tail index, per-tier capacity, rate-limit ceilings, cold-start delay, error-injection rate, static/dynamic recipe mix, reaction-model probabilities, agent-population mix (coding / research / browser). Perturbation classes follow When Simulation Lies: observation (dropped spans), action-space (unknown tool), reward-metadata (wrong cost), transition (latency spikes, quota changes).

### 3.6 Layer 6 — labels and the oracle

- Outcome labels come free from the simulator: admission waits, 429s, timeouts, deadlocks (detector [v1 R29] as oracle), waste, fairness.
- **Hindsight oracle:** for each generated exogenous sequence (arrivals + demands + durations), run a hindsight planner (Sinclair) that knows the whole sequence and computes the best reservation schedule. Store its actions as `a*` (imitation target) and its outcome as the per-scenario upper bound.

### 3.7 Pipeline and sizes

```
real traces ──► WfChef-style miner ──► recipes.json (structure) + marginals.json (quantities, timing)
                                          │
knobs ──► generator(recipes, marginals) ──► workflows (WfFormat-like JSON) ──► partly-open arrival stream
                                          │
                        program-centric discrete-event simulator (tiers, dynamics, reaction model)
                                          │
                 traces (OTel-shaped) + outcome labels + hindsight oracle actions/outcomes
```
- Scale: 10³–10⁵ concurrent workflows per scenario on CPU (AgentServeSim runs as a CPU fitness evaluator); 10²–10³ scenarios across the knob grid.
- Format: emit OTel GenAI span records (`invoke_agent` → `chat` / `execute_tool` with `gen_ai.tool.name`, `gen_ai.tool.call.id`) so real and synthetic traces are byte-compatible for the Observer.

### 3.8 Validating the generator before trusting it (in this order)

1. **Marginal fidelity** — KS / Wasserstein on durations, tokens, memory; tail-index within CI; entropy R within tolerance.
2. **Structural fidelity** — n-gram divergence of node sequences; loop-length and step-count distributions; fan-out distribution.
3. **Dynamics fidelity** — the failure-rate-vs-concurrency curve of the *uncoordinated* baseline in sim matches E1 on the real society.
4. **Behavioural fidelity (decisive)** — the **ordering of policies** (uncoordinated < reactive gate < fixed-rule lease) and the *size* of their gaps in sim match the real shadow/A-B measurements within a stated tolerance. This is how Decima and AgentServeSim justified their simulators. A generator that fails 4 is not used, whatever it scores on 1–3.

---

## 4. Limitations of the synthetic dataset / simulated agentic log

Ordered by how much they can bias the controller's measured gains. "Residual" = what remains after the mitigation.

| # | Limitation | Why it matters here | Mitigation | Residual |
|---|---|---|---|---|
| 4.1 | **Off-policy traces and the agent's reaction.** Recorded traces show what agents did under *no* controller. A controller that delays or denies a step changes what the agent does next (timeouts → retries → replans). | The very interventions we evaluate are absent from the data. | Closed-loop simulation with a fitted reaction model (§3.4); Jev's reaction oracle (§2.3 point C) fitted on real error episodes; deliberate intervention canaries in shadow-enforce. | The reaction model is fitted from *rare* error episodes; the tail of agent behaviour under delay stays unknown until enforcement. **Cannot be fully mitigated offline.** |
| 4.2 | **Semantic blindness.** Branching is decided by text the generator does not have. | Structure predictability in sim is exactly the generator's entropy; real predictability may be higher (PBKV uses the prefill hidden state) or lower. | Hybrid: use real *content* trajectories (10⁵ HF) for structure, synthesise only timing/quantities; report gains as a function of R (§4.11). | Real-content trajectories are timing-free and SFT-filtered toward successes (§4.3). |
| 4.3 | **The largest public datasets have no timestamps** and are curated for SFT (biased to successful, shorter trajectories). | Durations must be imputed from a different population (TraceLab/Copilot), assuming independence of timing from content. | Impute conditionally on tool type + step index; weight by failure rate to de-bias. | Independence assumption is untestable without our own timed traces. |
| 4.4 | **Heavy-tail estimation from small samples.** The tail (4.9% of calls, 92% of time) is where reservation matters and where 10³ sessions give the noisiest estimate. | Under-sampled extremes → optimistic waste/failure numbers. | Parametric tail fits with CIs; importance-sample tail scenarios; stress scenarios at 2× fitted tail index. | The true tail of a new society is unknown until observed. |
| 4.5 | **Provider dynamics are opaque and non-stationary.** 429 behaviour, priority tiers, backend congestion change without notice. | The sim's 429 model is a guess; HiveMind evaluated on mock APIs for this reason. | Calibrate to E1 on the real endpoints; treat capacity as a learned quantity with an uncertainty margin (AIMD + UCB, §1.3); re-calibrate on a cadence. | Regime changes mid-experiment can invalidate results; log provider headers. |
| 4.6 | **Wrong arrival model flips conclusions** (Schroeder). | Closed-only replay hides overload; open-only ignores per-workflow dependency. | Partly-open generator (§3.3). | Think-time distributions are borrowed from coding agents; other agent types differ. |
| 4.7 | **Missing cross-tier coupling.** Real bottleneck shifts arise from interactions (sandbox CPU contention → slower tools → longer KV residency → memory pressure → evictions). | Independently built tier models miss the coupling that creates the failures. | Model the couplings that AgentSysBench and SIC identify explicitly; validate by dynamics fidelity (§3.8-3). | Unknown couplings stay unknown. |
| 4.8 | **Non-stationarity of the society itself.** Model version switches, prompt/tool updates, context compaction (Copilot: caches invalidated). | Traces age in weeks; a controller tuned to last month's recipes drifts. | Drift monitors on R, calibration coverage, tail index; scheduled re-fit. | Every re-fit is a new small-sample problem (4.4). |
| 4.9 | **Agent monoculture in public data.** SWE-bench coding agents dominate; deep-research/browser/scientific agents differ (PASTE, PrefixGuard evaluate them separately). | Gains measured on coding recipes may not transfer. | Recipe classes as a knob; report per-class. | Non-coding timed traces are essentially unavailable publicly. |
| 4.10 | **LLM non-determinism.** Same task → different trajectories run to run. | A single synthetic sample per scenario understates variance. | ≥5–10 seeds per scenario; paired designs with common random numbers. | Variance itself may be non-stationary. |
| 4.11 | **Predictability decides the gain.** A more regular generator inflates every prediction-based rung. | Reviewers (and we) will not know whether gains are real. | Report every gain as a **curve over R** (generator entropy), and mark the real society's measured R on the curve. | — (this is a reporting discipline, not a fix). |
| 4.12 | **Privacy / sharing.** Real traces contain code, prompts, secrets. | Cannot publish, cannot crowd-validate. | Ditto-style clone: keep structure + systems marginals, drop content. | Clones cannot be re-labeled semantically later. |
| 4.13 | **Simulator fidelity ceiling.** Even validated simulators are ~5% off on mean JCT (AgentServeSim) and unvalidated on failure tails. | Sub-5% gains are within noise; tail behaviour is where our metrics live. | Only claim gains > 2× simulator error; validate tails in the paired cells too. | Tail validation needs real overload runs, which are expensive. |

---

## 5. How to realistically get this solved

### 5.1 Build a control plane, not a serving engine

| Component | Reuse | Build |
|---|---|---|
| **Observer** | OTel GenAI semantic conventions [S34]: `gen_ai.operation.name ∈ {invoke_agent, invoke_workflow, chat, execute_tool, retrieval, plan, …}`, `gen_ai.tool.name`, `gen_ai.tool.call.id`; exporters exist in LangGraph/AutoGen/CrewAI/OpenAI Agents SDK stacks (status: Development, not yet 1.0 — pin a version) | a stream consumer that builds `G_w` per workflow and per-tier occupancy; ~500 lines |
| **Gate (model tier)** | K8s Gateway API Inference Extension Flow Control / llm-d EPP (policy-aware queues at the gateway, KV- and cost-aware) [S12]; or LiteLLM's priority-reserved TPM/RPM with Redis counters [S13]; or Envoy `ext_proc` | a plugin that consults the ledger before forwarding; throws on `NoLease`/`LeaseExceeded` |
| **Gate (tool tier)** | tool/MCP gateway with per-tool semaphores; circuit breakers | same plugin, tool side |
| **Gate (service tier)** | service mesh rate limits / circuit breakers | same plugin, service side (later) |
| **Reserver / ledger** | Redis (as LiteLLM does) | single-node lease ledger with invariants I1–I6; virtual-time fair queue; SIC-safe admission rule |
| **Predictor** | Jev (System One) as async step-boundary featurizer and offline annotator (§2.3) | rung-dependent: EWMA + Markov (rung 3) → small supervised model (rung 4) → + Jev features/calibrated answers (rung 4b); served as ONNX behind the ledger |
| **Simulator** | AgentServeSim as reference architecture; WfCommons WfFormat/WfChef/WfGen for recipes | tool + service tiers, SIC dynamics, reaction model, hindsight oracle |
| **Oracle** | Hindsight Learning code (public) [S23] | adaptation to leases |
| **Test oracles** | agent deadlock/livelock detectors [v1 R29] | — |

The single hard requirement: **no path from an agent to a provider that bypasses the Gate** (egress rule, not library convention) — otherwise H1 returns.

### 5.2 Sequence (why this order)

1. **Observe first** (rung 0). No design decision survives contact with the real R, tail index and failure-vs-concurrency curve.
2. **Ship the reactive floor** (rung 1) — HiveMind's primitives are the biggest single gain in the literature and need no prediction. Everything after is measured against this, not against "uncoordinated".
3. **Measure headroom before learning** (rung 2) — hindsight oracle minus reactive floor. If the headroom is small, stop; the problem was coordination (H1), not prediction.
4. **Fixed rule + cheap predictors** (rung 3). Two knobs. Conformal on real.
5. **Learned predictor** (rung 4) only if rung 3 leaves measured headroom.
6. **Jev features** (rung 4b) only if rung 4 leaves measured headroom *and* Jev's calibration on real traces passes (§2.5).
7. **Harden** (rung 6).

### 5.3 What "solved" means (acceptance)

On the real society, under a load sweep up to 2× the uncoordinated breaking point: workflow failure rate ≤ 1% (from 72–100% uncoordinated), p99 workflow completion within 1.3× of the reactive floor's p50-scaled baseline, reserved-but-unused capacity-time ≤ 10% per tier, worst-case admission delay bounded (virtual-time guarantee) and observed ≤ the bound, zero deadlocks in 10⁴ multi-tier steps (detector-verified), task success unchanged (±1 pp), gate overhead ≤ 5 ms p99. Each number is a target to be re-set after rung 0; the *shape* of the acceptance is fixed.

### 5.4 Effort sketch (one engineer-equivalent, calendar)

Rung 0: 2–3 weeks · Rung 1: 2 weeks · Rung 2: 4–6 weeks (simulator + generator + oracle is the largest build; the simulator, generator seeds and knobs, and a clairvoyant-gate oracle exist as of 2026-09-22; a hindsight *planner* does not) · Rung 3: 2–3 weeks · Rung 4: 3–4 weeks · Rung 4b (Jev): 2–3 weeks · Rung 6: ongoing. Rungs 0–3 (≈3 months) deliver a defensible system; 4–4b are research options with kill criteria.

---

## 6. The ladder — ideate → develop → tune → establish gains/losses

### 6.0 Evaluation protocol shared by every rung

- **Paired design**: every policy sees the *same* exogenous sequences (arrival seeds, demands, durations) — common random numbers; this is the input-dependent-baseline idea applied to evaluation, and it makes small differences detectable.
- **Seeds**: ≥5 (10 for rungs 4–5); report medians, 95% CIs, effect sizes, per-seed win counts; Holm–Bonferroni across metric families.
- **Load sweep**: gains are load-dependent (Decima: 2× at high load, ~21% average); always report the curve, not one point.
- **Predictability sweep**: gains vs generator entropy R, with the real society's R marked (§4.11).
- **Oracle-normalised score**: `(policy − reactive) / (oracle − reactive)` per metric — how much of the available headroom a rung captures.
- **Loss columns are mandatory**: peak throughput sacrificed (SAGA pays ~30%), waste, gate latency, engineering cost, task-success change.
- **Metrics** (v1 §6.7): failure rate; p50/p99 workflow completion; waste per tier; max/mean admission wait; fairness (Agent-Fair-Share-style deviation and Jain); throughput; gate overhead; $ cost; task success.

### 6.1 The rungs

| Rung | Goal | Build | Measure (gains) | Losses to watch | Exit / kill criterion |
|---|---|---|---|---|---|
| **0 Observe** | Know the real society | OTel export; shadow Observer; E1 (failure vs concurrency), E2 (R, Markov 1/2/3-step), E3 (tails), E4 (multi-tier cycles) — scripts exist in `rung0/` and were validated on synthetic data (RUNG0_REPORT.md) | the four curves; `marginals.json`; recipes | none (no enforcement) | H1–H5 re-confirmed on *our* system; if R is low (E2 accuracy <0.7 — as the seeded synthetic society already is), the ladder ends at rung 1 |
| **1 Reactive floor** | Remove H1 failures without prediction | Gate in the existing flow-control seat: admission `A<Cmax`, coordinated retry with jitter, token bucket ≈80% of tier, priority queue; tool-tier semaphores; SIC-safe admission | failure rate, p99, waste vs uncoordinated | added queueing latency; throughput under-utilisation from the 80% bucket | failure rate ≤ 5% at the load where uncoordinated fails 72–100%; this becomes the **baseline for everything after** |
| **2 Simulator + headroom** | Know if prediction is worth anything | generator (§3) + program-centric sim + hindsight oracle; validate by §3.8 (policy ordering must match rung 0/1 data) | oracle − reactive per metric (the headroom); gain-vs-R and gain-vs-load curves for the oracle | simulator error (≤ ~5% mean; tails unvalidated) | if headroom < 2× simulator error on every metric → **stop, ship rung 1**; else continue |
| **3 Fixed-rule leases, cheap predictors** | Capture headroom with two knobs | Reserver with leases (τ, h), virtual-time fair queuing, gang acquisition in global order, DRF cap; predictors = online Markov (structure) + EWMA/quantile-regression (quantity); conformal τ on real residuals | oracle-normalised score; failure, p99, waste; worst-case delay vs bound; deadlocks = 0 | over-reservation waste at high τ; delay from gang acquisition | captures ≥ 50% of headroom with waste ≤ 10% → ship; else tune τ/h; if τ/h sweep cannot reach 50%, proceed to rung 4 |
| **4 Learned predictor** | Close the gap the cheap predictors leave | small sequence model (PBKV-class) trained on synthetic, fine-tuned on real; same rule; E6 ablation (no-prediction / Markov / learned / oracle) | Δ over rung 3 on outcome metrics, p<0.05, ≥5 seeds; label-efficiency at 300/1k/3k real sessions | training + serving cost; drift sensitivity | Δ significant and > 2× simulator error → ship; else keep rung 3 |
| **4b Jev features** *(experimental)* | Semantic features and calibrated decisions at step boundaries | §2.3 points A–C; async integration; `ext.jev` reserved like a tool | Δ over rung 4 on outcomes at equal data; ECE on real traces; staleness sensitivity | vendor dependence; label errors; async lag | §2.5 decision gate; kill if ECE > 0.1 or no outcome gain |
| **6 Harden** | Run it for real | distributed ledger preserving I1–I6; per-tenant fairness keys; drift monitors (R, coverage, tail index) with auto-recalibration; provider-header ingestion; chaos tests (quota drops, cold starts) | sustained-load metrics over days, not minutes | operational cost | acceptance §5.3 met for 2 weeks of production load |

### 6.2 Ideation loop inside each rung

```
hypothesis (what will this rung gain, on which metric, at what load and R)
   → evidence in sim (paired, seeded, load & R sweeps)
   → evidence in shadow on the real society (no enforcement)
   → canary enforcement on a slice (A/B against previous rung)
   → decision: ship / tune knobs / kill rung
```
Tuning is confined to declared knobs (rung 1: `Cmax`, bucket fraction; rung 3: τ, h; rung 4: model size, fine-tune data size; rung 5: λ, horizon K, MPC cost weights α, β). Anything else is a design change and goes back to the hypothesis step.

### 6.3 Scoreboard template (one row per rung × scenario)

```
rung | scenario(load, R, tail-index, agent-mix) | seeds | failure% [CI] | p50 TCT | p99 TCT | waste%/tier | max wait vs bound | fairness dev | throughput | gate ms p99 | $ | task success | oracle-normalised score | loss notes
```

### 6.4 Expected shape of gains and losses (from the evidence, to be replaced by measurements)

*Measured on synthetic data, 2026-09-22 morning (RUNG0_REPORT.md §4–§9):* the reactive gate captured the whole failure-rate gain on the API tier and the clairvoyant oracle added nothing there; the oracle's only headroom on the hosted society came from idle-sandbox management and SRPT (failures 3.8% → 2.5%, +18% throughput at the highest load; its p99 gain is not significant); pinning the model slot across tools cost latency/throughput/fairness, not failures.

*Measured the same evening (RUNG0_REPORT.md §12), after building rung 3:* the `LeaseController` beats the gate as configured on every metric at 10 seeds — and the E6 ablation over the gate's own knobs shows ~90% of that is the sandbox idle-timeout PRIOR (300 s → 0 s), the rest virtual-time fair queuing (a fixed rule), and nothing measurable from idle-time prediction at any predictability level (`think_snr` 0–0.9) or cold start (3–90 s). Gang leases raise the parallel share of fan-outs but waste 72–77% of reservations on a saturated CPU pool; budget leases eliminate 429s but buy nothing over SDK retries. So on this generator the rung-2 kill criterion fires: ship rung 1 with the idle timeout at 0 and fair queuing; rungs 3–4b have no earnable headroom until a society is found (real or randomised, §3.5) where cold starts are dear, the pool has slack, or provider budgets bind without background noise. Two lessons for the protocol of §6.0: **sweep the baseline's knobs before crediting prediction** (an E6 ablation is mandatory, not optional), and with 5 seeds no exact test reaches α = 0.05 (F18).

The bullets below were written before those measurements and stand as the prior they were.


- Rung 1 will take most of the *failure-rate* gain (HiveMind: 72–100% → 0–18%). Prediction rungs mostly buy **latency, waste and fairness**, not failure rate.
- Prediction's value rises with load (Decima) and with R (CacheScout), and collapses when R ≈ random routing.
- A fixed rule with a decent demand model has historically been within ~10% of an oracle (Hermes). Expect rungs 4–4b to fight over that last 10% — which is why their kill criteria are strict.
- Every reservation rung pays throughput (SAGA −30% for atomicity). The lease's backfill (v1 §6.2) is what should keep our loss well below that; measure it explicitly.

---

## 7. References (this document; v1 references remain valid as [v1 R#])

Jev / System One
- [S35] Almeida, *Introducing System One Models & Jev*, TypeSafe AI blog, 15 Sep 2026. https://typesafe.ai/blog/introducing-system-one-models-and-jev

JEPA family (retired reading of "JEV"; kept for the record)
- [S29] Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, 2025. https://arxiv.org/abs/2511.08544
- [S30] Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, 2025. https://arxiv.org/abs/2506.09985
- [S31] Bagatella et al., *TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning*, ICLR 2026. https://arxiv.org/abs/2510.00739
- [S32] Sobal et al., *Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models*, 2025. https://arxiv.org/abs/2502.14819
- [S33] Petersen et al., *HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series*, 2026. https://arxiv.org/abs/2605.11130
- Also seen: LeWorldModel (https://arxiv.org/abs/2603.19312), VL-JEPA (ICLR 2026, https://arxiv.org/abs/2512.10942), MTS-JEPA (https://arxiv.org/abs/2602.04643), Phys-JEPA (https://arxiv.org/abs/2606.16076), LaT-PFN (https://arxiv.org/abs/2405.10093), JEPA4Rec (https://arxiv.org/abs/2504.10512), *A Generalization Theory for JEPA-Based World Models* (https://arxiv.org/abs/2606.27014). V-JEPA 2-AC planning speed (~16 s/action vs Cosmos ~4 min) *(reported by secondary sources)*.

Serving, scheduling, fairness, admission
- [S1] Parrot, OSDI 2024 — v1 R1. [S2] Tan et al., *Teola/Ayo*, ASPLOS 2025. https://arxiv.org/abs/2407.00326 · https://github.com/NetX-lab/Ayo
- [S3] Autellix — v1 R2. [S4] HexAGenT — v1 R6.
- [S5] Guo, Wu, Yiu, *SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters*, 2026. https://arxiv.org/abs/2605.00528
- [S6] Hu Wei, *From Agent Loops to Structured Graphs: A Scheduler-Theoretic Framework for LLM Agent Execution*, 2026 (position). https://arxiv.org/abs/2604.11378
- [S7] Sheng et al., *Fairness in Serving Large Language Models* (VTC), OSDI 2024. https://arxiv.org/abs/2401.00588
- [S8] Wei et al., *Equinox: Holistic Fair Scheduling in Serving Large Language Models*, 2025. https://arxiv.org/abs/2508.16646
- [S9] Yang et al., *Justitia: Fair and Efficient Scheduling of Task-parallel LLM Agents with Selective Pampering*, 2025. https://arxiv.org/abs/2510.17015
- [S10] Ao, Dong, Luo, Simchi-Levi, *Service-Induced Congestion in Memory-Constrained LLM Serving*, 2026. https://arxiv.org/abs/2606.15555
- [S11] Patke et al., *Queue Management for SLO-Oriented Large Language Model Serving* (QLM), SoCC 2024. https://arxiv.org/abs/2407.00047
- [S12] Kubernetes Gateway API Inference Extension (Flow Control) and llm-d inference scheduler. https://gateway-api-inference-extension.sigs.k8s.io/guides/flow-control/ · https://github.com/llm-d/llm-d-inference-scheduler/
- [S13] LiteLLM: priority-based rate limiting; dynamic TPM/RPM. https://docs.litellm.ai/release_notes/v1.77.3-stable/v1-77-3 · https://docs.litellm.ai/docs/proxy/dynamic_rate_limit
- [S14] *Decomposing Predictive Kubernetes Autoscaling for LLM Serving Under Long Startup Delays*, 2026. https://arxiv.org/abs/2609.20874
- [S15] Sui et al., *Parallelizing Tool Execution and LLM Generation for Low-Latency Agent Serving* (PASTE), 2026. https://arxiv.org/abs/2603.18897
- [S16] Huang et al., *PrefixGuard: From LLM-Agent Traces to Online Failure-Warning Monitors*, 2026. https://arxiv.org/abs/2605.06455
- [S24] SageServe (forecast-aware autoscaling), SIGMETRICS. https://arxiv.org/abs/2502.14617 ; TokenScale https://arxiv.org/abs/2512.03416
- [S20] *ProMCP: Profiling Token Flows and Latency Costs in MCP-Based LLM Agents*, ACL 2026 Findings. https://aclanthology.org/2026.findings-acl.1967/

Microservice / serverless lineage
- [S17] Gan et al., *Seer*, ASPLOS 2019. https://dl.acm.org/doi/10.1145/3297858.3304004
- [S18] Zhang et al., *Sinan*, ASPLOS 2021. https://dl.acm.org/doi/10.1145/3445814.3446693
- [S19] Zhou et al., *AQUATOPE*, ASPLOS 2023. https://arxiv.org/abs/2212.13882
- [S21] Liang et al., *Ditto: End-to-End Application Cloning for Networked Cloud Services*, ASPLOS 2023. https://people.csail.mit.edu/delimitrou/papers/2023.asplos.ditto.pdf
- FIRM, OSDI 2020. https://www.usenix.org/system/files/osdi20-qiu.pdf

Learning with exogenous inputs, simulation methodology
- [S22] Mao et al., *Variance Reduction for RL in Input-Driven Environments*, ICLR 2019. https://arxiv.org/abs/1807.02264
- [S23] Sinclair et al., *Hindsight Learning for MDPs with Exogenous Inputs*, ICML 2023. https://arxiv.org/abs/2207.06272 · https://github.com/seanrsinclair/hindsight-learning
- [S25] Schroeder, Wierman, Harchol-Balter, *Open Versus Closed: A Cautionary Tale*, NSDI 2006. https://www.usenix.org/legacy/event/nsdi06/tech/full_papers/schroeder/schroeder.pdf
- [S26] WfCommons (WfChef, WfGen, WfBench, WfFormat, WfInstances). https://wfcommons.org/ · https://arxiv.org/abs/2105.14352
- AgentServeSim — v1 R24. When Simulation Lies — v1 R30. Decima — v1 R20.

Traces and datasets
- [S27] nebius/SWE-agent-trajectories (80,036) https://huggingface.co/datasets/nebius/SWE-agent-trajectories ; nebius/SWE-rebench-openhands-trajectories ; nvidia/Open-SWE-Traces (200k+) ; nvidia/SWE-Zero-openhands-trajectories (318k)
- [S28] Wang et al., *BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems*, KDD 2025. https://arxiv.org/abs/2401.17644 · https://github.com/HPMLL/BurstGPT
- TraceLab, Copilot traces, AgentSysBench, Exgentic, Alibaba — v1 R21–R26.

Observability
- [S34] OpenTelemetry GenAI semantic conventions — agent spans. https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md · https://opentelemetry.io/blog/2026/genai-observability/

Verification note: MCP timeout behaviour (~7–10 s → failed-dependency) is a practitioner report, not a paper. The "tool execution is 16–37% of end-to-end latency, median 1.19 s, p95 70 s" figure surfaced in search summaries without a verifiable source and is **not used** above.
