# Control — understanding the solution

A reader's guide to what this project set out to do, how the problem was attacked, what was actually found, and what is shipped. Ten minutes. Every number here is a pointer into [RUNG0_REPORT.md](RUNG0_REPORT.md) (Phases 0–5) or [NEEDS_REPORT.md](NEEDS_REPORT.md) (Phases 6–7), the only places numbers live; the reasoning and literature are in [SERVICE_GRAPH_RESERVATION.md](SERVICE_GRAPH_RESERVATION.md) (v1), [DEEP_DIVE_AND_LADDER.md](DEEP_DIVE_AND_LADDER.md) (v2), [PREDICTOR_DESIGN.md](PREDICTOR_DESIGN.md) (v3) and [JEV_SURVEY.md](JEV_SURVEY.md).

---

## 1. The problem in one picture

```
   N agentic workflows (coding / research agents), each a loop of
        LLM step → tool calls → LLM step → … → (human thinks) → next request

   all sharing three tiers of capacity:

   ┌──────────────┐   ┌──────────────────────┐   ┌──────────────────┐
   │  MODEL tier  │   │      TOOL tier        │   │  SERVICE tier    │
   │ API slots    │   │ sandboxes (mem, CPU)  │   │ retrieval, RPC   │
   │ TPM buckets  │   │ external APIs (RPM)   │   │                  │
   │ KV cache     │   │                       │   │                  │
   └──────────────┘   └──────────────────────┘   └──────────────────┘
```

Uncoordinated, the workflows call providers directly, retry on error, and collectively behave like a thundering herd: **429 storms, client timeouts, hold-and-wait deadlocks, starvation** — even when aggregate capacity is sufficient. The simulator reproduces the published shape (§4): 10 coding agents on 8 API slots at 40% utilisation already lose ~15% of requests to retry storms; 20 agents lose 98%.

The goal: a controller that keeps a society of agents from failing, without over-allocating, starving anyone, or adding a new hot-path system.

## 2. The idea

Every agent already talks to providers through a gateway (LiteLLM, the Kubernetes Gateway API inference extension, a tool/MCP proxy). Put **one Gate** in that seat and give it four responsibilities, each a separate component (v1 §6.1):

```
   agent runtime ──step request──▶ ┌────────┐
                                   │  GATE  │──admit / wait──▶ provider
                                   └───┬────┘
        observes every step/end        │ consults
   ┌──────────┐   ┌───────────┐   ┌────▼──────┐
   │ Observer │──▶│ Predictor │──▶│ Reserver  │   ledger of leases with
   │ (spans)  │   │ (next     │   │ (leases,  │   invariants: Σ leases ≤ capacity,
   └──────────┘   │  step,    │   │  expiry,  │   nothing past its expiry,
                  │  demand)  │   │  fairness)│   acquisition in global order
                  └───────────┘   └───────────┘
```

- **Observer**: builds each workflow's revealed graph from OpenTelemetry GenAI spans — the same span format for real and synthetic traces.
- **Predictor**: the only learned part. From what the gateway can *see* (step kinds, realised durations, request boundaries) it estimates the next step and its demand.
- **Reserver**: turns predictions into **leases** — "this workflow may use *q* units of *this* resource from *t₀* until *t₀+h*". Leases are backfillable (others may use what is not yet consumed), expire unconditionally, are capped at a fair share, and are acquired in a fixed global order — so deadlock and starvation are impossible *by construction*, not by tuning.
- **Gate**: admits or queues a step by consulting the ledger. It never predicts and holds no state beyond the ledger.

Two knobs only: τ (which quantile of the demand distribution to reserve) and h (how far ahead). Uncertainty widens the distribution; there is no "fallback policy".

## 3. How we attacked it — a ladder, not a leap

The design was not built first and justified later. Each rung had to earn the next (v2 §6):

| Rung | Question | What was built |
|---|---|---|
| 0 Observe | What does a real society look like? | E1–E5 scripts on OTel spans: failure vs concurrency, predictability of the next step, tails, deadlock cycles, idle-time predictability |
| 1 Reactive floor | How much does coordination alone fix? | `ReactiveGate`: queue everything, jittered retries, header-based pausing, AIMD |
| 2 Headroom | Is prediction worth anything? | The simulator + a **clairvoyant oracle** (sees every duration, token, think time, provider draw) — headroom = oracle − gate |
| 3 Fixed-rule leases | Can a real controller capture it? | `LeaseController`: online quantile trackers and Markov chains, leases with invariants, gang and budget reservations, virtual-time fair queuing |
| 4 Learned predictor | Only if rung 3 leaves headroom | — (not reached by the ladder) |
| 6–7 The full design, built anyway | What does the whole thing buy, and where? | `NeedsController` (`policy.type=needs`): Observer → multi-head calibrated predictor → demand forecast → expected-value-gated leases; System One (Jev) integration to the real API with a local stand-in and a live calibration judge; tenant budgets over costed tools / MCP servers / APIs; a feedback loop on the controller's own prediction error |

Three disciplines made the answers trustworthy: **paired seeds** (every policy sees the same workload via common random numbers), an **oracle bound** on prediction, and **ablations over the baseline's own knobs** before crediting anything to prediction. The last one turned out to matter most.

## 4. What the evidence said

**Coordination is the win on the API tier (§4).** The reactive gate takes 98% failure at 2.5× the slot count to 0.5%; the clairvoyant oracle adds nothing on top. Rung 1 is the deliverable there.

**On a self-hosted society the "headroom" was a knob (§12).** The lease controller beat the gate on every metric at 10 seeds — failure 4.2% → 1.3%, below even the oracle. The ablation then showed ~90% of that came from the gate's *idle-sandbox timeout* (300 s → 0 s: park the moment a session goes idle), the rest from fair queuing — both fixed rules — and **nothing measurable from the learned idle-time, KV or budget predictions**, at any predictability level or cold-start cost.

**Real traces agree (§13.5).** On 5,312 real Claude Code sessions (TraceLab, CC BY 4.0): tails as published, next-step predictability just over the 0.7 gate, and **idle time unpredictable from what a gateway sees (R² 0.05)**. A society fitted from those traces ranks the policies the same way.

**Prediction pays in exactly one place (§13.3–13.4).** A randomised sweep over 40 knob-sampled societies located the oracle's headroom where the **model tier saturates**. There, sorting chats by *predicted service time* — the prefill size (known exactly at request time) plus the learned mean output per recipe — captures ~all of the oracle's failure headroom and matches its throughput within 2% (p50 latency 388 s → 21 s). Its cost: a small failure increase in societies whose model tier is *not* saturated (long requests starve).

**The full controller, built without the ladder's gate (NEEDS_REPORT.md).** With the whole design in place — content-aware quantile predictor, conformal calibration, demand forecast, expected-value gating, family-owned gang leases for sub-agents, a System One model asked at every boundary and judged live — the outcome on the hosted, multi-agent, TraceLab-fitted and stress societies is **the same as the three fixed rules'**, at every level of content signal. Two places differ: where the **model tier saturates** the controller switches to shortest-predicted-first by itself and gets `srpt`'s outcome (failure 0.046 → 0.001 and 0.192 → 0.054; throughput ×4 and ×2.2 over the shipped gate, within 2–5 % of `srpt`), and where a **tenant budget binds** its forecast-aware pacing is the largest gain the project has found (10 seeds: p50 −43 % and throughput +10 % significant, zero refused payments, spend wasted on abandoned work 2.9 % → 1.6 %, 9 % cheaper per completed request, failure 0.035 → 0.019 at 7/10 wins; the cost: more client timeouts while paced work waits — the clairvoyant gate: 0.017 failure, +30 % throughput). Three of the controller's own defaults hurt and were caught only by per-switch ablation (the starvation guard, the spawn gang's lifetime, predicted KV reserves) — the same lesson as §8, one level up.

## 5. What is shipped

Three fixed rules for the gate, no new infrastructure, no model-serving path, invariants asserted:

| Rule | Where it applies | Why |
|---|---|---|
| **Park idle sandboxes immediately** (`sandbox_idle_timeout_s = 0`) | everywhere | The sandbox pool is what binds in coding-agent societies; a cold start (seconds) is cheaper than an idle slot (minutes) |
| **Virtual-time fair queuing** on realised memory-centric cost (`queue = vtfq`) | where the sandbox pool binds | Bounded worst-case wait; −20% failures at overload, at a p50 cost |
| **Shortest-predicted-chat-first** (`queue = srpt`) | where the model tier binds | The one prediction that pays; trivial to compute from observable inputs |

Everything else that was built — leases, ledger, gang and budget reservations, the predictors — exists, is tested, and is honest (a selftest feeds the controller contradictory hidden values and asserts identical decisions). It is *not* shipped, because on every society we could test it does not beat the rules above.

| Rule (Phase 7) | Where it applies | Why |
|---|---|---|
| **Forecast-aware budget pacing** (`policy.type = needs`, its `pacing` and `feedback` switches on) | where a tenant budget in dollars or tokens binds | Admit new requests only when the in-flight ones can still be paid for; an AIMD floor on refusals; no refused payments, half the spend on abandoned work, p50 −43 %, throughput +10 % (NEEDS_REPORT §5.6) |

## 6. Where it lives in the code

```
agentsim/policies.py    ReactiveGate (queue = fifo | vtfq | srpt)  ← the shipped rules
                        ClairvoyantGate (the oracle)
                        LeaseController (rung 3: leases, gang/budget reservations, predictions from on_event only)
agentsim/resources.py   the three tiers; Lease + Ledger with invariants I1/I3
agentsim/engine.py      program-centric discrete-event engine; the Gate's admission path is _try_run
agentsim/predict.py     streaming quantiles and online Markov — what a gateway can learn on its own stream
agentsim/fit.py         rung 0 proper: structure and quantities mined from real OTel spans
rung0/                  E1–E5, compare.py (paired statistics), headroom.py (where does prediction pay), tracelab_to_spans.py
scenarios/              societies (synthetic, stress, fitted-from-real), 19 grids, knob ranges for domain randomisation
```

Run it: `uv run python -m agentsim selftest` (47 checks, ~2 s); the README lists every grid and script.

## 7. What is not solved

- The trace we have is one population (single coding agents, one provider). Cold-start costs, pool slack and provider budgets of *your* society decide which of the three rules bind — shadow traces from it are the next input.
- Multi-agent spawn/join (the fan-out where gang reservation is hardest) is not in the generator.
- `srpt` needs a queue-length-conditional switch to remove its small off-saturation failure cost.
- The learned rungs (4, 4b) were never reached by the ladder — and, built anyway in Phases 6–7, they changed no outcome on the societies the ladder had already settled; their value is in the two regimes above and in the machinery (calibration, judging, pacing) that a real society with costs and text can now be measured on.
- The data: enough for mechanism-level results and relative comparisons under stated priors; not for real-world claims about costs, MCP servers, sub-agents or a System One model on text. Shadow OTel traces from the target society are the next input (HANDOFF §9).

## 8. The lesson worth keeping

A controller that "beats the baseline on every metric" is not a result until the baseline's knobs have been swept. The oracle showed headroom; the ablation showed most of it was a 300-second default. Ask "weak baseline?" of every knob before "does prediction pay?" — that question, asked one rung late, is what this project's evidence chain is built to force.
