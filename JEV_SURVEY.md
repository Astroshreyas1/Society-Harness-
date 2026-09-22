# System One models (TypeSafe's Jev) — survey, and how this controller uses them

*Written 2026-09-22. What is publicly documented about Jev, what it is good and bad at, the patterns its authors recommend, how every one of those patterns is used in this repo (`agentsim/jev.py`, `reserver.py`, `train.py`), what changes the day the vendor API is available, and where the integration goes next. Sources are listed at the end; quotations are from them.*

---

## 1. What a System One model is

TypeSafe describes System One models as "a new class of frontier models built to make fast, structured decisions that software can use directly", and Jev as "a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out". Three properties define the class, and each one matters for a resource controller:

| Property | What the vendor states | Why it matters here |
|---|---|---|
| **Typed, schema-guaranteed output** | "gives up string generation"; valid outputs are defined in advance, so it "cannot hallucinate or produce type errors" | A controller consumes numbers and enums, never text; a typed answer can be wired straight into an admission rule |
| **Calibrated probabilities** | trained with "Reinforcement Learning for Calibrated Decisions (RLCD)", which optimises "answers with epistemically honest probabilities": across many predictions, answers given 90 % probability should be right about 90 % of the time; "higher confidence really does mean higher accuracy" | Reservations are on quantiles (H2/H3 of the design); a probability the operator can trust is the artefact a lease consumes |
| **Non-autoregressive, parallel** | "Generates all outputs in a single query"; "System One models evaluate every question in a request in parallel. Adding questions barely changes the response time" | One call per step boundary can answer everything the Reserver might want, speculatively |

Reported operating point: "end-to-end response time is 70 ms–500 ms, most around 100"; "$0.042 per million input tokens", output free; "40×–200× faster" than frontier LLMs on comparable classification tasks. Stated limits: cardinality up to 255 per choice; 2–10 levels per score; ~64 k tokens per request (state plus all questions), ~32 k for the longest single question; 1,200 requests/min and 250 k tokens/s during early access (429 with `retry-after`); text input only.

## 2. The API, as documented

`POST https://api.typesafe.ai/v1/systemone`, bearer auth, JSON:

```json
{ "model": "jev-latest",
  "state": { "...": "any text / object / array the questions refer to" },
  "questions": {
    "department":      { "type": "choice", "instructions": "...", "criteria": { "billing": "...", "technical": "..." } },
    "frustration":     { "type": "score",  "instructions": "...", "criteria": ["calm", "annoyed", "angry"] },
    "refund_requested":{ "type": "noul",   "instructions": "..." } } }
```

Answers come back under the same keys: a **choice** returns `choice`, `probabilities` (one per option, summing to 1) and `confidence`; a **score** returns `score` ("the probability-weighted mean of the level numbers", so it can fall between levels), `probabilities` per level and `confidence`; a **noul** returns `noul`, the probability that the statement is true. The response carries `model` and `usage.input_tokens`. SDKs (`typesafe-sdk` for Python, `@typesafe-ai/sdk`) "retry with exponential backoff and honour `retry-after`". LiteLLM exposes the API as a pass-through (`<proxy>/typesafe/v1/systemone`) with spend tracking from `usage` — relevant because the Gate of this design lives in exactly that seat.

**Confidence versus probability.** "The answer tells you what; confidence tells you whether to act." Confidence is a second axis about the distribution itself (a score of 1.3 with confidence 0.9 is a firm "between levels 1 and 2"; the same score with confidence 0.4 is "unsure"). The recommended use is confidence-gated routing: below a threshold, do not act on the answer (route to a human, gather more, or fall back).

## 3. The patterns the authors recommend

1. **Speculative fan-out.** "Ask every independent question in one call, even those that only matter for some inputs"; a test of 13 questions in one call was "12.2× cheaper and 10× faster" than 13 sequential calls. "One call feeds the whole decision tree."
2. **Confidence-gated action.** Thresholds per action set by the cost of a wrong action.
3. **Retrieve, then judge.** "Send only the fields the question needs"; accuracy "falls as the state fills with unrelated content."
4. **Composite scoring.** "Break a complex judgment into atomic scores, combine with weights you control in code" — re-weighting needs no re-prompting.
5. **Verification.** "Score, judge, verify, guardrail, and detect jailbreaks": the LLM writes, Jev checks (a citation supports its claim; a tool call is risky; a request is done).
6. **Map-reduce.** "Turn petabytes of data into features and insights" — labelling whole corpora offline at $0.04 per million tokens.
7. **Instruction hygiene.** "Write the exact condition — the model reads your words, not your intent"; "one judgment per question"; "describe situations rather than degrees" for score levels; reference state by path in backticks.

**Documented jagged edges** (the vendor's own list for jev-1.13): reads literally (negations and implied conditions "land at face value"); "unreliable at counting"; dates as text are unreliable ("use Choice over enumerated options"); indirection (properties of properties); degrades with irrelevant context; "does not extract reliably — use regex or a generative model for candidates, then Jev picks"; cannot generate text.

## 4. How this controller uses each pattern

The integration is `agentsim/jev.py` (catalogue, records, local and remote models, judge, channel), consumed by `reserver.py` and trained/evaluated by `train.py`.

| Pattern | Where | What it does here |
|---|---|---|
| Typed catalogue | `catalogue()` | 18 questions over the three primitives, each with instructions and criteria written to the rules of §3.7, tagged with the decision points that ask them and the role they serve (feature / judge / verify / annotate). Cardinalities and level counts are validated against the API's limits |
| Speculative fan-out | `questions_at(cat, dp)`; `NeedsController._ask_jev` | At every boundary the controller asks every question that boundary can use, in one call: 9 at `chat_end`, 6 at `tool_ready`, 5 at `chat_ready`, 2 at `request_end`, 3 at `spawn` |
| Retrieve, then judge | `features.state_of` | The state sent is the session's observable view — recipe, counts, node history, the response's plan cue, the submitted content, the revealed tool calls, occupancy — nothing else; a `candidate` field describes the action being judged |
| Confidence as a second axis | `JevRecord.ok(q)` (`CONFIDENCE_MIN` = 0.55); `features.featurize` | Below the threshold an answer is only a feature (with its probability bucket and age); above it, judge questions may drive decisions. Noul answers carry no confidence: their probability is the axis |
| Composite scoring in code | `reserver.idle_timeout`, `join_timeout`, `leases` | `P(safe_to_park)` multiplies the cold-start cost in the park decision; `P(will_use_reservation)` is blended 50/50 with the learned pay rate in the gang gate; the Reserver's own forecast prices the other side |
| Verification / judging | roles `judge` and `verify` | `will_use_reservation`, `safe_to_park`, `is_final_turn`, `join_soon` judge the Reserver's candidates; `tool_will_error` and `reaction` judge calls before they run (labels exist on real traces: TraceLab's `is_error`) |
| Map-reduce | `agentsim annotate`, `JevChannel.flush` / `RemoteSystemOne.decide_batch` | Offline: label every chat of a trace with the typed answers (`jev_phase` feeds `rung0/e2_predictability.py --phase-attr`). Online: with `batch_window_s` > 0 decision points inside a window share one request whose state is `{"sessions": [...]}` and whose questions are prefixed per session |
| Calibration, measured | `LocalSystemOne.fit` (held-out sessions), `evaluate_system_one` (any model, incl. the vendor's), `Judge` (live) | ECE, accuracy vs majority, Brier per question; the design's kill rule (ECE > 0.1 → drop the question) runs **online, per question**: a drifted question stops being asked and stops driving decisions until its window recovers |
| Rate limits, latency, cost | `JevChannel` | `ext.jev` is a tool-tier resource: concurrency, an RPM bucket with background load, lognormal latency (median 150 ms, p99 500 ms by default), token usage priced at the vendor's rate; over budget the call is dropped, never queued — the on-path model works from the previous record and its age |
| The vendor seam | `RemoteSystemOne` | Exact request/response shape of §2; `JEV_API_KEY`, `JEV_API_BASE` (direct or a LiteLLM `/typesafe` pass-through), `JEV_MODEL`; retries on 429/5xx with backoff honouring `retry-after`; usage and cost accounting; `evaluate-jev --remote` scores it on labelled traces before any answer is trusted |

The local model (`LocalSystemOne`, one sparse multinomial per question over hashed n-grams of the state, log-loss trained, temperature-calibrated) is **not** a System One model; it reproduces the *contract* so that the integration could be built and measured before vendor access. On the synthetic hosted society at `content_snr` 0.6 it reaches ECE ≤ 0.04 on every question with signal (phase 0.75 vs 0.19 majority; duration class 0.86 vs 0.69; output class 0.71 vs 0.36; next chat 0.79 vs 0.70; tier of the next step 0.93 vs 0.76) and says so where there is none (idle class: temperature ≈ 4, flat). The vendor model reads the actual text with frontier understanding; the local one reads hashed tokens. Expect the vendor model to be *better on real content* and *no better on the synthetic cues*, which are already read losslessly by the local model.

## 5. What changes the day the API is available

1. `policy.needs.jev.remote = true` and the three environment variables; nothing else in the controller changes.
2. Run `agentsim evaluate-jev --remote --traces <labelled traces>` first (D2 in ISSUES.md): accuracy / ECE per question on held-out sessions, at $0.042 per million tokens (a 300-session evaluation costs cents). Questions with ECE > 0.1 stay untrusted; the Judge re-checks live.
3. Re-run the `needs_jev` variant of the grids (`scenarios/variants_needs*.json`) — the outcome gate: ship the Jev features only if the paired, Holm-adjusted differences over `needs_pre` are significant at 10 seeds (NEEDS_REPORT.md).
4. Cost at this society's scale: a decision point sends ~300–600 tokens of state; at 2 sessions/min × ~30 boundaries per session that is ~60 calls/min, ~25 k tokens/min, ~$0.06/hour — and within the 1,200 rpm limit without batching. Batching (`batch_window_s`) is for denser societies or a shared key.
5. Instructions and criteria in the catalogue were written to the vendor's rules but never tested against the vendor model; the first day's work is to read the per-question ECE and rewrite the ones that read badly (typical fixes: split a two-condition instruction; replace a degree word with a situation; add a `not stated` option).

## 6. Where the integration goes next (Phase 7, in progress)

The user's scope: feedback-loop allocation across CPU, GPU, tool calls, MCP servers and APIs (cost, token and rate limits), every one of them predicted, with an error-correction algorithm — and Jev guiding on top. The plan, each item measurable with the machinery that exists:

- **Resource model.** Each tool-tier resource (`ext.*`, and new `mcp.<server>` entries) carries concurrency, RPM, **cost per call, tokens per call**, latency, timeout; per-tenant / per-session **budgets** in dollars and tokens; the model tier is named for what it is (GPU concurrency / memory / throughput) with per-model TPM and price. The engine accounts spend per session and per tenant.
- **Prediction.** New heads / questions for what the resource model needs: `cost_class` and `tokens_class` of a call before it runs (choice), `budget_will_exceed` (noul: the session will exceed its budget before completing), `calls_remaining` (score) — labels from realised spend.
- **Feedback-loop allocation with error correction.** Per resource, the controller compares its predicted usage over the horizon with the usage that materialised and corrects: the conformal level (already: coverage error → quantile level), a **reservation scale** per resource (PI on the reserved-vs-used error, bounded), **budget pacing** (spend-to-plan error → the admission fraction of the provider bucket, AIMD-style), and a **fairness correction** (per-session share error → VTFQ weights). Every error signal and correction is reported in the run summary and ablatable.
- **Jev on top, supervisory.** `priority_class` (interactive / batch), `stuck_in_loop` (the agent repeats a failing call), `allocation_ok` (a judge question on the controller's proposed allocation for a session), `cost_class` — confidence-gated, ECE-judged, consumed by the allocation controller as priors and vetoes, never as the only signal.

## 7. Sources

- TypeSafe AI — [System One (concepts)](https://docs.typesafe.ai/concepts/system-one); [Introducing System One models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev); [LLM-readable docs](https://docs.typesafe.ai/llms.txt); [API reference](https://docs.typesafe.ai/api)
- [How to use Jev: a practical guide](https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e) (request/response JSON, limits, batching figures, pricing)
- [A deep dive into Jev](https://flaviocopes.com/jev/) (score semantics, confidence thresholds, instruction rules, jagged edges, SDKs)
- [Building a harness with Jev — LangChain](https://www.langchain.com/blog/building-a-harness-with-jev) (routing and tool-risk gating inside an agent loop)
- [TypeSafe AI (Jev) pass-through — LiteLLM](https://docs.litellm.ai/docs/pass_through/typesafe)
- Secondary: [DataCamp](https://www.datacamp.com/blog/system-one-models-jev), [MarkTechPost](https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/), [MindStudio](https://www.mindstudio.ai/blog/jev-system-one-model-launch), [The Register](https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711)
