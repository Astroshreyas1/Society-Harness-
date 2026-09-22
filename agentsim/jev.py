"""A System One model for the controller (JEV_SURVEY.md; DEEP_DIVE_AND_LADDER §2; PREDICTOR_DESIGN §2 D): unstructured
program state in, typed probabilistic decisions out — every answer with a calibrated probability, every question of a
request answered in one parallel pass, never on the admission path.

Built to the shape of TypeSafe's Jev API (POST /v1/systemone: {model, state, questions:{name:{type, instructions,
criteria}}} -> {answers, usage, model}) and its three primitives:
  choice   one option of <= 255, each with criteria           -> choice, probabilities, confidence
  score    2–10 ordered levels                                 -> score (probability-weighted level), probabilities, confidence
  noul     a yes/no probability                                -> noul
and its documented patterns: speculative fan-out (ask everything at a boundary in one call), confidence as a second
axis (act only above a threshold), retrieve-then-judge (send only the fields a question needs), composite scores in
code, verification / judging, map-reduce over traces.

Parts:
  Question / catalogue      the typed questions (instructions and criteria written to the API's rules: one judgment per
                            question, situations not degrees, no counting or dates), which decision points ask them, and
                            where their labels come from (realised outcomes; the hidden phase for offline annotation).
  JevRecord                 answers + calibrated pmfs + confidences + issue/ready times; `prob`, `score`, `noul`, `ok` (confident).
  SystemOne                 `decide(state, question_names, now, latency) -> JevRecord`, and `decide_batch` for many states.
    LocalSystemOne          numpy: one sparse multinomial per question (choice / score levels / noul as 2 classes) over hashed
                            n-grams of the state's text and structure, log-loss trained, temperature-calibrated per question on
                            held-out sessions; ECE per question. The contract, reproduced locally so the integration can be
                            built and measured before vendor access.
    RemoteSystemOne         the vendor client: bearer auth, JEV_API_BASE (direct or a LiteLLM /typesafe pass-through), retries
                            with exponential backoff honouring retry-after, timeouts, usage and cost accounting.
  Judge                     online calibration monitor: every answer is scored against the realised label the Observer later
                            emits; ECE / accuracy / Brier per question on a rolling window; `trusted(q)` gates the use of a
                            question (features and decision judging) — a question whose ECE drifts past 0.1 is dropped, as the
                            design's kill rule says, automatically and per question.
  JevChannel                `ext.jev` as a tool-tier resource the controller lives within: concurrency, RPM bucket with
                            background load, lognormal latency, optional batching of same-window decision points into one
                            call (map-reduce over sessions), dropped when over budget; answers arrive asynchronously.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import SessionView, fhash, state_of  # noqa: F401  (state_of is the state builder callers use)
from .resources import CallBucket
from .workload import MAX_SPAWN

DURATION_CLASSES = ("ms", "s", "min", "long")
OUT_CLASSES = ("short", "medium", "long", "xlong")
IDLE_CLASSES = ("<30s", "30-120s", "2-10min", ">10min")
NEXT_CHAT = ("final", "tools", "no_tools", "spawn")
TIERS_NEXT = ("model", "sandbox", "external", "service", "none")
REMAINING_LEVELS = ("done", "one more step", "a few steps", "many steps", "long task")
TAIL_LEVELS = ("instant", "seconds", "a minute", "several minutes", "very long")
RISK_LEVELS = ("none", "low", "medium", "high")
REACTIONS = ("retry", "replan", "abort")
COST_CLASSES = ("free", "cents", "dime", "dollar")            # a step's cost: 0 / < 1 c / < 10 c / more


def cost_class(usd: float) -> str:
    return COST_CLASSES[0] if usd <= 0 else COST_CLASSES[1] if usd < 0.01 else COST_CLASSES[2] if usd < 0.1 else COST_CLASSES[3]
CONFIDENCE_MIN = 0.55            # confidence-gated use: below this an answer is a feature, never a decision
ECE_MAX = 0.10                   # the design's kill rule per question, applied online


def idle_class(gap_s: float) -> str:
    return IDLE_CLASSES[0] if gap_s < 30 else IDLE_CLASSES[1] if gap_s < 120 else IDLE_CLASSES[2] if gap_s < 600 else IDLE_CLASSES[3]


@dataclass(frozen=True)
class Question:
    name: str
    type: str                                   # choice | score | noul
    instructions: str
    options: tuple[str, ...]                    # choice options / score levels / ("no", "yes") for noul
    criteria: object = None                     # dict per option (choice), tuple per level (score), str (noul)
    asked_at: tuple[str, ...] = ()              # decision points
    label: str = ""                             # where the training label comes from (see train.jev_examples)
    role: str = "feature"                       # feature | judge | verify | annotate

    def payload(self) -> dict:
        """The question as the vendor API takes it."""
        if self.type == "choice":
            crit = self.criteria if isinstance(self.criteria, dict) else {o: o for o in self.options}
            return {"type": "choice", "instructions": self.instructions, "criteria": crit}
        if self.type == "score":
            return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria) if self.criteria else list(self.options)}
        return {"type": "noul", "instructions": self.instructions, "criteria": self.criteria if isinstance(self.criteria, str) else self.instructions}


def catalogue(tool_kinds: list[str], phases: list[str]) -> dict[str, Question]:
    """Every question the controller may ask, keyed by name. Instructions follow the API's rules: the exact condition, one
    judgment each, situations rather than degrees, state fields referenced by path, nothing that needs counting or dates."""
    kinds = tuple(sorted(tool_kinds))
    Q = [
        # ---- step-boundary features (role: feature) ----------------------------------------------------------------
        Question("next_chat", "choice",
                 "After the tool calls listed in `revealed` finish, what does the agent's next model turn do?",
                 NEXT_CHAT, {"final": "the turn answers the user and the request ends (no tool calls)",
                             "tools": "the turn issues one or more tool calls", "no_tools": "the turn continues without tool calls and without ending",
                             "spawn": "the turn launches sub-agents"},
                 ("chat_end",), "S2/S3 of the same record", "feature"),
        Question("next_tool", "choice", "The kind of the first tool call the agent's next model turn will issue, or `none`.",
                 ("none",) + kinds, {"none": "the next turn issues no tool call", **{k: f"the first call of the next turn is a `{k}` tool" for k in kinds}},
                 ("chat_end",), "S3", "feature"),
        Question("duration_class", "choice", "How long the tool call described in `content` will run before it returns.",
                 DURATION_CLASSES, {"ms": "returns within one second: reads, small edits, quick searches", "s": "one to ten seconds: short commands, installs of small packages",
                                    "min": "ten seconds to a minute: builds, test subsets, large greps", "long": "over a minute: full test suites, long builds, training runs, downloads"},
                 ("tool_ready",), "duration bin of the longest member", "feature"),
        Question("tail_risk", "score", "How long the tool call in `content` is likely to occupy its sandbox.",
                 TAIL_LEVELS, ("returns immediately", "takes seconds", "about a minute", "several minutes", "very long, may hit the limit"),
                 ("tool_ready",), "duration bin (0..3 -> levels 0..4)", "feature"),
        Question("out_class", "choice", "How long the model's answer to the request in `content` will be.",
                 OUT_CLASSES, {"short": "under a hundred tokens: a confirmation, a short answer, a tool call", "medium": "a few paragraphs or a small edit",
                               "long": "a long explanation or a sizeable code change", "xlong": "a very long answer: whole files, long reports"},
                 ("chat_ready",), "output-token bin", "feature"),
        Question("remaining_work", "score", "How much of the current request remains after this model turn.",
                 REMAINING_LEVELS, ("the request is complete after this turn", "one more turn will finish it", "a few more turns", "many more turns",
                                    "a long task: many turns and long tool calls"),
                 ("chat_ready", "chat_end"), "remaining chats in the request (binned)", "feature"),
        Question("tier_next", "choice", "Which capacity the agent's next step needs.",
                 TIERS_NEXT, {"model": "the next step is a model turn", "sandbox": "the next step runs a local tool in the sandbox",
                              "external": "the next step calls an external API (search, web, api)", "service": "the next step is a retrieval",
                              "none": "the request ends"},
                 ("chat_end", "tool_ready"), "resource of the next node", "feature"),
        Question("idle_class", "choice", "How long the user will take to send the next message after this turn closes.",
                 IDLE_CLASSES, {"<30s": "the user replies at once", "30-120s": "a short pause", "2-10min": "the user reads or tests something first",
                                ">10min": "the user leaves for a while"},
                 ("request_end",), "idle gap bin", "feature"),
        Question("spawn_width", "choice", "How many sub-agents the model's next turn launches (`0` when it launches none).",
                 tuple(str(i) for i in range(MAX_SPAWN + 1)), None, ("chat_ready",), "spawn count", "feature"),
        Question("fail_soon", "noul", "The current request will be abandoned by the client within the next ten minutes.",
                 ("no", "yes"), "the request is abandoned after repeated errors or timeouts within ten minutes",
                 ("chat_ready", "tool_ready", "chat_end"), "R", "feature"),
        Question("risk", "score", "How likely the current request is to fail (be abandoned) rather than complete.",
                 RISK_LEVELS, ("will complete normally", "a small chance of failure", "a real chance of failure", "likely to fail"),
                 ("chat_ready", "chat_end"), "R (binned)", "feature"),
        # ---- decision judging (role: judge) — the Reserver's candidate actions ----------------------------------------
        Question("will_use_reservation", "noul", "The units a reservation would hold for this session (`candidate`) will actually be taken within the horizon.",
                 ("no", "yes"), "the session (or its sub-agents) acquires the reserved resource before the reservation expires",
                 ("chat_end", "chat_ready", "spawn"), "gang / spawn lease consumed", "judge"),
        Question("safe_to_park", "noul", "The session's sandbox will not be needed again within a cold start's time.",
                 ("no", "yes"), "no local tool call from this session starts within the next few seconds",
                 ("request_end", "spawn", "chat_end"), "idle gap >= cold start / no local tool within cold start", "judge"),
        Question("is_final_turn", "noul", "The model turn now at the gate is the last turn of the request.",
                 ("no", "yes"), "the turn answers the user and issues no tool calls", ("chat_ready",), "is_final of the chat", "judge"),
        Question("join_soon", "noul", "All sub-agents of this orchestrator will finish within the horizon.",
                 ("no", "yes"), "every child request completes or is abandoned before the horizon", ("spawn",), "join wait <= h", "judge"),
        # ---- verification (role: verify) — reaction and outcome ---------------------------------------------------------
        Question("tool_will_error", "noul", "The tool call in `content` will return an error rather than a result.",
                 ("no", "yes"), "the call exits with an error, a non-zero status or a not-found", ("tool_ready",), "errors attr (real traces)", "verify"),
        Question("reaction", "choice", "What the agent does after this call fails.",
                 REACTIONS, {"retry": "the same call is issued again", "replan": "the model reconsiders and issues different calls", "abort": "the request is abandoned"},
                 ("tool_ready",), "observed reaction after an error", "verify"),
        # ---- supervisory (role: judge) — the allocation loop's questions (Phase 7) ----------------------------------------
        Question("cost_class", "choice", "What the step now at the gate will cost the tenant (model tokens or a priced tool call).",
                 COST_CLASSES, {"free": "no metered cost: a local tool", "cents": "under one cent: a short model turn or a cheap API call",
                                "dime": "a few cents: a long model turn or a priced MCP call", "dollar": "ten cents or more: a very long context or a costly API"},
                 ("chat_ready", "tool_ready"), "usd of the step (binned)", "judge"),
        Question("budget_will_exceed", "noul", "The tenant budget will refuse a payment for this request before it completes.",
                 ("no", "yes"), "a step of this request is refused by the budget (outcome budget) before the request ends",
                 ("chat_ready", "chat_end"), "BUDGET_FAIL", "judge"),
        Question("stuck_in_loop", "noul", "The agent's next turn will issue the same tool calls as this turn did (it is repeating itself).",
                 ("no", "yes"), "the next turn's tool calls have the same kinds and cues as this turn's", ("chat_end",), "REPEAT", "judge"),
        # ---- offline annotation (role: annotate) ---------------------------------------------------------------------------
        Question("phase", "choice", "The phase of the agent's next model turn, from what it said and did so far.",
                 tuple(phases), {p: f"the next turn belongs to phase `{p}`" for p in phases}, ("chat_end",), "hidden phase of the next chat", "annotate"),
    ]
    out = {q.name: q for q in Q}
    for q in out.values():
        if not 2 <= len(q.options) <= 255:
            raise ValueError(f"question {q.name!r}: cardinality {len(q.options)} outside 2..255")
        if q.type == "score" and not 2 <= len(q.options) <= 10:
            raise ValueError(f"score {q.name!r}: {len(q.options)} levels outside 2..10")
    return out


def question_schema(tool_kinds: list[str], phases: list[str]) -> dict[str, tuple[str, ...]]:
    """Answer sets per question (for callers that only need the cardinalities)."""
    return {q.name: q.options for q in catalogue(tool_kinds, phases).values()}


def questions_at(cat: dict[str, Question], dp: str, roles: tuple[str, ...] = ("feature", "judge", "verify", "annotate")) -> tuple[str, ...]:
    """Speculative fan-out: every question this decision point can ask, in one call."""
    return tuple(q.name for q in cat.values() if dp in q.asked_at and q.role in roles)


@dataclass
class JevRecord:
    dp: str
    t_issued: float
    t_ready: float
    answers: dict[str, tuple[str, float]] = field(default_factory=dict)   # question -> (answer, probability of it)
    dists: dict[str, np.ndarray] = field(default_factory=dict)            # question -> calibrated pmf over the options / levels / (no, yes)
    confidence: dict[str, float] = field(default_factory=dict)            # choice / score: certainty about the distribution itself
    schema: dict[str, tuple[str, ...]] = field(default_factory=dict)
    usage_tokens: int = 0

    def prob(self, q: str, ans: str) -> float:
        d = self.dists.get(q)
        if d is None or q not in self.schema or ans not in self.schema[q]:
            return 0.0
        return float(d[self.schema[q].index(ans)])

    def noul(self, q: str) -> float | None:
        """P(yes) of a noul question; None if not answered."""
        d = self.dists.get(q)
        return None if d is None else float(d[-1])

    def score(self, q: str) -> float | None:
        """The probability-weighted level of a score question (0 .. levels-1); None if not answered."""
        d = self.dists.get(q)
        return None if d is None else float(np.dot(d, np.arange(len(d))))

    def ok(self, q: str, min_confidence: float = CONFIDENCE_MIN) -> bool:
        """Confidence-gated: may this answer drive a decision? (noul answers carry no confidence: their probability is the axis.)"""
        return q in self.answers and self.confidence.get(q, 1.0) >= min_confidence

    @property
    def confident(self) -> dict[str, bool]:
        return {q: self.ok(q) for q in self.answers}


def confidence_of(p: np.ndarray) -> float:
    """Certainty about a distribution: 1 - normalised entropy (1 = a point mass, 0 = uniform)."""
    k = len(p)
    if k < 2:
        return 1.0
    h = -float(np.sum(p * np.log(np.maximum(p, 1e-12))))
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def tokens_of_state(st: dict) -> list[int]:
    """Hashed unigrams / bigrams over the state's text fields plus its structural tokens (the local model's reading)."""
    toks = [f"dp:{st['dp']}", f"recipe:{st['recipe']}", f"child:{int(st['child'])}", f"chats:{min(st['chats'], 12)}",
            f"tools:{min(st['tools'], 20)}", f"req:{min(st['request_idx'], 10)}", f"last:{st['last_tool']}", f"jw:{st['join_width']}",
            f"tin:{int(math.log1p(st['tokens_in']))}"]
    h = st["history"]
    toks += [f"h{i}:{t}" for i, t in enumerate(reversed(h[-6:]))]
    toks += [f"bi:{a}>{b}" for a, b in zip(h, h[1:])]
    text = " ".join([st["plan"]] + st["content"] + st["revealed_content"])
    words = text.split()
    toks += [f"w:{w}" for w in words] + [f"ww:{a} {b}" for a, b in zip(words, words[1:])]
    toks += [f"rev:{k}" for k in st["revealed"][:6]]
    toks += [f"occ:{k}:{int(min(10, v * 10))}" for k, v in st["occupancy"].items()]
    cand = st.get("candidate")
    if cand:
        toks += [f"cand:{k}={v}" for k, v in cand.items()]
    return sorted({fhash(t) for t in toks})


def answers_from_labels(dp: str, labels: dict, cold_start_s: float, horizon_s: float) -> dict[str, str]:
    """The typed answers the realised outcomes imply for a decision point — one mapping for offline training
    (train.jev_examples) and for the online Judge, so both score the same thing."""
    from .workload import duration_bin, out_bin
    ans: dict[str, str] = {}
    if dp == "tool_ready" and "Q_tool_max" in labels:
        b = duration_bin(math.exp(labels["Q_tool_max"]))
        ans["duration_class"] = DURATION_CLASSES[b]
        ans["tail_risk"] = TAIL_LEVELS[(0, 1, 2, 4)[b]]
    if dp == "chat_ready":
        if "Q_out" in labels:
            ans["out_class"] = OUT_CLASSES[out_bin(int(round(math.exp(labels["Q_out"]) - 1)))]
        if "G" in labels:
            ans["spawn_width"] = str(int(labels["G"]))
        if "FINAL" in labels:
            ans["is_final_turn"] = "yes" if labels["FINAL"] > 0.5 else "no"
    if dp == "chat_end" and "S2" in labels:
        s2, s3 = labels["S2"], labels["S3"]
        ans["next_chat"] = "final" if s2 == "final" else "spawn" if s2 == "spawn" else ("tools" if s3.startswith("tool:") else "no_tools")
        ans["next_tool"] = s3[5:] if s3.startswith("tool:") else "none"
    if dp in ("chat_end", "tool_ready") and "NEXT_RES" in labels:
        r = labels["NEXT_RES"]
        ans["tier_next"] = "none" if r == "none" else "model" if r == "model" else "service" if r.startswith("svc") else "external" if r.startswith("ext") else "sandbox"
    if dp == "chat_end" and "PHASE" in labels:
        ans["phase"] = labels["PHASE"]
    if dp in ("chat_ready", "chat_end") and "REMAIN" in labels:
        n = int(labels["REMAIN"])
        ans["remaining_work"] = REMAINING_LEVELS[0 if n == 0 else 1 if n == 1 else 2 if n <= 4 else 3 if n <= 10 else 4]
    if dp == "request_end" and "I_gap" in labels:
        gap = math.exp(labels["I_gap"])
        ans["idle_class"] = idle_class(gap)
        ans["safe_to_park"] = "yes" if gap >= cold_start_s else "no"
    if dp == "spawn" and "JOIN" in labels:
        ans["join_soon"] = "yes" if labels["JOIN"] <= horizon_s else "no"
    if "R" in labels and dp in ("chat_ready", "tool_ready", "chat_end"):
        ans["fail_soon"] = "yes" if labels["R"] > 0.5 else "no"
        ans["risk"] = RISK_LEVELS[3] if labels["R"] > 0.5 else RISK_LEVELS[0]
    if dp == "tool_ready" and "E_err" in labels:
        ans["tool_will_error"] = "yes" if labels["E_err"] > 0 else "no"
    if "PAID" in labels:
        ans["will_use_reservation"] = "yes" if labels["PAID"] > 0.5 else "no"
    if dp in ("chat_ready", "tool_ready") and "USD" in labels:
        ans["cost_class"] = cost_class(float(labels["USD"]))
    if dp in ("chat_ready", "chat_end") and "BUDGET_FAIL" in labels:
        ans["budget_will_exceed"] = "yes" if labels["BUDGET_FAIL"] > 0.5 else "no"
    if dp == "chat_end" and "REPEAT" in labels:
        ans["stuck_in_loop"] = "yes" if labels["REPEAT"] > 0.5 else "no"
    return ans


def evaluate_system_one(model: "SystemOne", examples: list[tuple[dict, dict[str, str]]], max_n: int | None = None) -> dict:
    """Accuracy / ECE / Brier per question of any SystemOne (local or remote) on labelled (state, answers) examples,
    through `decide` — the D2 validation, usable on the vendor model."""
    if max_n is not None:
        examples = examples[:max_n]
    per_q: dict[str, list[tuple[np.ndarray, int]]] = {}
    usage = 0
    for st, ans in examples:
        qs = tuple(q for q in ans if q in model.schema)
        if not qs:
            continue
        rec = model.decide(st, qs, 0.0)
        usage += rec.usage_tokens
        for q in qs:
            per_q.setdefault(q, []).append((rec.dists[q], model.schema[q].index(ans[q])))
    out = {"n_examples": len(examples), "input_tokens": usage}
    for q, rows in per_q.items():
        if len(rows) < 20:
            continue
        P = np.stack([p for p, _ in rows])
        Y = np.array([y for _, y in rows])
        conf, hit = P.max(axis=1), (P.argmax(axis=1) == Y).astype(float)
        e = 0.0
        for i in range(10):
            m = (conf > i / 10) & (conf <= (i + 1) / 10)
            if m.any():
                e += m.mean() * abs(hit[m].mean() - conf[m].mean())
        onehot = np.zeros_like(P)
        onehot[np.arange(len(Y)), Y] = 1.0
        out[q] = {"n": int(len(Y)), "accuracy": round(float(hit.mean()), 4), "majority": round(float(np.bincount(Y, minlength=P.shape[1]).max() / len(Y)), 4),
                  "ece": round(float(e), 4), "brier": round(float(np.mean(np.sum((P - onehot) ** 2, axis=1))), 4), "trusted": e <= ECE_MAX}
    return out


class SystemOne:
    name = "system_one"

    def __init__(self, cat: dict[str, Question]):
        self.cat = cat
        self.schema = {q.name: q.options for q in cat.values()}

    def decide(self, state: dict, questions: tuple[str, ...], now: float, latency: float = 0.0) -> JevRecord:
        raise NotImplementedError

    def decide_batch(self, states: list[dict], questions: tuple[str, ...], now: float, latency: float = 0.0) -> list[JevRecord]:
        """Many states in one request (map-reduce over sessions). Default: one decision per state."""
        return [self.decide(st, questions, now, latency) for st in states]


class LocalSystemOne(SystemOne):
    """One sparse multinomial per question (hashed features, Adagrad), temperature-calibrated per question. Score questions
    are multinomials over their levels (score = expected level); noul questions are two-class."""

    N = 2 ** 16

    def __init__(self, cat: dict[str, Question], seed: int = 0, lr: float = 0.3):
        super().__init__(cat)
        self.W = {q: np.zeros((self.N, len(a)), dtype=np.float32) for q, a in self.schema.items()}
        self.b = {q: np.zeros(len(a)) for q, a in self.schema.items()}
        self.G = {q: np.full(self.N, 1e-2, dtype=np.float32) for q in self.schema}
        self.T = {q: 1.0 for q in self.schema}
        self.ece = {q: float("nan") for q in self.schema}
        self.n_train = {q: 0 for q in self.schema}
        self.lr = lr
        self.rng = np.random.default_rng(seed)

    def _logits(self, q: str, ids: list[int]) -> np.ndarray:
        return self.W[q][ids].sum(axis=0).astype(np.float64) + self.b[q]

    @staticmethod
    def _softmax(o: np.ndarray) -> np.ndarray:
        o = o - o.max()
        p = np.exp(o)
        return p / p.sum()

    def _step(self, q: str, ids: list[int], y: int) -> float:
        p = self._softmax(self._logits(q, ids))
        loss = -math.log(max(p[y], 1e-12))
        p[y] -= 1.0
        g = p.astype(np.float32)
        rows = np.array(ids)
        self.G[q][rows] += float(g @ g)
        self.W[q][rows] -= (self.lr / np.sqrt(self.G[q][rows]))[:, None] * g[None, :]
        self.b[q] -= 0.1 * self.lr * p
        return loss

    def fit(self, examples: list[tuple[dict, dict[str, str]]], epochs: int = 3, holdout: float = 0.2, seed: int = 0,
            groups: list[str] | None = None) -> dict:
        """Train on (state, {question: answer}) pairs; calibrate temperatures on a held-out share (whole groups, e.g.
        sessions, when `groups` is given); return ECE / accuracy per question on that share."""
        if not examples:
            raise ValueError("no examples")
        rng = np.random.default_rng(seed)
        if groups is not None:
            if len(groups) != len(examples):
                raise ValueError("groups must align with examples")
            gs = sorted(set(groups))
            hold_g = set(rng.choice(gs, size=max(1, int(len(gs) * holdout)), replace=False).tolist())
            hold = np.array([i for i, g in enumerate(groups) if g in hold_g])
            train = np.array([i for i, g in enumerate(groups) if g not in hold_g])
        else:
            idx = np.arange(len(examples))
            rng.shuffle(idx)
            n_hold = int(len(idx) * holdout)
            hold, train = idx[:n_hold], idx[n_hold:]
        feats = [tokens_of_state(st) for st, _ in examples]
        for ep in range(epochs):
            rng.shuffle(train)
            for i in train:
                st, ans = examples[i]
                for q, a in ans.items():
                    if q in self.W:
                        self._step(q, feats[i], self.schema[q].index(a))
                        self.n_train[q] += 1
        return self.evaluate(examples, hold, feats, fit_temperature=True)

    def evaluate(self, examples: list[tuple[dict, dict[str, str]]], idx=None, feats: list[list[int]] | None = None,
                 fit_temperature: bool = False) -> dict:
        """Accuracy / ECE / NLL per question on `examples[idx]` (all when idx is None); optionally refit the temperatures."""
        idx = np.arange(len(examples)) if idx is None else idx
        feats = feats if feats is not None else [tokens_of_state(st) for st, _ in examples]
        report = {}
        for q in self.schema:
            L, Y = [], []
            for i in idx:
                a = examples[i][1].get(q)
                if a is not None:
                    L.append(self._logits(q, feats[i]))
                    Y.append(self.schema[q].index(a))
            if len(Y) < 20:
                continue
            L, Y = np.stack(L), np.array(Y)
            if fit_temperature:
                self.T[q] = self._fit_temperature(L, Y)
            P = self._probs_matrix(L, self.T[q])
            self.ece[q] = self._ece(P, Y)
            majority = float(np.bincount(Y, minlength=len(self.schema[q])).max() / len(Y))
            report[q] = {"type": self.cat[q].type, "n_hold": int(len(Y)), "accuracy": round(float((P.argmax(axis=1) == Y).mean()), 4),
                         "majority": round(majority, 4), "ece": round(self.ece[q], 4), "temperature": round(self.T[q], 3),
                         "nll": round(self._nll(L, Y, self.T[q]), 4)}
            if self.cat[q].type == "score":
                report[q]["mae_levels"] = round(float(np.mean(np.abs(P @ np.arange(P.shape[1]) - Y))), 3)
        return report

    @staticmethod
    def _probs_matrix(L: np.ndarray, T: float) -> np.ndarray:
        o = L / T
        o = o - o.max(axis=1, keepdims=True)
        P = np.exp(o)
        return P / P.sum(axis=1, keepdims=True)

    @classmethod
    def _nll(cls, L: np.ndarray, Y: np.ndarray, T: float) -> float:
        P = cls._probs_matrix(L, T)
        return float(-np.mean(np.log(np.maximum(P[np.arange(len(Y)), Y], 1e-12))))

    @classmethod
    def _fit_temperature(cls, L: np.ndarray, Y: np.ndarray) -> float:
        a, b = math.log(0.2), math.log(10.0)
        phi = (math.sqrt(5) - 1) / 2
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = cls._nll(L, Y, math.exp(c)), cls._nll(L, Y, math.exp(d))
        for _ in range(25):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = cls._nll(L, Y, math.exp(c))
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = cls._nll(L, Y, math.exp(d))
        return math.exp((a + b) / 2)

    @staticmethod
    def _ece(P: np.ndarray, Y: np.ndarray, bins: int = 10) -> float:
        conf, hit = P.max(axis=1), (P.argmax(axis=1) == Y).astype(float)
        e = 0.0
        for i in range(bins):
            m = (conf > i / bins) & (conf <= (i + 1) / bins)
            if m.any():
                e += m.mean() * abs(hit[m].mean() - conf[m].mean())
        return float(e)

    def decide(self, state: dict, questions: tuple[str, ...], now: float, latency: float = 0.0) -> JevRecord:
        ids = tokens_of_state(state)
        rec = JevRecord(dp=state["dp"], t_issued=now, t_ready=now + latency, schema=self.schema, usage_tokens=len(ids) + 12 * len(questions))
        for q in questions:
            if q not in self.W:
                raise KeyError(f"unknown question {q!r}")
            p = self._softmax(self._logits(q, ids) / self.T[q])
            i = int(p.argmax())
            rec.dists[q] = p
            rec.answers[q] = (self.schema[q][i], float(p[i]))
            if self.cat[q].type != "noul":
                rec.confidence[q] = confidence_of(p)
        return rec

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for q in self.schema:
            arrays[f"W:{q}"] = self.W[q]
            arrays[f"b:{q}"] = self.b[q]
            arrays[f"G:{q}"] = self.G[q]
        meta = {"schema": {q: list(a) for q, a in self.schema.items()}, "T": self.T, "ece": self.ece, "n_train": self.n_train,
                "catalogue": {q.name: {"type": q.type, "instructions": q.instructions, "options": list(q.options), "asked_at": list(q.asked_at),
                                       "role": q.role, "label": q.label, "criteria": list(q.criteria) if isinstance(q.criteria, tuple) else q.criteria}
                              for q in self.cat.values()}}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: Path) -> "LocalSystemOne":
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        cat = {}
        for name, d in meta["catalogue"].items():
            crit = d["criteria"]
            cat[name] = Question(name, d["type"], d["instructions"], tuple(d["options"]), tuple(crit) if isinstance(crit, list) else crit,
                                 tuple(d["asked_at"]), d["label"], d["role"])
        m = cls(cat)
        for q in m.schema:
            m.W[q][...] = z[f"W:{q}"]
            m.b[q][...] = z[f"b:{q}"]
            m.G[q][...] = z[f"G:{q}"]
        m.T, m.ece, m.n_train = meta["T"], meta["ece"], meta["n_train"]
        return m


class RemoteSystemOne(SystemOne):
    """The vendor client (TypeSafe System One API). Environment: JEV_API_KEY (required), JEV_API_BASE (default
    https://api.typesafe.ai; a LiteLLM proxy's `<base>/typesafe` works unchanged), JEV_MODEL (default jev-latest).
    Retries 429 / 5xx with exponential backoff honouring `retry-after`; accounts usage and cost ($0.042 per million input
    tokens, output free). Not exercised by the simulator; `evaluate-jev --remote` scores it on labelled examples."""

    PRICE_PER_MTOK = 0.042

    def __init__(self, cat: dict[str, Question], max_retries: int = 4, timeout_s: float = 5.0):
        super().__init__(cat)
        self.key = os.environ.get("JEV_API_KEY", "")
        self.base = os.environ.get("JEV_API_BASE", "https://api.typesafe.ai").rstrip("/")
        self.model = os.environ.get("JEV_MODEL", "jev-latest")
        if not self.key:
            raise RuntimeError("RemoteSystemOne needs JEV_API_KEY in the environment (and optionally JEV_API_BASE, JEV_MODEL)")
        self.max_retries, self.timeout_s = max_retries, timeout_s
        self.calls, self.retries, self.input_tokens, self.latency_s = 0, 0, 0, []

    def _post(self, body: dict) -> dict:
        import urllib.error
        import urllib.request
        data = json.dumps(body).encode("utf-8")
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(f"{self.base}/v1/systemone", data=data,
                                         headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                self.latency_s.append(time.perf_counter() - t0)
                self.calls += 1
                return out
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    ra = e.headers.get("retry-after")
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                    self.retries += 1
                    time.sleep(wait)
                    delay = min(8.0, delay * 2)
                    continue
                raise RuntimeError(f"System One API error {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
        raise RuntimeError("unreachable")

    def _parse(self, ans: dict, q: str) -> tuple[np.ndarray, tuple[str, float], float | None]:
        opts = self.schema[q]
        qt = self.cat[q].type
        if qt == "noul":
            p_yes = float(ans["noul"])
            dist = np.array([1.0 - p_yes, p_yes])
            return dist, (("yes" if p_yes >= 0.5 else "no"), max(p_yes, 1 - p_yes)), None
        if qt == "choice":
            probs = ans["probabilities"]
            dist = np.array([float(probs.get(o, 0.0)) for o in opts], dtype=float)
            if dist.sum() <= 0:
                dist[opts.index(ans["choice"])] = 1.0
            dist = dist / dist.sum()
            return dist, (ans["choice"], float(dist[opts.index(ans["choice"])])), float(ans.get("confidence", confidence_of(dist)))
        probs = np.array(ans["probabilities"], dtype=float)                 # score: one probability per level, in order
        dist = probs / probs.sum() if probs.sum() > 0 else np.full(len(opts), 1.0 / len(opts))
        i = int(dist.argmax())
        return dist, (opts[i], float(dist[i])), float(ans.get("confidence", confidence_of(dist)))

    def decide(self, state: dict, questions: tuple[str, ...], now: float, latency: float = 0.0) -> JevRecord:
        out = self._post({"model": self.model, "state": state, "questions": {q: self.cat[q].payload() for q in questions}})
        rec = JevRecord(dp=state["dp"], t_issued=now, t_ready=now + latency, schema=self.schema,
                        usage_tokens=int(out.get("usage", {}).get("input_tokens", 0)))
        self.input_tokens += rec.usage_tokens
        for q in questions:
            dist, ans, conf = self._parse(out["answers"][q], q)
            rec.dists[q], rec.answers[q] = dist, ans
            if conf is not None:
                rec.confidence[q] = conf
        return rec

    def decide_batch(self, states: list[dict], questions: tuple[str, ...], now: float, latency: float = 0.0) -> list[JevRecord]:
        """Map-reduce over sessions: one request whose state is the list of session states and whose questions are the
        catalogue's questions per session (`s<i>.<name>`, instructions pointing at `sessions[i]`)."""
        if len(states) == 1:
            return [self.decide(states[0], questions, now, latency)]
        qs = {}
        for i in range(len(states)):
            for q in questions:
                pay = self.cat[q].payload()
                pay["instructions"] = f"For `sessions[{i}]`: " + pay["instructions"]
                qs[f"s{i}.{q}"] = pay
        out = self._post({"model": self.model, "state": {"sessions": states}, "questions": qs})
        used = int(out.get("usage", {}).get("input_tokens", 0))
        self.input_tokens += used
        recs = []
        for i, st in enumerate(states):
            rec = JevRecord(dp=st["dp"], t_issued=now, t_ready=now + latency, schema=self.schema, usage_tokens=used // len(states))
            for q in questions:
                dist, ans, conf = self._parse(out["answers"][f"s{i}.{q}"], q)
                rec.dists[q], rec.answers[q] = dist, ans
                if conf is not None:
                    rec.confidence[q] = conf
            recs.append(rec)
        return recs

    def report(self) -> dict:
        return {"calls": self.calls, "retries": self.retries, "input_tokens": self.input_tokens,
                "cost_usd": round(self.input_tokens / 1e6 * self.PRICE_PER_MTOK, 6),
                "latency_p50_s": round(float(np.median(self.latency_s)), 4) if self.latency_s else None}


class Judge:
    """Online calibration of a System One model against realised outcomes (the Observer's labels): rolling ECE, accuracy
    and Brier per question; `trusted(q)` implements the design's kill rule per question, live."""

    def __init__(self, cat: dict[str, Question], window: int = 400, min_n: int = 60):
        self.cat = cat
        self.buf: dict[str, deque] = {q: deque(maxlen=window) for q in cat}
        self.min_n = min_n

    def observe(self, rec: JevRecord, q: str, label: str) -> None:
        d = rec.dists.get(q)
        if d is None or q not in rec.schema or label not in rec.schema[q]:
            return
        self.buf[q].append((np.asarray(d, dtype=float), rec.schema[q].index(label)))

    def stats(self, q: str) -> dict | None:
        b = self.buf.get(q)
        if not b or len(b) < self.min_n:
            return None
        P = np.stack([p for p, _ in b])
        Y = np.array([y for _, y in b])
        conf, hit = P.max(axis=1), (P.argmax(axis=1) == Y).astype(float)
        e = 0.0
        for i in range(10):
            m = (conf > i / 10) & (conf <= (i + 1) / 10)
            if m.any():
                e += m.mean() * abs(hit[m].mean() - conf[m].mean())
        onehot = np.zeros_like(P)
        onehot[np.arange(len(Y)), Y] = 1.0
        return {"n": int(len(Y)), "ece": round(float(e), 4), "accuracy": round(float(hit.mean()), 4),
                "majority": round(float(np.bincount(Y, minlength=P.shape[1]).max() / len(Y)), 4),
                "brier": round(float(np.mean(np.sum((P - onehot) ** 2, axis=1))), 4)}

    def trusted(self, q: str) -> bool:
        """Unknown until min_n labels; then trusted while ECE <= ECE_MAX."""
        s = self.stats(q)
        return True if s is None else s["ece"] <= ECE_MAX

    def report(self) -> dict:
        out = {}
        for q in self.cat:
            s = self.stats(q)
            if s is not None:
                out[q] = dict(s, trusted=self.trusted(q))
        return out


class JevChannel:
    """`ext.jev`: the System One model as a rate-limited, latency-bearing tool the controller must live within. With
    `batch_window_s` > 0, decision points inside a window share one call (the map-reduce pattern); every record still
    lands at the call's ready time."""

    def __init__(self, model: SystemOne, spec: dict, rng: np.random.Generator):
        self.model = model
        self.capacity, self.rpm = int(spec["concurrency"]), float(spec["rpm"])
        self.background = float(spec["background_load"])
        med, p99 = float(spec["latency_median_s"]), float(spec["latency_p99_s"])
        if not 0 < med < p99:
            raise ValueError("jev latency: need 0 < median < p99")
        self.mu, self.sigma = math.log(med), math.log(p99 / med) / 2.3263478740408408
        self.batch_window = float(spec.get("batch_window_s", 0.0))
        self.bucket = CallBucket(self.rpm)
        self.rng = rng
        self.inflight: list[float] = []            # ready times of calls in flight
        self.calls, self.dropped, self.lat_sum, self.tokens, self.records = 0, 0, 0.0, 0, 0
        self.by_dp: dict[str, int] = {}
        self.pending: list[tuple[str, dict, tuple[str, ...]]] = []     # (sid, state, questions) waiting for the batch flush
        self.pending_since = -1.0

    def _admit(self, now: float) -> bool:
        self.inflight = [t for t in self.inflight if t > now]
        cost = 1.0 + (1.0 if self.rng.random() < self.background else 0.0)
        if len(self.inflight) >= self.capacity or not self.bucket.can_call(now, cost):
            self.dropped += 1
            return False
        self.bucket.take(now, cost)
        return True

    def ask(self, now: float, state: dict, questions: tuple[str, ...], sid: str) -> JevRecord | None:
        """One decision point: a call now — or, with batching, queued until the window's flush (see `flush`)."""
        if self.batch_window > 0:
            self.pending.append((sid, state, questions))
            if self.pending_since < 0:
                self.pending_since = now
            return None
        if not self._admit(now):
            return None
        latency = float(min(5.0, self.rng.lognormal(self.mu, self.sigma)))
        rec = self.model.decide(state, questions, now, latency)
        self._account(rec, latency, state["dp"])
        self.calls += 1
        return rec

    def flush_at(self) -> float | None:
        return None if self.pending_since < 0 else self.pending_since + self.batch_window

    def flush(self, now: float) -> list[tuple[str, JevRecord]]:
        """The batch: one call for every pending decision point (questions = the union; each record keeps its own)."""
        if not self.pending:
            return []
        pend, self.pending, self.pending_since = self.pending, [], -1.0
        if not self._admit(now):
            return []
        latency = float(min(5.0, self.rng.lognormal(self.mu, self.sigma))) + 0.01 * len(pend)   # a bigger request takes a little longer
        union = tuple(sorted({q for _, _, qs in pend for q in qs}))
        recs = self.model.decide_batch([st for _, st, _ in pend], union, now, latency)
        out = []
        for (sid, st, qs), rec in zip(pend, recs):
            keep = set(qs)
            rec.answers = {q: a for q, a in rec.answers.items() if q in keep}
            rec.dists = {q: d for q, d in rec.dists.items() if q in keep}
            rec.confidence = {q: c for q, c in rec.confidence.items() if q in keep}
            self._account(rec, latency, st["dp"])
            out.append((sid, rec))
        self.calls += 1
        return out

    def _account(self, rec: JevRecord, latency: float, dp: str) -> None:
        self.inflight.append(rec.t_ready)
        self.records += 1
        self.lat_sum += latency
        self.tokens += rec.usage_tokens
        self.by_dp[dp] = self.by_dp.get(dp, 0) + 1

    def report(self) -> dict:
        rep = {"calls": self.calls, "records": self.records, "dropped": self.dropped,
               "mean_latency_s": round(self.lat_sum / max(1, self.records), 4) if self.records else 0.0,
               "input_tokens": self.tokens, "cost_usd_at_vendor_price": round(self.tokens / 1e6 * RemoteSystemOne.PRICE_PER_MTOK, 6), "by_dp": dict(self.by_dp)}
        if hasattr(self.model, "report"):
            rep["remote"] = self.model.report()
        return rep
