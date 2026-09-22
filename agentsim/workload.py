"""Workload: recipes (structure), programs (the PCB), lazy closed-loop step sampling, arrivals.

A program's next step is sampled only when its predecessor completes (causal successor
release, AgentServeSim eq. r_{p,k+1} = c_{p,k} + g_{p,k}), so the schedule a policy induces
changes what follows. Nothing here touches resources; the engine does that.

Randomness is split into streams so policy comparisons are paired (common random numbers):
  program.rng_steps  what the agent does (steps, tokens, durations, think) — keyed by an
                     exogenous identity (MMPP arrival index, or (slot, k) for closed populations)
  program.rng_env    what happens to it (provider lotteries, agent reactions)
  Arrivals.rng       when sessions arrive
The policy gets its own stream from the engine.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .marginals import Dist, Marginals

THINK_BUCKETS = ((0, 0, "r0"), (1, 2, "r1-2"), (3, 7, "r3-7"), (8, 10 ** 9, "r8+"))   # request-index buckets (B7 state)

# Content proxies (PREDICTOR_DESIGN §8): every tool call and every chat carries a latent class that its *content*
# (the command skeleton, the prompt) would reveal to a gateway that proxies bodies. `content_snr` is the probability
# that the emitted cue is the true class; otherwise it is a uniformly random class. The cue is drawn from its own RNG
# stream, so at any snr the steps themselves are bit-identical to a run without cues.
DURATION_BINS = ((0.0, 1.0), (1.0, 10.0), (10.0, 60.0), (60.0, math.inf))       # ms / s / min / long
OUT_BINS = ((0, 100), (100, 500), (500, 2000), (2000, 10 ** 9))                  # output-token classes
TOOL_COUNT_BINS = ((0, 1), (1, 2), (2, 10 ** 9))                                  # no tools / one / fan-out (half-open)
MAX_SPAWN = 8                                                                      # sub-agents one chat may launch


def _bin(x: float, bins: tuple) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= x < hi or (i == len(bins) - 1 and x >= lo):
            return i
    raise ValueError(f"{x} outside every bin")


def duration_bin(d: float) -> int:
    return _bin(d, DURATION_BINS)


def out_bin(tokens: int) -> int:
    return _bin(tokens, OUT_BINS)


def tool_count_bin(n: int) -> int:
    return _bin(n, TOOL_COUNT_BINS)


def think_bucket(request_idx: int) -> str:
    for lo, hi, name in THINK_BUCKETS:
        if lo <= request_idx <= hi:
            return name
    raise ValueError(f"request_idx {request_idx} outside every bucket")


def _validate_pmf(name: str, pmf: dict) -> None:
    s = sum(float(v) for v in pmf.values())
    if pmf and abs(s - 1.0) > 1e-9:
        raise ValueError(f"{name}: probabilities sum to {s}, not 1")


def _temper(pmf: dict, temperature: float) -> dict:
    """Sharpen (T < 1) or flatten (T > 1) a pmf: p_i^(1/T) renormalised (A7's predictability knob).
    T == 1 returns the row untouched so the draws stay bit-identical to an untempered run."""
    if temperature == 1.0:
        return dict(pmf)
    w = {k: float(v) ** (1.0 / temperature) for k, v in pmf.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def _temper_p(prob: float, temperature: float) -> float:
    """The Bernoulli case of _temper: P(yes) sharpened or flattened against P(no)."""
    return _temper({"yes": prob, "no": 1.0 - prob}, temperature)["yes"]


def _scale_sigma(d: Dist, scale: float) -> Dist:
    """Widen (scale > 1) or narrow every lognormal component's log-sd, keeping medians (A8 tail knob)."""
    if scale == 1.0:
        return d
    if d.kind == "lognormal":
        return Dist("lognormal", mu=d.mu, sigma=d.sigma * scale, cap=d.cap)
    if d.kind == "mixture":
        return Dist("mixture", weights=d.weights, parts=tuple(_scale_sigma(x, scale) for x in d.parts), cap=d.cap)
    raise ValueError(f"tool_tail_scale cannot widen a {d.kind!r} distribution")


def implied_phase(chat: "Step") -> str:
    """Observable phase after a chat: the kind of its last tool call, or none (fitted recipes' '*')."""
    if not chat.follow_tools:
        return "after:none"
    return f"after:{chat.follow_tools[-1].members()[-1].name}"


def _draw(rng: np.random.Generator, pmf: dict) -> str:
    keys = list(pmf)
    return keys[int(rng.choice(len(keys), p=[float(pmf[k]) for k in keys]))]


class Recipes:
    def __init__(self, spec: dict):
        self.tool_resources: dict[str, str] = spec["tool_resources"]
        self.recipes: dict[str, dict] = spec["recipes"]
        for name, r in self.recipes.items():
            phases = set(r["transitions"]) | {"final"}
            for ph, row in r["transitions"].items():
                _validate_pmf(f"{name}.transitions.{ph}", row)
                unknown = set(row) - phases - {"*"}           # "*": the phase implied by the chat's own tools (fitted recipes)
                if unknown:
                    raise ValueError(f"{name}.transitions.{ph} -> unknown phases {unknown}")
            for ph in phases:
                _validate_pmf(f"{name}.tools_per_chat.{ph}", r["tools_per_chat"][ph])
                _validate_pmf(f"{name}.tool_kind.{ph}", r["tool_kind"][ph])
                for kind in r["tool_kind"][ph]:
                    if kind not in self.tool_resources:
                        raise ValueError(f"{name}: tool kind {kind!r} has no resource mapping")
                if ph not in r["retrieval_prob"]:
                    raise KeyError(f"{name}.retrieval_prob missing {ph}")
            if not 0 <= float(r["p_parallel"]) <= 1:
                raise ValueError(f"{name}.p_parallel must be in [0,1]")
        for name, r in self.recipes.items():                        # multi-agent spawn/join (B10), optional per recipe
            sp = r.get("spawn")
            if sp is None:
                continue
            for k in ("prob", "width", "child_recipe"):
                if k not in sp:
                    raise KeyError(f"{name}.spawn missing {k!r}")
            for ph, pr in sp["prob"].items():
                if ph not in r["transitions"] or not 0 <= float(pr) <= 1:
                    raise ValueError(f"{name}.spawn.prob: bad phase {ph!r} or probability {pr!r}")
            _validate_pmf(f"{name}.spawn.width", sp["width"])
            if any(not 1 <= int(w) <= MAX_SPAWN for w in sp["width"]):
                raise ValueError(f"{name}.spawn.width keys must be in 1..{MAX_SPAWN}")
            child = sp["child_recipe"]
            if child not in self.recipes:
                raise KeyError(f"{name}.spawn.child_recipe {child!r} is not a recipe")
            if self.recipes[child].get("spawn") is not None:
                raise ValueError(f"{name}.spawn.child_recipe {child!r} spawns itself: one level of sub-agents only")

    @classmethod
    def load(cls, path: Path) -> "Recipes":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class Step:
    """One unit of work the engine must place on resources. A parallel tool group is one Step
    whose `siblings` run together (gang need: k units at once; duration = max; any 429 fails all)."""
    kind: str                 # "chat" | "tool" | "retrieval"
    name: str                 # phase for chat, tool kind for tool, "retrieval", "parallel" for a group
    tokens_in: int = 0
    tokens_out: int = 0
    append: int = 0
    duration: float = 0.0     # hidden from policies until it happens (except the clairvoyant oracle)
    resource: str | None = None
    follow_tools: list["Step"] = field(default_factory=list)   # tools spawned by a chat, in order
    siblings: list["Step"] = field(default_factory=list)       # members of a parallel group
    is_final: bool = False
    content: str = ""         # OBSERVABLE cue a gateway proxying bodies sees when the step is submitted (skeleton / prompt)
    plan: str = ""            # OBSERVABLE cue in a chat's response (its stated plan), set when the chat ends
    spawn: int = 0            # sub-agents this chat launches at its end (revealed by its response; 0 = none)

    def members(self) -> list["Step"]:
        return self.siblings if self.siblings else [self]


@dataclass
class Program:
    """Program Control Block: everything about one session the engine and policies may read."""
    sid: str
    key: int                            # exogenous identity for common random numbers
    slot: int                           # closed populations: which agent slot; -1 for open arrivals
    recipe: str
    tokens_family: str
    arrival: float
    n_requests: int
    sandbox_gb: float
    rng_steps: np.random.Generator
    rng_env: np.random.Generator
    request_idx: int = 0
    requests_failed: int = 0
    step_idx: int = 0
    phase: str = "plan"
    context_tokens: int = 0
    attained_service: float = 0.0
    episodes: int = 0                   # agent-level error episodes (after SDK retries were exhausted)
    holds: dict[str, float] = field(default_factory=dict)   # resource -> amount currently held
    sandbox_cold: bool = True           # next local tool pays the cold start
    status: str = "active"              # active | done | aborted | censored
    phase_end: str = ""                 # last phase before the final chat of the current request (B7 state; hidden)
    last_tool: str = "none"             # kind of the last completed tool in the current request (observable)
    chats_in_request: int = 0           # completed chats in the current request (observable)
    prewarmed_at: float = -1.0          # time a prewarm re-acquired the sandbox; -1 when not warm-by-prewarm
    prewarm_hits: int = 0               # requests that found a prewarmed sandbox
    rng_content: np.random.Generator | None = None   # the content-cue noise stream (never touches rng_steps)
    trace: str = ""                     # root trace id: the session for top-level programs, the parent's for children
    parent: str | None = None           # multi-agent (B10): the orchestrator this sub-agent was spawned by
    depth: int = 0
    spawn_seq: int = 0                  # fan-outs this program has launched
    join_left: int = 0                  # children still running (the parent waits while > 0)
    join_started: float = -1.0
    children_failed: int = 0
    last_plan: str = ""                 # OBSERVABLE: the plan cue of the last completed chat
    spend_usd: float = 0.0              # OBSERVABLE: what this session has cost so far (provider usage / headers)
    spend_tokens: float = 0.0


class Workload:
    def __init__(self, recipes: Recipes, marginals: Marginals, mix: dict[str, float],
                 hold_model_during_tool: bool, context_limit: int, compaction_reset: int, seed: int,
                 recipe_temperature: float, think_snr: float, think_state_seed: int,
                 tool_tail_scale: float, p_parallel_scale: float, content_snr: float):
        _validate_pmf("recipe_mix", mix)
        if not (isinstance(content_snr, (int, float)) and 0.0 <= content_snr <= 1.0):
            raise ValueError(f"generator.content_snr must be in [0, 1], got {content_snr!r}")
        self.content_snr = float(content_snr)
        for r in mix:
            if r not in recipes.recipes:
                raise KeyError(f"recipe_mix references unknown recipe {r!r}")
        if not (isinstance(recipe_temperature, (int, float)) and recipe_temperature > 0):
            raise ValueError(f"generator.recipe_temperature must be > 0, got {recipe_temperature!r}")
        self.recipes, self.m, self.mix, self.seed = recipes, marginals, mix, seed
        self.hold_model_during_tool = hold_model_during_tool
        self.context_limit, self.compaction_reset = context_limit, compaction_reset
        self.temperature = float(recipe_temperature)
        for name, v in (("tool_tail_scale", tool_tail_scale), ("p_parallel_scale", p_parallel_scale)):
            if not (isinstance(v, (int, float)) and v > 0):
                raise ValueError(f"generator.{name} must be > 0, got {v!r}")
        # the structure rows every program samples from, tempered once (identity at T = 1)
        self.rec: dict[str, dict] = {}
        for name, r in recipes.recipes.items():
            self.rec[name] = {**r,
                              "transitions": {ph: _temper(row, self.temperature) for ph, row in r["transitions"].items()},
                              "tools_per_chat": {ph: _temper(row, self.temperature) for ph, row in r["tools_per_chat"].items()},
                              "tool_kind": {ph: _temper(row, self.temperature) for ph, row in r["tool_kind"].items()},
                              "retrieval_prob": {ph: _temper_p(float(v), self.temperature) for ph, v in r["retrieval_prob"].items()},
                              "p_parallel": min(1.0, _temper_p(float(r["p_parallel"]), self.temperature) * float(p_parallel_scale))}
        self.tool_dist: dict[str, Dist] = {kind: _scale_sigma(marginals.get(f"tool_duration.{kind}"), float(tool_tail_scale))
                                           for kind in recipes.tool_resources}
        # B7: structured think time. log think = mu + sigma * (sqrt(snr) * z_state + sqrt(1 - snr) * eps)
        if not (isinstance(think_snr, (int, float)) and 0.0 <= think_snr <= 1.0):
            raise ValueError(f"generator.think_snr must be in [0, 1], got {think_snr!r}")
        self.think_snr = float(think_snr)
        if not isinstance(think_state_seed, int) or isinstance(think_state_seed, bool):
            raise ValueError(f"generator.think_state_seed must be an int, got {think_state_seed!r}")
        self.think_state_seed = think_state_seed                # the society's idle structure; NOT the run seed
        self.think_dist = marginals.get("think_time")
        if self.think_dist.kind != "lognormal":
            raise ValueError("structured think time needs a lognormal think_time marginal")
        self.z_state: dict[tuple[str, str, str], float] = self._state_offsets() if self.think_snr > 0 else {}

    # ---- B7: per-state offsets, standardised under the generator's own state frequencies ------
    def _phase_end_probs(self, r: dict) -> dict[str, float]:
        """P(last phase before final = X) for the absorbing phase chain of a recipe (fundamental matrix)."""
        phases = list(r["transitions"])
        idx = {ph: i for i, ph in enumerate(phases)}
        n = len(phases)
        q, f = np.zeros((n, n)), np.zeros(n)
        for ph, row in r["transitions"].items():
            for nxt, pr in row.items():
                if nxt == "final":
                    f[idx[ph]] += float(pr)
                elif nxt == "*":                                     # resolve through this phase's own tool lottery
                    p_none = float(r["tools_per_chat"][ph].get("0", 0.0))
                    for kind, pk in r["tool_kind"][ph].items():
                        q[idx[ph], idx[f"after:{kind}"]] += float(pr) * (1.0 - p_none) * float(pk)
                    if p_none > 0:
                        q[idx[ph], idx["after:none"]] += float(pr) * p_none
                else:
                    q[idx[ph], idx[nxt]] += float(pr)
        visits = np.linalg.inv(np.eye(n) - q)[idx[r["start"]]]
        probs = visits * f
        if abs(probs.sum() - 1.0) > 1e-6:
            raise ValueError(f"recipe {r.get('tokens')}: final is not reached with probability 1 (got {probs.sum():.4f})")
        return {ph: float(probs[idx[ph]]) for ph in phases}

    def _state_offsets(self) -> dict[tuple[str, str, str], float]:
        """One offset per (recipe, phase_end, request bucket). Within every (recipe, bucket) group the
        offsets are standardised under the recipe's analytic phase_end probabilities (mean 0, variance 1),
        so whatever mix of recipes and buckets a horizon realises, the think-time marginal keeps its
        mean and variance in log space. Keyed by generator.think_state_seed, not the run seed: the
        mapping is a property of the society, identical across paired seeds; no program stream is touched."""
        rng = np.random.default_rng([self.think_state_seed, 5])
        out: dict[tuple[str, str, str], float] = {}
        for name, r in self.rec.items():
            if float(self.mix.get(name, 0.0)) <= 0:
                continue
            pend = self._phase_end_probs(r)
            phases = [ph for ph in pend if pend[ph] > 0]
            if len(phases) < 2:
                raise ValueError(f"recipe {name}: fewer than two possible end phases; no state signal exists")
            w = np.array([pend[ph] for ph in phases])
            for _, _, b in THINK_BUCKETS:
                z = rng.standard_normal(len(phases))
                mean = float((w * z).sum())
                var = float((w * (z - mean) ** 2).sum())
                if var <= 0:
                    raise ValueError("degenerate state offsets")
                z = (z - mean) / math.sqrt(var)
                for ph, v in zip(phases, z):
                    out[(name, ph, b)] = float(v)
        return out

    # ---- program lifecycle -------------------------------------------------------------
    def new_program(self, sid: str, key: int, slot: int, arrival: float) -> Program:
        rs = np.random.default_rng([self.seed, 1, key])
        re = np.random.default_rng([self.seed, 2, key])
        name = _draw(rs, self.mix)
        r = self.rec[name]
        fam = r["tokens"]
        return Program(sid=sid, key=key, slot=slot, recipe=name, tokens_family=fam, arrival=arrival,
                       n_requests=int(round(self.m.sample("session_requests", rs))),
                       sandbox_gb=float(self.m.sample("sandbox_mem_gb", rs)),
                       rng_steps=rs, rng_env=re, phase=r["start"],
                       context_tokens=int(self.m.sample(f"initial_context_tokens.{fam}", rs)),
                       rng_content=np.random.default_rng([self.seed, 7, key]), trace=sid)

    def new_child(self, parent: Program, j: int, arrival: float) -> Program:
        """A sub-agent of `parent` (B10): its own recipe, sandbox, context and RNG streams keyed by the parent's
        identity and the fan-out index (common random numbers hold across policies); one request, no think time."""
        sp = self.rec[parent.recipe]["spawn"]
        name = sp["child_recipe"]
        r = self.rec[name]
        fam = r["tokens"]
        ident = [self.seed, parent.key, 100 + parent.spawn_seq, j]
        rs, re = np.random.default_rng(ident + [1]), np.random.default_rng(ident + [2])
        sid = f"{parent.sid}/{parent.spawn_seq}.{j}"
        return Program(sid=sid, key=parent.key, slot=parent.slot, recipe=name, tokens_family=fam, arrival=arrival,
                       n_requests=1, sandbox_gb=float(self.m.sample("sandbox_mem_gb", rs)),
                       rng_steps=rs, rng_env=re, phase=r["start"],
                       context_tokens=int(self.m.sample(f"initial_context_tokens.{fam}", rs)),
                       rng_content=np.random.default_rng(ident + [7]), trace=parent.trace, parent=parent.sid,
                       depth=parent.depth + 1)

    # ---- content cues (observable; noise from rng_content only) ----------------------------
    def _cue(self, p: Program, true_cls: int, n: int) -> int:
        """The class a gateway reads off the content: the truth with probability content_snr, else uniform noise."""
        truthful = p.rng_content.random() < self.content_snr
        noise = int(p.rng_content.integers(n))                   # always drawn: the stream advances identically at any snr
        return true_cls if truthful else noise

    def start_request(self, p: Program) -> None:
        p.phase = self.rec[p.recipe]["start"]
        p.context_tokens += int(self.m.sample(f"append_tokens.{p.tokens_family}", p.rng_steps))  # user message
        self._compact_if_needed(p)

    def _tool(self, p: Program, kind: str) -> Step:
        d = float(self.tool_dist[kind].sample(p.rng_steps))
        append = int(self.m.sample(f"append_tokens.{p.tokens_family}", p.rng_steps))
        return Step(kind="tool", name=kind, resource=self.recipes.tool_resources[kind], duration=d, append=append,
                    content=f"sk:{kind}:{self._cue(p, duration_bin(d), len(DURATION_BINS))}")

    def next_chat(self, p: Program) -> Step:
        """Sample the next LLM step and the tool calls it will issue (closed-loop, lazy)."""
        r = self.rec[p.recipe]
        ph = p.phase
        out = int(self.m.sample(f"output_tokens.{p.tokens_family}", p.rng_steps))
        step = Step(kind="chat", name=ph, tokens_in=p.context_tokens, tokens_out=out, is_final=(ph == "final"))
        sp = r.get("spawn")
        if sp is not None and ph in sp["prob"] and p.parent is None and p.rng_steps.random() < float(sp["prob"][ph]):
            step.spawn = int(_draw(p.rng_steps, sp["width"]))       # an orchestrator turn: launches sub-agents, no tools
        elif ph != "final":
            n_tools = int(_draw(p.rng_steps, r["tools_per_chat"][ph]))
            tools = [self._tool(p, _draw(p.rng_steps, r["tool_kind"][ph])) for _ in range(n_tools)]
            if n_tools >= 2 and p.rng_steps.random() < float(r["p_parallel"]):
                group = Step(kind="tool", name="parallel", siblings=tools,
                             duration=max(t.duration for t in tools), append=sum(t.append for t in tools))
                step.follow_tools = [group]
            else:
                step.follow_tools = tools
        n_tools = sum(len(t.members()) for t in step.follow_tools)
        o, t = self._cue(p, out_bin(out), len(OUT_BINS)), self._cue(p, tool_count_bin(n_tools), len(TOOL_COUNT_BINS))
        step.content = f"pc:o{o} pc:t{t}"                            # the prompt's cue: how long an answer, will it call tools
        return step

    def wants_retrieval(self, p: Program) -> Step | None:
        r = self.rec[p.recipe]
        if p.rng_steps.random() < float(r["retrieval_prob"][p.phase]):
            return Step(kind="retrieval", name="retrieval", resource="svc.retrieval",
                        duration=float(self.m.sample("retrieval_duration", p.rng_steps)),
                        append=int(self.m.sample(f"append_tokens.{p.tokens_family}", p.rng_steps)) // 4)
        return None

    def after_chat(self, p: Program, step: Step) -> None:
        p.context_tokens += step.tokens_out
        r = self.rec[p.recipe]
        phases = list(r["transitions"]) + ["final"]
        if not step.is_final:
            nxt = _draw(p.rng_steps, r["transitions"][p.phase])
            if nxt == "*":
                nxt = implied_phase(step)
                if nxt not in r["transitions"]:
                    raise KeyError(f"{p.recipe}: implied phase {nxt!r} has no transition row")
            if nxt == "final":
                p.phase_end = p.phase
            p.phase = nxt
            step.plan = "pl:" + phases[self._cue(p, phases.index(nxt), len(phases))]   # the response's stated next move
        else:
            step.plan = "pl:final"                                   # the turn closed without tool calls: observable
        p.last_plan = step.plan
        self._compact_if_needed(p)

    def after_tool(self, p: Program, step: Step) -> None:
        p.context_tokens += step.append
        self._compact_if_needed(p)

    def think_time(self, p: Program) -> float:
        """Human idle time after a request. At think_snr = 0 the i.i.d. draw of the seeds (bit-identical);
        above it, part of the log-variance is a fixed offset of the observable state (recipe, phase_end,
        request bucket) so that idle time can be *learned*, with the marginal's mean/variance preserved."""
        d = self.think_dist
        if self.think_snr == 0.0:
            return float(d.sample(p.rng_steps))
        eps = float(p.rng_steps.standard_normal())
        key = (p.recipe, p.phase_end, think_bucket(p.request_idx))
        if key not in self.z_state:
            raise KeyError(f"no state offset for {key}; phase_end was not recorded before the final chat")
        snr = self.think_snr
        x = math.exp(d.mu + d.sigma * (math.sqrt(snr) * self.z_state[key] + math.sqrt(1.0 - snr) * eps))
        return float(min(x, d.cap))

    def _compact_if_needed(self, p: Program) -> None:
        if p.context_tokens > self.context_limit:          # context compaction invalidates KV (Copilot traces)
            p.context_tokens = self.compaction_reset

    # ---- reaction to errors: SDK retries first, then the agent (PRIOR; fit at rung 0) ----
    def react(self, p: Program, sdk_tries: int, sdk_retries: int, after_sdk: dict[str, float], max_episodes: int) -> str:
        """Return 'retry' (SDK layer) | 'replan' | 'abort' (agent layer)."""
        if sdk_tries < sdk_retries:
            return "retry"
        _validate_pmf("reaction.after_sdk", after_sdk)
        p.episodes += 1
        if p.episodes > max_episodes:
            return "abort"
        return _draw(p.rng_env, after_sdk)


class Arrivals:
    """Partly-open arrival process (Schroeder et al. 2006).

    type "mmpp": sessions arrive as a two-state Markov-modulated Poisson process (bursty, open);
    inside a session, requests are closed-loop (next request after think time).
    type "closed": a fixed population of N agents; a new session starts the moment one ends
    (HiveMind's "N parallel agents" setting, used for E1); initial starts staggered over stagger_s.
    """

    def __init__(self, spec: dict, seed: int):
        self.spec, self.rng = spec, np.random.default_rng([seed, 3])
        t = spec["type"]
        if t == "mmpp":
            for k in ("rate_per_min", "burst_factor", "mean_low_s", "mean_high_s"):
                if k not in spec:
                    raise KeyError(f"arrivals.mmpp missing {k}")
            self.state_high = False
            self.switch_at = self.rng.exponential(spec["mean_low_s"])
        elif t == "closed":
            for k in ("population", "stagger_s"):
                if k not in spec:
                    raise KeyError(f"arrivals.closed missing {k}")
        else:
            raise ValueError(f"unknown arrivals type {t!r}")

    @property
    def closed(self) -> bool:
        return self.spec["type"] == "closed"

    def next_interarrival(self, now: float) -> float:
        s = self.spec
        if now >= self.switch_at:
            self.state_high = not self.state_high
            self.switch_at = now + self.rng.exponential(s["mean_high_s"] if self.state_high else s["mean_low_s"])
        rate = s["rate_per_min"] * (s["burst_factor"] if self.state_high else 1.0) / 60.0
        return float(self.rng.exponential(1.0 / rate))
