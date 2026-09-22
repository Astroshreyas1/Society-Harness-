"""Fit the generator's seeds from OTel-shaped traces (rung 0 proper; ISSUES C1/C2).

Two outputs, each with provenance per entry:
  marginals  quantities: tool_duration by kind, output_tokens / append_tokens / initial_context_tokens by
             token family, think_time, session_requests, sandbox_mem_gb, retrieval_duration — lognormal by
             quantile matching (median + p99, or p90 for think time). Entries with too few samples keep
             their seed and say so.
  recipes    structure: one recipe per root name, as a first-order chain over OBSERVABLE phases —
             "start" (first chat of a request), "after:<kind>" (a chat that followed a tool of that kind),
             "after:none" (a chat that followed a chat with no tools). A phase's transition row has two
             targets: "final" (the request's closing chat comes next) and "*" (the phase implied by this
             chat's own tools; the generator resolves it after drawing them). tools_per_chat, tool_kind and
             retrieval_prob are per phase; p_parallel per recipe. Laplace 0.5 on "final" keeps it reachable.

Ids are only unique within one run's file, so several files are namespaced by file index (F15).
Only spans with outcome "ok" describe structure; retries of the same step are not new nodes.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .schema import Span, read_jsonl

MIN_N = 50            # samples needed to replace a seed marginal
MIN_SESSIONS = 20     # completed sessions needed to fit session_requests


def load_files(paths: list[str | Path]) -> list[list[Span]]:
    out = []
    for p in paths:
        spans = list(read_jsonl(Path(p)))
        if not spans:
            raise ValueError(f"{p}: no spans")
        out.append(spans)
    return out


def _lognormal(xs: list[float], cap: float, key: str = "p99") -> dict | None:
    if len(xs) < MIN_N:
        return None
    q = 99 if key == "p99" else 90
    med, hi = float(np.percentile(xs, 50)), float(np.percentile(xs, q))
    if not hi > med > 0:
        return None
    return {"type": "lognormal", "median": med, key: hi, "cap": cap}


def _em_lognormal_mixture(xs: list[float], k: int, iters: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """EM for a k-component Gaussian mixture on log x (deterministic quantile init). Returns weights, mu, sigma, log-lik."""
    y = np.log(np.maximum(np.asarray(xs, dtype=float), 1e-3))     # real traces have 0 ms calls: floor at 1 ms
    n = len(y)
    mu = np.percentile(y, np.linspace(10, 90, k))
    sigma = np.full(k, max(y.std(), 1e-3))
    w = np.full(k, 1.0 / k)
    ll = -np.inf
    for _ in range(iters):
        logp = -0.5 * ((y[:, None] - mu[None, :]) / sigma[None, :]) ** 2 - np.log(sigma[None, :]) - 0.5 * np.log(2 * np.pi) + np.log(w[None, :])
        m = logp.max(axis=1, keepdims=True)
        resp = np.exp(logp - m)
        tot = resp.sum(axis=1, keepdims=True)
        resp /= tot
        new_ll = float((m[:, 0] + np.log(tot[:, 0])).sum())
        nk = resp.sum(axis=0) + 1e-12
        w = nk / n
        mu = (resp * y[:, None]).sum(axis=0) / nk
        sigma = np.sqrt((resp * (y[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk)
        sigma = np.maximum(sigma, 0.05)
        if new_ll - ll < 1e-6:
            ll = new_ll
            break
        ll = new_ll
    return w, mu, sigma, ll


def _tool_duration_spec(xs: list[float], cap: float) -> dict | None:
    """Lognormal mixture with 1-3 components chosen by BIC (the seeds are mixtures; a single lognormal
    matched at median/p99 loses the body-to-tail shape that sets the tool-time tail shares)."""
    if len(xs) < MIN_N:
        return None
    best = None
    for k in (1, 2, 3):
        if len(xs) < 40 * k:
            break
        w, mu, sigma, ll = _em_lognormal_mixture(xs, k)
        bic = -2 * ll + (3 * k - 1) * np.log(len(xs))
        if best is None or bic < best[0]:
            best = (bic, w, mu, sigma)
    _, w, mu, sigma = best
    parts = [{"type": "lognormal", "median": float(np.exp(m)), "p99": float(np.exp(m + 2.3263478740408408 * sg))} for m, sg in zip(mu, sigma)]
    if len(parts) == 1:
        return {**parts[0], "cap": cap}
    return {"type": "mixture", "weights": [float(v) for v in w], "parts": parts, "cap": cap}


def _pmf(counts: Counter) -> dict:
    tot = sum(counts.values())
    if tot <= 0:
        raise ValueError("empty pmf")
    return {str(k): v / tot for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))}


class _Request:
    """Ordered ok-spans of one request: chats with the tools that followed each, retrievals before each."""

    def __init__(self, req: Span):
        self.req = req
        self.chats: list[tuple[Span, list[Span], bool]] = []     # (chat, tool spans after it, retrieval before it)


def _requests(spans: list[Span]) -> tuple[dict[str, Span], list[_Request]]:
    roots = {s.trace_id: s for s in spans if s.op == "invoke_agent"}
    by_trace: dict[str, list[Span]] = defaultdict(list)
    for s in spans:
        if s.op in ("chat", "execute_tool", "retrieval"):
            by_trace[s.trace_id].append(s)
    reqs: list[_Request] = []
    for r in spans:
        if r.op != "invoke_workflow":
            continue
        inside = sorted((s for s in by_trace.get(r.trace_id, ()) if r.t_start <= s.t_start <= r.t_end), key=lambda s: (s.t_start, s.span_id))
        cur = _Request(r)
        pending_retrieval, failed_since_chat = False, False
        for s in inside:
            if s.outcome != "ok":
                failed_since_chat = True                         # a 429 / timeout / eviction: what follows is a reaction
                continue
            if s.op == "retrieval":
                pending_retrieval = True
            elif s.op == "chat":
                if failed_since_chat and cur.chats:              # replan chat: the agent's reaction, not its plan (folded)
                    failed_since_chat = False
                    continue
                cur.chats.append((s, [], pending_retrieval))
                pending_retrieval = False
                failed_since_chat = False
            elif cur.chats:
                cur.chats[-1][1].append(s)
        if cur.chats:
            reqs.append(cur)
    return roots, reqs


def _members(t: Span) -> list[str]:
    return list(t.attrs.get("members") or [t.name])


def fit_recipes(files: list[list[Span]], seed_recipes: dict) -> dict:
    """Mine observable-phase recipes; token families and tool→resource mapping come from the seeds."""
    tool_resources = seed_recipes["tool_resources"]
    families = {name: r["tokens"] for name, r in seed_recipes["recipes"].items()}
    trans: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    n_tools: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    kinds: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    retr: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    par: dict[str, Counter] = defaultdict(Counter)
    n_req: Counter = Counter()
    for spans in files:
        roots, reqs = _requests(spans)
        for rq in reqs:
            rec = roots[rq.req.trace_id].name
            n_req[rec] += 1
            ok = rq.req.outcome == "ok"
            last = len(rq.chats) - 1
            phase = "start"
            for i, (chat, tools, had_retrieval) in enumerate(rq.chats):
                if ok and i == last:
                    break                                            # the closing chat is the "final" phase itself
                members = [m for t in tools for m in _members(t)]
                n_tools[rec][phase][len(members)] += 1
                for m in members:
                    if m not in tool_resources:
                        raise KeyError(f"tool kind {m!r} in traces has no resource mapping in the seed recipes")
                    kinds[rec][phase][m] += 1
                retr[rec][phase]["yes" if had_retrieval else "no"] += 1
                if len(members) >= 2:                                    # a chat that issued >= 2 tool calls: did any run in parallel?
                    par[rec]["parallel" if any(t.attrs.get("parallel") for t in tools) else "sequential"] += 1
                if i < last:
                    trans[rec][phase]["final" if (ok and i + 1 == last) else "*"] += 1
                phase = f"after:{members[-1]}" if members else "after:none"
    recipes: dict[str, dict] = {}
    for rec in sorted(n_req):
        if rec not in families:
            raise KeyError(f"root name {rec!r} is not a seed recipe (token family unknown)")
        implied = {f"after:{k}" for ph in kinds[rec] for k in kinds[rec][ph]}        # every phase the generator can imply
        if any(n_tools[rec][ph][0] for ph in n_tools[rec]):
            implied.add("after:none")
        phases = sorted(set(trans[rec]) | set(n_tools[rec]) | implied)
        t_rows, tpc, tk, rp = {}, {}, {}, {}
        for ph in phases:
            c = Counter(trans[rec][ph])
            c["final"] += 0.5                                        # Laplace: final always reachable
            t_rows[ph] = _pmf(c)
            tpc[ph] = _pmf(n_tools[rec][ph]) if n_tools[rec][ph] else {"0": 1.0}
            tk[ph] = _pmf(kinds[rec][ph]) if kinds[rec][ph] else {}
            rc = retr[rec][ph]
            rp[ph] = rc["yes"] / (rc["yes"] + rc["no"]) if (rc["yes"] + rc["no"]) else 0.0
        tpc["final"], tk["final"], rp["final"] = {"0": 1.0}, {}, 0.0
        p = par[rec]
        recipes[rec] = {"tokens": families[rec], "start": "start", "transitions": t_rows, "tools_per_chat": tpc,
                        "tool_kind": tk, "retrieval_prob": rp,
                        "p_parallel": p["parallel"] / (p["parallel"] + p["sequential"]) if (p["parallel"] + p["sequential"]) else 0.0,
                        "_fitted": {"requests": n_req[rec], "phases": len(phases)}}
    return {"provenance": {"note": "Fitted from traces by agentsim.fit: first-order chain over observable phases "
                                   "(start / after:<tool kind> / after:none); '*' = the phase this chat's own tools imply; "
                                   "'final' smoothed with Laplace 0.5. Token families and tool_resources from the seed recipes.",
                           "requests_per_recipe": dict(n_req)},
            "tool_resources": tool_resources, "recipes": recipes}


def fit_marginals(files: list[list[Span]], seed: dict, seed_recipes: dict) -> dict:
    m = json.loads(json.dumps(seed))
    prov = m.setdefault("provenance", {})
    families = {name: r["tokens"] for name, r in seed_recipes["recipes"].items()}
    tool_d: dict[str, list[float]] = defaultdict(list)
    out_t: dict[str, list[float]] = defaultdict(list)
    app_t: dict[str, list[float]] = defaultdict(list)
    init_t: dict[str, list[float]] = defaultdict(list)
    thinks: list[float] = []
    retr_d: list[float] = []
    mem: list[float] = []
    n_requests: list[int] = []
    for spans in files:
        roots, reqs = _requests(spans)
        for s in spans:
            if s.op == "execute_tool" and s.outcome == "ok" and s.name != "parallel":
                tool_d[s.name].append(s.duration)
            elif s.op == "think":
                thinks.append(s.duration)
            elif s.op == "retrieval" and s.outcome == "ok":
                retr_d.append(s.duration)
        seen_mem: set[str] = set()
        for s in spans:
            if s.op == "execute_tool" and s.mem > 0 and s.trace_id not in seen_mem:
                seen_mem.add(s.trace_id)
                mem.append(s.mem)
        for r in roots.values():
            if r.outcome == "ok":
                n_requests.append(int(r.attrs["requests_done"]))
        first_of: dict[str, _Request] = {}
        for rq in reqs:
            if rq.req.attrs.get("request_idx") == 0:
                first_of[rq.req.trace_id] = rq
            fam = families[roots[rq.req.trace_id].name]
            for i, (chat, tools, _) in enumerate(rq.chats):
                if chat.tokens_out > 0:
                    out_t[fam].append(chat.tokens_out)
                if i + 1 < len(rq.chats):
                    nxt = rq.chats[i + 1][0]
                    n_members = sum(len(_members(t)) for t in tools)
                    delta = nxt.tokens_in - chat.tokens_in - chat.tokens_out
                    if delta > 0 and n_members > 0:                # compaction makes deltas negative: dropped
                        app_t[fam].append(delta / n_members)
        for rq in first_of.values():
            init_t[families[roots[rq.req.trace_id].name]].append(rq.chats[0][0].tokens_in)
    fitted, kept = [], []
    for kind, xs in tool_d.items():
        d = _tool_duration_spec(xs, 3600)
        if d:
            m["tool_duration"][kind] = d
            k = len(d["parts"]) if d["type"] == "mixture" else 1
            prov[f"tool_duration.{kind}"] = f"fitted {k}-component lognormal mixture by EM (BIC over 1-3), n={len(xs)}"
            fitted.append(f"tool_duration.{kind}")
    for fam_key, src, cap in (("output_tokens", out_t, 32000), ("append_tokens", app_t, 120000), ("initial_context_tokens", init_t, 400000)):
        for fam, xs in src.items():
            d = _lognormal([float(x) for x in xs], cap)
            if d:
                m[fam_key][fam] = d
                note = {"append_tokens": " (per-tool share of the context growth between consecutive chats; negative deltas from compaction dropped)",
                        "initial_context_tokens": " (tokens_in of the first chat, includes the first user message)"}.get(fam_key, "")
                prov[f"{fam_key}.{fam}"] = f"fitted, n={len(xs)}{note}"
                fitted.append(f"{fam_key}.{fam}")
    d = _lognormal(thinks, 14400, "p90")
    if d:
        m["think_time"] = d
        prov["think_time"] = f"fitted, n={len(thinks)}"
        fitted.append("think_time")
    d = _lognormal(retr_d, 600)
    if d:
        m["retrieval_duration"] = d
        prov["retrieval_duration"] = f"fitted, n={len(retr_d)}"
        fitted.append("retrieval_duration")
    d = _lognormal(mem, 64)
    if d:
        m["sandbox_mem_gb"] = d
        prov["sandbox_mem_gb"] = f"fitted from per-session sandbox memory, n={len(mem)}"
        fitted.append("sandbox_mem_gb")
    if len(n_requests) >= MIN_SESSIONS:
        ones = sum(1 for n in n_requests if n <= 1)
        rest = [float(n - 1) for n in n_requests if n > 1]
        then = _lognormal(rest, 300) if len(rest) >= MIN_SESSIONS else None
        if then:
            m["session_requests"] = {"type": "one_or", "p_one": ones / len(n_requests), "then": then}
            prov["session_requests"] = f"fitted from {len(n_requests)} completed sessions (censored sessions excluded: biased short)"
            fitted.append("session_requests")
    for key in ("tool_duration", "output_tokens", "append_tokens", "initial_context_tokens", "think_time", "session_requests", "sandbox_mem_gb", "retrieval_duration"):
        entries = [f"{key}.{k}" for k in m[key]] if isinstance(m[key], dict) and "type" not in m[key] else [key]
        for e in entries:
            if e not in fitted:
                prov[e] = prov.get(e, "") + " [unfitted: seed kept]" if "[unfitted" not in prov.get(e, "") else prov[e]
                kept.append(e)
    prov["fitted_entries"] = sorted(fitted)
    prov["unfitted_entries"] = sorted(kept)
    return m


def fit_all(files: list[list[Span]], seed_marginals: dict, seed_recipes: dict) -> tuple[dict, dict]:
    if len(files) > 1:                                               # namespace ids across runs (F15)
        for i, spans in enumerate(files):
            for s in spans:
                s.trace_id = f"{i}:{s.trace_id}"
    return fit_marginals(files, seed_marginals, seed_recipes), fit_recipes(files, seed_recipes)
