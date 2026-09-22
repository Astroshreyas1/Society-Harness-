"""Offline training and annotation on OTel-shaped traces (synthetic or real), through the same Observer the gateway runs.

  train-needs   spans -> replay_events -> Observer -> (record, labels) -> NeedsPredictor.fit -> data/models/<name>.npz
                Synthetic pre-training; the controller loads the file (policy.needs.model) and keeps learning online.
  train-jev     spans -> the same records, converted to typed (state, answers) examples -> LocalSystemOne.fit
                Held-out accuracy / ECE / temperature per question is the calibration report (PREDICTOR_DESIGN §7's ECE gate).
  annotate      spans + a System One model -> chat spans gain attrs["jev"] = {phase, next_chat, ...} with probabilities:
                integration point A (offline labelling), consumable by rung0/e2_predictability.py --phase-attr jev_phase.

Labels come from realised outcomes only, except `phase`, which the replayer reads from the chat spans' hidden `phase`
attribute (the generator's truth; on real traces, a human or the vendor model would supply it).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .features import Observer, Record, chat_tool_lists, replay_events
from .jev import LocalSystemOne, RemoteSystemOne, answers_from_labels, catalogue, evaluate_system_one
from .needs import NeedsModel, NeedsPredictor
from .schema import Span, read_jsonl
from .workload import DURATION_BINS, MAX_SPAWN, OUT_BINS, Recipes, duration_bin, out_bin

ROOT = Path(__file__).resolve().parent.parent


def load_traces(paths: list[str | Path], scenario: Path | None = None, max_sessions: int | None = None) -> list[tuple[list[Span], dict]]:
    """Each trace file with the scenario.json beside it (capacities, recipes) — or one `scenario` for all (real traces).
    Session ids are namespaced per file; `max_sessions` keeps the first N sessions of each file (large real traces)."""
    out = []
    for i, p in enumerate(paths):
        p = Path(p)
        spans = list(read_jsonl(p))
        if not spans:
            raise ValueError(f"{p}: no spans")
        if max_sessions is not None:
            keep: list[str] = []
            seen: set[str] = set()
            for s in spans:
                if s.trace_id not in seen:
                    seen.add(s.trace_id)
                    keep.append(s.trace_id)
                    if len(keep) >= max_sessions:
                        break
            keep_set = set(keep)
            spans = [s for s in spans if s.trace_id in keep_set]
        sc = scenario if scenario is not None else p.parent / "scenario.json"
        if not sc.exists():
            raise FileNotFoundError(f"{sc}: the trainer needs the scenario next to each trace (capacities, recipes), or --scenario")
        cfg = json.loads(sc.read_text(encoding="utf-8"))
        if len(paths) > 1:
            for s in spans:
                s.trace_id = f"{i}:{s.trace_id}"
                s.span_id = f"{i}:{s.span_id}"
                if s.parent_span_id:
                    s.parent_span_id = f"{i}:{s.parent_span_id}"
        chat_tool_lists(spans)
        out.append((spans, cfg))
    return out


def replay_pairs(spans: list[Span], cfg: dict, keep_state: bool, use_content: bool = True, H: float | None = None) -> tuple[list[tuple[Record, dict]], Observer]:
    recipes = Recipes.load(ROOT / cfg["recipes"])
    caps = {n: float(r["capacity"]) for n, r in cfg["resources"].items()}
    events, res = replay_events(spans, caps)
    obs = Observer(sorted(recipes.tool_resources), recipes.tool_resources, res, float(cfg["framework"]["sandbox_cold_start_s"]),
                   H if H is not None else float(cfg["policy"]["h"]), use_content=use_content, keep_state=keep_state)
    for t, kind, sid, recipe, is_child, depth, info in events:
        obs.on_event(kind, sid, t, recipe=recipe, is_child=is_child, depth=depth, **info)
    merged: dict[int, tuple[Record, dict]] = {}
    for rec, labels in obs.drain():
        k = id(rec)
        if k in merged:
            merged[k][1].update(labels)
        else:
            merged[k] = (rec, dict(labels))
    pairs = sorted(merged.values(), key=lambda x: x[0].t)
    return pairs, obs


def train_needs(paths: list[str], out: Path, epochs: int, seed: int = 0, use_content: bool = True, scenario: Path | None = None,
                max_sessions: int | None = None) -> dict:
    traces = load_traces(paths, scenario, max_sessions)
    all_pairs: list[tuple[Record, dict]] = []
    vocab = None
    for spans, cfg in traces:
        pairs, obs = replay_pairs(spans, cfg, keep_state=False, use_content=use_content)
        if vocab is None:
            vocab = obs.vocab
        elif obs.vocab != vocab:
            raise ValueError("traces from societies with different tool kinds cannot train one model")
        all_pairs.extend(pairs)
    if not all_pairs:
        raise ValueError("no labelled records")
    rng = np.random.default_rng(seed)
    sids = sorted({rec.sid.split("/")[0] for rec, _ in all_pairs})       # hold out whole sessions (children with their parent)
    hold_sids = set(rng.choice(sids, size=max(1, len(sids) // 5), replace=False).tolist())
    hold = [pr for pr in all_pairs if pr[0].sid.split("/")[0] in hold_sids]
    train = [pr for pr in all_pairs if pr[0].sid.split("/")[0] not in hold_sids]
    pred = NeedsPredictor(vocab, tau=0.8, seed=seed)
    hist = pred.fit(train, epochs, seed)
    ev = evaluate_needs(pred, hold)
    pred.model.save(out)
    return {"records": len(all_pairs), "train": len(train), "holdout": len(hold), "loss_per_epoch": [round(x, 4) for x in hist],
            "holdout": ev, "saved": str(out)}


def evaluate_needs(pred: NeedsPredictor, pairs: list[tuple[Record, dict]]) -> dict:
    """Per head on held-out pairs: quantile coverage / pinball, categorical accuracy vs majority, binary Brier."""
    by_head: dict[str, list] = defaultdict(list)
    for rec, labels in pairs:
        pr = pred.predict(rec)
        for k, y in labels.items():
            spec = pred.model.heads.spec.get(k)
            if spec is None:
                continue
            kind = spec[0]
            if kind == "quantile":
                q50, q80, q95 = (pr.quantile(k, t, conformal=False) for t in (0.5, 0.8, 0.95))
                by_head[k].append((float(y), q50, q80, q95))
            elif kind == "softmax":
                by_head[k].append((pred.model.vidx[y] if isinstance(y, str) else int(y), int(pr.probs(k).argmax()), float(pr.probs(k).max())))
            else:
                by_head[k].append((float(y), pr.p_true(k)))
    out = {}
    for k, rows in by_head.items():
        kind = pred.model.heads.spec[k][0]
        a = np.array(rows)
        if kind == "quantile":
            y = a[:, 0]
            pin = lambda t, q: float(np.mean(np.maximum(t * (y - q), (t - 1) * (y - q))))
            base = np.percentile(y, [50, 80, 95])
            out[k] = {"n": len(y), "mae_median": round(float(np.mean(np.abs(y - a[:, 1]))), 4),
                      "mae_const": round(float(np.mean(np.abs(y - np.median(y)))), 4),
                      "cover80": round(float(np.mean(y <= a[:, 2])), 3), "cover95": round(float(np.mean(y <= a[:, 3])), 3),
                      "pinball80_ratio": round(pin(0.8, a[:, 2]) / max(1e-9, pin(0.8, base[1])), 3),
                      "pinball95_ratio": round(pin(0.95, a[:, 3]) / max(1e-9, pin(0.95, base[2])), 3),
                      "r2_median": round(float(1 - np.sum((y - a[:, 1]) ** 2) / np.sum((y - y.mean()) ** 2)), 3) if len(y) >= 30 and np.var(y) > 1e-6 else float("nan")}
        elif kind == "softmax":
            y = a[:, 0].astype(int)
            maj = float(np.bincount(y).max() / len(y))
            out[k] = {"n": len(y), "accuracy": round(float(np.mean(a[:, 1] == y)), 3), "majority": round(maj, 3)}
        else:
            y, p = a[:, 0], a[:, 1]
            out[k] = {"n": len(y), "brier": round(float(np.mean((p - y) ** 2)), 4), "brier_const": round(float(np.mean((y.mean() - y) ** 2)), 4),
                      "base_rate": round(float(y.mean()), 4)}
    return out


# ---- Jev ------------------------------------------------------------------------------------------------
def jev_examples(pairs: list[tuple[Record, dict]], cold_start_s: float, horizon_s: float) -> list[tuple[dict, dict[str, str]]]:
    """Typed answers from realised outcomes (and, for `phase`, the replayed hidden label), through the same mapping the
    online Judge uses. Tool records of one step collapse into one example (the longest member's class)."""
    out: list[tuple[dict, dict[str, str]]] = []
    seen_tool: set[tuple[str, float]] = set()
    for rec, labels in pairs:
        st = rec.meta.get("state")
        if st is None:
            raise ValueError("records were built without keep_state=True")
        if rec.dp == "tool_ready":
            key = (rec.sid, rec.t)
            if key in seen_tool or "Q_tool_max" not in labels:
                continue
            seen_tool.add(key)
        ans = answers_from_labels(rec.dp, labels, cold_start_s, horizon_s)
        if ans:
            out.append((st, ans))
    return out


def _society(cfg: dict) -> tuple[Recipes, list[str]]:
    recipes = Recipes.load(ROOT / cfg["recipes"])
    phases = sorted({ph for r in recipes.recipes.values() for ph in list(r["transitions"]) + ["final"]})
    return recipes, phases


def jev_examples_from(paths: list[str], scenario: Path | None = None, max_sessions: int | None = None) -> tuple[list, dict]:
    traces = load_traces(paths, scenario, max_sessions)
    examples, cat = [], None
    for spans, cfg in traces:
        pairs, obs = replay_pairs(spans, cfg, keep_state=True)
        recipes, phases = _society(cfg)
        c = catalogue(sorted(recipes.tool_resources), phases)
        if cat is None:
            cat = c
        elif {q: v.options for q, v in c.items()} != {q: v.options for q, v in cat.items()}:
            raise ValueError("traces from societies with different question schemas cannot train one model")
        examples.extend(jev_examples(pairs, float(cfg["framework"]["sandbox_cold_start_s"]), float(cfg["policy"]["h"])))
    if not examples:
        raise ValueError("no examples")
    return examples, cat


def train_jev(paths: list[str], out: Path, epochs: int, seed: int = 0, scenario: Path | None = None, max_sessions: int | None = None) -> dict:
    examples, cat = jev_examples_from(paths, scenario, max_sessions)
    model = LocalSystemOne(cat, seed)
    report = model.fit(examples, epochs=epochs, seed=seed, groups=[st["sid"] for st, _ in examples])
    model.save(out)
    return {"examples": len(examples), "per_question": report, "saved": str(out)}


def evaluate_jev(paths: list[str], model_path: Path | None, remote: bool, max_n: int | None = None, scenario: Path | None = None,
                 max_sessions: int | None = None) -> dict:
    """Score a System One model — the local file, or the vendor model through RemoteSystemOne — on labelled examples from
    traces: accuracy / ECE / Brier per question (the D2 validation before any answer is trusted)."""
    examples, cat = jev_examples_from(paths, scenario, max_sessions)
    model = RemoteSystemOne(cat) if remote else LocalSystemOne.load(model_path)
    rep = evaluate_system_one(model, examples, max_n)
    if remote:
        rep["remote"] = model.report()
    return rep


def annotate(trace: Path, model_path: Path, out: Path) -> dict:
    """Ask the System One model at every chat end; its `phase` answer is about the *next* chat, so it is written onto
    that chat's span (attrs['jev_phase'], plus the full typed record under attrs['jev'] on the chat that was asked)."""
    spans = list(read_jsonl(trace))
    cfg = json.loads((trace.parent / "scenario.json").read_text(encoding="utf-8"))
    chat_tool_lists(spans)
    model = LocalSystemOne.load(model_path)
    pairs, obs = replay_pairs(spans, cfg, keep_state=True)
    by_key = {(rec.sid, round(rec.t, 6)): rec for rec, _ in pairs if rec.dp == "chat_end"}
    by_sid: dict[str, list[Span]] = defaultdict(list)
    for s in spans:
        if s.op == "chat" and s.outcome == "ok":
            by_sid[s.span_id.rsplit("-", 1)[0]].append(s)
    n = 0
    for sid, chats in by_sid.items():
        chats.sort(key=lambda x: x.t_start)
        for cur, nxt in zip(chats, chats[1:]):
            rec = by_key.get((sid, round(cur.t_end, 6)))
            if rec is None:
                continue
            jr = model.decide(rec.meta["state"], tuple(q for q in ("next_chat", "next_tool", "phase", "fail_soon") if q in model.schema), rec.t)
            cur.attrs["jev"] = {q: {"answer": a, "p": round(pr, 4)} for q, (a, pr) in jr.answers.items()}
            if not cur.attrs.get("is_final", cur.name == "final"):
                nxt.attrs["jev_phase"] = jr.answers["phase"][0]
                n += 1
    from .schema import write_jsonl
    write_jsonl(out, spans)
    acc = [s.attrs["jev_phase"] == s.attrs.get("phase") for s in spans if s.op == "chat" and "jev_phase" in s.attrs and "phase" in s.attrs]
    return {"annotated": n, "phase_accuracy_vs_hidden": round(float(np.mean(acc)), 4) if acc else float("nan"), "wrote": str(out)}
