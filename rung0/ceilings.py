"""Ceilings for the Needs Predictor's content heads, measured on the real TraceLab trace before any model is
trained (PREDICTOR_DESIGN.md §7, experiments 1–2).

  python rung0/ceilings.py tools data/real/tracelab/syfi_coding_trace.jsonl.gz [--provider claude] [--max-calls 200000]
      Head Q ceiling: Bash calls with a sanitized `command_skeleton` -> hashed n-grams -> sparse multinomial logistic
      regression for the duration class {<1 s, 1–10 s, 10–60 s, >60 s} and a ridge fit of log duration; session-level
      split. Reports accuracy vs majority, tail recall/precision (>60 s), R² of log duration, and the pinball ratio at
      τ = 0.9 of per-predicted-class quantiles vs the global quantile — what a lease on the tool would use.

  python rung0/ceilings.py idle data/real/tracelab/spans_claude.jsonl
      Head I ceiling: think spans with observable attrs (user, hour, weekday, last tool, request bucket, prev_think,
      user_message_chars) -> per-key means for nested key sets and a ridge on one-hots; session-level split.

numpy only. Hashing needs no vocabulary; the classifier is a few dozen lines of SGD on sparse feature ids.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from observer import load  # noqa: E402

N_FEATS = 2 ** 16
CLASSES = ((0.0, 1.0, "<1s"), (1.0, 10.0, "1-10s"), (10.0, 60.0, "10-60s"), (60.0, math.inf, ">60s"))


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little") % N_FEATS


def hashed_features(skeleton: str) -> list[int]:
    """Unigram + bigram hashes over whitespace tokens, plus a length bucket. Deterministic; no vocabulary."""
    toks = skeleton.split()
    ids = [_h("u:" + t) for t in toks] + [_h("b:" + a + " " + b) for a, b in zip(toks, toks[1:])]
    ids.append(_h(f"len:{min(len(toks), 12)}"))
    return sorted(set(ids))


def duration_class(d: float) -> int:
    for i, (lo, hi, _) in enumerate(CLASSES):
        if lo <= d < hi:
            return i
    raise ValueError(d)


class SparseLogReg:
    """Multinomial logistic regression on sparse binary features; minibatch SGD with L2; numpy only."""

    def __init__(self, n_classes: int, n_feats: int = N_FEATS, lr: float = 0.5, l2: float = 1e-6, seed: int = 0):
        self.k, self.n, self.lr, self.l2 = n_classes, n_feats, lr, l2
        self.W = np.zeros((n_feats, n_classes), dtype=np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def _logits(self, ids: list[int]) -> np.ndarray:
        return self.W[ids].sum(axis=0) + self.b

    def predict_proba(self, X: list[list[int]]) -> np.ndarray:
        out = np.empty((len(X), self.k), dtype=np.float64)
        for i, ids in enumerate(X):
            z = self._logits(ids).astype(np.float64)
            z -= z.max()
            e = np.exp(z)
            out[i] = e / e.sum()
        return out

    def predict(self, X: list[list[int]]) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def fit(self, X: list[list[int]], y: np.ndarray, epochs: int = 5, batch: int = 64) -> "SparseLogReg":
        y = np.asarray(y)
        idx = np.arange(len(X))
        for ep in range(epochs):
            self.rng.shuffle(idx)
            lr = self.lr / (1 + ep)
            for s in range(0, len(idx), batch):
                b = idx[s:s + batch]
                grad_b = np.zeros(self.k, dtype=np.float32)
                rows, grads = [], []
                for i in b:
                    ids = X[i]
                    p = self._logits(ids).astype(np.float64)
                    p -= p.max()
                    p = np.exp(p)
                    p /= p.sum()
                    p[y[i]] -= 1.0                                  # dL/dz for cross-entropy
                    g = (p / len(b)).astype(np.float32)
                    rows.extend(ids)
                    grads.append(np.repeat(g[None, :], len(ids), axis=0))
                    grad_b += g
                rows = np.array(rows)
                gmat = np.concatenate(grads, axis=0)
                np.add.at(self.W, rows, -lr * gmat)
                self.W[rows] -= lr * self.l2 * self.W[rows]
                self.b -= lr * grad_b
        return self


def ridge_sparse(X: list[list[int]], y: np.ndarray, X_test: list[list[int]], lam: float = 1.0, epochs: int = 6, lr: float = 0.05) -> np.ndarray:
    """Ridge regression on sparse binary features by SGD; returns predictions for X_test."""
    w = np.zeros(N_FEATS, dtype=np.float32)
    b = float(y.mean())
    yc = y - b
    idx = np.arange(len(X))
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        rng.shuffle(idx)
        step = lr / (1 + ep)
        for i in idx:
            ids = X[i]
            err = float(w[ids].sum()) - yc[i]
            w[ids] -= step * (err + lam * w[ids] / len(X))
    return np.array([b + float(w[ids].sum()) for ids in X_test])


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


# ---- experiment 1: tool duration from command skeleton ------------------------------------------
def tools(path: Path, provider: str, max_calls: int) -> None:
    calls: list[tuple[str, str, float]] = []                       # (session, skeleton, duration s)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if provider != "all" and r["provider"] != provider:
                continue
            for t in r["tools"]:
                sk = t.get("command_skeleton")
                ms = t.get("tool_wall_latency_ms")
                if sk and ms is not None and t["tool_name"] == "Bash":
                    calls.append((r["session_id"], sk if isinstance(sk, str) else " ".join(map(str, sk)), max(float(ms) / 1000.0, 1e-3)))
    if len(calls) > max_calls:
        rng = np.random.default_rng(0)
        calls = [calls[i] for i in sorted(rng.choice(len(calls), max_calls, replace=False))]
    sessions = sorted({c[0] for c in calls})
    rng = np.random.default_rng(0)
    test_sessions = set(np.array(sessions)[rng.permutation(len(sessions))[: len(sessions) * 3 // 10]])
    train = [c for c in calls if c[0] not in test_sessions]
    test = [c for c in calls if c[0] in test_sessions]
    print(f"Bash calls with a skeleton: {len(calls)} ({len(sessions)} sessions); train {len(train)} / test {len(test)} (session-level split)")
    print("example skeletons:", [c[1][:60] for c in calls[:4]])
    Xtr, Xte = [hashed_features(c[1]) for c in train], [hashed_features(c[1]) for c in test]
    ytr, yte = np.array([duration_class(c[2]) for c in train]), np.array([duration_class(c[2]) for c in test])
    dtr, dte = np.array([c[2] for c in train]), np.array([c[2] for c in test])
    prior = Counter(ytr.tolist())
    print("class shares (train):", {CLASSES[k][2]: f"{v / len(ytr):.3f}" for k, v in sorted(prior.items())})
    clf = SparseLogReg(n_classes=4).fit(Xtr, ytr, epochs=6)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    maj = max(prior, key=prior.get)
    acc, acc_maj = float(np.mean(pred == yte)), float(np.mean(yte == maj))
    print(f"\nduration class from skeleton: accuracy {acc:.3f} (majority {acc_maj:.3f})")
    long_true, long_pred = yte == 3, pred == 3
    tp = int((long_true & long_pred).sum())
    print(f"tail (>60 s): recall {tp / max(1, long_true.sum()):.3f}, precision {tp / max(1, long_pred.sum()):.3f}; "
          f"share of tool TIME in calls flagged long: {dte[long_pred].sum() / dte.sum():.3f} (true >60 s share {dte[long_true].sum() / dte.sum():.3f})")
    p_long = proba[:, 3]
    order = np.argsort(-p_long)
    print("ranking by P(>60 s) — top deciles of test calls: share of calls truly >60 s, share of all tool time captured")
    cum_t = 0.0
    for d in range(1, 6):
        sel = order[(d - 1) * len(order) // 10: d * len(order) // 10]
        cum = order[: d * len(order) // 10]
        print(f"  decile {d}: >60 s share {np.mean(yte[sel] == 3):.3f}; time captured by top {10 * d}%: {dte[cum].sum() / dte.sum():.3f}")
    p_tr = clf.predict_proba(Xtr)[:, 3]
    edges = np.percentile(p_tr, np.linspace(0, 100, 11)[1:-1])
    bin_tr, bin_te = np.searchsorted(edges, p_tr), np.searchsorted(edges, p_long)
    for tau in (0.9, 0.95):
        g = float(np.percentile(dtr, 100 * tau))
        qb = {k: float(np.percentile(dtr[bin_tr == k], 100 * tau)) for k in range(10)}
        q = np.array([qb[k] for k in bin_te])
        ratio = pinball(dte, q, tau) / pinball(dte, np.full_like(dte, g), tau)
        print(f"  lease quantile τ={tau:.2f} per P(long) DECILE vs global — pinball ratio {ratio:.3f}, coverage {np.mean(dte <= q):.3f} vs {np.mean(dte <= g):.3f}, "
              f"reserved unit-seconds {q.sum() / g / len(q):.2f}x of global; top decile reserves {qb[9]:.0f} s, bottom {qb[0]:.1f} s")
    ltr, lte = np.log(dtr), np.log(dte)
    pr = ridge_sparse(Xtr, ltr, Xte)
    r2 = 1 - float(((lte - pr) ** 2).sum() / ((lte - lte.mean()) ** 2).sum())
    print(f"log duration from skeleton (ridge): R² {r2:.3f}")
    for tau in (0.5, 0.9, 0.95):
        g = float(np.percentile(dtr, 100 * tau))
        per = {}
        for k in range(4):
            sel = clf.predict(Xtr) == k
            per[k] = float(np.percentile(dtr[sel], 100 * tau)) if sel.sum() >= 20 else g
        q = np.array([per[k] for k in pred])
        ratio = pinball(dte, q, tau) / pinball(dte, np.full_like(dte, g), tau)
        print(f"  lease quantile τ={tau:.2f}: per-predicted-class vs global — pinball ratio {ratio:.3f}, coverage {np.mean(dte <= q):.3f} vs {np.mean(dte <= g):.3f}, "
              f"mean reserved {q.mean():.1f} s vs {g:.1f} s")


# ---- experiment 2: idle gap from who / when / what ------------------------------------------------
def idle(path: Path) -> None:
    spans = load([path])
    recs = []
    for s in spans:
        if s.op == "think" and s.duration > 0:
            a = s.attrs
            for k in ("user", "hour", "weekday", "prev_think", "user_message_chars", "last_tool", "request_idx"):
                if k not in a:
                    raise KeyError(f"think span lacks {k!r}; regenerate spans with the current adapter")
            recs.append((s.trace_id, a, math.log(s.duration)))
    sessions = sorted({r[0] for r in recs})
    rng = np.random.default_rng(0)
    test_sessions = set(np.array(sessions)[rng.permutation(len(sessions))[: len(sessions) * 3 // 10]])
    train = [r for r in recs if r[0] not in test_sessions]
    test = [r for r in recs if r[0] in test_sessions]
    ytr, yte = np.array([r[2] for r in train]), np.array([r[2] for r in test])
    print(f"think gaps: {len(recs)} in {len(sessions)} sessions; train {len(train)} / test {len(test)} (session-level split; users shared)")
    bucket = lambda i: "r0" if i == 0 else "r1-2" if i <= 2 else "r3-7" if i <= 7 else "r8+"
    keysets = {
        "last tool (E5 today)": lambda a: (a["last_tool"],),
        "user": lambda a: (a["user"],),
        "user, hour": lambda a: (a["user"], a["hour"]),
        "user, hour, weekday": lambda a: (a["user"], a["hour"], a["weekday"]),
        "user, hour, weekday, last tool, bucket": lambda a: (a["user"], a["hour"], a["weekday"], a["last_tool"], bucket(int(a["request_idx"]))),
        "prev think (log bucket)": lambda a: (int(math.log10(max(a["prev_think"], 1.0))),) if a["prev_think"] is not None else ("none",),
        "user message chars (log bucket)": lambda a: (int(math.log10(max(a["user_message_chars"], 1))),),
    }
    print(f"\n{'key set':45s} {'R²':>7s} {'pinball ratio τ=.8 (lower q)':>28s} {'τ=.9':>6s} {'states':>7s}")
    for name, kf in keysets.items():
        by = defaultdict(list)
        for _, a, y in train:
            by[kf(a)].append(y)
        gmean = float(ytr.mean())
        means = {k: float(np.mean(v)) for k, v in by.items() if len(v) >= 5}
        pred = np.array([means.get(kf(a), gmean) for _, a, _ in test])
        r2 = 1 - float(((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
        ratios = []
        for tau in (0.8, 0.9):
            lq = 100 * (1 - tau)
            g = float(np.percentile(ytr, lq))
            qs = {k: float(np.percentile(v, lq)) for k, v in by.items() if len(v) >= 10}
            q = np.array([qs.get(kf(a), g) for _, a, _ in test])
            ratios.append(pinball(yte, q, 1 - tau) / pinball(yte, np.full_like(yte, g), 1 - tau))
        print(f"{name:45s} {r2:7.3f} {ratios[0]:28.3f} {ratios[1]:6.3f} {len(means):7d}")
    feats = lambda a: [_h(f"user:{a['user']}"), _h(f"hour:{a['hour']}"), _h(f"wd:{a['weekday']}"), _h(f"tool:{a['last_tool']}"),
                       _h(f"bucket:{bucket(int(a['request_idx']))}"), _h(f"prev:{int(math.log10(max(a['prev_think'], 1.0))) if a['prev_think'] is not None else 'none'}"),
                       _h(f"umc:{int(math.log10(max(a['user_message_chars'], 1)))}"), _h(f"uh:{a['user']}:{a['hour']}")]
    pr = ridge_sparse([feats(a) for _, a, _ in train], ytr, [feats(a) for _, a, _ in test], lam=0.1, epochs=8, lr=0.02)
    r2 = 1 - float(((yte - pr) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
    print(f"\nridge on all one-hots (user, hour, weekday, last tool, bucket, prev think, user-message length, user×hour): R² {r2:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=("tools", "idle"))
    ap.add_argument("path")
    ap.add_argument("--provider", default="claude")
    ap.add_argument("--max-calls", type=int, default=200000)
    args = ap.parse_args()
    if args.experiment == "tools":
        tools(Path(args.path), args.provider, args.max_calls)
    else:
        idle(Path(args.path))


if __name__ == "__main__":
    main()
