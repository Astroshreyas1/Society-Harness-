"""E2 - predictability of the next node (H5). From node sequences:
  H_k = conditional entropy (bits) of the next node given the last k nodes, k = 0..3
  R_k = 1 - H_k / H_0   (entropy reduction; CacheScout's predictability metric)
  k-th order Markov top-1 accuracy at 1, 2 and 3 steps ahead (greedy chaining), trace-level split.
Decides the Predictor size and the lease horizon h (v2 §6.1 rung 0).

  python rung0/e2_predictability.py data/synthetic/hosted/*/traces.jsonl [--with-phase] [--split 0.7]
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict

import numpy as np

from observer import load, sequences


def cond_entropy(seqs: list[list[str]], k: int) -> float:
    ctx = defaultdict(Counter)
    for s in seqs:
        for i in range(k, len(s)):
            ctx[tuple(s[i - k:i])][s[i]] += 1
    total = sum(sum(c.values()) for c in ctx.values())
    h = 0.0
    for c in ctx.values():
        n = sum(c.values())
        for v in c.values():
            h -= (v / total) * math.log2(v / n)
    return h


class Markov:
    def __init__(self, k: int):
        self.k, self.ctx = k, defaultdict(Counter)

    def fit(self, seqs: list[list[str]]) -> "Markov":
        for s in seqs:
            for i in range(self.k, len(s)):
                self.ctx[tuple(s[i - self.k:i])][s[i]] += 1
        self.prior = Counter(t for s in seqs for t in s)
        return self

    def predict(self, hist: tuple[str, ...]) -> str:
        for j in range(min(self.k, len(hist)), -1, -1):            # back off to shorter contexts
            c = self.ctx.get(tuple(hist[len(hist) - j:]) if j else ())
            if c:
                return c.most_common(1)[0][0]
        return self.prior.most_common(1)[0][0]

    def accuracy(self, seqs: list[list[str]], steps: int) -> float:
        hit = n = 0
        for s in seqs:
            for i in range(self.k, len(s) - steps + 1):
                hist = tuple(s[max(0, i - self.k):i])
                pred = list(hist)
                for _ in range(steps):
                    pred.append(self.predict(tuple(pred[-self.k:]) if self.k else ()))
                n += 1
                hit += int(pred[-1] == s[i + steps - 1])
        return hit / n if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--with-phase", action="store_true", help="expose chat phases (upper bound if a labeler supplies them)")
    ap.add_argument("--phase-attr", default=None, help="use a labeler's phase annotation on chat spans (e.g. jev_phase from `agentsim annotate`)")
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seqs = list(sequences(load(args.traces), with_phase=args.with_phase, phase_attr=args.phase_attr).values())
    seqs = [s for s in seqs if len(s) >= 4]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(seqs))
    cut = int(args.split * len(seqs))
    train, test = [seqs[i] for i in idx[:cut]], [seqs[i] for i in idx[cut:]]
    vocab = Counter(t for s in seqs for t in s)
    print(f"traces={len(seqs)} nodes={sum(map(len, seqs))} vocab={len(vocab)} with_phase={args.with_phase} phase_attr={args.phase_attr}")
    print("top nodes:", ", ".join(f"{t}={c}" for t, c in vocab.most_common(8)))
    h0 = cond_entropy(seqs, 0)
    print(f"\n{'k':>2s} {'H_k bits':>9s} {'R_k':>6s} {'acc@1':>7s} {'acc@2':>7s} {'acc@3':>7s}")
    best = None
    for k in range(0, 4):
        hk = cond_entropy(seqs, k)
        m = Markov(k).fit(train)
        a1, a2, a3 = (m.accuracy(test, s) for s in (1, 2, 3))
        print(f"{k:2d} {hk:9.3f} {1 - hk / h0:6.3f} {a1:7.3f} {a2:7.3f} {a3:7.3f}")
        if best is None or a1 > best[1]:
            best = (k, a1, a3)
    k, a1, a3 = best
    print(f"\nH5 verdict: best 1-step accuracy {a1:.3f} at k={k}; 3-step {a3:.3f}.",
          "Short-horizon structure IS predictable (h = 1-2 steps)." if a1 >= 0.7 else
          "Below the 0.7 threshold - reservations must be near-zero-horizon; ship the reactive gate only (v2 §6.1 rung 0).")


if __name__ == "__main__":
    main()
