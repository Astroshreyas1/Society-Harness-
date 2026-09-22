"""The Needs Predictor (PREDICTOR_DESIGN §3–§4): a small multi-head model with distributional outputs, calibrated online.

    record = [ hashed sparse ids (2^16) | dense features (N_DENSE) ]
             -> embedding sum (d) + dense projection -> relu -> hidden (relu) -> heads
    heads:   S2, S3    softmax over the node vocabulary (the two nodes after the revealed ones)
             Q_tool    quantiles of log tool duration            (knots 0.05 / 0.2 / 0.5 / 0.8 / 0.95, monotone by construction)
             Q_out     quantiles of log(1 + output tokens)
             T_gap     quantiles of log seconds until the session's next chat is submitted
             I_gap     quantiles of log idle seconds after a request;  I_cold  P(idle >= cold start)
             R         P(the request aborts within H seconds of this decision point)
             G         softmax over spawn width 0..MAX_SPAWN

Every output the Reserver consumes is calibrated on the live stream: categorical heads by temperature scaling on a
rolling window (ECE reported), quantile heads by adaptive conformal inference per (head, key) — the requested level
moves so that the realised miss rate tracks 1 - tau (Gibbs & Candès). A bad model therefore yields wide honest
intervals, never confident wrong ones; the crude trackers' quantiles enter as features so the model can defer to them.

numpy only; per-example updates (Adagrad rows for the embedding table, Adam for the dense parameters), so the
same object trains offline on replayed traces and keeps learning online inside the gateway.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from .features import N_DENSE, N_FEATS, QUANTILES, Record
from .workload import MAX_SPAWN

D_EMB, HIDDEN = 48, 96
LOGIT_CLIP = 30.0


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation; |error| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"ppf needs 0 < p < 1, got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


Z_KNOTS = tuple(norm_ppf(t) for t in QUANTILES)


def quantile_at(knots: np.ndarray, level: float) -> float:
    """Interpolate / extrapolate the head's 5 knots linearly on the normal-score scale (lognormal-like tails)."""
    z = norm_ppf(level)
    if z <= Z_KNOTS[0]:
        s = (knots[1] - knots[0]) / (Z_KNOTS[1] - Z_KNOTS[0])
        return float(knots[0] + s * (z - Z_KNOTS[0]))
    if z >= Z_KNOTS[-1]:
        s = (knots[-1] - knots[-2]) / (Z_KNOTS[-1] - Z_KNOTS[-2])
        return float(knots[-1] + s * (z - Z_KNOTS[-1]))
    for i in range(len(Z_KNOTS) - 1):
        if Z_KNOTS[i] <= z <= Z_KNOTS[i + 1]:
            w = (z - Z_KNOTS[i]) / (Z_KNOTS[i + 1] - Z_KNOTS[i])
            return float(knots[i] * (1 - w) + knots[i + 1] * w)
    raise AssertionError("unreachable")


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-LOGIT_CLIP, min(LOGIT_CLIP, x))))


def knots_from_raw(r: np.ndarray) -> np.ndarray:
    """raw (5,) -> monotone knots q05 <= q20 <= q50 <= q80 <= q95: median r[2], increments softplus(r[3]), softplus(r[4]),
    decrements softplus(r[1]), softplus(r[0])."""
    sp = _softplus(r)
    m = float(r[2])
    dn2, dn1, up1, up2 = float(sp[0]), float(sp[1]), float(sp[3]), float(sp[4])
    return np.array([m - dn1 - dn2, m - dn1, m, m + up1, m + up1 + up2])


class Heads:
    def __init__(self, n_vocab: int):
        self.spec: dict[str, tuple[str, int]] = {
            "S2": ("softmax", n_vocab), "S3": ("softmax", n_vocab),
            "Q_tool": ("quantile", 5), "Q_out": ("quantile", 5), "T_gap": ("quantile", 5), "I_gap": ("quantile", 5),
            "I_cold": ("binary", 1), "R": ("binary", 1), "G": ("softmax", MAX_SPAWN + 1)}
        self.offset: dict[str, int] = {}
        n = 0
        for k, (_, size) in self.spec.items():
            self.offset[k] = n
            n += size
        self.n_out = n

    def slice(self, k: str) -> slice:
        return slice(self.offset[k], self.offset[k] + self.spec[k][1])


class NeedsModel:
    """Parameters, forward pass and per-example gradient step. Losses: cross-entropy (softmax), BCE (binary),
    summed pinball over the 5 knots (quantile). `learn` takes any subset of heads in `labels`."""

    def __init__(self, vocab: list[str], seed: int = 0, lr: float = 0.01, lr_emb: float = 0.05):
        self.vocab = list(vocab)
        self.vidx = {t: i for i, t in enumerate(self.vocab)}
        self.heads = Heads(len(self.vocab))
        rng = np.random.default_rng(seed)
        self.E = (rng.standard_normal((N_FEATS, D_EMB)) * 0.05).astype(np.float32)
        self.Wd = rng.standard_normal((N_DENSE, D_EMB)) * (1.0 / math.sqrt(N_DENSE))
        self.b0 = np.zeros(D_EMB)
        self.W1 = rng.standard_normal((D_EMB, HIDDEN)) * (1.0 / math.sqrt(D_EMB))
        self.b1 = np.zeros(HIDDEN)
        self.W2 = rng.standard_normal((HIDDEN, self.heads.n_out)) * (0.3 / math.sqrt(HIDDEN))
        self.b2 = np.zeros(self.heads.n_out)
        self.lr, self.lr_emb = lr, lr_emb
        self.G_E = np.zeros((N_FEATS,), dtype=np.float32) + 1e-3        # Adagrad accumulator per embedding row
        self.dense_params = ["Wd", "b0", "W1", "b1", "W2", "b2"]
        self._flatten()                                                  # one parameter vector: one Adam step per mini-batch
        self.t = 0
        self.batch, self._in_batch = 4, 0                                # dense gradients accumulate over 4 examples per step
        self.dense_mu = np.zeros(N_DENSE)          # running standardisation of the dense features
        self.dense_var = np.ones(N_DENSE)
        self.n_seen = 0

    def _flatten(self) -> None:
        """Store the dense parameters as views into one flat vector (theta) with flat Adam moments."""
        sizes = [getattr(self, k).size for k in self.dense_params]
        self.theta = np.zeros(sum(sizes))
        self.grad = np.zeros_like(self.theta)
        self.m_adam = np.zeros_like(self.theta)
        self.v_adam = np.zeros_like(self.theta)
        self._views: dict[str, tuple[int, int, tuple]] = {}
        off = 0
        for k, n in zip(self.dense_params, sizes):
            arr = getattr(self, k)
            self.theta[off:off + n] = arr.ravel()
            setattr(self, k, self.theta[off:off + n].reshape(arr.shape))
            self._views[k] = (off, off + n, arr.shape)
            off += n

    def _gview(self, k: str) -> np.ndarray:
        a, b, shape = self._views[k]
        return self.grad[a:b].reshape(shape)

    # ---- forward ----------------------------------------------------------------------------
    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.dense_mu) / np.sqrt(np.maximum(self.dense_var, 1.0))   # centre; shrink wide features, never amplify

    def _update_norm(self, x: np.ndarray) -> None:
        self.n_seen += 1
        a = 1.0 / min(self.n_seen, 2000)
        d = x - self.dense_mu
        self.dense_mu += a * d
        self.dense_var += a * (d * d - self.dense_var)

    def forward(self, rec: Record) -> tuple[np.ndarray, dict]:
        ids = rec.ids if rec.ids else [0]
        x = self._norm(rec.dense)
        e = self.E[ids].sum(axis=0).astype(np.float64) / math.sqrt(len(ids)) + x @ self.Wd + self.b0
        z = np.maximum(e, 0.0)
        h = np.maximum(z @ self.W1 + self.b1, 0.0)
        out = h @ self.W2 + self.b2
        return out, {"ids": ids, "x": x, "e": e, "z": z, "h": h}

    def outputs(self, out: np.ndarray) -> dict:
        """Raw head outputs -> probabilities (uncalibrated) and quantile knots."""
        res = {}
        for k, (kind, _) in self.heads.spec.items():
            o = out[self.heads.slice(k)]
            if kind == "softmax":
                o = o - o.max()
                p = np.exp(o)
                res[k] = p / p.sum()
            elif kind == "binary":
                res[k] = _sigmoid(float(o[0]))
            else:
                res[k] = knots_from_raw(o)
        return res

    # ---- backward ---------------------------------------------------------------------------
    def learn(self, rec: Record, labels: dict, weight: float = 1.0) -> float:
        out, cache = self.forward(rec)
        self._update_norm(rec.dense)
        g = np.zeros_like(out)
        loss = 0.0
        for k, y in labels.items():
            if k not in self.heads.spec:
                continue
            kind, _ = self.heads.spec[k]
            sl = self.heads.slice(k)
            o = out[sl]
            if kind == "softmax":
                idx = self.vidx[y] if isinstance(y, str) else int(y)
                oo = o - o.max()
                p = np.exp(oo)
                p /= p.sum()
                loss -= math.log(max(p[idx], 1e-12))
                p[idx] -= 1.0
                g[sl] = p
            elif kind == "binary":
                p = _sigmoid(float(o[0]))
                y = float(y)
                loss -= y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12))
                g[sl] = p - y
            else:
                y = float(y)
                q = knots_from_raw(o)
                dq = np.array([(1.0 - t) if y < qi else -t for t, qi in zip(QUANTILES, q)])      # d pinball / d q_i
                loss += float(sum(max(t * (y - qi), (t - 1) * (y - qi)) for t, qi in zip(QUANTILES, q)))
                # q = [m-dn1-dn2, m-dn1, m, m+up1, m+up1+up2]; raw r = [dn2, dn1, m, up1, up2] through softplus
                gr = np.zeros(5)
                gr[2] = dq.sum()
                sp = lambda v: 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, v))))                # softplus'
                gr[3] = (dq[3] + dq[4]) * sp(o[3])
                gr[4] = dq[4] * sp(o[4])
                gr[1] = -(dq[0] + dq[1]) * sp(o[1])
                gr[0] = -dq[0] * sp(o[0])
                g[sl] = gr
        if not np.any(g):
            return 0.0
        g *= weight
        h, z, x, ids = cache["h"], cache["z"], cache["x"], cache["ids"]
        self._gview("W2")[...] += np.outer(h, g)
        self._gview("b2")[...] += g
        gh = (self.W2 @ g) * (h > 0)
        self._gview("W1")[...] += np.outer(z, gh)
        self._gview("b1")[...] += gh
        gz = (self.W1 @ gh) * (z > 0)
        self._gview("Wd")[...] += np.outer(x, gz)
        self._gview("b0")[...] += gz
        self._in_batch += 1
        if self._in_batch >= self.batch:
            self._adam()
        gE = (gz / math.sqrt(len(ids))).astype(np.float32)
        rows = np.array(ids)
        self.G_E[rows] += float(gE @ gE)
        self.E[rows] -= (self.lr_emb / np.sqrt(self.G_E[rows]))[:, None] * gE[None, :]
        return float(loss)

    def _adam(self, beta1: float = 0.9, beta2: float = 0.999) -> None:
        self.t += 1
        g = self.grad
        g /= self._in_batch
        self._in_batch = 0
        np.clip(g, -10.0, 10.0, out=g)
        self.m_adam *= beta1
        self.m_adam += (1 - beta1) * g
        self.v_adam *= beta2
        self.v_adam += (1 - beta2) * g * g
        mh = self.m_adam / (1 - beta1 ** self.t)
        vh = self.v_adam / (1 - beta2 ** self.t)
        self.theta -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        g[...] = 0.0

    # ---- persistence --------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, E=self.E, Wd=self.Wd, b0=self.b0, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, G_E=self.G_E,
                            dense_mu=self.dense_mu, dense_var=self.dense_var, n_seen=np.array([self.n_seen]), t=np.array([self.t]),
                            vocab=np.array(json.dumps(self.vocab)))

    @classmethod
    def load(cls, path: Path) -> "NeedsModel":
        z = np.load(Path(path), allow_pickle=False)
        m = cls(json.loads(str(z["vocab"])))
        for k in ("E", "Wd", "b0", "W1", "b1", "W2", "b2", "G_E", "dense_mu", "dense_var"):
            getattr(m, k)[...] = z[k]                                  # dense arrays are views into theta: writes land there
        m.n_seen, m.t = int(z["n_seen"][0]), int(z["t"][0])
        return m


class Calibrator:
    """Temperature scaling for one categorical head on a rolling window; reports ECE (10 bins)."""

    def __init__(self, window: int = 2000, refit_every: int = 250):
        self.buf: deque = deque(maxlen=window)
        self.T = 1.0
        self.refit_every, self.since = refit_every, 0

    def add(self, logits: np.ndarray, label: int) -> None:
        self.buf.append((np.asarray(logits, dtype=np.float64), int(label)))
        self.since += 1
        if self.since >= self.refit_every and len(self.buf) >= 50:
            self.since = 0
            self._refit()

    def _matrix(self) -> tuple[np.ndarray, np.ndarray]:
        L = np.stack([l for l, _ in self.buf])
        y = np.array([yy for _, yy in self.buf])
        return L, y

    @staticmethod
    def _nll(L: np.ndarray, y: np.ndarray, T: float) -> float:
        o = L / T
        o = o - o.max(axis=1, keepdims=True)
        lse = np.log(np.exp(o).sum(axis=1))
        return float(np.mean(lse - o[np.arange(len(y)), y]))

    def _refit(self) -> None:
        L, y = self._matrix()
        a, b = math.log(0.25), math.log(8.0)                          # golden-section on log T
        phi = (math.sqrt(5) - 1) / 2
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = self._nll(L, y, math.exp(c)), self._nll(L, y, math.exp(d))
        for _ in range(20):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = self._nll(L, y, math.exp(c))
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = self._nll(L, y, math.exp(d))
        self.T = math.exp((a + b) / 2)

    def apply(self, logits: np.ndarray) -> np.ndarray:
        o = np.asarray(logits, dtype=np.float64) / self.T
        o = o - o.max()
        p = np.exp(o)
        return p / p.sum()

    def ece(self, bins: int = 10) -> float:
        if not self.buf:
            return float("nan")
        L, y = self._matrix()
        o = L / self.T
        o = o - o.max(axis=1, keepdims=True)
        P = np.exp(o)
        P /= P.sum(axis=1, keepdims=True)
        conf, hit = P.max(axis=1), (P.argmax(axis=1) == y).astype(float)
        e = 0.0
        for i in range(bins):
            m = (conf > i / bins) & (conf <= (i + 1) / bins)
            if m.any():
                e += m.mean() * abs(hit[m].mean() - conf[m].mean())
        return float(e)


class ConformalLevel:
    """Adaptive conformal inference for one (head, key) at target level tau: the requested level moves so the
    realised miss rate of the upper quantile tracks 1 - tau; symmetric for lower quantiles (tau < 0.5)."""

    def __init__(self, tau: float, gamma: float = 0.02, window: int = 300):
        if not 0.0 < tau < 1.0:
            raise ValueError("tau in (0, 1)")
        self.tau, self.gamma = tau, gamma
        self.alpha = 1.0 - tau if tau >= 0.5 else tau          # miss rate targeted on the relevant side
        self.level = tau
        self.miss: deque = deque(maxlen=window)
        self.n = 0

    def update(self, y: float, q_at_level: float) -> None:
        err = float(y > q_at_level) if self.tau >= 0.5 else float(y < q_at_level)
        self.miss.append(err)
        self.n += 1
        cur = (1.0 - self.level) if self.tau >= 0.5 else self.level     # the miss rate currently targeted
        cur = min(0.5, max(0.002, cur + self.gamma * (self.alpha - err)))
        self.level = (1.0 - cur) if self.tau >= 0.5 else cur

    def coverage(self) -> float:
        return 1.0 - float(np.mean(self.miss)) if self.miss else float("nan")


class NeedsPredictor:
    """The model plus its calibration layers and drift counters; the object a Reserver talks to."""

    def __init__(self, vocab: list[str], tau: float, seed: int = 0, model: NeedsModel | None = None, replay: int = 4, learn: bool = True):
        self.model = model if model is not None else NeedsModel(vocab, seed)
        if model is not None and list(model.vocab) != list(vocab):
            raise ValueError("loaded model's node vocabulary differs from this society's")
        self.tau = tau
        self.cal = {k: Calibrator() for k, (kind, _) in self.model.heads.spec.items() if kind in ("softmax", "binary")}
        self.conf: dict[tuple[str, tuple, float], ConformalLevel] = {}
        self.buffer: deque = deque(maxlen=4000)
        self.rng = np.random.default_rng([seed, 9])
        self.replay = replay
        self.learning = learn
        self.n_learned, self.loss_ema = 0, 0.0
        self.cover: dict[str, deque] = {}

    # ---- inference ----------------------------------------------------------------------------
    def predict(self, rec: Record) -> "Prediction":
        out, _ = self.model.forward(rec)
        return Prediction(self, rec, out)

    def conformal(self, head: str, key: tuple, tau: float) -> ConformalLevel:
        k = (head, key, tau)
        c = self.conf.get(k)
        if c is None:
            c = self.conf[k] = ConformalLevel(tau)
        return c

    # ---- learning -------------------------------------------------------------------------------
    def learn(self, rec: Record, labels: dict) -> None:
        out, _ = self.model.forward(rec)
        for k, y in labels.items():
            spec = self.model.heads.spec.get(k)
            if spec is None:
                continue
            kind, _ = spec
            o = out[self.model.heads.slice(k)]
            if kind == "softmax":
                self.cal[k].add(o, self.model.vidx[y] if isinstance(y, str) else int(y))
            elif kind == "binary":
                self.cal[k].add(np.array([0.0, float(o[0])]), int(float(y) > 0.5))
            else:                                                     # conformal bookkeeping on the *pre-update* prediction
                knots = knots_from_raw(o)
                for tau in (self.tau, 1.0 - self.tau, 0.5):
                    c = self.conformal(k, rec.key, tau)
                    c.update(float(y), quantile_at(knots, c.level))
                self.cover.setdefault(k, deque(maxlen=500)).append(float(float(y) <= quantile_at(knots, self.tau)))
        if not self.learning:
            return
        loss = self.model.learn(rec, labels)
        self.n_learned += 1
        self.loss_ema = 0.99 * self.loss_ema + 0.01 * loss if self.n_learned > 1 else loss
        self.buffer.append((rec, labels))
        for _ in range(min(self.replay, len(self.buffer) - 1)):
            r2, l2 = self.buffer[int(self.rng.integers(len(self.buffer)))]
            self.model.learn(r2, l2, weight=0.5)

    def fit(self, pairs: list[tuple[Record, dict]], epochs: int, seed: int = 0) -> list[float]:
        """Offline training: shuffled passes over labelled records; returns the mean loss per epoch."""
        rng = np.random.default_rng(seed)
        idx = np.arange(len(pairs))
        hist = []
        for _ in range(epochs):
            rng.shuffle(idx)
            tot = 0.0
            for i in idx:
                rec, labels = pairs[i]
                tot += self.model.learn(rec, labels)
            hist.append(tot / max(1, len(pairs)))
        for rec, labels in pairs[-2000:]:                             # warm the calibration layers on the tail of the data
            self.learn_calibration_only(rec, labels)
        return hist

    def learn_calibration_only(self, rec: Record, labels: dict) -> None:
        was = self.learning
        self.learning = False
        try:
            self.learn(rec, labels)
        finally:
            self.learning = was

    def report(self) -> dict:
        return {"learned": self.n_learned, "loss_ema": round(self.loss_ema, 4),
                "ece": {k: round(c.ece(), 4) for k, c in self.cal.items() if c.buf},
                "temperature": {k: round(c.T, 3) for k, c in self.cal.items() if c.buf},
                "coverage_at_tau": {k: round(float(np.mean(v)), 3) for k, v in self.cover.items() if v},
                "conformal_levels": {f"{h}:{'/'.join(map(str, key))}@{t}": round(c.level, 3) for (h, key, t), c in self.conf.items() if c.n >= 50}}


class Prediction:
    """Calibrated view of one forward pass."""

    def __init__(self, pred: NeedsPredictor, rec: Record, out: np.ndarray):
        self.pred, self.rec, self.out = pred, rec, out
        self.raw = pred.model.outputs(out)

    def probs(self, head: str) -> np.ndarray:
        o = self.out[self.pred.model.heads.slice(head)]
        kind = self.pred.model.heads.spec[head][0]
        if kind == "softmax":
            return self.pred.cal[head].apply(o)
        if kind == "binary":
            return self.pred.cal[head].apply(np.array([0.0, float(o[0])]))
        raise ValueError(f"{head} is not categorical")

    def p(self, head: str, cls) -> float:
        p = self.probs(head)
        if isinstance(cls, str):
            return float(p[self.pred.model.vidx[cls]])
        return float(p[int(cls)])

    def p_true(self, head: str) -> float:
        return float(self.probs(head)[1])

    def knots(self, head: str) -> np.ndarray:
        return self.raw[head]

    def quantile(self, head: str, tau: float, key: tuple | None = None, conformal: bool = True) -> float:
        """The tau-quantile (log scale) — conformalised: the requested level is the one that has been covering
        at rate tau on the live stream for this (head, key)."""
        level = tau
        if conformal:
            c = self.pred.conformal(head, key if key is not None else self.rec.key, tau)
            level = c.level
        return quantile_at(self.raw[head], level)

    def q(self, head: str, tau: float, key: tuple | None = None, conformal: bool = True) -> float:
        """The same on the natural scale (seconds / tokens); Q_out is log1p so it subtracts 1."""
        v = math.exp(self.quantile(head, tau, key, conformal))
        return max(0.0, v - 1.0) if head == "Q_out" else v

    def mixture_quantile(self, head_by_type: dict[str, tuple[str, tuple]], type_probs: dict[str, float], tau: float) -> float:
        """tau-quantile of the mixture sum_j P(type_j) * D_j, each D_j known through its knots: the smallest x with
        sum_j P_j * F_j(x) >= tau, by bisection on the log scale over the components' quantile functions."""
        comps = [(pj, self.raw[head], key, head) for t, (head, key) in head_by_type.items() for pj in [type_probs.get(t, 0.0)] if pj > 0]
        if not comps:
            return 0.0
        z = sum(pj for pj, *_ in comps)
        lo = min(quantile_at(k, 0.01) for _, k, _, _ in comps)
        hi = max(quantile_at(k, 0.995) for _, k, _, _ in comps)

        def cdf(x: float) -> float:
            tot = 0.0
            for pj, knots, key, head in comps:
                a, b = 0.001, 0.999
                for _ in range(30):                                   # invert the quantile function numerically
                    mid = (a + b) / 2
                    if quantile_at(knots, mid) < x:
                        a = mid
                    else:
                        b = mid
                tot += pj / z * a
            return tot

        for _ in range(40):
            mid = (lo + hi) / 2
            if cdf(mid) < tau:
                lo = mid
            else:
                hi = mid
        return hi
