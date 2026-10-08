"""Knowledge-tracing models: EMA baseline, BKT (EM-fitted), PFA (logistic regression)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.ml.skills import SKILL_IDS

MASTERED_AT = 0.95


# --- EMA -------------------------------------------------------------------

def ema_predict(seq: list[int], alpha: float = 0.3, init: float = 0.5) -> list[float]:
    """Prediction before each observation."""
    preds, m = [], init
    for o in seq:
        preds.append(m)
        m = m + alpha * (o - m)
    return preds


# --- BKT -------------------------------------------------------------------

@dataclass
class BKTParams:
    L0: float
    T: float
    G: float
    S: float

    def as_dict(self) -> dict:
        return {k: round(float(getattr(self, k)), 4) for k in ("L0", "T", "G", "S")}


def bkt_trace(p: BKTParams, seq: list[int]) -> tuple[list[float], list[float]]:
    """Returns (P(correct) predicted before each obs, P(learned) after each obs)."""
    L = p.L0
    preds, mastery = [], []
    for o in seq:
        preds.append(float(L * (1 - p.S) + (1 - L) * p.G))
        if o:
            post = L * (1 - p.S) / (L * (1 - p.S) + (1 - L) * p.G)
        else:
            post = L * p.S / (L * p.S + (1 - L) * (1 - p.G))
        L = post + (1 - post) * p.T
        mastery.append(float(L))
    return preds, mastery


def fit_bkt(sequences: list[list[int]], n_iter: int = 60, max_guess: float = 0.3, max_slip: float = 0.3,
            max_len: int = 400) -> BKTParams:
    """Baum-Welch (EM) for the 2-state BKT HMM without forgetting.

    Sequences are padded into an (N x T) matrix and the forward-backward pass is
    vectorised across sequences. Padded steps get emission probability 1 for
    both states, which leaves alpha/beta unchanged, and are masked out of the
    sufficient statistics. Sequences longer than `max_len` are truncated for
    fitting only.
    """
    seqs = [np.asarray(s[:max_len], dtype=np.int8) for s in sequences if len(s)]
    if not seqs:
        return BKTParams(0.4, 0.1, 0.2, 0.1)
    N, T = len(seqs), max(len(s) for s in seqs)
    O = np.zeros((N, T), dtype=np.int8)
    M = np.zeros((N, T), dtype=bool)
    for i, s in enumerate(seqs):
        O[i, : len(s)] = s
        M[i, : len(s)] = True
    trans_mask = M[:, 1:]  # transition t -> t+1 exists

    p = BKTParams(0.4, 0.1, 0.2, 0.1)
    for _ in range(n_iter):
        # emissions: B[n, t, state]
        B = np.empty((N, T, 2))
        B[..., 0] = np.where(O == 1, p.G, 1 - p.G)
        B[..., 1] = np.where(O == 1, 1 - p.S, p.S)
        B[~M] = 1.0

        alpha = np.empty((N, T, 2)); c = np.empty((N, T))
        a = np.stack([np.full(N, 1 - p.L0), np.full(N, p.L0)], axis=1) * B[:, 0]
        c[:, 0] = a.sum(1); alpha[:, 0] = a / c[:, :1]
        for t in range(1, T):
            prev = alpha[:, t - 1]
            a = np.stack([prev[:, 0] * (1 - p.T), prev[:, 0] * p.T + prev[:, 1]], axis=1) * B[:, t]
            c[:, t] = a.sum(1); alpha[:, t] = a / c[:, t:t + 1]

        beta = np.ones((N, T, 2))
        for t in range(T - 2, -1, -1):
            nb = B[:, t + 1] * beta[:, t + 1]
            beta[:, t, 0] = ((1 - p.T) * nb[:, 0] + p.T * nb[:, 1]) / c[:, t + 1]
            beta[:, t, 1] = nb[:, 1] / c[:, t + 1]

        gamma = alpha * beta
        gamma /= gamma.sum(-1, keepdims=True)
        xi01 = alpha[:, :-1, 0] * p.T * B[:, 1:, 1] * beta[:, 1:, 1] / c[:, 1:]

        L0 = gamma[:, 0, 1].mean()
        T_ = (xi01 * trans_mask).sum() / max((gamma[:, :-1, 0] * trans_mask).sum(), 1e-9)
        w = gamma * M[..., None]
        correct = (w * O[..., None]).sum((0, 1))
        total = w.sum((0, 1))
        G = correct[0] / max(total[0], 1e-9)
        S = 1 - correct[1] / max(total[1], 1e-9)
        p = BKTParams(float(np.clip(L0, 0.01, 0.99)), float(np.clip(T_, 0.005, 0.6)),
                      float(np.clip(G, 0.01, max_guess)), float(np.clip(S, 0.01, max_slip)))
    return p


# --- PFA -------------------------------------------------------------------

def pfa_features(skill: str, successes: int, failures: int, item_b: float) -> np.ndarray:
    """One-hot skill intercept + per-skill success/failure counts + item difficulty."""
    k = len(SKILL_IDS)
    v = np.zeros(3 * k + 1)
    i = SKILL_IDS.index(skill)
    v[i] = 1.0
    v[k + i] = np.log1p(successes)
    v[2 * k + i] = np.log1p(failures)
    v[-1] = item_b
    return v


def pfa_design(attempts, item_b: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds (X, y, learner) in time order, counts reset per learner."""
    counts: dict = defaultdict(lambda: [0, 0])
    X, y, g = [], [], []
    for a in sorted(attempts, key=lambda a: (a.learner, a.t)):
        s, f = counts[(a.learner, a.skill)]
        X.append(pfa_features(a.skill, s, f, item_b.get(a.item_id, 0.0)))
        y.append(a.correct)
        g.append(a.learner)
        counts[(a.learner, a.skill)][0 if a.correct else 1] += 1
    return np.array(X), np.array(y), np.array(g)


def fit_pfa(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(X, y)
