"""Synthetic learners for developing and evaluating knowledge tracing,
IRT, the recommender and clustering.

Generative model (documented so results can be interpreted):
- each learner has a hidden learned/unlearned state per skill (BKT dynamics
  with per-skill ground-truth L0, T, G, S) and a general ability theta;
- P(correct) = sigmoid(logit(base) + 0.6*theta - b_item), where base is
  (1 - S) if the skill is learned else G;
- after every attempt an unlearned skill becomes learned with probability T;
- item difficulty b_item = linear function of the item's features + noise.
Archetypes lower L0 on specific skills to create recognizable groups.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.irt.item_bank import FEATURE_NAMES, build_item_bank
from app.ml.skills import SKILL_IDS

TRUE_BKT = {  # L0, T, G, S
    "S1": (0.60, 0.20, 0.25, 0.08),
    "S2": (0.45, 0.15, 0.20, 0.10),
    "S3": (0.30, 0.12, 0.20, 0.10),
    "S4": (0.30, 0.12, 0.15, 0.10),
    "S5": (0.40, 0.15, 0.15, 0.08),
    "S6": (0.25, 0.10, 0.12, 0.10),
    "S7": (0.20, 0.10, 0.12, 0.12),
    "S8": (0.15, 0.08, 0.10, 0.12),
}
_DIFF_WEIGHTS = np.array([0.5, 0.3, 0.25, 0.2, 0.2, 0.25, 0.3])  # aligned with FEATURE_NAMES

ARCHETYPES = {
    "balanced": {},
    "negatives": {"S3": 0.05, "S8": 0.03},
    "moving_terms": {"S7": 0.03, "S6": 0.08},
    "foundations": {"S1": 0.2, "S2": 0.1, "S4": 0.08},
    "strong": {s: 0.85 for s in SKILL_IDS},
}


@dataclass
class Attempt:
    learner: int
    skill: str
    item_id: str
    correct: int
    t: int


def true_item_difficulty(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed + 991)
    items = build_item_bank()
    X = np.array([[it.features[f] for f in FEATURE_NAMES] for it in items])
    b = X @ _DIFF_WEIGHTS
    b = b - b.mean() + rng.normal(0, 0.25, len(items))
    return {it.id: float(v) for it, v in zip(items, b)}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    return np.log(p / (1 - p))


class SimLearner:
    def __init__(self, idx: int, rng: np.random.Generator, archetype: str = "balanced"):
        self.idx = idx
        self.archetype = archetype
        self.theta = float(rng.normal(0, 0.8))
        overrides = ARCHETYPES[archetype]
        self.learned = {s: bool(rng.random() < overrides.get(s, TRUE_BKT[s][0])) for s in SKILL_IDS}
        self.rng = rng

    def answer(self, skill: str, b_item: float) -> int:
        _, T, G, S = TRUE_BKT[skill]
        base = (1 - S) if self.learned[skill] else G
        p = _sigmoid(_logit(base) + 0.6 * self.theta - b_item)
        correct = int(self.rng.random() < p)
        if not self.learned[skill] and self.rng.random() < T:
            self.learned[skill] = True
        return correct


def simulate_attempts(n_learners: int = 500, seed: int = 0, attempts_range=(10, 30),
                      archetype_mix: dict[str, float] | None = None) -> tuple[list[Attempt], list[SimLearner]]:
    """Each learner practices every skill with uniformly random items from that skill."""
    rng = np.random.default_rng(seed)
    b_true = true_item_difficulty(seed)
    items_by_skill: dict[str, list[str]] = {s: [] for s in SKILL_IDS}
    for it in build_item_bank():
        items_by_skill[it.skill].append(it.id)
    names = list((archetype_mix or {"balanced": 1.0}).keys())
    probs = np.array(list((archetype_mix or {"balanced": 1.0}).values()), dtype=float)
    probs /= probs.sum()

    attempts, learners = [], []
    for li in range(n_learners):
        learner = SimLearner(li, rng, str(rng.choice(names, p=probs)))
        learners.append(learner)
        t = 0
        schedule = []
        for s in SKILL_IDS:
            schedule += [s] * int(rng.integers(*attempts_range))
        rng.shuffle(schedule)
        for s in schedule:
            item = str(rng.choice(items_by_skill[s]))
            attempts.append(Attempt(li, s, item, learner.answer(s, b_true[item]), t))
            t += 1
    return attempts, learners
