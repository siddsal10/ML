"""Learner model + adaptive item recommender used at runtime.

- Mastery per skill = BKT P(learned), replayed over the student's observed
  outcomes (practice answers and diagnosed submissions) with the per-skill
  parameters fitted by EM (app.ml.tracing.train).
- P(correct) for a candidate item = PFA logistic regression using the
  student's success/failure counts on that skill and the item's IRT difficulty.
- Policy: practise the diagnosed skill (or the least-mastered unmastered skill);
  choose items whose predicted P(correct) is closest to TARGET_P, preferring
  a spread of template levels and skipping items answered correctly before.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.ml.artifacts import MODEL_DIR, load_bundle
from app.ml.irt.item_bank import Item, build_item_bank
from app.ml.skills import SKILL_IDS
from app.ml.tracing.models import MASTERED_AT, BKTParams, bkt_trace, pfa_features

KT_MODEL_PATH = MODEL_DIR / "knowledge_tracing.joblib"
TARGET_P = 0.7
DEFAULT_BKT = BKTParams(0.3, 0.1, 0.2, 0.1)


@dataclass
class SkillState:
    skill: str
    mastery: float
    attempts: int
    successes: int
    failures: int
    history: list[float] = field(default_factory=list)  # mastery after each observation

    @property
    def mastered(self) -> bool:
        return self.mastery >= MASTERED_AT


class LearnerModel:
    def __init__(self, bundle: dict | None = None):
        self.bundle = bundle if bundle is not None else load_bundle(str(KT_MODEL_PATH))
        self.trained = self.bundle is not None
        if self.trained:
            self.bkt = {s: BKTParams(**p) for s, p in self.bundle["bkt_params"].items()}
            self.pfa = self.bundle["pfa"]
            self.item_b = self.bundle["item_difficulty"]
        else:
            self.bkt = {s: DEFAULT_BKT for s in SKILL_IDS}
            self.pfa = None
            self.item_b = {}

    def skill_states(self, outcomes: dict[str, list[int]]) -> dict[str, SkillState]:
        states = {}
        for s in SKILL_IDS:
            seq = outcomes.get(s, [])
            p = self.bkt[s]
            _, hist = bkt_trace(p, seq)
            states[s] = SkillState(s, float(hist[-1] if hist else p.L0), len(seq), int(sum(seq)), len(seq) - int(sum(seq)), hist)
        return states

    def p_correct(self, item: Item, state: SkillState) -> float:
        if self.pfa is None:
            p = self.bkt[item.skill]
            return float(state.mastery * (1 - p.S) + (1 - state.mastery) * p.G)
        x = pfa_features(item.skill, state.successes, state.failures, self.item_b.get(item.id, 0.0))
        return float(self.pfa.predict_proba(x.reshape(1, -1))[0, 1])

    def target_skill(self, states: dict[str, SkillState], diagnosed_skill: str | None = None) -> str:
        if diagnosed_skill and not states[diagnosed_skill].mastered:
            return diagnosed_skill
        open_skills = [s for s in SKILL_IDS if not states[s].mastered] or SKILL_IDS
        return min(open_skills, key=lambda s: states[s].mastery)

    def recommend(self, skill: str, state: SkillState, n: int = 3, exclude: set[str] | None = None,
                  bank: tuple[Item, ...] | None = None) -> list[tuple[Item, float]]:
        exclude = exclude or set()
        candidates = [it for it in (bank or build_item_bank()) if it.skill == skill and it.id not in exclude]
        scored = sorted(((it, self.p_correct(it, state)) for it in candidates), key=lambda t: abs(t[1] - TARGET_P))
        chosen, levels = [], set()
        for it, p in scored:  # first pass: one per template level
            if it.level not in levels:
                chosen.append((it, p)); levels.add(it.level)
            if len(chosen) == n:
                break
        for it, p in scored:
            if len(chosen) == n:
                break
            if all(it.id != c.id for c, _ in chosen):
                chosen.append((it, p))
        return sorted(chosen, key=lambda t: -t[1])  # easiest first


def policy_choose(policy: str, model: LearnerModel, outcomes: dict[str, list[int]], step: int,
                  rng: np.random.Generator, bank: tuple[Item, ...]) -> Item:
    """Item-selection policies compared in the simulation study."""
    if policy == "random":
        return bank[int(rng.integers(len(bank)))]
    if policy == "round_robin":
        skill = SKILL_IDS[step % len(SKILL_IDS)]
        pool = [it for it in bank if it.skill == skill]
        return pool[int(rng.integers(len(pool)))]
    states = model.skill_states(outcomes)
    skill = model.target_skill(states)
    return model.recommend(skill, states[skill], n=1, bank=bank)[0][0]
