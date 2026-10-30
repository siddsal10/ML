"""Is the adaptive policy's advantage an artefact of the learner simulator?

Review-3 compared practice policies on learners from our own BKT + IRT
simulator, which favours a BKT-based policy. Here the deployed learner model
(BKT for S1-S7 fitted on real ASSISTments students, S8 simulated) chooses the
items, and the simulated learners are varied:

    original         the Review-3 simulator (hand-set parameters)
    real_calibrated  S1-S7 prior/learn/guess/slip set to the values fitted on real students
    forgetting       a learned skill is forgotten with probability 0.02 whenever another skill is practised
    heterogeneous    each learner's learning rates scaled by a log-normal factor (sd 0.6 on log scale)
    all_combined     real_calibrated + forgetting + heterogeneous

Outcomes after 40 items: share of the 8 skills truly learned (latent) and the
expected score on a 16-item post-test (2 mid-difficulty items per skill),
which is what a classroom study can observe. The between-learner effect size
of adaptive vs random practice then sizes that study (docs/REAL_WORLD_VALIDATION.md).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats

from app.ml.irt.item_bank import build_item_bank
from app.ml.irt.recommender import LearnerModel, policy_choose
from app.ml.skills import SKILL_IDS
from app.ml.tracing.simulate import TRUE_BKT, true_item_difficulty

POLICIES = ("random", "round_robin", "adaptive")
FORGET = 0.02
T_SPREAD = 0.6


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    return np.log(p / (1 - p))


class Learner:
    def __init__(self, rng: np.random.Generator, params: dict[str, tuple], forget: float = 0.0, heterogeneous: bool = False):
        self.rng = rng
        self.theta = float(rng.normal(0, 0.8))
        mult = float(np.exp(rng.normal(0, T_SPREAD))) if heterogeneous else 1.0
        self.params = {s: (L0, min(0.9, T * mult), G, S) for s, (L0, T, G, S) in params.items()}
        self.learned = {s: bool(rng.random() < self.params[s][0]) for s in SKILL_IDS}
        self.forget = forget

    def p_correct(self, skill: str, b_item: float) -> float:
        _, _, G, S = self.params[skill]
        return float(_sigmoid(_logit((1 - S) if self.learned[skill] else G) + 0.6 * self.theta - b_item))

    def answer(self, skill: str, b_item: float) -> int:
        correct = int(self.rng.random() < self.p_correct(skill, b_item))
        if not self.learned[skill] and self.rng.random() < self.params[skill][1]:
            self.learned[skill] = True
        if self.forget:
            for s in SKILL_IDS:
                if s != skill and self.learned[s] and self.rng.random() < self.forget:
                    self.learned[s] = False
        return correct


def variants(real_params: dict[str, dict]) -> dict[str, dict]:
    real = {s: tuple(real_params[s][k] for k in ("L0", "T", "G", "S")) if s in real_params else TRUE_BKT[s]
            for s in SKILL_IDS}
    return {
        "original": {"params": TRUE_BKT},
        "real_calibrated": {"params": real},
        "forgetting": {"params": TRUE_BKT, "forget": FORGET},
        "heterogeneous": {"params": TRUE_BKT, "heterogeneous": True},
        "all_combined": {"params": real, "forget": FORGET, "heterogeneous": True},
    }


def n_per_arm(d: float, rho: float = 0.0, alpha: float = 0.05, power: float = 0.8) -> int | None:
    """Students per arm for a two-sided two-sample t-test; rho is the pre/post-test
    correlation used by an ANCOVA adjustment (effect size grows by 1/sqrt(1 - rho^2))."""
    if d <= 0:
        return None
    d_eff = d / np.sqrt(1 - rho ** 2)
    for n in range(4, 20_000):
        df = 2 * n - 2
        t = stats.t.ppf(1 - alpha / 2, df)
        nc = d_eff * np.sqrt(n / 2)
        if 1 - stats.nct.cdf(t, df, nc) + stats.nct.cdf(-t, df, nc) >= power:
            return n
    return None


def simulate(model: LearnerModel, spec: dict, seeds=(0, 1, 2, 3, 4), n_learners: int = 150, budget: int = 40) -> dict:
    bank = build_item_bank()
    b_true = true_item_difficulty(0)
    post_items = {}
    for s in SKILL_IDS:  # two mid-difficulty items per skill
        items = sorted((it for it in bank if it.skill == s), key=lambda it: b_true[it.id])
        post_items[s] = [items[len(items) // 2 - 1], items[len(items) // 2]]
    per_policy = {}
    for policy in POLICIES:
        learned, post, seed_means = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            seed_learned = []
            for i in range(n_learners):
                learner = Learner(np.random.default_rng(seed * 10_000 + i), spec["params"], spec.get("forget", 0.0),
                                  spec.get("heterogeneous", False))
                outcomes: dict[str, list[int]] = defaultdict(list)
                for step in range(budget):
                    item = policy_choose(policy, model, outcomes, step, rng, bank)
                    outcomes[item.skill].append(learner.answer(item.skill, b_true[item.id]))
                seed_learned.append(np.mean(list(learner.learned.values())))
                post.append(np.mean([learner.p_correct(s, b_true[it.id]) for s in SKILL_IDS for it in post_items[s]]))
            learned += seed_learned
            seed_means.append(np.mean(seed_learned))
        per_policy[policy] = {"learned": np.array(learned), "post": np.array(post), "seed_means": seed_means}

    out = {}
    for policy, r in per_policy.items():
        out[policy] = {"skills_learned_mean": round(float(r["learned"].mean()), 4),
                       "skills_learned_seed_std": round(float(np.std(r["seed_means"])), 4),
                       "post_test_mean": round(float(r["post"].mean()), 4)}
    a, z = per_policy["adaptive"], per_policy["random"]
    pooled = np.sqrt((a["post"].var(ddof=1) + z["post"].var(ddof=1)) / 2)
    d = float((a["post"].mean() - z["post"].mean()) / pooled) if pooled > 0 else 0.0
    return {"policies": out,
            "adaptive_minus_random_skills_learned": round(out["adaptive"]["skills_learned_mean"] - out["random"]["skills_learned_mean"], 4),
            "adaptive_minus_random_post_test": round(out["adaptive"]["post_test_mean"] - out["random"]["post_test_mean"], 4),
            "post_test_effect_size_d": round(d, 3)}


def run(log=print, n_learners: int = 150, seeds=(0, 1, 2, 3, 4)) -> dict:
    model = LearnerModel()
    results = {}
    for name, spec in variants(model.bundle["bkt_params"]).items():
        results[name] = simulate(model, spec, seeds=seeds, n_learners=n_learners)
        log(f"  {name:16s} learned: adaptive {results[name]['policies']['adaptive']['skills_learned_mean']:.3f} "
            f"vs random {results[name]['policies']['random']['skills_learned_mean']:.3f}; post-test d={results[name]['post_test_effect_size_d']}")
    planning = {}
    for label, d in [*((f"simulated_{k}", r["post_test_effect_size_d"]) for k, r in results.items()),
                     ("conventional_0.3", 0.3), ("conventional_0.5", 0.5)]:
        planning[label] = {"d": round(d, 3), "n_per_arm_t_test": n_per_arm(d), "n_per_arm_ancova_rho_0.5": n_per_arm(d, rho=0.5)}
    return {
        "task": "practice_policy_robustness_to_simulator",
        "policy_model": "deployed learner model (BKT S1-S7 fitted on ASSISTments 2009, S8 simulated; PFA + IRT item choice)",
        "setup": f"{n_learners} learners x {len(seeds)} seeds per policy and variant, 40 items each; same learner seeds across policies",
        "variants": {"forgetting_prob": FORGET, "heterogeneous_log_sd": T_SPREAD},
        "results": results,
        "classroom_study_sample_size": planning,
        "note": "Still simulations: they test robustness to modelling assumptions, not real learning gains.",
    }
