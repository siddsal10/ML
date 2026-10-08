"""Train + evaluate knowledge tracing (EMA vs BKT vs PFA), Rasch IRT, the
difficulty model, and the recommender policy - all on simulated learners."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from app.ml.artifacts import save_bundle
from app.ml.irt.item_bank import FEATURE_NAMES, build_item_bank
from app.ml.irt.rasch import fit_difficulty_model, fit_rasch
from app.ml.irt.recommender import KT_MODEL_PATH, LearnerModel, policy_choose
from app.ml.skills import SKILL_IDS
from app.ml.tracing.models import bkt_trace, ema_predict, fit_bkt, fit_pfa, pfa_design
from app.ml.tracing.simulate import TRUE_BKT, SimLearner, simulate_attempts, true_item_difficulty


def _metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
    }


def _sequences(attempts, learners: set[int]) -> dict:
    seqs = defaultdict(list)
    for a in sorted(attempts, key=lambda a: (a.learner, a.t)):
        if a.learner in learners:
            seqs[(a.learner, a.skill)].append(a.correct)
    return seqs


def simulate_policies(model: LearnerModel, seeds=(0, 1, 2, 3, 4), n_learners: int = 150, budget: int = 40) -> dict:
    bank = build_item_bank()
    out = {}
    for policy in ("random", "round_robin", "adaptive"):
        per_seed_learned, per_seed_waste = [], []
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            b_true = true_item_difficulty(0)
            learned, wasted = [], []
            for i in range(n_learners):
                learner = SimLearner(i, np.random.default_rng(seed * 10_000 + i))
                outcomes: dict[str, list[int]] = defaultdict(list)
                waste = 0
                for step in range(budget):
                    item = policy_choose(policy, model, outcomes, step, rng, bank)
                    waste += learner.learned[item.skill]
                    outcomes[item.skill].append(learner.answer(item.skill, b_true[item.id]))
                learned.append(np.mean(list(learner.learned.values())))
                wasted.append(waste / budget)
            per_seed_learned.append(np.mean(learned))
            per_seed_waste.append(np.mean(wasted))
        out[policy] = {
            "skills_mastered_frac_mean": round(float(np.mean(per_seed_learned)), 4),
            "skills_mastered_frac_std": round(float(np.std(per_seed_learned)), 4),
            "practice_on_already_learned_frac": round(float(np.mean(per_seed_waste)), 4),
        }
    return out


def train_tracing(seed: int = 0, n_learners: int = 600, out_path: Path = KT_MODEL_PATH, quick: bool = False) -> dict:
    attempts, _ = simulate_attempts(n_learners=n_learners, seed=seed)
    learner_ids = np.arange(n_learners)
    rng = np.random.default_rng(seed)
    rng.shuffle(learner_ids)
    n_test = n_learners // 5
    test_l, train_l = set(learner_ids[:n_test].tolist()), set(learner_ids[n_test:].tolist())
    train_att = [a for a in attempts if a.learner in train_l]
    test_att = [a for a in attempts if a.learner in test_l]

    # --- IRT on training learners ---
    bank = build_item_bank()
    item_index = {it.id: i for i, it in enumerate(bank)}
    learner_map = {l: i for i, l in enumerate(sorted(train_l))}
    theta, b = fit_rasch(
        np.array([learner_map[a.learner] for a in train_att]),
        np.array([item_index[a.item_id] for a in train_att]),
        np.array([a.correct for a in train_att], dtype=float),
        len(learner_map), len(bank),
    )
    b_true = true_item_difficulty(seed)
    b_true_vec = np.array([b_true[it.id] for it in bank])
    X_items = np.array([[it.features[f] for f in FEATURE_NAMES] for it in bank])
    diff_model, diff_r2 = fit_difficulty_model(X_items, b)
    item_b = {it.id: float(v) for it, v in zip(bank, b)}

    # --- BKT per skill ---
    train_seqs = _sequences(attempts, train_l)
    test_seqs = _sequences(attempts, test_l)
    bkt = {s: fit_bkt([seq for (l, sk), seq in train_seqs.items() if sk == s], n_iter=25 if quick else 60) for s in SKILL_IDS}

    y_all, p_ema, p_bkt = [], [], []
    for (l, s), seq in test_seqs.items():
        y_all += seq
        p_ema += ema_predict(seq)
        p_bkt += bkt_trace(bkt[s], seq)[0]

    # --- PFA ---
    X_tr, y_tr, _ = pfa_design(train_att, item_b)
    pfa = fit_pfa(X_tr, y_tr)
    X_te, y_te, _ = pfa_design(test_att, item_b)
    p_pfa = pfa.predict_proba(X_te)[:, 1]

    # BKT/EMA sequences are grouped by (learner, skill) - align PFA metrics on its own ordering.
    models = {"ema_baseline": _metrics(y_all, p_ema), "bkt": _metrics(y_all, p_bkt), "pfa": _metrics(y_te, p_pfa)}
    recovery = {s: {"fitted": bkt[s].as_dict(), "true": dict(zip(["L0", "T", "G", "S"], TRUE_BKT[s]))} for s in SKILL_IDS}
    mae = float(np.mean([abs(bkt[s].as_dict()[k] - TRUE_BKT[s][i]) for s in SKILL_IDS for i, k in enumerate(["L0", "T", "G", "S"])]))

    bundle = {
        "bkt_params": {s: bkt[s].as_dict() for s in SKILL_IDS},
        "pfa": pfa,
        "item_difficulty": item_b,
        "difficulty_model": diff_model,
        "feature_version": 1,
    }
    learner_model = LearnerModel(bundle)
    policy = simulate_policies(learner_model, seeds=(0, 1) if quick else (0, 1, 2, 3, 4), n_learners=40 if quick else 150)

    report = {
        "task": "knowledge_tracing_and_irt",
        "data": f"{n_learners} simulated learners x 8 skills ({len(attempts)} attempts); generative model in app/ml/tracing/simulate.py",
        "split": "learner-disjoint 80/20",
        "knowledge_tracing": models,
        "bkt_parameter_recovery_mae": round(mae, 4),
        "bkt_parameters": recovery,
        "irt": {
            "n_items": len(bank),
            "difficulty_corr_with_truth": round(float(np.corrcoef(b, b_true_vec)[0, 1]), 4),
            "difficulty_model_cv_r2": round(diff_r2, 4),
            "difficulty_model_coefficients": dict(zip(FEATURE_NAMES, [round(float(c), 4) for c in diff_model.coef_])),
        },
        "recommender_simulation": {
            "setup": "40 practice items per simulated learner; metric = fraction of the 8 skills truly learned at the end (mean over seeds)",
            "policies": policy,
        },
        "limitations": "All numbers are on simulated learners whose generative model is BKT+IRT-like, which favours BKT/IRT; "
                       "they validate the implementation, not real-classroom accuracy.",
    }
    bundle["metrics"] = report
    save_bundle(bundle, out_path)
    return report
