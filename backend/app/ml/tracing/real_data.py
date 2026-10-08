"""Knowledge tracing on REAL student data: ASSISTments 2009-2010 skill builder.

- Benchmark: EMA vs BKT (EM per skill) vs PFA (logistic regression) vs PFA +
  Rasch item difficulty vs PFA + IRT + recency features, over all 110 skills,
  learner-disjoint split. The Rasch regulariser is chosen on validation
  learners carved out of the training set (never on the test learners).
- Deployment: BKT parameters for StepWise's skills are fitted on the
  ASSISTments skills that correspond to them (SKILL_MAP).

Preprocessing (standard for this dataset): drop rows without a skill tag,
order each student's rows by order_id; rows tagged with several skills appear
once per skill ("unrolled"), as in pyBKT / DKT papers.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from app.ml.artifacts import BACKEND_ROOT
from app.ml.irt.rasch import fit_rasch
from app.ml.tracing.models import bkt_trace, ema_predict, fit_bkt

ASSISTMENTS_PATH = BACKEND_ROOT / "data" / "external" / "assistments_2009.csv"

# StepWise skill -> ASSISTments skills it corresponds to (approximate where noted).
SKILL_MAP: dict[str, list[str]] = {
    "S1": ["Addition Whole Numbers", "Subtraction Whole Numbers"],
    "S2": ["Multiplication Whole Numbers", "Multiplication and Division Positive Decimals"],  # approx.
    "S3": ["Addition and Subtraction Integers", "Multiplication and Division Integers"],
    "S4": ["Order of Operations All", "Order of Operations +,-,/,* () positive reals"],
    "S5": ["Solving for a variable"],
    "S6": ["Equation Solving Two or Fewer Steps"],
    "S7": ["Equation Solving More Than Two Steps"],  # approx.: multi-step = repeated inverse operations
    # S8 (equations with negatives) has no ASSISTments counterpart -> keeps simulated parameters
}


@dataclass
class RealAttempt:
    learner: int
    skill: str
    item_id: str
    correct: int
    t: int


def load_assistments(path: Path = ASSISTMENTS_PATH) -> list[RealAttempt]:
    import pandas as pd

    df = pd.read_csv(path, encoding="latin-1", low_memory=False,
                     usecols=["order_id", "user_id", "problem_id", "skill_name", "correct"])
    df = df.dropna(subset=["skill_name"])
    df = df[df["correct"].isin([0, 1])].sort_values(["user_id", "order_id"])
    df["t"] = df.groupby("user_id").cumcount()
    return [RealAttempt(int(u), str(s), str(p), int(c), int(t))
            for u, s, p, c, t in df[["user_id", "skill_name", "problem_id", "correct", "t"]].itertuples(index=False)]


def _metrics(y, p) -> dict:
    from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"auc": round(float(roc_auc_score(y, p)), 4), "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
            "log_loss": round(float(log_loss(y, p)), 4), "n": int(len(y))}


def _sequences(attempts) -> dict:
    seqs = defaultdict(list)
    for a in attempts:  # already time-ordered per learner
        seqs[(a.learner, a.skill)].append(a.correct)
    return seqs


def _pfa_design(attempts, skill_index: dict[str, int], item_b: dict[str, float] | None, recency: bool = False):
    """Sparse PFA design: skill intercept + log success/failure counts per skill
    [+ item difficulty] [+ recency: EMA of this skill's answers, last answer right/wrong]."""
    k = len(skill_index)
    counts = defaultdict(lambda: [0, 0])
    ema = defaultdict(lambda: 0.5)
    last: dict = {}
    rows, cols, vals, y = [], [], [], []
    for r, a in enumerate(attempts):
        key = (a.learner, a.skill)
        i = skill_index[a.skill]
        s, f = counts[key]
        feats = [(i, 1.0), (k + i, np.log1p(s)), (2 * k + i, np.log1p(f))]
        if item_b is not None:
            feats.append((3 * k, item_b.get(a.item_id, 0.0)))
        if recency:
            prev = last.get(key)
            feats += [(3 * k + 1, ema[key]), (3 * k + 2, float(prev == 1)), (3 * k + 3, float(prev == 0))]
        for c, v in feats:
            rows.append(r); cols.append(c); vals.append(v)
        y.append(a.correct)
        counts[key][0 if a.correct else 1] += 1
        ema[key] += 0.3 * (a.correct - ema[key])
        last[key] = a.correct
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(y), 3 * k + 4))
    return X, np.array(y)


def _fit_item_difficulty(train, l2: float) -> dict[str, float]:
    items = sorted({a.item_id for a in train})
    item_idx = {it: i for i, it in enumerate(items)}
    learner_idx = {l: i for i, l in enumerate(sorted({a.learner for a in train}))}
    _, b = fit_rasch(np.array([learner_idx[a.learner] for a in train]), np.array([item_idx[a.item_id] for a in train]),
                     np.array([a.correct for a in train], float), len(learner_idx), len(items), l2=l2)
    return dict(zip(items, b.tolist()))


def _fit_predict_pfa(train, test, skill_index, item_b, recency):
    X_tr, y_tr = _pfa_design(train, skill_index, item_b, recency)
    X_te, y_te = _pfa_design(test, skill_index, item_b, recency)
    model = LogisticRegression(C=1.0, max_iter=3000, fit_intercept=False).fit(X_tr, y_tr)
    return y_te, model.predict_proba(X_te)[:, 1]


def evaluate_assistments(seed: int = 0, n_iter: int = 30) -> tuple[dict, dict]:
    """Returns (report, fitted StepWise BKT params for mapped skills)."""
    attempts = load_assistments()
    learners = np.array(sorted({a.learner for a in attempts}))
    rng = np.random.default_rng(seed)
    rng.shuffle(learners)
    test_l = set(learners[: len(learners) // 5].tolist())
    train = [a for a in attempts if a.learner not in test_l]
    test = [a for a in attempts if a.learner in test_l]
    skills = sorted({a.skill for a in attempts})
    skill_index = {s: i for i, s in enumerate(skills)}

    # EMA + BKT on (learner, skill) sequences
    train_seqs, test_seqs = _sequences(train), _sequences(test)
    by_skill = defaultdict(list)
    for (_, s), seq in train_seqs.items():
        by_skill[s].append(seq)
    bkt = {s: fit_bkt(by_skill[s], n_iter=n_iter) for s in skills if by_skill[s]}
    y_seq, p_ema, p_bkt = [], [], []
    for (_, s), seq in test_seqs.items():
        y_seq += seq
        p_ema += ema_predict(seq)
        p_bkt += bkt_trace(bkt[s], seq)[0] if s in bkt else [float(np.mean(seq))] * len(seq)

    # Rasch regulariser chosen on validation learners inside the training split.
    from sklearn.metrics import roc_auc_score

    train_l = np.array(sorted({a.learner for a in train}))
    rng.shuffle(train_l)
    val_l = set(train_l[: len(train_l) // 8].tolist())
    inner = [a for a in train if a.learner not in val_l]
    val = [a for a in train if a.learner in val_l]
    l2_scores = {}
    for l2 in (0.05, 1.0, 5.0):
        y_v, p_v = _fit_predict_pfa(inner, val, skill_index, _fit_item_difficulty(inner, l2), recency=True)
        l2_scores[l2] = round(float(roc_auc_score(y_v, p_v)), 4)
    best_l2 = max(l2_scores, key=l2_scores.get)
    item_b = _fit_item_difficulty(train, best_l2)

    results = {"ema_baseline": _metrics(y_seq, p_ema), "bkt": _metrics(y_seq, p_bkt)}
    for name, ib, rec in (("pfa", None, False), ("pfa_irt", item_b, False), ("pfa_irt_recency", item_b, True)):
        results[name] = _metrics(*_fit_predict_pfa(train, test, skill_index, ib, rec))

    # Deployment parameters for StepWise skills: fit on ALL learners of the mapped ASSISTments skills.
    all_seqs = _sequences(attempts)
    deployed, mapped_counts = {}, {}
    for ours, theirs in SKILL_MAP.items():
        seqs = [seq for (_, s), seq in all_seqs.items() if s in theirs]
        if seqs:
            deployed[ours] = fit_bkt(seqs, n_iter=n_iter).as_dict()
            mapped_counts[ours] = {"assistments_skills": theirs, "sequences": len(seqs), "attempts": int(sum(map(len, seqs)))}

    report = {
        "dataset": "ASSISTments 2009-2010 skill builder (real students)",
        "preprocessing": "rows with a skill tag; ordered by order_id; multi-skill rows unrolled per skill",
        "n_attempts": len(attempts),
        "n_learners": int(len(learners)),
        "n_skills": len(skills),
        "n_items": len({a.item_id for a in attempts}),
        "split": "learner-disjoint 80/20",
        "models": results,
        "rasch_l2_validation_auc": {str(k): v for k, v in l2_scores.items()},
        "rasch_l2_selected": best_l2,
        "stepwise_skill_mapping": mapped_counts,
        "stepwise_bkt_params_from_real_data": deployed,
        "note": "BKT here has no forgetting and one parameter set per skill. On this dataset recency is highly "
                "predictive (skill-builder sets end after 3 correct in a row), which is why EMA beats plain BKT/PFA "
                "and why adding recency features to PFA gives the best model.",
    }
    return report, deployed
