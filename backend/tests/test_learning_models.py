from __future__ import annotations

import numpy as np
import pytest

from app.ml.clustering.cluster import build_profile, cluster_students
from app.ml.irt.item_bank import ITEMS_PER_SKILL, build_item_bank
from app.ml.irt.rasch import fit_rasch
from app.ml.irt.recommender import TARGET_P, LearnerModel
from app.ml.skills import SKILL_IDS, infer_skill
from app.ml.tracing.models import BKTParams, bkt_trace, ema_predict, fit_bkt
from app.services.math_parser import Eq, X, compare_final_answer, safe_parse


# --- skills -------------------------------------------------------------------

@pytest.mark.parametrize("text, category, skill", [
    ("12 + 15 = 27", "correct_solution", "S1"),
    ("6 * 3 = 9", "incorrect_operation", "S2"),
    ("-3 + 5 = -8", "sign_error", "S3"),
    ("(4 + 6) * 2 = 20", "correct_solution", "S4"),
    ("x + 4 = 10\nx = 6", "correct_solution", "S5"),
    ("2x + 4 = 10\nx = 3", "correct_solution", "S6"),
    ("2x + 4 = 10\nx = 7", "algebraic_manipulation_error", "S7"),
    ("2x - 4 = -10\nx = -3", "correct_solution", "S8"),
])
def test_infer_skill(text, category, skill):
    assert infer_skill(text, category) == skill


# --- item bank ------------------------------------------------------------------

def test_item_bank_size_and_unique_ids():
    bank = build_item_bank()
    assert len(bank) == ITEMS_PER_SKILL * len(SKILL_IDS)
    assert len({it.id for it in bank}) == len(bank)
    assert {it.level for it in bank} == {0, 1, 2}


def test_every_item_answer_is_mathematically_correct():
    for it in build_item_bank():
        if it.prompt.startswith("Solve for x: "):
            eq = safe_parse(it.prompt.removeprefix("Solve for x: "))
            assert isinstance(eq, Eq)
            import sympy
            assert sympy.solve(eq, X) == [sympy.Integer(it.answer)], it.prompt
        else:
            expr = it.prompt.removeprefix("What is ").rstrip("?")
            assert compare_final_answer(expr, it.answer) is True, it.prompt


# --- knowledge tracing ------------------------------------------------------------

def test_bkt_mastery_rises_with_correct_answers_and_falls_with_wrong():
    p = BKTParams(0.3, 0.1, 0.2, 0.1)
    _, up = bkt_trace(p, [1, 1, 1, 1])
    _, down = bkt_trace(p, [0, 0, 0, 0])
    assert up[-1] > 0.95 > 0.3 > down[0]
    assert all(b >= a for a, b in zip(up, up[1:]))


def test_bkt_em_recovers_simulated_parameters():
    rng = np.random.default_rng(0)
    true = BKTParams(0.3, 0.15, 0.2, 0.1)
    seqs = []
    for _ in range(400):
        learned = rng.random() < true.L0
        seq = []
        for _ in range(15):
            seq.append(int(rng.random() < ((1 - true.S) if learned else true.G)))
            learned = learned or rng.random() < true.T
        seqs.append(seq)
    fit = fit_bkt(seqs, n_iter=40)
    for k in ("L0", "T", "G", "S"):
        assert abs(getattr(fit, k) - getattr(true, k)) < 0.08, k


def test_ema_prediction_starts_at_half():
    assert ema_predict([1, 0])[0] == 0.5


def test_rasch_recovers_difficulty_order():
    rng = np.random.default_rng(1)
    b_true = np.linspace(-2, 2, 12)
    theta = rng.normal(0, 1, 300)
    L, I = np.meshgrid(np.arange(300), np.arange(12), indexing="ij")
    L, I = L.ravel(), I.ravel()
    y = (rng.random(L.size) < 1 / (1 + np.exp(-(theta[L] - b_true[I])))).astype(float)
    _, b = fit_rasch(L, I, y, 300, 12)
    assert np.corrcoef(b, b_true)[0, 1] > 0.95


# --- recommender ---------------------------------------------------------------------

def test_recommender_targets_skill_and_excludes_items():
    model = LearnerModel()
    states = model.skill_states({"S3": [0, 1, 0]})
    first = model.recommend("S3", states["S3"], n=3)
    assert len(first) == 3 and all(it.skill == "S3" for it, _ in first)
    assert all(0 <= p <= 1 for _, p in first)
    again = model.recommend("S3", states["S3"], n=3, exclude={it.id for it, _ in first})
    assert not {it.id for it, _ in first} & {it.id for it, _ in again}


def test_recommender_picks_items_near_target_probability():
    model = LearnerModel()
    if not model.trained:
        pytest.skip("knowledge tracing model not trained")
    states = model.skill_states({})
    recs = model.recommend("S6", states["S6"], n=3)
    candidates = [model.p_correct(it, states["S6"]) for it in build_item_bank() if it.skill == "S6"]
    best_gap = min(abs(p - TARGET_P) for p in candidates)
    assert min(abs(p - TARGET_P) for _, p in recs) == pytest.approx(best_gap)


def test_target_skill_prefers_diagnosed_unmastered_skill():
    model = LearnerModel()
    states = model.skill_states({"S1": [1] * 10})
    assert model.target_skill(states, "S7") == "S7"
    assert model.target_skill(states, "S1") != "S1"  # mastered -> pick weakest instead


# --- clustering -------------------------------------------------------------------------

def test_clustering_separates_two_obvious_groups():
    sign = [build_profile(["sign_error"] * 5 + ["correct_solution"], {s: 0.4 for s in SKILL_IDS}) for _ in range(6)]
    strong = [build_profile(["correct_solution"] * 6, {s: 0.95 for s in SKILL_IDS}) for _ in range(6)]
    profiles = np.array(sign + strong) + np.random.default_rng(0).normal(0, 0.01, (12, len(sign[0])))
    result = cluster_students(profiles, k_range=range(2, 4))
    labels = result["labels"]
    assert len(set(labels[:6])) == 1 and len(set(labels[6:])) == 1 and labels[0] != labels[6]
    names = set(result["names"].values())
    assert any("sign" in n for n in names) and "Strong & consistent" in names


def test_clustering_handles_tiny_class():
    result = cluster_students(np.zeros((2, 14)))
    assert result["labels"] == [0, 0]


# --- real-data helpers ---------------------------------------------------------------

def test_crohme_token_filters():
    from app.ml.real_symbols import TOKEN_TO_CLASS, in_vocab, is_linear

    assert is_linear("2 x + 3 = 7".split()) and in_vocab("2 x + 3 = 7".split())
    assert not is_linear("\\frac { 1 } { 2 }".split())
    assert not is_linear("x ^ { 2 }".split())
    assert is_linear("a + b".split()) and not in_vocab("a + b".split())
    assert TOKEN_TO_CLASS["\\times"] == "x"


def test_crohme_alignment_labels_symbols_in_reading_order():
    import cv2

    from app.ml.real_symbols import _ink_band
    from app.ml.segmentation import binarize, segment_symbols

    img = np.full((120, 400), 255, np.uint8)
    cv2.putText(img, "1 + 2 = 3", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)
    crops = segment_symbols(_ink_band(binarize(img)))
    assert len(crops) == len("1 + 2 = 3".split())
    assert [c.x for c in crops] == sorted(c.x for c in crops)


def test_real_pfa_recency_features():
    from app.ml.tracing.real_data import RealAttempt, _pfa_design

    attempts = [RealAttempt(1, "A", "p1", 1, 0), RealAttempt(1, "A", "p2", 0, 1), RealAttempt(1, "A", "p3", 1, 2)]
    X, y = _pfa_design(attempts, {"A": 0}, {"p1": 0.5}, recency=True)
    X = X.toarray()
    k = 1
    assert list(y) == [1, 0, 1]
    assert X[0, 3 * k] == 0.5 and X[1, 3 * k] == 0.0          # item difficulty (unknown item -> 0)
    assert X[0, 3 * k + 1] == 0.5 and X[1, 3 * k + 1] == pytest.approx(0.65)  # EMA before answering
    assert X[1, 3 * k + 2] == 1.0 and X[2, 3 * k + 3] == 1.0   # last answer right / wrong
    assert X[2, k] == pytest.approx(np.log1p(1)) and X[2, 2 * k] == pytest.approx(np.log1p(1))


def test_vectorised_bkt_matches_reference_on_uneven_lengths():
    seqs = [[1, 0, 1, 1], [0, 0], [1], [0, 1, 1, 1, 1, 1, 0]]
    p = fit_bkt(seqs, n_iter=10)
    assert 0 < p.L0 < 1 and 0 < p.T <= 0.6 and p.G <= 0.3 and p.S <= 0.3
