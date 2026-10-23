from __future__ import annotations

import cv2
import numpy as np
from sklearn.metrics import accuracy_score

from app.ml.paper_experiments import DEGRADATIONS, bootstrap_ci, degrade, mcnemar
from app.ml.segmentation import binarize, normalize_illumination, segment_page


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.where(rng.random(500) < 0.8, y, 1 - y)
    lo, hi = bootstrap_ci(y, p, accuracy_score, n=300)
    assert lo < accuracy_score(y, p) < hi and hi - lo < 0.1


def test_bootstrap_ci_grouped():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    p = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    lo, hi = bootstrap_ci(y, p, accuracy_score, n=200, groups=[1, 1, 2, 2, 3, 3, 4, 4])
    assert 0 <= lo <= hi <= 1


def test_mcnemar_identical_models_not_significant():
    y = np.array([0, 1, 1, 0, 1])
    r = mcnemar(y, y, y)
    assert r["model1_only_correct"] == r["model2_only_correct"] == 0 and r["p_value"] == 1.0


def test_mcnemar_detects_clear_difference():
    y = np.ones(100, int)
    better = np.ones(100, int)
    worse = np.r_[np.ones(60, int), np.zeros(40, int)]
    assert mcnemar(y, better, worse)["p_value"] < 0.001


def _page() -> np.ndarray:
    img = np.full((120, 400), 255, np.uint8)
    cv2.putText(img, "1 + 2 = 3", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5)
    return img


def test_degradations_keep_shape_and_dtype():
    rng = np.random.default_rng(0)
    for kind in DEGRADATIONS:
        out = degrade(_page(), kind, rng)
        assert out.shape == (120, 400) and out.dtype == np.uint8


def test_illumination_normalisation_flattens_background():
    page = _page().astype(np.float32)
    shaded = np.clip(page * np.linspace(0.4, 1.0, 400)[None, :], 0, 255).astype(np.uint8)
    flat = normalize_illumination(shaded)
    background = flat[5:20, :]  # no ink in the top rows
    assert background.std() < 10 and background.mean() > 200


def test_normalisation_rescues_segmentation_under_shading():
    page = _page().astype(np.float32)
    shaded = np.clip((page * 0.75 + 40) * np.linspace(0.45, 1.0, 400)[None, :], 0, 255).astype(np.uint8)
    with_norm = segment_page(shaded, normalize=True)
    assert len(with_norm) == 1 and len(with_norm[0]) == 5
    # clean input is unaffected by normalisation
    assert np.array_equal(binarize(_page(), normalize=True) > 0, binarize(_page()) > 0) or \
        np.mean((binarize(_page(), normalize=True) > 0) != (binarize(_page()) > 0)) < 0.01
