"""Review-3 additions: reading safeguards, crash fixes, local-computation features,
confirm-before-diagnose, and the validation-study helpers."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.ml.artifacts import SYMBOL_MODEL_PATH
from app.ml.mistake_features import extract_features
from app.ml.segmentation import binarize, binarize_page, flatten_background, remove_ruled_lines
from app.ml.symbol_classifier import decode_line, is_well_formed
from app.ml.validation.cogtutor import _solutions, _to_x, is_dev_template
from app.ml.validation.crohme_glyphs import Glyph, compose_page, text_to_symbols
from app.ml.validation.photo import CONDITIONS, capture
from app.ml.validation.policy import n_per_arm
from app.services.diagnosis import CONFIRM_BELOW, diagnose
from app.services.math_parser import analyze_transcription

CLASSES = list("()+-/0123456789=x")


def _probs(rows: list[dict[str, float]]) -> np.ndarray:
    out = np.full((len(rows), len(CLASSES)), 1e-4)
    for i, row in enumerate(rows):
        for ch, p in row.items():
            out[i, CLASSES.index(ch)] = p
    return out / out.sum(1, keepdims=True)


# --- syntax-constrained decoding ---------------------------------------------------

@pytest.mark.parametrize("line, ok", [
    ("12+15=27", True), ("-3+5=-8", True), ("2x+4=10", True), ("(4+6)/2=5", True), ("9-(-2)=11", True),
    ("+5", True), ("1+)+1", False), ("++1", False), ("(1+2", False), ("3+", False), ("()", False), (")1(", False),
])
def test_line_grammar(line, ok):
    assert is_well_formed(list(line)) is ok


def test_decoder_repairs_ill_formed_line_with_runner_up():
    # argmax reads "1 + ) + 1"; ')' has '1' as runner-up
    probs = _probs([{"1": 0.9}, {"+": 0.9}, {")": 0.55, "1": 0.4}, {"+": 0.9}, {"1": 0.9}])
    assert "".join(CLASSES[i] for i in decode_line(probs, CLASSES)) == "1+1+1"


def test_decoder_keeps_well_formed_reading_and_wrong_answers():
    # a wrong answer is read as written: only syntax is used, never arithmetic
    probs = _probs([{"2": 0.9}, {"+": 0.9}, {"2": 0.9}, {"=": 0.9}, {"5": 0.6, "4": 0.39}])
    assert "".join(CLASSES[i] for i in decode_line(probs, CLASSES)) == "2+2=5"


def test_decoder_falls_back_to_argmax_without_valid_reading():
    probs = _probs([{"+": 0.99}, {"=": 0.99}])
    assert [CLASSES[i] for i in decode_line(probs, CLASSES, top_k=1)] == ["+", "="]


# --- photo-robust page binarization ----------------------------------------------

def _page_with_ink() -> np.ndarray:
    img = np.full((120, 400), 255, np.uint8)
    cv2.putText(img, "3 - 4 = 7", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 0, 4)
    return img


def test_flatten_background_is_identity_on_evenly_lit_page():
    page = _page_with_ink()
    assert flatten_background(page) is page


def test_flatten_background_removes_lighting_gradient():
    page = _page_with_ink().astype(np.float32)
    lit = (page * np.linspace(1.0, 0.35, page.shape[1])[None, :]).astype(np.uint8)
    clean = binarize(page.astype(np.uint8)) > 0
    assert (binarize_page(lit) > 0)[clean].mean() > 0.9  # ink still found
    assert (binarize_page(lit) > 0).sum() < 2 * clean.sum()  # dark side not taken as ink
    assert (binarize(lit) > 0).sum() > 2 * clean.sum()  # plain Otsu fails


def test_ruled_lines_removed_but_minus_kept():
    binary = np.zeros((200, 600), np.uint8)
    binary[100:103, :] = 255  # notebook rule across the page
    for x in (50, 150, 250):
        binary[60:95, x:x + 6] = 255  # three '1's
    binary[75:79, 300:340] = 255  # a minus sign
    out = remove_ruled_lines(binary)
    assert out[100:103].sum() == 0
    assert out[75:79, 300:340].all() and out[60:95, 50:56].all()


# --- crash fixes found by the propagation study ----------------------------------

def test_double_equals_misread_does_not_crash_parser():
    analysis = analyze_transcription("4 == 4\n2 = 2")
    assert analysis.steps[0].kind == "unparseable"


def test_x_in_denominator_does_not_crash_features():
    feats = extract_features("1/x + 2 = 5\nx = 3", None, 0.9)
    assert feats.vector.shape[0] == len(feats.values)


# --- local-computation features (feature version 2) --------------------------------

def test_local_computation_features_name_the_in_place_slip():
    wrong_op = extract_features("-75x = 4 - 54\n-75x = 58", None, 1.0).values
    assert wrong_op["local_computation"] == 1.0 and wrong_op["local_op_swap_add_sub"] == 1.0
    sign = extract_features("9x - 13 = 77\n9x = 77 + 13\n9x = -90", None, 1.0).values
    assert sign["local_sign_flip_whole"] == 1.0
    clean = extract_features("3x - 6 = 9\n3x = 15\nx = 5", "5", 1.0).values
    assert clean["local_computation"] == 0.0


# --- confirm-before-diagnose -----------------------------------------------------------

def test_low_confidence_symbol_holds_the_diagnosis():
    analysis = analyze_transcription("12 + 15 = 26")
    held = diagnose(analysis, "27", 0.9, min_symbol_confidence=CONFIRM_BELOW - 0.1)
    assert held.category == "transcription_uncertainty" and held.is_correct is None
    assert diagnose(analysis, "27", 0.9, min_symbol_confidence=None).category != "transcription_uncertainty"
    assert diagnose(analysis, "27", 0.9, min_symbol_confidence=1.0).category != "transcription_uncertainty"


# --- study helpers ---------------------------------------------------------------------

def test_cogtutor_helpers():
    assert _to_x("-6=3+y/9") == "-6=3+x/9" and _to_x("X=4*4") == "x=4*4" and _to_x("y=x") is None
    assert _solutions("2x=6") == _solutions("x=3") != _solutions("x=-3")
    assert _solutions("5=5") == "all" and _solutions("1=0") == "none"
    assert is_dev_template("2x=6") == is_dev_template("7x=1")  # same template, same split


def test_text_to_symbols_writes_multiplication_as_a_cross():
    assert text_to_symbols("3*x + 2*4 = 11") == list("3x+2x4=11")
    assert text_to_symbols("idk") is None


def test_compose_and_capture_produce_grey_pages():
    rng = np.random.default_rng(0)
    blob = np.zeros((30, 20), np.uint8)
    blob[5:25, 8:12] = 255
    pool = {s: [Glyph(s, blob, 1.0, 0.0)] for s in "123+="}
    page, baselines = compose_page([list("1+2=3"), list("3=3")], pool, rng)
    assert page.dtype == np.uint8 and len(baselines) == 2 and (page == 0).any()
    for name in ("clean", "ruled_paper", "phone_severe"):
        img = capture(page, np.random.default_rng(1), baselines, **CONDITIONS[name])
        assert img.dtype == np.uint8 and img.ndim == 2


def test_sample_size_matches_textbook_values():
    assert n_per_arm(0.5) == 64  # two-sample t-test, alpha 0.05, power 0.8
    assert n_per_arm(0.5, rho=0.5) < 64
    assert n_per_arm(0.0) is None


@pytest.mark.skipif(not SYMBOL_MODEL_PATH.exists(), reason="run python -m app.ml.train")
def test_real_world_evaluation_on_a_tiny_manifest(tmp_path):
    from app.ml.validation.real_world import evaluate_manifest

    img = np.full((120, 420), 255, np.uint8)
    cv2.putText(img, "12 + 15 = 27", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 4)
    cv2.imwrite(str(tmp_path / "p1.png"), img)
    rows = [{"sample_id": "p1", "image_path": "p1.png", "reference_text": "12+15=27", "expected_answer": "27",
             "mistake_label": "correct_solution", "mistake_label_b": "correct_solution", "device": "scan"},
            {"sample_id": "gone", "image_path": "missing.png", "reference_text": "1 = 1"}]
    (tmp_path / "m.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    report = evaluate_manifest(tmp_path / "m.jsonl")
    assert report["n_pages"] == 2 and report["n_missing_images"] == 1
    assert "by_device" in report["reading"] and report["diagnosis"]["from_true_transcription"]["n"] == 1
    assert report["annotator_agreement"]["raw"] == 1.0
