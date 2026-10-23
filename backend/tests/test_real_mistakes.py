from __future__ import annotations

import numpy as np
import pytest

from app.ml.real_mistakes import (
    PORTABLE_FEATURES,
    REAL_MISTAKE_MODEL_PATH,
    answer_features,
    image_features,
    parse_value,
    portable_vector,
    split_alternatives,
    split_blanks,
    to_slip_label,
)


@pytest.mark.parametrize("raw, value", [
    ("400", 400), ("1.2", 1.2), ("-2", -2), ("\\frac{3}{2}", 1.5), ("1\\frac{1}{2}", 1.5),
    ("\\frac{{29}}{{13}}", 29 / 13), ("$2$", 2), ("二", 2), ("45 - 30 = 15", 15), ("50%", 0.5), ("15厘米", 15),
])
def test_parse_value(raw, value):
    v = parse_value(raw)
    assert v is not None and float(v) == pytest.approx(value)


def test_parse_value_rejects_non_values():
    assert parse_value("") is None
    assert parse_value("1,2") is None  # tuple, not a single value


def test_blank_and_alternative_splitting():
    assert split_blanks("420; 42000") == ["420", "42000"]
    assert split_alternatives("\\frac{3}{2}, 1\\frac{1}{2}, 1.5") == ["\\frac{3}{2}", "1\\frac{1}{2}", "1.5"]
    assert split_alternatives("$2$\n或$二$") == ["$2$", "$二$"]


@pytest.mark.parametrize("correct, student, feature", [
    ("2", "-2", "sign_flip"),
    ("12", "1.2", "decimal_shift"),
    ("64", "65", "one_digit_diff"),
    ("13", "31", "digit_permutation"),
    ("4; 49", "49; 4", "blanks_swapped"),
    ("30", "10", "int_ratio"),
])
def test_answer_pattern_features(correct, student, feature):
    assert answer_features(correct, student)[feature] == 1.0


def test_answer_features_use_question_and_solution_numbers():
    f = answer_features("39", "32", question="从田边到池塘有32米", solution="32 / 4 = 8; 8 * 4 + 7 = 39")
    assert f["student_in_question"] == 1.0
    f2 = answer_features("39", "8", question="32米", solution="32 / 4 = 8")
    assert f2["student_in_solution"] == 1.0


def test_alternative_answer_forms_count_as_correct():
    f = answer_features("\\frac{1}{2}, 0.5", "0.5")
    assert f["n_wrong_blanks"] == 0.0


def test_image_features_handle_missing_and_blank_images():
    assert set(image_features(None)) == set(image_features(np.full((50, 50), 255, np.uint8)))
    ink = np.full((100, 200), 255, np.uint8)
    ink[40:60, 20:180] = 0
    assert image_features(ink)["img_ink"] > image_features(np.full((100, 200), 255, np.uint8))["img_ink"]


def test_portable_vector_shape():
    v = portable_vector("27", "26", "What is 12 + 15?", "", "primary", None)
    assert v.shape == (len(PORTABLE_FEATURES),) and np.isfinite(v).all()


def test_slip_grouping():
    assert list(to_slip_label(["calculation_error", "concept_error", "copying_error"])) == ["slip", "understanding", "slip"]


@pytest.mark.skipif(not REAL_MISTAKE_MODEL_PATH.exists(), reason="run python -m app.ml.train --only real_mistakes")
def test_predict_error_type_with_trained_model():
    from app.ml.artifacts import load_bundle
    from app.ml.real_mistakes import predict_error_type

    out = predict_error_type(load_bundle(str(REAL_MISTAKE_MODEL_PATH)), "27", "26", "What is 12 + 15?", None, "primary")
    assert len(out["top"]) == 3 and 0 <= out["slip_probability"] <= 1
    assert sum(t["probability"] for t in out["top"]) <= 1.0001
