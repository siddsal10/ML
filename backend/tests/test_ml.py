from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.ml.artifacts import MISTAKE_MODEL_PATH, SYMBOL_MODEL_PATH
from app.ml.mistake_features import FEATURE_NAMES, extract_features
from app.ml.segmentation import segment_page
from app.ml.symbol_classifier import format_symbols
from app.services.diagnosis import diagnose
from app.services.math_parser import analyze_transcription

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEMO_MANIFEST = BACKEND_ROOT / "data" / "demo_samples" / "manifest.json"

needs_symbol_model = pytest.mark.skipif(not SYMBOL_MODEL_PATH.exists(), reason="run python -m app.ml.train")
needs_mistake_model = pytest.mark.skipif(not MISTAKE_MODEL_PATH.exists(), reason="run python -m app.ml.train")


# --- symbol recognition ----------------------------------------------------------

@pytest.mark.parametrize(
    "symbols, expected",
    [
        (list("12+15=27"), "12 + 15 = 27"),
        (list("-3+5=-8"), "-3 + 5 = -8"),
        (list("2x+4=10"), "2x + 4 = 10"),
        (list("7x8=56"), "7 * 8 = 56"),  # handwritten multiplication sign
        (list("(4+6)/2=5"), "(4 + 6) / 2 = 5"),
        (list("9-(-2)=11"), "9 - (-2) = 11"),
    ],
)
def test_format_symbols_canonical_spacing(symbols, expected):
    assert format_symbols(symbols) == expected


def _render(text: str) -> np.ndarray:
    img = Image.new("L", (500, 120 * (text.count("\n") + 1)), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=56)
    for i, line in enumerate(text.split("\n")):
        draw.text((20, 20 + 110 * i), line, fill=0, font=font)
    return np.array(img)


def test_segmentation_merges_equals_bars_and_splits_lines():
    lines = segment_page(_render("1 = 2\n3 = 4"))
    assert len(lines) == 2
    assert [len(crops) for crops in lines] == [3, 3]


def test_segmentation_blank_page_has_no_lines():
    assert segment_page(np.full((100, 100), 255, dtype=np.uint8)) == []


@needs_symbol_model
@pytest.mark.skipif(not DEMO_MANIFEST.exists(), reason="demo images not generated")
def test_ml_provider_reads_demo_images():
    from app.providers.ml_provider import ClassicalMLProvider

    provider = ClassicalMLProvider()
    for entry in json.loads(DEMO_MANIFEST.read_text()).values():
        result = provider.transcribe(BACKEND_ROOT / entry["path"])
        assert result.text == entry["text"]
        assert result.is_mock is False
        assert 0.0 <= result.confidence <= 1.0
        assert result.raw_regions and "alternatives" in result.raw_regions[0]


def test_ml_provider_unavailable_without_model(monkeypatch, tmp_path):
    from app.providers import ml_provider
    from app.providers.base import ProviderUnavailableError

    monkeypatch.setattr(ml_provider, "SYMBOL_MODEL_PATH", tmp_path / "missing.joblib")
    provider = ml_provider.ClassicalMLProvider()
    available, reason = provider.is_available()
    assert available is False and "train" in reason

    img = tmp_path / "x.png"
    Image.new("RGB", (10, 10), "white").save(img)
    with pytest.raises(ProviderUnavailableError):
        provider.transcribe(img)


# --- mistake features ------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected, feature",
    [
        ("-3 + 5 = -8", "2", "sign_flip_term"),
        ("6 * 3 = 9", "18", "op_swap_cross"),
        ("12 + 15 = 26", "27", "off_by_small"),
        ("2x + 4 = 10\nx = 7", "3", "cand_transpose_no_sign_change"),
        ("2x + 4 = 10\nx = 6", "3", "cand_forgot_divide"),
        ("2x + 4 = 10\nx = -3", "3", "cand_negated_answer"),
    ],
)
def test_mistake_feature_fires(text, expected, feature):
    feats = extract_features(text, expected, 0.9)
    assert feats.values[feature] == 1.0
    assert feats.vector.shape == (len(FEATURE_NAMES),)


def test_consistent_algebra_work_has_no_error_features():
    feats = extract_features("2x + 4 = 10\n2x = 6\nx = 3", "3", 0.9)
    assert feats.values["work_consistent"] == 1.0
    assert feats.values["nonequiv_found"] == 0.0
    assert feats.suspected_step is None


def test_unbalanced_parentheses_do_not_crash_parser():
    analysis = analyze_transcription("(4 + 6 = 10")
    assert analysis.steps[0].kind == "unparseable"


# --- ML diagnosis ---------------------------------------------------------------

@needs_mistake_model
@pytest.mark.parametrize(
    "text, expected, category",
    [
        ("12 + 15 = 27", "27", "correct_solution"),
        ("2x + 4 = 10\n2x = 6\nx = 3", "3", "correct_solution"),
        ("6 * 3 = 9", "18", "incorrect_operation"),
        ("-3 + 5 = -8", "2", "sign_error"),
        ("2x + 4 = 10\n2x = 14\nx = 7", "3", "algebraic_manipulation_error"),
        ("3x + 5 = 20\nx = 15", "5", "skipped_step"),
        ("12 + 15 = 27", "28", "final_answer_only_error"),
    ],
)
def test_ml_diagnosis_categories(text, expected, category):
    diag = diagnose(analyze_transcription(text), expected, 0.9, transcription=text)
    assert diag.source == "ml"
    assert diag.category == category
    assert diag.evidence


@needs_mistake_model
def test_ml_diagnosis_points_at_wrong_step():
    text = "2x + 4 = 10\n2x = 14\nx = 7"
    diag = diagnose(analyze_transcription(text), "3", 0.9, transcription=text)
    assert diag.suspected_step == "2x = 14"
    assert diag.is_correct is False


def test_diagnosis_falls_back_to_rules_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.diagnosis.MISTAKE_MODEL_PATH", tmp_path / "missing.joblib")
    diag = diagnose(analyze_transcription("12 + 15 = 27"), "27", 0.9)
    assert diag.source == "deterministic"
    assert diag.category == "correct_solution"


# --- training smoke test ----------------------------------------------------------

def test_training_pipeline_runs_on_tiny_data(tmp_path):
    from app.ml.glyph_data import generate_dataset
    from app.ml.mistake_classifier import train_mistake_classifier
    from app.ml.mistake_data import generate_samples
    from app.ml.symbol_classifier import train_symbol_classifier

    glyphs = generate_dataset(samples_per_font=2, seed=0)
    report, recognizer, _ = train_symbol_classifier(glyphs, out_path=tmp_path / "sym.joblib")
    assert report["selected_model"] in report["models"]
    assert (tmp_path / "sym.joblib").exists()
    assert recognizer.classes

    mreport = train_mistake_classifier(generate_samples(n_per_label=15, seed=0), out_path=tmp_path / "mis.joblib")
    assert "rule_based_baseline" in mreport["models"]
    assert (tmp_path / "mis.joblib").exists()
