from __future__ import annotations

import pytest

from app.services.math_parser import (
    UnsafeExpressionError,
    analyze_transcription,
    compare_final_answer,
    normalize_ocr_text,
    safe_parse,
    solve_linear_equation,
)


def test_safe_parse_arithmetic():
    result = safe_parse("12 + 15")
    assert result == 27


def test_safe_parse_equation():
    from sympy import Eq

    result = safe_parse("12 + 15 = 27")
    assert isinstance(result, Eq)


def test_safe_parse_rejects_dunder_access():
    with pytest.raises(UnsafeExpressionError):
        safe_parse("().__class__.__bases__[0]")


def test_safe_parse_rejects_import_statement():
    with pytest.raises(UnsafeExpressionError):
        safe_parse("__import__('os').system('echo hi')")


def test_safe_parse_rejects_arbitrary_identifiers():
    with pytest.raises(UnsafeExpressionError):
        safe_parse("os.system(1)")


def test_normalize_ocr_text_fixes_common_symbols():
    assert normalize_ocr_text("12 × 3") == "12 * 3"
    assert normalize_ocr_text("10 ÷ 2") == "10 / 2"


def test_analyze_transcription_detects_valid_equation():
    analysis = analyze_transcription("12 + 15 = 27")
    assert analysis.all_steps_valid is True
    assert analysis.steps[0].is_valid is True


def test_analyze_transcription_detects_invalid_equation():
    analysis = analyze_transcription("-3 + 5 = -8")
    assert analysis.all_steps_valid is False
    assert analysis.steps[0].is_valid is False


def test_analyze_transcription_handles_unparseable_line():
    analysis = analyze_transcription("this is not math at all !!!")
    assert analysis.has_unparseable_step is True


def test_compare_final_answer_equal():
    assert compare_final_answer("27", "27") is True


def test_compare_final_answer_not_equal():
    assert compare_final_answer("27", "28") is False


def test_compare_final_answer_unparseable_returns_none():
    assert compare_final_answer("garbage!!!", "27") is None


def test_solve_linear_equation():
    assert solve_linear_equation("2*x + 4 = 10") == "x = 3"
