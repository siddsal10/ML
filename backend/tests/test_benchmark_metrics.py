from __future__ import annotations

from app.utils.text_metrics import (
    character_error_rate,
    exact_match,
    macro_f1,
    normalized_edit_distance,
    word_error_rate,
)


def test_character_error_rate_identical_strings():
    assert character_error_rate("hello", "hello") == 0.0


def test_character_error_rate_completely_different():
    assert character_error_rate("abc", "xyz") == 1.0


def test_character_error_rate_partial():
    # "12+15=27" vs "12+15=26" -> 1 char substitution / 8 ref chars
    cer = character_error_rate("12+15=27", "12+15=26")
    assert 0 < cer < 0.2


def test_word_error_rate_identical():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0


def test_word_error_rate_one_word_different():
    wer = word_error_rate("the cat sat", "the dog sat")
    assert wer == round(1 / 3, 4)


def test_normalized_edit_distance_bounds():
    assert normalized_edit_distance("", "") == 0.0
    ned = normalized_edit_distance("abc", "abcd")
    assert 0 <= ned <= 1


def test_exact_match_true():
    assert exact_match(" 27 ", "27") is True


def test_exact_match_false():
    assert exact_match("27", "28") is False


def test_macro_f1_perfect_predictions():
    pairs = [("a", "a"), ("b", "b"), ("c", "c")]
    assert macro_f1(pairs) == 1.0


def test_macro_f1_all_wrong():
    pairs = [("a", "b"), ("b", "a")]
    assert macro_f1(pairs) == 0.0


def test_macro_f1_empty_returns_none():
    assert macro_f1([]) is None
