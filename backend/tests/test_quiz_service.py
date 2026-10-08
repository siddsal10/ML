from __future__ import annotations

from app.services.quiz_service import (
    generate_quiz_questions,
    grade_answer,
    update_mastery,
)


def test_generate_quiz_questions_returns_three_difficulties():
    questions = generate_quiz_questions("arithmetic", weak_concept=None, seed=42)
    assert len(questions) == 3
    assert {q.difficulty for q in questions} == {"easy", "medium", "hard"}


def test_generate_quiz_questions_deterministic_with_seed():
    q1 = generate_quiz_questions("algebra", weak_concept="algebraic_manipulation_error", seed=7)
    q2 = generate_quiz_questions("algebra", weak_concept="algebraic_manipulation_error", seed=7)
    assert [q.prompt for q in q1] == [q.prompt for q in q2]


def test_generate_quiz_questions_targets_sign_error_concept():
    questions = generate_quiz_questions("arithmetic", weak_concept="sign_error", seed=1)
    assert all("-" in q.prompt for q in questions)


def test_generate_quiz_questions_every_question_has_expected_answer():
    questions = generate_quiz_questions("arithmetic", weak_concept=None, seed=99)
    for q in questions:
        assert q.expected_answer
        assert q.explanation


def test_grade_answer_correct():
    assert grade_answer("27", "27") is True


def test_grade_answer_incorrect():
    assert grade_answer("26", "27") is False


def test_grade_answer_handles_whitespace():
    assert grade_answer(" 27 ", "27") is True


def test_update_mastery_moves_toward_score():
    new_mastery = update_mastery(old_mastery=0.5, quiz_score_fraction=1.0)
    assert new_mastery > 0.5
    assert new_mastery <= 1.0


def test_update_mastery_moves_down_on_poor_score():
    new_mastery = update_mastery(old_mastery=0.8, quiz_score_fraction=0.0)
    assert new_mastery < 0.8


def test_update_mastery_clamped_between_zero_and_one():
    assert update_mastery(old_mastery=1.0, quiz_score_fraction=1.0) <= 1.0
    assert update_mastery(old_mastery=0.0, quiz_score_fraction=0.0) >= 0.0


def test_update_mastery_matches_documented_formula():
    old, score, lr = 0.4, 1.0, 0.4
    expected = round(old + lr * (score - old), 4)
    assert update_mastery(old, score) == expected
