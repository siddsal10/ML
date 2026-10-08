"""Adaptive quiz engine.

Generates three follow-up questions (easy / medium / hard) targeted at a
detected weak concept, using deterministic templates for supported
arithmetic/algebra topics. Grading uses the same safe SymPy comparison used
elsewhere (no eval). Mastery is updated with a simple, documented
exponential-moving-average rule:

    new_mastery = old_mastery + LEARNING_RATE * (quiz_score - old_mastery)

where quiz_score is in [0, 1] (fraction correct) and LEARNING_RATE = 0.4.
This gives more weight to recent performance while still smoothing across
attempts. It is intentionally simple and fully documented rather than a
black-box learned model.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from app.services.math_parser import compare_final_answer

LEARNING_RATE = 0.4


@dataclass
class GeneratedQuestion:
    difficulty: str
    prompt: str
    expected_answer: str
    explanation: str
    source: str = "template"


def _arithmetic_question(rng: random.Random, difficulty: str) -> GeneratedQuestion:
    ranges = {"easy": (1, 10), "medium": (10, 50), "hard": (50, 200)}
    lo, hi = ranges[difficulty]
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    op = rng.choice(["+", "-"])
    answer = a + b if op == "+" else a - b
    return GeneratedQuestion(
        difficulty=difficulty,
        prompt=f"What is {a} {op} {b}?",
        expected_answer=str(answer),
        explanation=f"{a} {op} {b} = {answer}. Combine the two numbers using the {'sum' if op == '+' else 'difference'} operation.",
    )


def _sign_error_question(rng: random.Random, difficulty: str) -> GeneratedQuestion:
    magnitude = {"easy": 10, "medium": 30, "hard": 100}[difficulty]
    a = rng.randint(1, magnitude)
    b = rng.randint(1, magnitude)
    answer = -a + b
    return GeneratedQuestion(
        difficulty=difficulty,
        prompt=f"What is -{a} + {b}?",
        expected_answer=str(answer),
        explanation=(
            f"-{a} + {b} = {answer}. Think of it as starting at -{a} on the number line and "
            f"moving {b} steps to the right."
        ),
    )


def _linear_algebra_question(rng: random.Random, difficulty: str) -> GeneratedQuestion:
    coeff_range, const_range = {
        "easy": ((1, 4), (1, 10)),
        "medium": ((2, 6), (5, 20)),
        "hard": ((3, 9), (10, 40)),
    }[difficulty]
    coeff = rng.randint(*coeff_range)
    x_value = rng.randint(1, 10)
    const = rng.randint(*const_range)
    rhs = coeff * x_value + const
    return GeneratedQuestion(
        difficulty=difficulty,
        prompt=f"Solve for x: {coeff}x + {const} = {rhs}",
        expected_answer=str(x_value),
        explanation=(
            f"Subtract {const} from both sides: {coeff}x = {rhs - const}. "
            f"Divide both sides by {coeff}: x = {x_value}."
        ),
    )


_TOPIC_GENERATORS = {
    "sign_error": _sign_error_question,
    "algebraic_manipulation_error": _linear_algebra_question,
    "formula_selection_error": _linear_algebra_question,
}


def generate_quiz_questions(
    topic: str, weak_concept: str | None, seed: int | None = None
) -> list[GeneratedQuestion]:
    rng = random.Random(seed)
    generator = _TOPIC_GENERATORS.get(weak_concept or "", None)

    if generator is None:
        # Fall back on topic string heuristics.
        topic_lower = (topic or "").lower()
        if "algebra" in topic_lower or "equation" in topic_lower:
            generator = _linear_algebra_question
        else:
            generator = _arithmetic_question

    return [generator(rng, difficulty) for difficulty in ("easy", "medium", "hard")]


def grade_answer(submitted_answer: str, expected_answer: str) -> bool:
    result = compare_final_answer(submitted_answer.strip(), expected_answer.strip())
    if result is not None:
        return result
    # Fall back to exact string match if SymPy comparison is inconclusive
    # (e.g. non-numeric expected answer).
    return submitted_answer.strip() == expected_answer.strip()


def update_mastery(old_mastery: float, quiz_score_fraction: float) -> float:
    """quiz_score_fraction in [0, 1]. Returns new mastery in [0, 1]."""
    new_value = old_mastery + LEARNING_RATE * (quiz_score_fraction - old_mastery)
    return max(0.0, min(1.0, round(new_value, 4)))
