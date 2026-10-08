"""Feedback generation.

Produces a student-friendly explanation, step-by-step corrected solution,
one hint, and a prerequisite concept, keyed off the diagnosed mistake
category. Templates are deterministic (no LLM call) so behavior is
reproducible and auditable; this is documented in MODEL_COMPARISON.md /
ARCHITECTURE.md as deterministic software logic, not model inference.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.diagnosis import Diagnosis
from app.services.math_parser import MathAnalysis

_TEMPLATES: dict[str, dict[str, str]] = {
    "correct_solution": {
        "explanation": "Nice work - every step checks out and your final answer is correct.",
        "hint": "Try a slightly harder version of this problem to keep building confidence.",
        "prerequisite": "N/A",
    },
    "arithmetic_error": {
        "explanation": (
            "Your method looks right, but one of the basic calculations (addition, "
            "subtraction, multiplication, or division) doesn't come out correctly."
        ),
        "hint": "Recompute the flagged step by hand, one operation at a time.",
        "prerequisite": "Basic arithmetic operations",
    },
    "sign_error": {
        "explanation": (
            "It looks like a plus/minus sign got flipped somewhere in the flagged step. "
            "This is one of the most common slips in arithmetic and algebra."
        ),
        "hint": "Rewrite the step and double-check each sign before combining terms.",
        "prerequisite": "Adding and subtracting signed numbers",
    },
    "incorrect_operation": {
        "explanation": "The operation used in this step doesn't match what the problem calls for.",
        "hint": "Reread the question and confirm whether you should add, subtract, multiply, or divide.",
        "prerequisite": "Choosing the correct operation from a word or symbolic problem",
    },
    "algebraic_manipulation_error": {
        "explanation": (
            "The equation-solving step doesn't balance. When you do something to one side "
            "of an equation, the same thing needs to happen to the other side."
        ),
        "hint": "Undo the last operation you applied and redo it symmetrically on both sides.",
        "prerequisite": "Solving linear equations by performing the same operation on both sides",
    },
    "formula_selection_error": {
        "explanation": "This looks like the wrong formula or rule was applied for this type of problem.",
        "hint": "Identify what type of problem this is, then recall the specific rule that applies.",
        "prerequisite": "Matching problem types to the correct formula",
    },
    "skipped_step": {
        "explanation": (
            "The jump from the problem to the final answer is missing intermediate work, "
            "which makes it hard to verify - and easy to make an unnoticed mistake."
        ),
        "hint": "Write out every intermediate step, even ones that feel obvious.",
        "prerequisite": "Showing complete step-by-step work",
    },
    "transcription_uncertainty": {
        "explanation": (
            "The handwriting could not be read with enough confidence to check the math. "
            "This is a scanning/legibility issue, not necessarily a math mistake."
        ),
        "hint": "Try correcting the transcription text box, or retake the photo with better lighting.",
        "prerequisite": "N/A",
    },
    "final_answer_only_error": {
        "explanation": (
            "Your working steps check out, but the final boxed answer doesn't match what "
            "was expected. Double-check the very last line."
        ),
        "hint": "Recheck the final simplification or substitution step.",
        "prerequisite": "Careful final-answer verification",
    },
    "insufficient_information": {
        "explanation": "There wasn't enough clearly readable or parseable content to diagnose a specific mistake.",
        "hint": "Add a clearer photo or edit the transcription so each step is on its own line.",
        "prerequisite": "N/A",
    },
}


@dataclass
class Feedback:
    feedback_text: str
    corrected_solution_text: str | None
    hint_text: str
    prerequisite_concept: str
    confidence: float
    warnings: list[str]
    recommended_next_action: str


def _format_corrected_solution(math_analysis: MathAnalysis, canonical_solution: list[str]) -> str | None:
    if canonical_solution:
        return "\n".join(canonical_solution)
    return None


def generate_feedback(
    diagnosis: Diagnosis,
    math_analysis: MathAnalysis,
    canonical_solution: list[str],
    ocr_confidence: float,
) -> Feedback:
    template = _TEMPLATES.get(diagnosis.category, _TEMPLATES["insufficient_information"])

    warnings: list[str] = []
    if ocr_confidence < 0.5:
        warnings.append(
            "OCR confidence is low; please verify the extracted transcription is accurate before trusting this feedback."
        )
    if math_analysis.has_unparseable_step:
        warnings.append("One or more lines could not be parsed as math and were skipped in analysis.")

    if diagnosis.category == "correct_solution":
        next_action = "generate_adaptive_quiz_for_mastery_check"
    elif diagnosis.category in ("transcription_uncertainty", "insufficient_information"):
        next_action = "correct_transcription_and_reanalyze"
    else:
        next_action = "review_explanation_then_generate_practice_quiz"

    return Feedback(
        feedback_text=template["explanation"],
        corrected_solution_text=_format_corrected_solution(math_analysis, canonical_solution),
        hint_text=template["hint"],
        prerequisite_concept=template["prerequisite"],
        confidence=diagnosis.confidence,
        warnings=warnings,
        recommended_next_action=next_action,
    )
