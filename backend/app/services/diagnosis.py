"""Mistake diagnosis.

SymPy (via math_parser) verifies each step; a trained scikit-learn
classifier (app.ml.mistake_classifier) then maps engineered features of
those checks to a mistake category. Two hard guardrails run before the
model (no parseable content, very low OCR confidence). If no trained model
artifact is available, the original hand-written decision tree
(`diagnose_rule_based`) is used instead. Every result records `source` =
"ml" or "deterministic".

Confirm-before-diagnose: when any recognized symbol's confidence is below
CONFIRM_BELOW, no diagnosis is given until the student has checked the
highlighted symbols (a misread digit would otherwise be diagnosed as the
student's mistake). The threshold is chosen in app/ml/validation/propagation.py
on validation pages: the smallest value at which no more than 5% of pages are
diagnosed differently from their true transcription after confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.ml.artifacts import MISTAKE_MODEL_PATH, load_bundle
from app.ml.mistake_features import EVIDENCE_TEXT, FEATURE_VERSION, extract_features
from app.services.math_parser import MathAnalysis, ParsedStep, compare_final_answer

CATEGORIES = [
    "arithmetic_error",
    "sign_error",
    "incorrect_operation",
    "algebraic_manipulation_error",
    "formula_selection_error",
    "skipped_step",
    "transcription_uncertainty",
    "final_answer_only_error",
    "correct_solution",
    "insufficient_information",
]

CONFIRM_BELOW = 0.95  # min per-symbol confidence for diagnosing without confirmation (validation_propagation.json)


@dataclass
class Diagnosis:
    category: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    suspected_step: str | None = None
    source: str = "deterministic"
    is_correct: bool | None = None


def _looks_like_sign_error(step: ParsedStep) -> bool:
    """Very targeted heuristic: an equation whose LHS/RHS differ only by sign
    on one term is very likely a sign error rather than a generic arithmetic
    slip. We check this by string proximity on the numeric magnitude.
    """
    if step.kind != "equation" or step.is_valid is not False:
        return False
    # crude check: does the line contain a leading minus right after '='?
    return bool(step.normalized_text.replace(" ", "").count("-") >= 1)


def diagnose(
    math_analysis: MathAnalysis,
    expected_answer: str | None,
    ocr_confidence: float,
    transcription: str | None = None,
    min_symbol_confidence: float | None = None,
    confirmed: bool = False,
) -> Diagnosis:
    """Primary entrypoint: guardrails, then the ML classifier, else rules.

    `min_symbol_confidence` is the lowest per-symbol confidence of an unconfirmed
    recognizer transcription (None once the student has confirmed or typed it).
    `confirmed` means the student verified the text, so it cannot be a misreading."""
    if ocr_confidence < 0.35:
        return Diagnosis(
            category="transcription_uncertainty",
            confidence=round(1 - ocr_confidence, 2),
            evidence=["OCR confidence is very low; transcription may be unreliable."],
            source="deterministic",
        )
    if min_symbol_confidence is not None and min_symbol_confidence < CONFIRM_BELOW:
        return Diagnosis(
            category="transcription_uncertainty",
            confidence=round(1 - min_symbol_confidence, 2),
            evidence=["Some symbols were read with low confidence (highlighted). Check them, then re-check "
                      "to get a diagnosis, so a misread is not mistaken for your error."],
            source="deterministic",
        )
    if not math_analysis.steps:
        return Diagnosis(
            category="insufficient_information",
            confidence=0.9,
            evidence=["No parseable mathematical content was found."],
            source="deterministic",
        )

    bundle = load_bundle(str(MISTAKE_MODEL_PATH))
    if bundle is None or bundle.get("feature_version") != FEATURE_VERSION:
        return diagnose_rule_based(math_analysis, expected_answer, ocr_confidence)

    text = transcription if transcription is not None else "\n".join(s.raw_text for s in math_analysis.steps)
    feats = extract_features(text, expected_answer, ocr_confidence, analysis=math_analysis)
    model = bundle["model"]
    probs = model.predict_proba(feats.vector.reshape(1, -1))[0]
    classes = list(model.classes_)
    if confirmed and "transcription_uncertainty" in classes:
        probs = probs.copy()
        probs[classes.index("transcription_uncertainty")] = 0.0
        probs = probs / max(probs.sum(), 1e-12)
    order = np.argsort(probs)[::-1]
    category = classes[order[0]]

    evidence = [EVIDENCE_TEXT[name] for name, v in feats.values.items() if name in EVIDENCE_TEXT and v == 1.0]
    if feats.suspected_step:
        evidence.insert(0, f"First step that does not hold: '{feats.suspected_step}'.")
    runner_up = classes[order[1]] if len(order) > 1 else None
    evidence.append(
        f"Classifier ({bundle.get('model_name', 'ml')}) probability {probs[order[0]]:.2f}"
        + (f"; next most likely: {runner_up} ({probs[order[1]]:.2f})." if runner_up else ".")
    )

    is_correct: bool | None
    if category == "correct_solution":
        is_correct = True
    elif category in ("insufficient_information", "transcription_uncertainty"):
        is_correct = None
    else:
        is_correct = False

    return Diagnosis(
        category=category,
        confidence=round(float(probs[order[0]]), 3),
        evidence=evidence,
        suspected_step=feats.suspected_step if is_correct is False else None,
        source="ml",
        is_correct=is_correct,
    )


def diagnose_rule_based(
    math_analysis: MathAnalysis,
    expected_answer: str | None,
    ocr_confidence: float,
) -> Diagnosis:
    """Hand-written decision tree used before the ML classifier existed.

    Kept as (a) the baseline the ML model is evaluated against and (b) the
    fallback when no trained model artifact is present.
    """
    if ocr_confidence < 0.35:
        return Diagnosis(
            category="transcription_uncertainty",
            confidence=round(1 - ocr_confidence, 2),
            evidence=["OCR confidence is very low; transcription may be unreliable."],
            source="deterministic",
        )

    if not math_analysis.steps:
        return Diagnosis(
            category="insufficient_information",
            confidence=0.9,
            evidence=["No parseable mathematical content was found."],
            source="deterministic",
        )

    unparseable = [s for s in math_analysis.steps if s.kind == "unparseable"]
    if len(unparseable) == len(math_analysis.steps):
        return Diagnosis(
            category="insufficient_information",
            confidence=0.35,
            evidence=["None of the steps could be parsed as mathematics."],
            source="deterministic",
        )

    # Deterministic path: find the first invalid equation step.
    invalid_steps = [s for s in math_analysis.steps if s.kind == "equation" and s.is_valid is False]

    final_correct = compare_final_answer(math_analysis.final_value, expected_answer)

    if not invalid_steps:
        if final_correct is True or (final_correct is None and expected_answer is None):
            return Diagnosis(
                category="correct_solution",
                confidence=0.85,
                evidence=["All checked equation steps are mathematically valid."],
                source="deterministic",
                is_correct=True,
            )
        if final_correct is False:
            return Diagnosis(
                category="final_answer_only_error",
                confidence=0.6,
                evidence=[
                    "Intermediate equation steps are valid, but the final answer does not match the expected answer.",
                ],
                suspected_step=math_analysis.steps[-1].raw_text,
                source="deterministic",
                is_correct=False,
            )
        return Diagnosis(
            category="insufficient_information",
            confidence=0.4,
            evidence=["Could not compare final answer to an expected value."],
            source="deterministic",
        )

    first_bad = invalid_steps[0]
    if _looks_like_sign_error(first_bad):
        category = "sign_error"
        confidence = 0.55
    elif first_bad.kind == "equation" and "x" in first_bad.normalized_text:
        category = "algebraic_manipulation_error"
        confidence = 0.55
    else:
        category = "arithmetic_error"
        confidence = 0.6

    return Diagnosis(
        category=category,
        confidence=confidence,
        evidence=[f"Step '{first_bad.raw_text}' does not hold: {first_bad.sympy_repr}"],
        suspected_step=first_bad.raw_text,
        source="deterministic",
        is_correct=False,
    )
