"""Orchestrates the full analyze pipeline for a submission.

Pipeline: preprocess -> detect regions -> transcribe (provider) -> parse
math -> diagnose mistake -> generate canonical solution -> generate
feedback -> persist everything.
"""
from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ml import active_learning, registry
from app.ml.skills import infer_skill
from app.models.analysis import AnalysisResult
from app.models.learning import RecognizedSymbol
from app.models.mistake import Mistake
from app.models.region import ExtractedRegion
from app.models.submission import Submission
from app.providers.base import ProviderUnavailableError
from app.providers.registry import get_provider
from app.services import diagnosis as diagnosis_service
from app.services import feedback as feedback_service
from app.services import math_parser
from app.services import region_detection
from app.services import solution_generator
from app.services.preprocessing import preprocess_image


def run_analysis_pipeline(
    db: Session,
    submission: Submission,
    provider_name: str,
    corrected_transcription: str | None = None,
) -> AnalysisResult:
    """Run (or re-run) the full pipeline for an existing Submission row.

    If `corrected_transcription` is provided, OCR is skipped and the
    supplied text is used directly (the "correct OCR and reanalyze"
    workflow), with full confidence since it was verified by the student.
    """
    settings = get_settings()
    timings: dict[str, float] = dict(submission.stage_timings or {})
    clock = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[stage] = round((now - clock) * 1000, 1)
        clock = now

    # --- 1. Preprocess (only needed once, but safe to skip if already done) ---
    if not submission.processed_image_path or not Path(submission.processed_image_path).exists():
        processed_path = Path(settings.PROCESSED_DIR) / f"{submission.id}.png"
        meta = preprocess_image(submission.original_image_path, processed_path)
        submission.processed_image_path = meta["output_path"]
        lap("preprocess")

    # --- 2. Detect regions (only if not already stored) ---
    if not submission.regions:
        detected = region_detection.detect_regions(submission.processed_image_path)
        for region in detected:
            db.add(
                ExtractedRegion(
                    submission_id=submission.id,
                    region_type=region.region_type,
                    x=region.x,
                    y=region.y,
                    width=region.width,
                    height=region.height,
                    confidence=region.confidence,
                    order_index=region.order_index,
                )
            )
        db.flush()
        lap("regions")

    warnings: list[str] = []
    provider_used = provider_name
    is_demo_mode = provider_name == "demo"
    min_symbol_confidence = None  # set for unconfirmed recognizer output (confirm-before-diagnose)

    # --- 3. Transcribe ---
    if corrected_transcription is not None:
        raw_text = corrected_transcription
        ocr_confidence = 1.0
        submission.corrected_transcription = corrected_transcription
    else:
        provider = get_provider(provider_name)
        try:
            result = provider.transcribe(submission.original_image_path)
            raw_text = result.text
            ocr_confidence = result.confidence
            warnings.extend(result.warnings)
            provider_used = result.provider
            is_demo_mode = result.is_mock
            if result.provider == "ml":
                if result.raw_regions:
                    min_symbol_confidence = min(r["confidence"] for r in result.raw_regions)
                _store_symbols(db, submission, result.raw_regions)
        except ProviderUnavailableError as exc:
            # Never fabricate a prediction - fall back to the demo provider
            # explicitly and say so, rather than silently guessing.
            warnings.append(f"Provider '{provider_name}' unavailable ({exc.reason}); using demo provider instead.")
            fallback = get_provider("demo")
            result = fallback.transcribe(submission.original_image_path)
            raw_text = result.text
            ocr_confidence = result.confidence
            warnings.extend(result.warnings)
            provider_used = "demo"
            is_demo_mode = True

        submission.raw_transcription = raw_text
        lap("ocr")

    submission.provider_used = provider_used
    submission.is_demo_mode = is_demo_mode
    submission.ocr_confidence = ocr_confidence

    # --- 4. Parse math ---
    text_for_math = corrected_transcription if corrected_transcription is not None else raw_text
    math_analysis = math_parser.analyze_transcription(text_for_math or "")
    lap("verify")

    # --- 5. Diagnose ---
    diag = diagnosis_service.diagnose(
        math_analysis, submission.expected_answer, ocr_confidence, transcription=text_for_math or "",
        min_symbol_confidence=min_symbol_confidence,
    )
    submission.skill = infer_skill(text_for_math or "", diag.category)
    lap("diagnose")

    # --- 6. Canonical solution ---
    first_line = math_analysis.steps[0].raw_text if math_analysis.steps else (text_for_math or "")
    canonical_solution = solution_generator.generate_canonical_solution(
        first_line, submission.expected_answer
    )

    # --- 7. Feedback ---
    fb = feedback_service.generate_feedback(diag, math_analysis, canonical_solution, ocr_confidence)
    fb.warnings = list(dict.fromkeys(warnings + fb.warnings))  # dedupe, preserve order

    # --- 8. Persist ---
    interpreted_steps = [
        {
            "raw_text": s.raw_text,
            "kind": s.kind,
            "sympy_repr": s.sympy_repr,
            "is_valid": s.is_valid,
            "error": s.error,
        }
        for s in math_analysis.steps
    ]

    analysis = AnalysisResult(
        submission_id=submission.id,
        interpreted_steps=interpreted_steps,
        is_correct=diag.is_correct,
        canonical_solution=canonical_solution,
        corrected_solution_text=fb.corrected_solution_text,
        feedback_text=fb.feedback_text,
        hint_text=fb.hint_text,
        prerequisite_concept=fb.prerequisite_concept,
        confidence=fb.confidence,
        warnings=fb.warnings,
        recommended_next_action=fb.recommended_next_action,
    )
    db.add(analysis)
    db.flush()

    mistake = Mistake(
        analysis_id=analysis.id,
        category=diag.category,
        confidence=diag.confidence,
        evidence=diag.evidence,
        suspected_step=diag.suspected_step,
        source=diag.source,
    )
    db.add(mistake)
    lap("feedback")
    submission.stage_timings = timings
    db.commit()
    db.refresh(analysis)
    return analysis


def _store_symbols(db: Session, submission: Submission, regions: list[dict]) -> None:
    """Persist recognized symbols (+ crops for active learning), replacing any previous OCR run."""
    db.query(RecognizedSymbol).filter(RecognizedSymbol.submission_id == submission.id).delete()
    version = registry.active_version(active_learning.SYMBOL_MODEL)
    for idx, region in enumerate(regions):
        mask = region.pop("mask", None)
        x, y, w, h = region["bbox"]
        db.add(RecognizedSymbol(
            submission_id=submission.id, idx=idx, line=region["line"], x=x, y=y, w=w, h=h,
            predicted=region["symbol"], confidence=region["confidence"], alternatives=region["alternatives"],
            crop_path=active_learning.save_crop(submission.id, idx, mask) if mask is not None else None,
            model_version=version,
        ))
    db.flush()
