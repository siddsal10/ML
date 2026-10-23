from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.db.database import get_db
from app.models.student import Student
from app.models.submission import Submission
from app.ml import active_learning
from app.ml.symbol_classifier import format_symbols
from app.models.learning import RecognizedSymbol
from app.schemas.submission import (
    AnalyzeResponse,
    RegionOut,
    SubmissionOut,
    SymbolCorrectionsIn,
    SymbolOut,
    TranscriptionUpdate,
)
from app.services.submission_service import run_analysis_pipeline
from app.utils.file_utils import UploadValidationError, build_processed_path, build_upload_path, validate_upload

router = APIRouter(prefix="/api/submissions", tags=["submissions"])
logger = get_logger(__name__)


def _symbols(db: Session, submission_id: str) -> list[RecognizedSymbol]:
    return (
        db.query(RecognizedSymbol)
        .filter(RecognizedSymbol.submission_id == submission_id)
        .order_by(RecognizedSymbol.idx)
        .all()
    )


def _image_size(path: str) -> tuple[int | None, int | None]:
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).size
    except (OSError, ValueError):
        return None, None


def _real_error_type(submission: Submission, analysis) -> dict | None:
    """Second opinion from the model trained on real student mistakes (ScratchMath).
    Only meaningful for wrong answers with an answer key."""
    from app.ml.artifacts import load_bundle
    from app.ml.real_mistakes import REAL_MISTAKE_MODEL_PATH, predict_error_type
    from app.ml.symbol_classifier import load_gray
    from app.services.math_parser import analyze_transcription

    if analysis.is_correct is not False or not submission.expected_answer:
        return None
    bundle = load_bundle(str(REAL_MISTAKE_MODEL_PATH))
    if bundle is None:
        return None
    text = submission.corrected_transcription or submission.raw_transcription or ""
    final = analyze_transcription(text).final_value
    if final is None:
        return None
    level = "middle" if "x" in text else "primary"
    result = predict_error_type(bundle, submission.expected_answer, str(final), submission.original_question or "",
                                load_gray(submission.original_image_path), level)
    result["model"] = f"{bundle['model_name']} / {bundle['slip_model_name']}"
    result["trained_on"] = "ScratchMath: 1,720 real student mistakes"
    return result


def _to_analyze_response(submission: Submission, analysis, db: Session | None = None) -> AnalyzeResponse:
    regions = [RegionOut.model_validate(r) for r in submission.regions]
    mistake = analysis.mistake
    width, height = _image_size(submission.original_image_path)
    symbols = [SymbolOut.model_validate(s) for s in _symbols(db, submission.id)] if db is not None else []
    return AnalyzeResponse(
        submission_id=submission.id,
        extracted_text=submission.corrected_transcription or submission.raw_transcription or "",
        detected_regions=regions,
        ocr_confidence=submission.ocr_confidence or 0.0,
        interpreted_steps=analysis.interpreted_steps,
        correctness=analysis.is_correct,
        mistake_category=mistake.category if mistake else None,
        suspected_incorrect_step=mistake.suspected_step if mistake else None,
        diagnostic_evidence=mistake.evidence if mistake else [],
        corrected_solution=analysis.corrected_solution_text,
        feedback=analysis.feedback_text,
        hint=analysis.hint_text,
        prerequisite_concept=analysis.prerequisite_concept,
        confidence=analysis.confidence,
        warnings=analysis.warnings,
        recommended_next_action=analysis.recommended_next_action,
        provider_used=submission.provider_used or "unknown",
        is_demo_mode=submission.is_demo_mode,
        mistake_source=mistake.source if mistake else None,
        skill=submission.skill,
        stage_timings=submission.stage_timings or {},
        symbols=symbols,
        image_width=width,
        image_height=height,
        expected_answer=submission.expected_answer,
        original_question=submission.original_question,
        topic=submission.topic,
        real_error_type=_real_error_type(submission, analysis),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_submission(
    student_id: str = Form(...),
    subject: str = Form("math"),
    topic: str = Form(...),
    original_question: str = Form(...),
    expected_answer: str | None = Form(None),
    provider: str | None = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")

    settings = get_settings()
    content = await image.read()
    try:
        validate_upload(image, content)
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    upload_path = build_upload_path(student_id, image.filename or "upload")
    upload_path.write_bytes(content)

    provider_name = provider or settings.DEFAULT_OCR_PROVIDER

    submission = Submission(
        student_id=student_id,
        subject=subject,
        topic=topic,
        original_question=original_question,
        expected_answer=expected_answer,
        provider_requested=provider_name,
        original_image_path=str(upload_path),
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    try:
        analysis = run_analysis_pipeline(db, submission, provider_name)
    except Exception:
        logger.exception("Analysis pipeline failed for submission %s", submission.id)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.") from None

    db.commit()
    db.refresh(submission)
    return _to_analyze_response(submission, analysis, db)


@router.get("/{submission_id}", response_model=SubmissionOut)
def get_submission(submission_id: str, db: Session = Depends(get_db)) -> Submission:
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    return submission


@router.get("/{submission_id}/image")
def get_original_image(submission_id: str, db: Session = Depends(get_db)):
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    path = Path(submission.original_image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    return FileResponse(path)


@router.get("/{submission_id}/processed-image")
def get_processed_image(submission_id: str, db: Session = Depends(get_db)):
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    if not submission.processed_image_path:
        raise HTTPException(status_code=404, detail="Submission has not been processed yet")
    path = Path(submission.processed_image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Processed image file missing on disk")
    return FileResponse(path)


@router.post("/{submission_id}/reanalyze", response_model=AnalyzeResponse)
def reanalyze_submission(
    submission_id: str,
    payload: TranscriptionUpdate,
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    """Re-run math analysis using a student-corrected transcription.

    This is the "OCR correction" workflow: skips OCR entirely and treats
    the supplied text as ground truth for this submission.
    """
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    try:
        analysis = run_analysis_pipeline(
            db, submission, submission.provider_used or "demo", corrected_transcription=payload.corrected_transcription
        )
    except Exception:
        logger.exception("Reanalysis failed for submission %s", submission_id)
        raise HTTPException(status_code=500, detail="Reanalysis failed. Please try again.") from None

    db.commit()
    db.refresh(submission)
    return _to_analyze_response(submission, analysis, db)


@router.get("/{submission_id}/analysis", response_model=AnalyzeResponse)
def get_latest_analysis(submission_id: str, db: Session = Depends(get_db)) -> AnalyzeResponse:
    submission = db.get(Submission, submission_id)
    if submission is None or not submission.analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    latest = sorted(submission.analyses, key=lambda a: a.created_at)[-1]
    return _to_analyze_response(submission, latest, db)


@router.post("/{submission_id}/symbols/corrections", response_model=AnalyzeResponse)
def correct_symbols(submission_id: str, payload: SymbolCorrectionsIn, db: Session = Depends(get_db)) -> AnalyzeResponse:
    """Student fixes misread symbols: the corrections are saved as labelled
    training crops (active learning) and the work is re-analyzed with the
    corrected transcription."""
    submission = db.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    symbols = {s.idx: s for s in _symbols(db, submission_id)}
    if not symbols:
        raise HTTPException(status_code=400, detail="This submission has no recognized symbols to correct")
    for c in payload.corrections:
        sym = symbols.get(c.idx)
        if sym is None:
            raise HTTPException(status_code=422, detail=f"Unknown symbol index {c.idx}")
        if c.label not in active_learning.SYMBOLS:
            raise HTTPException(status_code=422, detail=f"Unsupported symbol {c.label!r}")
        sym.corrected_label = None if c.label == sym.predicted else c.label
        if sym.corrected_label:
            active_learning.save_correction(sym.crop_path, c.label)

    lines: dict[int, list[str]] = {}
    for sym in sorted(symbols.values(), key=lambda s: s.idx):
        lines.setdefault(sym.line, []).append(sym.label)
    text = "\n".join(format_symbols(chars) for _, chars in sorted(lines.items()))
    db.flush()

    analysis = run_analysis_pipeline(db, submission, submission.provider_used or "ml", corrected_transcription=text)
    db.commit()
    db.refresh(submission)
    return _to_analyze_response(submission, analysis, db)
