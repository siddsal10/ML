from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RegionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    region_type: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    extracted_content: str | None
    order_index: int


class MistakeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    category: str
    confidence: float
    evidence: list[str]
    suspected_step: str | None
    source: str


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    interpreted_steps: list[dict[str, Any]]
    is_correct: bool | None
    canonical_solution: list[str]
    corrected_solution_text: str | None
    feedback_text: str | None
    hint_text: str | None
    prerequisite_concept: str | None
    confidence: float
    warnings: list[str]
    recommended_next_action: str | None
    created_at: datetime
    mistake: MistakeOut | None = None


class SubmissionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    student_id: str
    subject: str
    topic: str
    original_question: str
    expected_answer: str | None
    provider_requested: str | None
    provider_used: str | None
    raw_transcription: str | None
    corrected_transcription: str | None
    ocr_confidence: float | None
    is_demo_mode: bool
    created_at: datetime


class SymbolOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    idx: int
    line: int
    x: int
    y: int
    w: int
    h: int
    predicted: str
    confidence: float
    alternatives: list[list[Any]]
    corrected_label: str | None


class SymbolCorrection(BaseModel):
    idx: int
    label: str = Field(..., min_length=1, max_length=2)


class SymbolCorrectionsIn(BaseModel):
    corrections: list[SymbolCorrection] = Field(..., min_length=0)  # empty = "the reading is right"


class AnalyzeResponse(BaseModel):
    """Full structured response returned by POST /api/submissions/analyze."""

    submission_id: str
    extracted_text: str
    detected_regions: list[RegionOut]
    ocr_confidence: float
    interpreted_steps: list[dict[str, Any]]
    correctness: bool | None
    mistake_category: str | None
    suspected_incorrect_step: str | None
    diagnostic_evidence: list[str]
    corrected_solution: str | None
    feedback: str | None
    hint: str | None
    prerequisite_concept: str | None
    confidence: float
    warnings: list[str]
    recommended_next_action: str | None
    provider_used: str
    is_demo_mode: bool
    mistake_source: str | None = None
    skill: str | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    symbols: list[SymbolOut] = Field(default_factory=list)
    symbol_check_below: float | None = None  # symbols below this confidence must be checked before diagnosis
    image_width: int | None = None
    image_height: int | None = None
    expected_answer: str | None = None
    original_question: str | None = None
    topic: str | None = None
    real_error_type: dict[str, Any] | None = None


class TranscriptionUpdate(BaseModel):
    corrected_transcription: str = Field(..., min_length=1)
