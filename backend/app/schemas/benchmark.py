from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ProviderInfo(BaseModel):
    name: str
    available: bool
    reason: str | None = None
    requires_api_key: bool = False
    is_local_model: bool = False


class BenchmarkRunRequest(BaseModel):
    manifest_path: str | None = None
    providers: list[str] | None = None  # if None, use all available


class BenchmarkResultOut(BaseModel):
    provider: str
    sample_id: str
    available: bool
    predicted_text: str | None
    reference_text: str | None
    cer: float | None
    wer: float | None
    exact_match: bool | None
    normalized_edit_distance: float | None
    inference_time_ms: float | None
    diagnosis_correct: bool | None


class BenchmarkSummary(BaseModel):
    provider: str
    available: bool
    num_samples: int
    mean_cer: float | None
    mean_wer: float | None
    exact_match_accuracy: float | None
    mean_normalized_edit_distance: float | None
    mean_inference_time_ms: float | None
    diagnosis_accuracy: float | None
    mistake_category_macro_f1: float | None


class BenchmarkRunOut(BaseModel):
    id: str
    manifest_path: str
    providers: list[str]
    status: str
    created_at: datetime
    summary: dict
    results: list[BenchmarkResultOut]
