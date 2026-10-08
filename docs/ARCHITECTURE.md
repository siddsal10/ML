# Architecture

## Overview

The system is a standard three-tier web application: a React SPA frontend, a FastAPI backend, and
a SQLite database (swappable for Postgres/MySQL via `DATABASE_URL`). The interesting engineering
is in the backend's **pipeline** (image → text → math → diagnosis → feedback → quiz → mastery)
and its **provider abstraction** (so the same pipeline works identically whether OCR comes from a
zero-dependency mock, a local model, or a hosted vision API).

## Full request flow: `POST /api/submissions/analyze`

1. **API layer** (`app/api/submissions.py`) validates the student exists, reads the uploaded
   file's bytes, and calls `validate_upload()` (`app/utils/file_utils.py`), which checks file
   extension, magic bytes, size limit, and non-emptiness. Filenames are sanitized
   (`sanitize_filename()`) to strip path components and disallowed characters before the file is
   written under a per-student upload directory — this is the path-traversal defense.

2. A `Submission` row is created with `provider_requested` recorded, then
   **`run_analysis_pipeline()`** (`app/services/submission_service.py`) orchestrates the rest:

   a. **Preprocessing** (`app/services/preprocessing.py`): EXIF orientation correction →
      grayscale → OpenCV `fastNlMeansDenoising` → CLAHE contrast enhancement → a
      minAreaRect-based deskew → adaptive Gaussian thresholding → aspect-preserving resize. The
      processed image is saved to a separate path (`PROCESSED_DIR`) from the original. Ink-ratio
      heuristics flag likely-blank or likely-overexposed images as warnings.

   b. **Region detection** (`app/services/region_detection.py`): contour-based segmentation on
      the binarized image (dilate to merge nearby strokes → `findContours` → heuristic
      classification into text/math/diagram by stroke density and connected-component count).
      This is explicitly documented as an MVP approximation, not a learned layout model.

   c. **Transcription**: the selected provider's `.transcribe()` is called
      (`app/providers/registry.py` → `get_provider(name)`). The default provider is `ml`
      (`app/providers/ml_provider.py`): OpenCV line/symbol segmentation (`app/ml/segmentation.py`),
      HOG + zoning features (`app/ml/symbol_features.py`) and the trained SVM
      (`app/ml/symbol_classifier.py`). Every recognized symbol (box, prediction, confidence,
      runner-ups, crop) is stored as a `RecognizedSymbol` row for the correction UI and active
      learning. If the provider raises
      `ProviderUnavailableError`, the pipeline **falls back to the demo provider and records a
      warning** rather than crashing or fabricating a result. If the student has already
      corrected the transcription (the "OCR correction" workflow), OCR is skipped entirely and
      the corrected text is used with confidence 1.0.

   d. **Math parsing** (`app/services/math_parser.py`): the transcription is normalized (common
      OCR substitutions like `×`→`*`, `÷`→`/`, en-dash→`-`) and split into lines. Each line is
      parsed with SymPy's `parse_expr` under a **strict character whitelist** plus a **restricted
      `local_dict`** (only `x`, `y`, `pi`, `sqrt`, `Abs` resolve) — `eval()`/`exec()` are never
      called directly anywhere in the codebase. Equations are checked with
      `Eq(lhs, rhs, evaluate=False)` (constructing with `evaluate=True` would silently collapse
      purely-numeric equalities into a plain Python boolean, losing the equation structure) and
      verified by simplifying `lhs - rhs`.

   e. **Diagnosis** (`app/services/diagnosis.py`): two guardrails (very low OCR confidence →
      `transcription_uncertainty`; no parseable steps → `insufficient_information`), then the
      **ML mistake classifier**: `app/ml/mistake_features.py` turns the SymPy checks into 33 named
      features (sign-flip match, operator-swap match, term moved without sign change, first
      non-equivalent algebra step, …) and the trained Random Forest returns category, probability,
      runner-up and the fired evidence. If no trained model is present, the original rule-based
      decision tree (`diagnose_rule_based`) is used. The submission is also mapped to one of 8
      skills (`app/ml/skills.py`). Per-stage timings are stored on the submission.

   f. **Canonical solution generation** (`app/services/solution_generator.py`): regex-matches
      supported question shapes (single arithmetic operation, single-variable linear equation)
      and produces a step-by-step solution via SymPy `solve()`. Unsupported shapes return an empty
      list rather than a fabricated solution.

   g. **Feedback generation** (`app/services/feedback.py`): a deterministic template keyed by the
      diagnosed category. No LLM call — this keeps feedback reproducible and auditable.

   h. **Persistence**: an `AnalysisResult` row (interpreted steps, correctness, canonical
      solution, feedback text, confidence, warnings, recommended next action) and a linked
      `Mistake` row (category, confidence, evidence, source: `ml`|`deterministic`) are written.
      A submission can have multiple `AnalysisResult`s over time (e.g. after re-analysis); the
      latest is authoritative.

3. The API layer maps the `AnalysisResult` + `Mistake` + `ExtractedRegion`s into the
   `AnalyzeResponse` schema and returns it.

## Practice, mastery and teacher flow

- `GET /api/students/{id}/skills` replays the student's observations (practice answers + diagnosed
  submissions, `app/services/learner_service.py`) through **BKT** with the EM-fitted per-skill
  parameters and returns mastery per skill plus a timeline.
- `GET /api/practice/{id}/next` picks the target skill (the latest diagnosed mistake's skill, else
  the least-mastered) and returns 3 items from the 320-item bank whose **PFA**-predicted P(correct)
  (using **Rasch** item difficulty) is closest to 0.7. `POST /api/practice/answer` grades with
  SymPy, stores a `PracticeAttempt`, and returns mastery before/after.
- `GET /api/teacher/class` builds each student's profile (error-category rates + mastery),
  clusters them with K-Means (k by silhouette), names the groups and projects them with PCA.
- The legacy `/api/quizzes` endpoints (template quiz + EMA mastery) are kept for compatibility.

## ML Lab and active learning

- `POST /api/submissions/{id}/symbols/corrections` saves corrected crops as labelled training data
  (`app/ml/active_learning.py`), rebuilds the transcription and re-runs the pipeline.
- `POST /api/lab/models/symbol_classifier/retrain` starts a background job: synthetic glyphs +
  corrections → model comparison → new version in the registry (`app/ml/registry.py`), not
  activated until `POST /api/lab/models/{name}/activate`.
- `GET /api/lab/models[/{name}]` serves model cards (`app/ml/model_cards.py`) and the training
  reports in `ml_models/reports/`; `POST /api/lab/symbols/predict` powers the playground.

## Provider architecture

```
BaseOCRProvider (abstract)
 ├── ClassicalMLProvider   - "ml" (default): OpenCV segmentation + HOG + SVM, CPU, ~4 MB model
 ├── DemoOCRProvider       - deterministic, hash-matched demo samples, zero deps
 ├── TrOCRProvider         - optional deep-learning baseline (requirements-dl.txt)
 ├── DonutProvider         - optional deep-learning baseline (benchmark adapter)
 └── VisionAPIProvider     - optional hosted vision-API baseline
```

Every provider implements `is_available() -> (bool, reason)` and
`transcribe(image_path) -> OCRResult`. `transcribe()` must raise `ProviderUnavailableError`
rather than return fabricated text if it cannot actually run — this contract is exercised
directly in `tests/test_providers.py` and indirectly via `submission_service`'s fallback-to-demo
behavior and the benchmark module's "unavailable" reporting.

## Data model

See `backend/app/models/` for the full SQLAlchemy schema: `Student` 1→N `Submission` 1→N
`ExtractedRegion`, `Submission` 1→N `AnalysisResult` 1→1 `Mistake`; `Student` 1→N `Quiz` 1→N
`QuizQuestion` 1→1 `QuizAttempt`; `Student` 1→N `MasteryRecord` (unique per topic);
`BenchmarkRun` 1→N `BenchmarkResult`; `Submission` 1→N `RecognizedSymbol`; `Student` 1→N
`PracticeAttempt`. Missing nullable columns are added to older SQLite databases on startup
(`init_db`). Cascade deletes are configured so deleting a student cleans
up all dependent rows.
