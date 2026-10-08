# Research Methodology

> **Note (current build):** the project now uses classical ML end to end — see
> [ML_APPROACH.md](ML_APPROACH.md). TrOCR/Donut/vision-API remain optional baselines, and the
> LLM diagnosis fallback described below has been replaced by the ML mistake classifier.

## Research questions

1. **How accurately do TrOCR, Donut, and a vision-language model transcribe handwritten
   educational content?** Measured via CER, WER, exact-match accuracy, and normalized edit
   distance against a reference-transcription manifest (`app/services/benchmark_service.py`).

2. **Does combining OCR output with image-aware reasoning improve mistake diagnosis?** This
   project's default pipeline is deterministic-math-first (OCR → SymPy parsing → rule-based
   diagnosis); the vision-API provider path is architecturally positioned to test whether a
   model that "sees" the image directly (rather than only reading a transcription) catches
   mistakes the deterministic path misses — e.g. crossed-out work, a diagram misread as text.
   This build does not yet wire a full image-aware diagnosis call (see `MODEL_COMPARISON.md`);
   the ablation below documents exactly what would need to change to test this.

3. **How much does OCR quality affect final diagnosis accuracy?** Measured by comparing
   `diagnosis_accuracy` (in the benchmark summary) across providers with different transcription
   quality, using the *same* deterministic diagnosis logic downstream — isolating OCR quality as
   the independent variable.

4. **Does adaptive question generation target the detected weakness correctly?** Qualitatively
   verifiable today: `quiz_service.generate_quiz_questions(topic, weak_concept)` maps specific
   diagnosed categories (`sign_error`, `algebraic_manipulation_error`, `formula_selection_error`)
   to concept-targeted generators (`_sign_error_question`, `_linear_algebra_question`); a
   rigorous answer would require a classroom study measuring whether students who receive
   concept-targeted quizzes improve mastery faster than students receiving generic
   difficulty-matched quizzes — out of scope for this prototype but the mastery-tracking
   infrastructure (`MasteryRecord`, `apply_quiz_result`) is in place to support such a study.

## Baselines

- **OCR baseline**: demo provider's mock output serves as a sanity-check floor (it is
  intentionally *not* a real baseline — it only "succeeds" on the four bundled samples and
  otherwise honestly reports low confidence).
- **Diagnosis baseline**: "always predict `correct_solution`" or "always predict the majority
  class in the manifest" — useful floors to compare `diagnosis_accuracy` and
  `mistake_category_macro_f1` against so a >baseline result is meaningful, not just a
  consequence of class imbalance in a small manifest.

## Ablations

As specified in the project brief, the following pipeline configurations are meaningful to
compare once a real evaluation manifest (see `DATASET_GUIDE.md`) is available:

1. **OCR only** — CER/WER/exact-match per provider, no downstream diagnosis.
2. **Vision-language model only** — `vision_api` provider's raw transcription quality in
   isolation.
3. **OCR + deterministic math checker** — this project's default pipeline (TrOCR/demo →
   `math_parser` → `diagnosis`).
4. **OCR + vision-language reasoning** — would require extending `VisionAPIProvider` (or a new
   provider) to accept a chat-completion-style prompt asking the model to both transcribe *and*
   diagnose from the image directly, then comparing its `mistake_category` output against
   `diagnosis_accuracy` from ablation 3. Not implemented in this build; the boundary is called
   out explicitly in `app/services/diagnosis.py::diagnose_with_llm`.
5. **Complete fusion pipeline** — combining deterministic checks as the primary signal with
   vision-LLM diagnosis as a secondary/tie-breaking signal when deterministic confidence is low
   (e.g. `ocr_confidence < 0.5` or all steps unparseable). The fallback boundary already exists
   as a hook point (`diagnose()` calls `diagnose_with_llm()` exactly in this situation) — the
   remaining work is the chat-completion wiring described in ablation 4.

## Metrics

| Metric | Formula / definition | Where implemented |
|---|---|---|
| Character Error Rate (CER) | Levenshtein distance / reference length (character-level) | `app/utils/text_metrics.py::character_error_rate` |
| Word Error Rate (WER) | Levenshtein distance / reference length (word-level) | `app/utils/text_metrics.py::word_error_rate` |
| Exact-match accuracy | Fraction of samples with `hypothesis.strip() == reference.strip()` | `app/utils/text_metrics.py::exact_match` |
| Normalized edit distance | Levenshtein distance / max(len(ref), len(hyp)) | `app/utils/text_metrics.py::normalized_edit_distance` |
| Inference time | Wall-clock ms per `.transcribe()` call | `OCRResult.inference_time_ms` |
| Diagnosis accuracy | Fraction of samples where predicted mistake category == manifest's `mistake_label` | `benchmark_service._summarize` |
| Mistake-category macro F1 | Unweighted mean of per-class F1 across all mistake categories present | `app/utils/text_metrics.py::macro_f1` |

All metrics are computed from scratch (standard Levenshtein DP, standard per-class
precision/recall/F1) with no external metrics library dependency, so the exact computation is
auditable in `app/utils/text_metrics.py`.

## Threats to validity

- **Manifest size**: the bundled demo manifest has 4 samples and is only suitable for
  smoke-testing the pipeline mechanics, not for drawing statistically meaningful conclusions
  about model quality. Any real comparison needs a manifest with (at minimum) dozens of samples
  per mistake category to get non-degenerate macro-F1 estimates.
- **Synthetic demo images**: the bundled demo images are rendered text (via PIL), not real
  handwriting, so they are not representative of TrOCR/Donut's real-world accuracy — they exist
  purely to make the deterministic demo provider's hash-matching mechanism testable/demoable.
  Real accuracy numbers require real handwritten samples (see `DATASET_GUIDE.md`).
- **Mistake-label subjectivity**: some mistakes are genuinely ambiguous between categories (e.g.
  an arithmetic slip that also happens to be a sign flip) — inter-annotator agreement should be
  measured if building a labeled manifest for research purposes, and is not accounted for in the
  current deterministic labeling scheme.
- **Confidence-score comparability**: TrOCR's confidence is a softmax-probability proxy, Donut's
  is a fixed placeholder, and vision-API's is a fixed placeholder — these are *not* calibrated or
  comparable across providers, and should not be treated as such in any downstream analysis (this
  is documented at each provider's definition site).
- **No held-out/train-test split concerns apply here** since no model is fine-tuned or trained as
  part of this build's default path — see `README.md`'s "Future improvements" for the optional
  fine-tuning direction, which would need standard train/val/test discipline if pursued.
