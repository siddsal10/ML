# Model Comparison: TrOCR vs. Donut vs. Vision-Language APIs (Qwen-VL / Claude)

> **Note (current build):** the default recognizer is the classical-ML `ml` provider (HOG + SVM);
> the models below are optional deep-learning baselines for the OCR benchmark. See
> [ML_APPROACH.md](ML_APPROACH.md).

This document describes the three OCR/vision approaches available in the provider architecture,
based on their published design and general characteristics. **No benchmark numbers here are
invented** — the `mean_cer` / `mean_wer` / etc. columns are only ever filled in by actually
running `evaluation/benchmark.py` against a manifest and reading `evaluation/results/*.json`; see
the empty result templates in that directory.

## TrOCR (default local provider)

- **Type**: encoder-decoder transformer (ViT image encoder + RoBERTa-style text decoder),
  purpose-built for OCR.
- **Checkpoint used**: `microsoft/trocr-base-handwritten`, fine-tuned specifically on handwritten
  text (the IAM dataset family).
- **Why it's the default**: it is the only model in this project's lineup that is *specifically*
  fine-tuned for handwritten text recognition, runs acceptably on CPU, and has a permissive,
  well-documented HuggingFace integration with no external API dependency.
- **Known limitation for this project**: TrOCR was trained for general handwritten text, not
  specifically mathematical notation — symbols like `Σ`, fraction bars, or exponents are not part
  of its training distribution, so accuracy on complex math expressions (beyond basic
  arithmetic/algebra written left-to-right) is expected to degrade. This project scopes supported
  math to arithmetic and linear algebra partly for this reason.
- **Confidence estimate**: approximated from the mean of per-token softmax max-probabilities
  during generation — a reasonable proxy, not a calibrated confidence score.

## Donut (benchmark adapter, not in the default pipeline)

- **Type**: OCR-free document understanding transformer — it maps image pixels directly to
  structured output tokens without a separate text-detection step.
- **Checkpoint used**: `naver-clova-ix/donut-base`, the *base* pretrained checkpoint (not
  fine-tuned for this task).
- **Why it's not the default**: Donut's base checkpoint is designed to be fine-tuned per
  downstream task (e.g. via a document-specific prompt/schema); using it zero-shot on
  freeform handwritten arithmetic is unproven and likely to underperform TrOCR, since there is no
  arithmetic-specific fine-tuning applied here. It is included as an adapter specifically so the
  benchmark module can measure this gap empirically rather than assert it.
- **Confidence estimate**: deliberately fixed at a conservative constant (0.5) rather than
  computed, because the base checkpoint has no confidence signal suited to this task — reporting
  a computed-but-meaningless number would be more misleading than an honest fixed placeholder.

## Vision-language APIs (Qwen-VL / Claude / other OpenAI-compatible endpoints)

- **Type**: general-purpose multimodal LLMs with image understanding, accessed via
  `VisionAPIProvider` (`app/providers/vision_api_provider.py`), which supports either the
  Anthropic Claude API or any OpenAI-compatible chat-completions endpoint (which is how a
  self-hosted Qwen-VL deployment would typically be exposed).
- **Strength**: because these are general reasoning models, they can potentially read messier
  handwriting *and* provide contextual understanding (e.g. recognizing that a crossed-out step
  was abandoned) — but this project does not use them for anything beyond transcription today.
  The current provider prompt (`_PROMPT` in `vision_api_provider.py`) asks only for a faithful
  transcription, not interpretation, to keep the deterministic math/diagnosis pipeline as the
  single source of truth for correctness judgments.
- **Cost/latency**: unlike TrOCR/Donut, every call is a network round-trip with per-token API
  cost — this is why the architecture treats it as *one option among several* rather than the
  default, and why the demo mode exists so the whole project is usable with zero API spend.
- **Confidence estimate**: these APIs don't expose token-level confidence for this use case, so a
  fixed conservative estimate (0.8) is reported rather than a fabricated calibrated score.

## Summary table (qualitative, not benchmarked)

| Provider | Runs offline? | Fine-tuned for handwriting? | Cost | Wired into default pipeline? |
|---|---|---|---|---|
| demo | Yes (always) | N/A (mock) | Free | Yes (default) |
| TrOCR | Yes (CPU-capable) | Yes | Free (local compute) | Yes (configurable default) |
| Donut | Yes (CPU-capable, slower) | No (base checkpoint) | Free (local compute) | Benchmark-only |
| vision_api | No (network call) | General-purpose, not handwriting-specific | Per-token API cost | Optional |

## How to fill in real benchmark numbers

```bash
cd evaluation
python benchmark.py --manifest manifests/demo_manifest.jsonl --providers demo trocr donut
```

This writes `results/run_summary.json` with `mean_cer`, `mean_wer`, `exact_match_accuracy`,
`mean_normalized_edit_distance`, `mean_inference_time_ms`, `diagnosis_accuracy`, and
`mistake_category_macro_f1` per provider, computed from whatever manifest you point it at (the
bundled demo manifest is intentionally tiny — 4 samples — and only useful for smoke-testing the
pipeline, not for drawing real conclusions about model quality; see `DATASET_GUIDE.md` for adding
a real evaluation set).
