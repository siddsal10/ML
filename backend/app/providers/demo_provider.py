"""Deterministic demo/mock OCR provider.

This provider lets the entire application be demonstrated with zero API
keys and zero downloaded models. It is NOT a real handwriting recognizer.

Behavior:
- It recognizes a small, fixed set of bundled demo images (matched by file
  hash) and returns a deterministic, pre-written transcription for each.
  This lets the seeded demo workflow (correct + incorrect examples) work
  reliably end to end.
- For any *other* uploaded image, it does not fabricate a transcription.
  Instead it returns a low-confidence placeholder together with an explicit
  warning, and the frontend must let the student edit/correct the
  transcription before diagnosis is requested (see /api endpoints and the
  "OCR correction" workflow in the frontend AnalysisPage).

Every OCRResult produced here has is_mock=True so the UI can render an
unmistakable "Demo mode" badge and callers know never to present this as
real model inference.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from app.providers.base import BaseOCRProvider, OCRResult

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _BACKEND_ROOT / "data" / "demo_samples" / "manifest.json"

# Deterministic demo transcriptions, keyed by a human-readable slug.
# scripts/generate_demo_images.py renders one PNG per entry and writes
# data/demo_samples/manifest.json mapping the *actual* sha256 of each
# rendered file back to its slug. This provider loads that manifest at
# runtime, so hash matching is always correct regardless of how PIL/OpenCV
# happen to encode the PNG bytes on a given platform - we never try to
# predict those bytes ourselves.
DEMO_SAMPLES: dict[str, dict] = {
    # "correct" arithmetic example: 12 + 15 = 27
    "demo_correct_arithmetic": {
        "text": "12 + 15 = 27",
        "confidence": 0.94,
    },
    # "incorrect" example containing a sign error: -3 + 5 = -8 (should be 2)
    "demo_sign_error": {
        "text": "-3 + 5 = -8",
        "confidence": 0.91,
    },
    # simple linear algebra, correct: 2x + 4 = 10 -> x = 3
    "demo_correct_algebra": {
        "text": "2x + 4 = 10\nx = 3",
        "confidence": 0.89,
    },
    # linear algebra with an algebraic manipulation error: 2x + 4 = 10 -> x = 7
    "demo_algebra_error": {
        "text": "2x + 4 = 10\nx = 7",
        "confidence": 0.87,
    },
}


def _file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_hash_table() -> dict[str, dict]:
    """Build file-hash -> demo sample lookup from the generated manifest.

    Returns an empty mapping (not an error) if the manifest hasn't been
    generated yet - in that case the demo provider still runs, it just
    won't recognize any uploaded image as a known demo sample until
    `python scripts/generate_demo_images.py` has been run.
    """
    if not _MANIFEST_PATH.exists():
        return {}
    try:
        manifest = json.loads(_MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    table = {}
    for slug, entry in manifest.items():
        sample = DEMO_SAMPLES.get(slug)
        file_hash = entry.get("sha256")
        if sample and file_hash:
            table[file_hash] = sample
    return table


# Built once at import time; the manifest only changes when the demo-image
# generation script is re-run (a dev-time operation), so caching is safe.
_HASH_TO_SAMPLE = _load_hash_table()


class DemoOCRProvider(BaseOCRProvider):
    name = "demo"
    requires_api_key = False
    is_local_model = False

    def is_available(self) -> tuple[bool, str | None]:
        # The demo provider is always available - that's the point.
        return True, None

    def transcribe(self, image_path: str | Path) -> OCRResult:
        start = time.perf_counter()
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        file_hash = _file_hash(image_path)
        sample = _HASH_TO_SAMPLE.get(file_hash)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if sample is not None:
            return OCRResult(
                text=sample["text"],
                confidence=sample["confidence"],
                provider=self.name,
                is_mock=True,
                inference_time_ms=elapsed_ms,
                warnings=[],
            )

        # Unknown image in demo mode: be honest, do not invent text.
        return OCRResult(
            text="",
            confidence=0.15,
            provider=self.name,
            is_mock=True,
            inference_time_ms=elapsed_ms,
            warnings=[
                "Demo mode cannot reliably transcribe arbitrary uploaded "
                "images because no real OCR model is loaded. Please type or "
                "correct the transcription manually, then re-run analysis.",
            ],
        )
