"""Classical-ML handwriting provider (no deep learning).

OpenCV segments the page into lines and symbols; each symbol is described by
HOG + zoning features and classified by the trained scikit-learn model in
ml_models/symbol_classifier.joblib (train with `python -m app.ml.train`).
Confidence is the mean top-class probability over all symbols, and each
symbol's bounding box, prediction and runner-up alternatives are returned
in raw_regions so low-confidence symbols can be highlighted for correction.
Each region also carries the symbol's binary "mask" (numpy) so the pipeline
can store crops for active learning; it is removed before persistence.
"""
from __future__ import annotations

import time
from pathlib import Path

from app.ml.artifacts import SYMBOL_MODEL_PATH
from app.providers.base import BaseOCRProvider, OCRResult, ProviderUnavailableError
from app.services.diagnosis import CONFIRM_BELOW

LOW_SYMBOL_CONFIDENCE = CONFIRM_BELOW  # same symbols the student is asked to check


class ClassicalMLProvider(BaseOCRProvider):
    name = "ml"
    requires_api_key = False
    is_local_model = True

    def is_available(self) -> tuple[bool, str | None]:
        if not SYMBOL_MODEL_PATH.exists():
            return False, "symbol model not trained yet (run: python -m app.ml.train)"
        return True, None

    def transcribe(self, image_path: str | Path) -> OCRResult:
        from app.ml.symbol_classifier import SymbolRecognizer

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            recognizer = SymbolRecognizer.load(SYMBOL_MODEL_PATH)
        except ValueError as exc:
            raise ProviderUnavailableError(self.name, str(exc)) from exc
        if recognizer is None:
            raise ProviderUnavailableError(self.name, self.is_available()[1] or "model missing")

        start = time.perf_counter()
        text, confidence, regions = recognizer.transcribe_file(image_path, with_masks=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        warnings = []
        if not regions:
            warnings.append("No handwriting was detected in the image.")
        uncertain = [r for r in regions if r["confidence"] < LOW_SYMBOL_CONFIDENCE]
        if uncertain:
            shown = ", ".join(f"'{r['symbol']}'" for r in uncertain[:5])
            warnings.append(f"{len(uncertain)} symbol(s) recognized with low confidence ({shown}); please check the transcription.")

        return OCRResult(
            text=text,
            confidence=round(confidence, 3),
            provider=self.name,
            is_mock=False,
            raw_regions=regions,
            inference_time_ms=elapsed_ms,
            warnings=warnings,
        )
