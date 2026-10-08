"""TrOCR local handwriting-recognition provider.

Loads microsoft/trocr-base-handwritten (or a configured alternative) lazily
on first use and caches the model/processor in-process for subsequent
requests. Never downloads anything at application startup.
"""
from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

from app.core.config import get_settings
from app.providers.base import BaseOCRProvider, OCRResult, ProviderUnavailableError

_load_lock = Lock()


class TrOCRProvider(BaseOCRProvider):
    name = "trocr"
    requires_api_key = False
    is_local_model = True

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
        self._load_error: str | None = None

    # -- availability -----------------------------------------------------
    def is_available(self) -> tuple[bool, str | None]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return False, f"required package not installed: {exc}"
        return True, None

    # -- lazy loading -------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        with _load_lock:
            if self._model is not None or self._load_error is not None:
                return
            try:
                import torch
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                settings = get_settings()
                device = settings.HF_DEVICE
                if device == "auto":
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"

                cache_dir = settings.HF_CACHE_DIR or None
                self._processor = TrOCRProcessor.from_pretrained(
                    settings.TROCR_MODEL_NAME, cache_dir=cache_dir
                )
                self._model = VisionEncoderDecoderModel.from_pretrained(
                    settings.TROCR_MODEL_NAME, cache_dir=cache_dir
                ).to(device)
                self._model.eval()
                self._device = device
            except Exception as exc:  # noqa: BLE001 - surface as unavailable
                self._load_error = str(exc)

    # -- inference ----------------------------------------------------------
    def transcribe(self, image_path: str | Path) -> OCRResult:
        available, reason = self.is_available()
        if not available:
            raise ProviderUnavailableError(self.name, reason or "unknown")

        self._ensure_loaded()
        if self._load_error is not None:
            raise ProviderUnavailableError(self.name, self._load_error)

        import torch
        from PIL import Image

        start = time.perf_counter()
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self._processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self._device)

            with torch.no_grad():
                generated = self._model.generate(
                    pixel_values,
                    max_length=128,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            token_ids = generated.sequences
            text = self._processor.batch_decode(token_ids, skip_special_tokens=True)[0]

            # Approximate a confidence score from generation scores when
            # available; otherwise fall back to a fixed heuristic value.
            confidence = 0.75
            if getattr(generated, "scores", None):
                import torch.nn.functional as F

                probs = []
                for score in generated.scores:
                    p = F.softmax(score, dim=-1)
                    probs.append(float(p.max()))
                if probs:
                    confidence = sum(probs) / len(probs)

        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailableError(self.name, f"inference failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        return OCRResult(
            text=text.strip(),
            confidence=round(min(max(confidence, 0.0), 1.0), 3),
            provider=self.name,
            is_mock=False,
            inference_time_ms=elapsed_ms,
        )


# Module-level singleton so the model is cached across requests within the
# same process (lazy: nothing loads until .transcribe() is first called).
_trocr_singleton: TrOCRProvider | None = None


def get_trocr_provider() -> TrOCRProvider:
    global _trocr_singleton
    if _trocr_singleton is None:
        _trocr_singleton = TrOCRProvider()
    return _trocr_singleton
