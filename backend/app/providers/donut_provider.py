"""Donut adapter provider.

Donut (naver-clova-ix/donut-base) is included as a benchmarking adapter per
the project brief. It is NOT wired into the default analyze pipeline (TrOCR
is the default local model) because Donut is heavier and its base
checkpoint is not fine-tuned for free-form handwritten arithmetic, so its
output quality on this task is unproven. It is provided so the benchmark
module can load it lazily and compare it against TrOCR / vision-API
providers when the user explicitly opts in.

Like TrOCRProvider, the model is loaded lazily on first use and cached.
"""
from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

from app.core.config import get_settings
from app.providers.base import BaseOCRProvider, OCRResult, ProviderUnavailableError

_load_lock = Lock()


class DonutProvider(BaseOCRProvider):
    name = "donut"
    requires_api_key = False
    is_local_model = True

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
        self._load_error: str | None = None

    def is_available(self) -> tuple[bool, str | None]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return False, f"required package not installed: {exc}"
        return True, None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        with _load_lock:
            if self._model is not None or self._load_error is not None:
                return
            try:
                import torch
                from transformers import DonutProcessor, VisionEncoderDecoderModel

                settings = get_settings()
                device = settings.HF_DEVICE
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                cache_dir = settings.HF_CACHE_DIR or None
                self._processor = DonutProcessor.from_pretrained(
                    settings.DONUT_MODEL_NAME, cache_dir=cache_dir
                )
                self._model = VisionEncoderDecoderModel.from_pretrained(
                    settings.DONUT_MODEL_NAME, cache_dir=cache_dir
                ).to(device)
                self._model.eval()
                self._device = device
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)

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
            task_prompt = "<s_synthdog>"
            decoder_input_ids = self._processor.tokenizer(
                task_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(self._device)
            pixel_values = self._processor(image, return_tensors="pt").pixel_values.to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self._model.decoder.config.max_position_embeddings,
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                    eos_token_id=self._processor.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
            sequence = self._processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self._processor.tokenizer.eos_token, "")
            sequence = sequence.replace(self._processor.tokenizer.pad_token, "")
            text = sequence.replace(task_prompt, "").strip()
        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailableError(self.name, f"inference failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        return OCRResult(
            text=text,
            # Donut's base checkpoint has no confidence head suited to this
            # task; we deliberately report a conservative fixed estimate
            # rather than compute a misleading number.
            confidence=0.5,
            provider=self.name,
            is_mock=False,
            inference_time_ms=elapsed_ms,
            warnings=["Donut base checkpoint is not fine-tuned for handwritten math; treat output as low confidence."],
        )


_donut_singleton: DonutProvider | None = None


def get_donut_provider() -> DonutProvider:
    global _donut_singleton
    if _donut_singleton is None:
        _donut_singleton = DonutProvider()
    return _donut_singleton
