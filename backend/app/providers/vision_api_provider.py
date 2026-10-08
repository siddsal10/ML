"""Vision-language API provider.

Wraps either the Anthropic Claude API or a generic OpenAI-compatible vision
endpoint (e.g. an open Qwen-VL deployment) behind the common provider
interface. Selected via VISION_API_BACKEND in the environment. If no
backend is configured or no key is present, is_available() returns False
and callers get a ProviderUnavailableError rather than a fabricated result.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from app.core.config import get_settings
from app.providers.base import BaseOCRProvider, OCRResult, ProviderUnavailableError

_PROMPT = (
    "You are transcribing a photo of a student's handwritten math or "
    "arithmetic work. Return ONLY the transcribed text/expressions, "
    "preserving line breaks between steps. Do not add commentary."
)


class VisionAPIProvider(BaseOCRProvider):
    name = "vision_api"
    requires_api_key = True
    is_local_model = False

    def is_available(self) -> tuple[bool, str | None]:
        settings = get_settings()
        if settings.VISION_API_BACKEND == "none":
            return False, "no vision API backend configured (set VISION_API_BACKEND)"
        if settings.VISION_API_BACKEND == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                return False, "ANTHROPIC_API_KEY not set"
            try:
                import anthropic  # noqa: F401
            except ImportError:
                return False, "anthropic package not installed"
            return True, None
        if settings.VISION_API_BACKEND == "openai_compatible":
            if not settings.OPENAI_COMPATIBLE_API_KEY or not settings.OPENAI_COMPATIBLE_BASE_URL:
                return False, "OPENAI_COMPATIBLE_API_KEY / OPENAI_COMPATIBLE_BASE_URL not set"
            try:
                import httpx  # noqa: F401
            except ImportError:
                return False, "httpx package not installed"
            return True, None
        return False, "unknown VISION_API_BACKEND"

    def transcribe(self, image_path: str | Path) -> OCRResult:
        available, reason = self.is_available()
        if not available:
            raise ProviderUnavailableError(self.name, reason or "unknown")

        settings = get_settings()
        start = time.perf_counter()
        image_path = Path(image_path)
        media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        b64_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        try:
            if settings.VISION_API_BACKEND == "anthropic":
                text = self._call_anthropic(b64_data, media_type, settings)
            else:
                text = self._call_openai_compatible(b64_data, media_type, settings)
        except ProviderUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailableError(self.name, f"API call failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        return OCRResult(
            text=text.strip(),
            # Vision APIs used here don't expose token-level confidence;
            # we report a fixed conservative estimate rather than invent one.
            confidence=0.8,
            provider=self.name,
            is_mock=False,
            inference_time_ms=elapsed_ms,
        )

    @staticmethod
    def _call_anthropic(b64_data: str, media_type: str, settings) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64_data},
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        )
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "\n".join(parts)

    @staticmethod
    def _call_openai_compatible(b64_data: str, media_type: str, settings) -> str:
        import httpx

        url = settings.OPENAI_COMPATIBLE_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_COMPATIBLE_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.OPENAI_COMPATIBLE_MODEL or "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
                        },
                    ],
                }
            ],
            "max_tokens": 512,
        }
        resp = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
