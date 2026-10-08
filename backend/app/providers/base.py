"""Abstract provider interface shared by all OCR / vision-language providers.

Every provider (demo, TrOCR, Donut, vision API) implements this common
interface so the rest of the application can treat them interchangeably.
Providers must NEVER fabricate a result: if a provider cannot run (missing
dependency, missing API key, model failed to load), it must raise
ProviderUnavailableError instead of returning invented text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class ProviderUnavailableError(Exception):
    """Raised when a provider cannot service a request right now."""

    def __init__(self, provider_name: str, reason: str):
        self.provider_name = provider_name
        self.reason = reason
        super().__init__(f"Provider '{provider_name}' unavailable: {reason}")


@dataclass
class OCRResult:
    """Standardized output of a provider's transcribe() call."""

    text: str
    confidence: float  # 0.0 - 1.0, provider's own estimate
    provider: str
    is_mock: bool = False
    raw_regions: list[dict] = field(default_factory=list)  # optional region-level detail
    inference_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


class BaseOCRProvider(ABC):
    """Common interface implemented by every OCR / vision-language provider."""

    name: str = "base"
    requires_api_key: bool = False
    is_local_model: bool = False

    @abstractmethod
    def is_available(self) -> tuple[bool, str | None]:
        """Return (available, reason_if_not)."""

    @abstractmethod
    def transcribe(self, image_path: str | Path) -> OCRResult:
        """Transcribe handwritten content from an image.

        Must raise ProviderUnavailableError rather than returning a
        fabricated result if the provider cannot actually run.
        """

    def describe(self) -> dict:
        available, reason = self.is_available()
        return {
            "name": self.name,
            "available": available,
            "reason": reason,
            "requires_api_key": self.requires_api_key,
            "is_local_model": self.is_local_model,
        }
