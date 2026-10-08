from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from app.providers.base import ProviderUnavailableError
from app.providers.demo_provider import DemoOCRProvider
from app.providers.registry import get_provider
from app.providers.vision_api_provider import VisionAPIProvider


def test_demo_provider_is_always_available():
    provider = DemoOCRProvider()
    available, reason = provider.is_available()
    assert available is True
    assert reason is None


def test_demo_provider_returns_mock_result_for_unknown_image():
    provider = DemoOCRProvider()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "unknown.png"
        Image.new("RGB", (20, 20), color="white").save(path)

        result = provider.transcribe(path)

        assert result.is_mock is True
        assert result.provider == "demo"
        # Must NOT fabricate a confident transcription for an unrecognized image.
        assert result.confidence < 0.5
        assert len(result.warnings) > 0


def test_demo_provider_raises_for_missing_file():
    provider = DemoOCRProvider()
    with pytest.raises(FileNotFoundError):
        provider.transcribe("/nonexistent/path/image.png")


def test_vision_api_provider_unavailable_without_config(temp_env, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("VISION_API_BACKEND", "none")
    get_settings.cache_clear()

    provider = VisionAPIProvider()
    available, reason = provider.is_available()
    assert available is False
    assert reason is not None


def test_vision_api_provider_raises_provider_unavailable_error(temp_env, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("VISION_API_BACKEND", "none")
    get_settings.cache_clear()

    provider = VisionAPIProvider()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "img.png"
        Image.new("RGB", (10, 10)).save(path)
        with pytest.raises(ProviderUnavailableError):
            provider.transcribe(path)


def test_registry_returns_correct_provider_instances():
    assert get_provider("ml").name == "ml"
    assert get_provider("demo").name == "demo"
    assert get_provider("trocr").name == "trocr"
    assert get_provider("donut").name == "donut"
    assert get_provider("vision_api").name == "vision_api"


def test_registry_raises_for_unknown_provider():
    with pytest.raises(ValueError):
        get_provider("not_a_real_provider")


def test_trocr_provider_unavailable_gracefully_when_deps_missing(monkeypatch):
    """Even if torch/transformers ARE installed in this environment, verify
    the availability-check contract: is_available() returns a tuple and
    transcribe() never fabricates output if the provider reports unavailable.
    """
    from app.providers.trocr_provider import TrOCRProvider

    provider = TrOCRProvider()
    available, reason = provider.is_available()
    assert isinstance(available, bool)
    if not available:
        with pytest.raises(ProviderUnavailableError):
            provider.transcribe("/nonexistent.png")
