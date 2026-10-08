"""Provider registry: maps provider name -> provider instance.

All provider selection in the application goes through get_provider() so
there is a single, auditable place that decides which implementation
services a request.
"""
from __future__ import annotations

from app.providers.base import BaseOCRProvider
from app.providers.demo_provider import DemoOCRProvider
from app.providers.donut_provider import get_donut_provider
from app.providers.ml_provider import ClassicalMLProvider
from app.providers.trocr_provider import get_trocr_provider
from app.providers.vision_api_provider import VisionAPIProvider

_demo_singleton = DemoOCRProvider()
_vision_api_singleton = VisionAPIProvider()
_ml_singleton = ClassicalMLProvider()


def get_provider(name: str) -> BaseOCRProvider:
    """Return the provider instance for `name`.

    "ml" is the classical-ML recognizer (the project's primary provider).
    trocr / donut are optional deep-learning baselines kept for comparison
    (they need requirements-dl.txt). Local model providers are returned as singletons that
    lazily load their weights on first .transcribe() call. Raises
    ValueError for unknown provider names.
    """
    if name == "demo":
        return _demo_singleton
    if name == "ml":
        return _ml_singleton
    if name == "trocr":
        return get_trocr_provider()
    if name == "donut":
        return get_donut_provider()
    if name == "vision_api":
        return _vision_api_singleton
    raise ValueError(f"Unknown provider: {name}")


def list_providers() -> list[BaseOCRProvider]:
    return [
        _ml_singleton,
        _demo_singleton,
        get_trocr_provider(),
        get_donut_provider(),
        _vision_api_singleton,
    ]


def describe_all_providers() -> list[dict]:
    return [p.describe() for p in list_providers()]
