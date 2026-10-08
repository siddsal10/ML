"""Where trained model artifacts live, and how they are loaded.

Models are saved as joblib bundles: {"model", "classes", "feature_version",
"metrics", ...}. Loading is cached per process. A missing artifact is not an
error at import time - callers check `load_bundle(...) is None` and degrade
honestly (provider unavailable / rule-based fallback).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("ML_MODEL_DIR", BACKEND_ROOT / "ml_models"))

SYMBOL_MODEL_PATH = MODEL_DIR / "symbol_classifier.joblib"
MISTAKE_MODEL_PATH = MODEL_DIR / "mistake_classifier.joblib"


@lru_cache(maxsize=None)
def load_bundle(path: str) -> dict | None:
    import joblib

    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


def save_bundle(bundle: dict, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    load_bundle.cache_clear()
