"""Versioned model registry.

Loaders always read the *active* artifact at ml_models/<name>.joblib.
Every trained model is also stored as ml_models/versions/<name>/<vN>.joblib
and listed in ml_models/registry.json with its headline metrics, so the ML
Lab can compare versions and switch the active one.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.ml.artifacts import MODEL_DIR, load_bundle, save_bundle

REGISTRY_PATH = MODEL_DIR / "registry.json"
VERSIONS_DIR = MODEL_DIR / "versions"


def _read() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _write(reg: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


def active_path(name: str) -> Path:
    return MODEL_DIR / f"{name}.joblib"


def _ensure_initialized(reg: dict, name: str) -> dict:
    """A shipped model with no registry entry is registered as v1."""
    if name not in reg and active_path(name).exists():
        dest = VERSIONS_DIR / name / "v1.joblib"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(active_path(name), dest)
        reg[name] = {"active": "v1", "versions": [{"version": "v1", "created_at": None, "notes": "shipped model", "metrics": {}}]}
    return reg


def list_versions(name: str) -> dict:
    reg = _ensure_initialized(_read(), name)
    _write(reg)
    return reg.get(name, {"active": None, "versions": []})


def register(name: str, bundle: dict, metrics: dict, notes: str = "", activate: bool = False) -> str:
    # A full training run that activates its result replaces the active file
    # itself, so it must not first snapshot that file as a "shipped" v1.
    reg = _read() if activate else _ensure_initialized(_read(), name)
    entry = reg.setdefault(name, {"active": None, "versions": []})
    version = f"v{len(entry['versions']) + 1}"
    save_bundle(bundle, VERSIONS_DIR / name / f"{version}.joblib")
    entry["versions"].append({
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
        "metrics": metrics,
    })
    _write(reg)
    if activate:
        activate_version(name, version)
    return version


def activate_version(name: str, version: str) -> None:
    reg = _ensure_initialized(_read(), name)
    src = VERSIONS_DIR / name / f"{version}.joblib"
    if name not in reg or not src.exists():
        raise ValueError(f"unknown version {name}:{version}")
    shutil.copy2(src, active_path(name))
    reg[name]["active"] = version
    _write(reg)
    load_bundle.cache_clear()


def active_version(name: str) -> str | None:
    return list_versions(name).get("active")
