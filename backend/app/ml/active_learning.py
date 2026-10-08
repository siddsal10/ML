"""Active learning for the symbol recognizer.

1. Every symbol the recognizer reads is stored with its crop and its
   confidence margin; the UI highlights the least-certain ones first
   (uncertainty sampling).
2. When a student corrects a symbol, the crop is saved as a labelled
   training example under data/symbol_crops/labelled/<label>/.
3. Retraining = synthetic glyphs + every collected correction (corrections are
   always in the training split). The new model is registered as a new
   version and compared with the active one on the same font-disjoint test
   split; activation is a separate, explicit step.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from app.ml import registry
from app.ml.artifacts import BACKEND_ROOT, load_bundle
from app.ml.glyph_data import SYMBOL_DIR_NAMES, SYMBOLS, GlyphDataset, generate_dataset
from app.ml.real_symbols import cached_real_sets
from app.ml.symbol_classifier import train_symbol_classifier

CROP_DIR = BACKEND_ROOT / "data" / "symbol_crops"
LABELLED_DIR = CROP_DIR / "labelled"
SYMBOL_MODEL = "symbol_classifier"


def save_crop(submission_id: str, idx: int, mask: np.ndarray) -> str:
    path = CROP_DIR / "raw" / submission_id / f"{idx}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), 255 - mask)  # store dark ink on white, like a scan
    return str(path)


def save_correction(crop_path: str, label: str) -> str | None:
    if label not in SYMBOLS or not crop_path or not Path(crop_path).exists():
        return None
    dest = LABELLED_DIR / SYMBOL_DIR_NAMES.get(label, label) / f"{uuid.uuid4().hex[:12]}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(crop_path).read_bytes())
    return str(dest)


def correction_counts() -> dict[str, int]:
    names = {v: k for k, v in SYMBOL_DIR_NAMES.items()}
    if not LABELLED_DIR.exists():
        return {}
    return {names.get(d.name, d.name): len(list(d.glob("*.png"))) for d in sorted(LABELLED_DIR.iterdir()) if d.is_dir()}


# --- background retraining jobs ------------------------------------------------

JOBS: dict[str, dict] = {}


def _evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {"accuracy": round(float(accuracy_score(y, pred)), 4), "macro_f1": round(float(f1_score(y, pred, average="macro")), 4)}


def retrain_symbol_model(samples_per_font: int = 30, seed: int = 0, job_id: str | None = None, use_real: bool = True) -> dict:
    def step(msg: str):
        if job_id:
            JOBS[job_id]["log"].append(msg)

    step("Generating synthetic glyphs + loading corrections")
    extra = LABELLED_DIR if LABELLED_DIR.exists() and any(LABELLED_DIR.iterdir()) else None
    data = generate_dataset(samples_per_font=samples_per_font, seed=seed, extra_dir=extra)
    always_train = {i for i, n in enumerate(data.group_names) if n.startswith("extra:")}

    real = cached_real_sets() if use_real else None
    if real is not None:
        step(f"Adding {len(real['train_y'])} real handwritten symbols (CROHME + HASYv2)")
        offset = len(data.group_names)
        groups = real["train_groups"]
        remap = {g: offset + i for i, g in enumerate(np.unique(groups))}
        data = GlyphDataset(
            np.vstack([data.X, real["train_X"]]), np.concatenate([data.y, real["train_y"]]),
            np.concatenate([data.groups, np.array([remap[g] for g in groups])]),
            data.group_names + [f"real:{g}" for g in np.unique(groups)],
        )

    step(f"Training candidate models on {len(data.y)} symbols")
    candidate_path = registry.VERSIONS_DIR / SYMBOL_MODEL / "_candidate.joblib"
    report, _, _ = train_symbol_classifier(data, seed=seed, out_path=candidate_path, always_train_groups=always_train)

    step("Comparing with the active model")
    new_bundle = load_bundle(str(candidate_path))
    old_bundle = load_bundle(str(registry.active_path(SYMBOL_MODEL)))
    if real is not None:
        # Fair comparison: neither model was trained on the CROHME test years.
        comparison = {"benchmark": "CROHME 2014/2016/2019 test symbols (real, unseen writers)",
                      "new": _evaluate(new_bundle["model"], real["test_X"], real["test_y"])}
        if old_bundle is not None:
            comparison["active"] = _evaluate(old_bundle["model"], real["test_X"], real["test_y"])
    else:
        # Deployed models are refit on every font, so compare held-out-font scores
        # from each version's own training run (same seed -> same held-out fonts).
        best = report["models"][report["selected_model"]]
        comparison = {"benchmark": "held-out fonts (synthetic)", "new": {"accuracy": best["accuracy"], "macro_f1": best["macro_f1"]}}
        versions = registry.list_versions(SYMBOL_MODEL)
        active_entry = next((v for v in versions["versions"] if v["version"] == versions["active"]), None)
        if active_entry and active_entry["metrics"].get("held_out_macro_f1") is not None:
            comparison["active"] = {"macro_f1": active_entry["metrics"]["held_out_macro_f1"]}
    corr_idx = np.array([i for i, g in enumerate(data.groups) if g in always_train], dtype=int)
    if len(corr_idx):
        comparison["corrections_in_sample"] = {
            "n": int(len(corr_idx)),
            "active_accuracy": _evaluate(old_bundle["model"], data.X[corr_idx], data.y[corr_idx])["accuracy"] if old_bundle else None,
            "new_accuracy": _evaluate(new_bundle["model"], data.X[corr_idx], data.y[corr_idx])["accuracy"],
            "note": "corrections are in the new model's training data, so new_accuracy is in-sample",
        }
    version = registry.register(
        SYMBOL_MODEL, new_bundle,
        metrics={"held_out_macro_f1": comparison["new"]["macro_f1"], "benchmark": comparison["benchmark"],
                 "n_corrections": int(len(corr_idx)), "algorithm": report["selected_model"]},
        notes=f"retrained with {len(corr_idx)} student corrections",
    )
    candidate_path.unlink(missing_ok=True)
    load_bundle.cache_clear()
    step(f"Registered {version} (not active yet)")
    return {"version": version, "comparison": comparison, "selected_model": report["selected_model"]}


def start_retrain_job(samples_per_font: int = 30) -> str:
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"id": job_id, "status": "running", "log": [], "result": None, "started_at": time.time()}

    def run():
        try:
            JOBS[job_id]["result"] = retrain_symbol_model(samples_per_font=samples_per_font, job_id=job_id)
            JOBS[job_id]["status"] = "done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["log"].append(f"Error: {exc}")
        JOBS[job_id]["finished_at"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return job_id
