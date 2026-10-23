"""ML Lab: model cards, metrics, version history, retraining and a live playground."""
from __future__ import annotations

import base64
import io
from functools import lru_cache

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from app.ml import active_learning, registry
from app.ml.glyph_data import SYMBOLS, augment, find_fonts, render_glyph, _load_font
from app.ml.model_cards import CARDS, headline, load_report
from app.ml.segmentation import segment_page
from app.ml.symbol_classifier import SymbolRecognizer, format_symbols
from app.ml.symbol_features import glyph_features, normalize_glyph

router = APIRouter(prefix="/api/lab", tags=["ml-lab"])


def _png_b64(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@router.get("/models")
def list_models() -> list[dict]:
    out = []
    for name, card in CARDS.items():
        report = load_report(name)
        versions = registry.list_versions(name) if card["retrainable"] else None
        out.append({"name": name, **card, "trained": report is not None, "headline": headline(name, report),
                    "active_version": versions["active"] if versions else None})
    return out


@router.get("/models/{name}")
def model_detail(name: str) -> dict:
    if name not in CARDS:
        raise HTTPException(status_code=404, detail="Unknown model")
    report = load_report(name)
    return {"name": name, **CARDS[name], "report": report, "real_report": load_report(f"{name}_real"),
            "experiments": load_report("paper_experiments"),
            "headline": headline(name, report),
            "versions": registry.list_versions(name) if CARDS[name]["retrainable"] else None}


@router.post("/models/symbol_classifier/retrain")
def retrain_symbols() -> dict:
    if any(j["status"] == "running" for j in active_learning.JOBS.values()):
        raise HTTPException(status_code=409, detail="A retraining job is already running")
    return {"job_id": active_learning.start_retrain_job()}


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = active_learning.JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job


class ActivateIn(BaseModel):
    version: str


@router.post("/models/{name}/activate")
def activate(name: str, payload: ActivateIn) -> dict:
    if name not in CARDS or not CARDS[name]["retrainable"]:
        raise HTTPException(status_code=404, detail="Model has no versions")
    try:
        registry.activate_version(name, payload.version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return registry.list_versions(name)


@router.post("/symbols/predict")
async def predict_symbols(image: UploadFile = File(...)) -> dict:
    recognizer = SymbolRecognizer.load(registry.active_path(active_learning.SYMBOL_MODEL))
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Symbol model not trained (run python -m app.ml.train)")
    try:
        img = Image.open(io.BytesIO(await image.read()))
    except OSError as exc:
        raise HTTPException(status_code=422, detail="Not an image") from exc
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    gray = np.array(img.convert("L"))

    symbols, lines = [], []
    for li, crops in enumerate(segment_page(gray)):
        chars = []
        for crop in crops:
            probs = recognizer.model.predict_proba(glyph_features(crop.mask).reshape(1, -1))[0]
            top = np.argsort(probs)[::-1][:5]
            chars.append(recognizer.classes[top[0]])
            symbols.append({
                "line": li, "bbox": [crop.x, crop.y, crop.w, crop.h],
                "top": [[recognizer.classes[i], round(float(probs[i]), 4)] for i in top],
                "model_view": _png_b64(cv2.resize(255 - normalize_glyph(crop.mask), (112, 112), interpolation=cv2.INTER_NEAREST)),
            })
        lines.append(format_symbols(chars))
    return {"text": "\n".join(lines), "symbols": symbols, "model": recognizer.model_name}


@lru_cache(maxsize=1)
def _dataset_preview() -> dict:
    fonts = find_fonts()
    rng = np.random.default_rng(3)
    picks = fonts[:: max(1, len(fonts) // 4)][:4] or [None]
    tiles = []
    for ch in SYMBOLS:
        row = []
        for fp in picks:
            base = render_glyph(_load_font(fp), ch)
            row.append(base)
            row.append(augment(base, rng))
        tiles.append(row)
    cell = 40
    grid = np.full((cell * len(tiles), cell * len(tiles[0])), 255, np.uint8)
    for r, row in enumerate(tiles):
        for c, mask in enumerate(row):
            g = cv2.resize(255 - normalize_glyph(mask), (cell - 4, cell - 4), interpolation=cv2.INTER_AREA)
            grid[r * cell + 2 : r * cell + cell - 2, c * cell + 2 : c * cell + cell - 2] = g
    return {"fonts": [f.split("/")[-1] for f in fonts], "grid": _png_b64(grid), "grid_rows": SYMBOLS,
            "grid_note": f"Each row: {len(picks)} fonts x (clean render, augmented copy)"}


@router.get("/dataset")
def dataset_info() -> dict:
    preview = _dataset_preview()
    real = load_report("symbol_classifier_real")
    kt_real = load_report("knowledge_tracing_real")
    return {**preview, "classes": SYMBOLS, "corrections": active_learning.correction_counts(),
            "real_datasets": real["datasets"] if real else None,
            "assistments": {k: kt_real[k] for k in ("n_attempts", "n_learners", "n_skills", "n_items")} if kt_real else None}
