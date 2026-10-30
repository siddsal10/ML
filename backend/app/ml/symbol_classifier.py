"""Symbol classifier: model comparison, selection, and page transcription."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.ml.artifacts import SYMBOL_MODEL_PATH, load_bundle, save_bundle
from app.ml.glyph_data import GlyphDataset
from app.ml.segmentation import segment_page
from app.ml.symbol_features import FEATURE_VERSION, glyph_features


def candidate_models(seed: int = 0) -> dict:
    return {
        "svm_rbf": Pipeline([("scale", StandardScaler()), ("clf", SVC(C=10, gamma="scale", probability=True, random_state=seed))]),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "knn": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))]),
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=0.5))]),
    }


def train_symbol_classifier(data: GlyphDataset, seed: int = 0, out_path: Path = SYMBOL_MODEL_PATH,
                            always_train_groups: set[int] | None = None) -> tuple[dict, "SymbolRecognizer", np.ndarray]:
    """Compare candidate models on a font-disjoint split, refit the best on all data, save it.

    Groups in `always_train_groups` (e.g. student corrections) never enter the
    test split. Returns the report, a recognizer built from the train-split
    model (never saw the held-out fonts) for honest end-to-end evaluation,
    and the test indices."""
    always = always_train_groups or set()
    eligible = np.array([i for i, g in enumerate(data.groups) if g not in always])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(splitter.split(data.X[eligible], data.y[eligible], data.groups[eligible]))
    test_idx = eligible[te]
    train_idx = np.setdiff1d(np.arange(len(data.y)), test_idx)
    held_out_fonts = sorted({data.group_names[g] for g in data.groups[test_idx]})

    results = {}
    for name, model in candidate_models(seed).items():
        t0 = time.perf_counter()
        model.fit(data.X[train_idx], data.y[train_idx])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = model.predict(data.X[test_idx])
        pred_ms = (time.perf_counter() - t0) * 1000 / len(test_idx)
        results[name] = {
            "accuracy": round(float(accuracy_score(data.y[test_idx], pred)), 4),
            "macro_f1": round(float(f1_score(data.y[test_idx], pred, average="macro")), 4),
            "fit_seconds": round(fit_s, 2),
            "predict_ms_per_symbol": round(pred_ms, 4),
        }

    best = max(results, key=lambda n: results[n]["macro_f1"])
    best_model = candidate_models(seed)[best]
    best_model.fit(data.X[train_idx], data.y[train_idx])
    labels = sorted(set(data.y))
    cm = confusion_matrix(data.y[test_idx], best_model.predict(data.X[test_idx]), labels=labels)

    # Final model is refit on every font so deployment sees all styles.
    final = candidate_models(seed)[best]
    final.fit(data.X, data.y)
    report = {
        "task": "symbol_recognition",
        "split": "font-disjoint (GroupShuffleSplit, 25% of fonts held out)",
        "n_samples": int(len(data.y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_fonts": len(data.group_names),
        "held_out_fonts": held_out_fonts,
        "models": results,
        "selected_model": best,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }
    save_bundle(
        {"model": final, "model_name": best, "classes": list(final.classes_), "feature_version": FEATURE_VERSION, "metrics": report},
        out_path,
    )
    held_out_recognizer = SymbolRecognizer({"model": best_model, "model_name": best, "feature_version": FEATURE_VERSION})
    return report, held_out_recognizer, test_idx


def format_symbols(symbols: list[str]) -> str:
    """Turn a recognized symbol sequence into canonical linear notation.

    - 'x' written between two numbers is the multiplication sign ('3 x 4').
    - Binary operators and '=' get one space on each side; a '-' at the start,
      after another operator, or after '(' is unary and stays attached.
    Handwritten gaps are too irregular to be trusted as word spaces.
    """
    syms = list(symbols)
    for i in range(1, len(syms) - 1):
        if syms[i] == "x" and (syms[i - 1].isdigit() or syms[i - 1] == ")") and (syms[i + 1].isdigit() or syms[i + 1] == "("):
            syms[i] = "*"
    out: list[str] = []
    prev = None
    for ch in syms:
        if ch in "+*/=" or (ch == "-" and prev is not None and prev not in "+-*/=("):
            out.append(f" {ch} ")
        else:
            out.append(ch)
        prev = ch
    return "".join(out).strip()


# Line grammar for syntax-constrained decoding: which kind of symbol may follow
# which in one line of arithmetic / linear-equation working. '+' and '-' can be
# signs ('+5', '3 * -2') as well as operators. 'x' is both the variable and the
# handwritten multiplication sign, so it may sit between digits.
_KIND = {**{d: "digit" for d in "0123456789"}, "x": "x", "+": "sign", "-": "sign", "/": "op",
         "=": "eq", "(": "lp", ")": "rp"}
_FOLLOWS = {
    None: {"digit", "x", "sign", "lp"},
    "digit": {"digit", "x", "op", "sign", "eq", "lp", "rp"},
    "x": {"digit", "x", "op", "sign", "eq", "lp", "rp"},
    "op": {"digit", "x", "sign", "lp"},
    "sign": {"digit", "x", "lp"},
    "eq": {"digit", "x", "sign", "lp"},
    "lp": {"digit", "x", "sign", "lp"},
    "rp": {"x", "op", "sign", "eq", "lp", "rp"},
}
_CAN_END = {"digit", "x", "rp"}
_MAX_DEPTH = 4


def is_well_formed(symbols: list[str]) -> bool:
    kind, depth = None, 0
    for s in symbols:
        k = _KIND[s]
        depth += (k == "lp") - (k == "rp")
        if k not in _FOLLOWS[kind] or not 0 <= depth <= _MAX_DEPTH:
            return False
        kind = k
    return kind in _CAN_END and depth == 0


def decode_line(probs: np.ndarray, classes: list[str], top_k: int = 3) -> list[int]:
    """Most probable class sequence (sum of log-probabilities over each symbol's
    top-k classes) that is a well-formed line, by Viterbi over (kind, bracket
    depth) states. Only syntax is used, never arithmetic truth, so a student's
    wrong answer is read as written. Falls back to per-symbol argmax when no
    well-formed reading exists."""
    argmax = [int(i) for i in np.argmax(probs, axis=1)]
    beams: dict[tuple, tuple[float, list[int]]] = {(None, 0): (0.0, [])}
    for p in probs:
        nxt: dict[tuple, tuple[float, list[int]]] = {}
        for (kind, depth), (score, path) in beams.items():
            for c in np.argsort(p)[::-1][:top_k]:
                k = _KIND[classes[c]]
                d = depth + (k == "lp") - (k == "rp")
                if k not in _FOLLOWS[kind] or not 0 <= d <= _MAX_DEPTH:
                    continue
                s = score + float(np.log(max(p[c], 1e-12)))
                if (k, d) not in nxt or s > nxt[(k, d)][0]:
                    nxt[(k, d)] = (s, path + [int(c)])
        if not nxt:
            return argmax
        beams = nxt
    ends = [v for (k, d), v in beams.items() if k in _CAN_END and d == 0]
    return max(ends, key=lambda v: v[0])[1] if ends else argmax


class SymbolRecognizer:
    """Transcribes a page image using the trained symbol classifier."""

    def __init__(self, bundle: dict, constrained: bool = True):
        if bundle.get("feature_version") != FEATURE_VERSION:
            raise ValueError("symbol model was trained with a different feature version; retrain it")
        self.model = bundle["model"]
        self.model_name = bundle.get("model_name", "unknown")
        self.classes = list(self.model.classes_)
        self.constrained = constrained

    @classmethod
    def load(cls, path: Path = SYMBOL_MODEL_PATH) -> "SymbolRecognizer | None":
        bundle = load_bundle(str(path))
        return cls(bundle) if bundle is not None else None

    def transcribe_gray(self, gray: np.ndarray, with_masks: bool = False, normalize: bool = True) -> tuple[str, float, list[dict]]:
        lines_out, confidences, regions = [], [], []
        for li, crops in enumerate(segment_page(gray, normalize=normalize)):
            feats = np.vstack([glyph_features(c.mask) for c in crops])
            probs = self.model.predict_proba(feats)
            chosen = decode_line(probs, self.classes) if self.constrained else np.argmax(probs, axis=1)
            symbols = []
            for crop, p, c in zip(crops, probs, chosen):
                ch = self.classes[c]
                others = [i for i in np.argsort(p)[::-1][:3] if i != c][:2]
                symbols.append(ch)
                confidences.append(float(p[c]))
                regions.append({
                    "line": li,
                    "bbox": [crop.x, crop.y, crop.w, crop.h],
                    "symbol": ch,
                    "confidence": round(float(p[c]), 3),
                    "alternatives": [[self.classes[i], round(float(p[i]), 3)] for i in others],
                    **({"mask": crop.mask} if with_masks else {}),
                })
            lines_out.append(format_symbols(symbols))
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return "\n".join(lines_out), confidence, regions

    def transcribe_file(self, image_path: str | Path, with_masks: bool = False) -> tuple[str, float, list[dict]]:
        gray = load_gray(image_path)
        if gray is None:
            raise ValueError(f"could not read image: {image_path}")
        return self.transcribe_gray(gray, with_masks=with_masks)


def load_gray(image_path: str | Path) -> np.ndarray | None:
    """Read an image as grayscale, honouring EXIF orientation and flattening
    transparency onto white (canvas drawings are transparent PNGs)."""
    from PIL import Image, ImageOps

    try:
        img = ImageOps.exif_transpose(Image.open(image_path))
    except (OSError, ValueError):
        return None
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return np.array(img.convert("L"))
