"""Real handwritten symbols for the recognizer: CROHME + HASYv2.

CROHME (Hugging Face mirror Neeze/CROHME-full) ships whole expressions with
LaTeX labels, not symbol boxes. Symbols are harvested by *count alignment*:
for a linear expression (no fractions, scripts, roots ...), segment the image
with the same OpenCV segmenter used at inference; if the number of segmented
symbols equals the number of LaTeX tokens, label the crops in left-to-right
order and keep those whose label is one of our 17 classes. Mismatched
expressions are skipped, which keeps label noise low.

The official CROHME test years (2014/2016/2019) are written by different
people from the training set, so they give a writer-independent benchmark both
at symbol level and end to end (whole expression transcription).

HASYv2 adds real single symbols (digits, + - x / and the multiplication sign)
from ~480 writers; it has no '=' or brackets.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.ml.artifacts import BACKEND_ROOT
from app.ml.glyph_data import SYMBOLS
from app.ml.segmentation import binarize, segment_symbols
from app.ml.symbol_features import glyph_features

EXTERNAL = BACKEND_ROOT / "data" / "external"
CROHME_DIR = EXTERNAL / "crohme"
HASY_DIR = EXTERNAL / "hasy"
TRAIN_FILES = ["train-00000-of-00002", "train-00001-of-00002"]
TEST_FILES = {"2014": "2014-00000-of-00001", "2016": "2016-00000-of-00001", "2019": "2019-00000-of-00001"}

TOKEN_TO_CLASS = {**{s: s for s in SYMBOLS}, "\\times": "x"}
# Tokens that are a single glyph on one baseline (others make the expression non-linear).
_STRUCTURAL = {"{", "}", "^", "_", "\\frac", "\\sqrt", "\\sum", "\\int", "\\limits", "\\lim", "\\prod",
               "\\left", "\\right", "\\begin", "\\end", "\\\\", "&", "\\mbox", "\\overline", "\\underline"}
HASY_MAP = {**{d: d for d in "0123456789"}, "+": "+", "-": "-", "x": "x", "\\times": "x", "/": "/"}


@dataclass
class SymbolSet:
    X: list = field(default_factory=list)
    y: list = field(default_factory=list)
    groups: list = field(default_factory=list)  # expression index (CROHME) or writer id (HASY)
    source: list = field(default_factory=list)


@dataclass
class Expression:
    gray: np.ndarray
    tokens: list[str]
    target: str  # expression in our linear notation (only for fully in-vocabulary expressions)


def _load_gray(image_bytes: bytes, max_width: int = 900) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr.mean() < 127:  # CROHME renders white ink on black
        arr = 255 - arr
    if arr.shape[1] > max_width:
        scale = max_width / arr.shape[1]
        arr = cv2.resize(arr, (max_width, int(arr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return arr


def _ink_band(binary: np.ndarray) -> np.ndarray:
    rows = np.where(binary.any(axis=1))[0]
    return binary[rows.min(): rows.max() + 1] if len(rows) else binary


def is_linear(tokens: list[str]) -> bool:
    return bool(tokens) and not any(t in _STRUCTURAL for t in tokens)


def in_vocab(tokens: list[str]) -> bool:
    return all(t in TOKEN_TO_CLASS for t in tokens)


def _iter_parquet(name: str):
    import pyarrow.parquet as pq

    table = pq.read_table(CROHME_DIR / f"{name}.parquet", columns=["image", "label"])
    for batch in table.to_batches(max_chunksize=256):
        for row in batch.to_pylist():
            yield row["image"]["bytes"], row["label"]


def harvest_crohme(files: list[str], min_tokens: int = 2, group_offset: int = 0,
                   collect_expressions: bool = False) -> tuple[SymbolSet, list[Expression], dict]:
    out, expressions = SymbolSet(), []
    stats = {"expressions": 0, "linear": 0, "aligned": 0, "symbols": 0}
    gid = group_offset
    for name in files:
        for image_bytes, label in _iter_parquet(name):
            stats["expressions"] += 1
            tokens = label.split()
            if len(tokens) < min_tokens or not is_linear(tokens):
                continue
            stats["linear"] += 1
            gray = _load_gray(image_bytes)
            if collect_expressions and in_vocab(tokens):
                from app.ml.symbol_classifier import format_symbols

                expressions.append(Expression(gray, tokens, format_symbols([TOKEN_TO_CLASS[t] for t in tokens])))
            if not any(t in TOKEN_TO_CLASS for t in tokens):
                continue
            crops = segment_symbols(_ink_band(binarize(gray)))
            if len(crops) != len(tokens):
                continue
            stats["aligned"] += 1
            gid += 1
            for crop, tok in zip(crops, tokens):
                if tok in TOKEN_TO_CLASS:
                    out.X.append(glyph_features(crop.mask))
                    out.y.append(TOKEN_TO_CLASS[tok])
                    out.groups.append(gid)
                    out.source.append("crohme")
                    stats["symbols"] += 1
    return out, expressions, stats


def load_hasy() -> tuple[SymbolSet, dict]:
    out = SymbolSet()
    labels = HASY_DIR / "hasy-data-labels.csv"
    with open(labels) as f:
        for row in csv.DictReader(f):
            cls = HASY_MAP.get(row["latex"])
            if cls is None:
                continue
            img = cv2.imread(str(HASY_DIR / row["path"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if np.count_nonzero(mask) < 5:
                continue
            out.X.append(glyph_features(mask))
            out.y.append(cls)
            out.groups.append(1_000_000 + int(row["user_id"]))
            out.source.append("hasy")
    counts = {c: out.y.count(c) for c in sorted(set(out.y))}
    return out, {"symbols": len(out.y), "per_class": counts, "writers": len(set(out.groups))}


def available() -> dict[str, bool]:
    return {
        "crohme": all((CROHME_DIR / f"{n}.parquet").exists() for n in TRAIN_FILES + list(TEST_FILES.values())),
        "hasy": (HASY_DIR / "hasy-data-labels.csv").exists(),
    }


CACHE_PATH = EXTERNAL / "cache" / "real_symbols.npz"


def cached_real_sets() -> dict | None:
    """Real training symbols (CROHME train + HASY) and CROHME test symbols as feature
    arrays, cached on disk so active-learning retrains don't re-harvest. None if the
    datasets have not been downloaded."""
    from app.ml.symbol_features import FEATURE_VERSION

    if not all(available().values()):
        return None
    if CACHE_PATH.exists():
        d = np.load(CACHE_PATH, allow_pickle=False)
        if int(d["feature_version"]) == FEATURE_VERSION:
            return {k: d[k] for k in d.files}
    train, _, _ = harvest_crohme(TRAIN_FILES)
    hasy, _ = load_hasy()
    test, _, _ = harvest_crohme(list(TEST_FILES.values()), group_offset=10_000_000)
    data = {
        "train_X": np.vstack(train.X + hasy.X), "train_y": np.array(train.y + hasy.y),
        "train_groups": np.array(train.groups + hasy.groups), "test_X": np.vstack(test.X),
        "test_y": np.array(test.y), "feature_version": np.array(FEATURE_VERSION),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, **data)
    return data
