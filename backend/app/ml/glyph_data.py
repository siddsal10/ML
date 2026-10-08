"""Synthetic training data for the symbol classifier.

Real handwritten-math datasets (CROHME, IAM) cannot be redistributed, so the
bundled training set is generated: each symbol is rendered in every
available font (several of them handwriting-style) and then distorted with
handwriting-like augmentation (affine jitter, elastic warp, stroke-width
change). Each sample records which font it came from so evaluation can hold
out entire fonts (unseen "writers") instead of random samples.

Real labelled crops can be added by placing PNGs in
  <extra_dir>/<symbol_label>/*.png     (ink dark on light background)
see SYMBOL_DIR_NAMES for the folder name of symbols that aren't valid
directory names.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.ml.symbol_features import glyph_features, tight_crop

SYMBOLS: list[str] = list("0123456789") + ["+", "-", "=", "x", "(", ")", "/"]

# Folder names for symbols that can't be used as directory names.
SYMBOL_DIR_NAMES = {"+": "plus", "-": "minus", "=": "equals", "(": "lparen", ")": "rparen", "/": "slash"}

# Fonts that cover all of SYMBOLS. Handwriting-style ones come first; plain
# sans/serif fonts add stroke-shape diversity. Missing fonts are skipped.
FONT_CANDIDATES = [
    # handwriting / informal
    "Bradley Hand Bold.ttf", "Chalkboard.ttc", "ChalkboardSE.ttc", "Chalkduster.ttf",
    "Comic Sans MS.ttf", "Comic Sans MS Bold.ttf", "Noteworthy.ttc", "MarkerFelt.ttc",
    "SignPainter.ttc", "Apple Chancery.ttf", "Savoye LET.ttc", "Snell Roundhand.ttc",
    # sans / serif / mono
    "Arial.ttf", "Arial Bold.ttf", "Verdana.ttf", "Georgia.ttf", "Trebuchet MS.ttf",
    "Tahoma.ttf", "Courier New.ttf", "Futura.ttc", "GillSans.ttc", "Avenir.ttc",
    "Helvetica.ttc", "Menlo.ttc", "Times New Roman.ttf",
    # common on Linux / Docker images
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf",
    "LiberationSans-Regular.ttf", "LiberationSerif-Regular.ttf", "FreeSans.ttf",
]

FONT_DIRS = [
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts",
    "C:/Windows/Fonts",
]

RENDER_SIZE = 64


def find_fonts() -> list[str]:
    """Absolute paths of available candidate fonts (deduplicated, ordered)."""
    found: dict[str, str] = {}
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            name = os.path.basename(path)
            if name in FONT_CANDIDATES and name not in found:
                found[name] = path
    return [found[n] for n in FONT_CANDIDATES if n in found]


def _load_font(path: str | None) -> ImageFont.FreeTypeFont:
    if path is None:
        return ImageFont.load_default(size=RENDER_SIZE)
    return ImageFont.truetype(path, RENDER_SIZE)


def render_glyph(font: ImageFont.FreeTypeFont, ch: str) -> np.ndarray:
    """Render one character; returns a tight binary mask (ink=255)."""
    canvas = Image.new("L", (RENDER_SIZE * 3, RENDER_SIZE * 3), color=0)
    draw = ImageDraw.Draw(canvas)
    draw.text((RENDER_SIZE * 1.5, RENDER_SIZE * 1.5), ch, fill=255, font=font, anchor="mm")
    arr = np.array(canvas)
    _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
    return tight_crop(binary)


def augment(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Handwriting-like distortion of a binary glyph mask."""
    pad = max(mask.shape) // 2 + 4
    img = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    h, w = img.shape

    # Random affine: rotation, shear, anisotropic scale.
    angle = rng.uniform(-12, 12)
    shear = rng.uniform(-0.3, 0.3)
    sx, sy = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    A = np.array([[sx, shear, 0], [0, sy, 0]], dtype=np.float64)
    A[:, 2] = np.array([w / 2, h / 2]) - A[:, :2] @ np.array([w / 2, h / 2])
    M = np.vstack([rot, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]])
    img = cv2.warpAffine(img, M[:2], (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    # Elastic distortion (Simard et al. 2003): smoothed random displacement field.
    if rng.random() < 0.8:
        alpha, sigma = rng.uniform(2, 6), rng.uniform(3, 5)
        dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha * 4
        dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha * 4
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        img = cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=0)

    # Stroke width: thinner or thicker pen.
    k = int(rng.integers(-2, 4))
    if k > 0:
        img = cv2.dilate(img, np.ones((k, k), np.uint8))
    elif k < 0:
        eroded = cv2.erode(img, np.ones((2, 2), np.uint8))
        if np.count_nonzero(eroded > 127) > 15:
            img = eroded

    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return tight_crop(img)


@dataclass
class GlyphDataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray  # source id (font index) for writer-disjoint splits
    group_names: list[str]


def generate_dataset(samples_per_font: int = 30, seed: int = 0, extra_dir: str | Path | None = None) -> GlyphDataset:
    rng = np.random.default_rng(seed)
    font_paths: list[str | None] = find_fonts() or [None]
    X, y, groups = [], [], []
    names = [Path(p).name if p else "pil-default" for p in font_paths]

    for gi, path in enumerate(font_paths):
        font = _load_font(path)
        for ch in SYMBOLS:
            base = render_glyph(font, ch)
            if np.count_nonzero(base) < 10:
                continue
            X.append(glyph_features(base))
            y.append(ch)
            groups.append(gi)
            for _ in range(samples_per_font - 1):
                X.append(glyph_features(augment(base, rng)))
                y.append(ch)
                groups.append(gi)

    if extra_dir is not None:
        gi = len(names)
        names.append(f"extra:{Path(extra_dir).name}")
        dir_to_symbol = {v: k for k, v in SYMBOL_DIR_NAMES.items()}
        for sub in sorted(Path(extra_dir).iterdir()):
            label = dir_to_symbol.get(sub.name, sub.name)
            if not sub.is_dir() or label not in SYMBOLS:
                continue
            for img_path in sorted(sub.glob("*.png")):
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                X.append(glyph_features(mask))
                y.append(label)
                groups.append(gi)

    return GlyphDataset(np.vstack(X), np.array(y), np.array(groups), names)


def render_expression(font_path: str | None, text: str, rng: np.random.Generator) -> np.ndarray:
    """Render a (possibly multi-line) expression as a grayscale page image,
    with per-character jitter. Used for end-to-end OCR evaluation."""
    font = _load_font(font_path)
    lines = text.split("\n")
    width = 80 + RENDER_SIZE * max(len(ln) for ln in lines)
    height = 60 + int(RENDER_SIZE * 1.6) * len(lines)
    page = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(page)
    y = 30
    for line in lines:
        x = 40
        for ch in line:
            if ch == " ":
                x += int(RENDER_SIZE * 0.45)
                continue
            jitter_y = int(rng.integers(-3, 4))
            draw.text((x, y + jitter_y), ch, fill=0, font=font)
            bbox = draw.textbbox((x, y), ch, font=font)
            x = bbox[2] + int(rng.integers(3, 9))
        y += int(RENDER_SIZE * 1.6)
    return np.array(page)
