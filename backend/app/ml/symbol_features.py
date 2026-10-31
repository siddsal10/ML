"""Hand-engineered feature extraction for single handwritten symbols.

Input to every function here is a binary uint8 mask where ink = 255.
Feature vector (FEATURE_VERSION 2, 1398 dims):
- 324  HOG (28x28 window, 14x14 blocks, 7x7 stride/cells, 9 bins)
- 49   7x7 zoning ink density
- 1    log aspect ratio of the tight bounding box (separates '-' from '1')
- 1024 the glyph itself, 32x32 in [0, 1] (input of the CNN; the classical
       models read only the first N_HANDCRAFTED columns)
"""
from __future__ import annotations

import cv2
import numpy as np

FEATURE_VERSION = 2
IMG_SIZE = 28
CNN_SIZE = 32
N_HANDCRAFTED = 374
_INNER = 22  # glyph is scaled into a 22x22 box and centered in 28x28 (MNIST-style margin)

_HOG = cv2.HOGDescriptor((IMG_SIZE, IMG_SIZE), (14, 14), (7, 7), (7, 7), 9)


def tight_crop(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def normalize_glyph(mask: np.ndarray, size: int = IMG_SIZE, inner: int = _INNER) -> np.ndarray:
    """Scale a tight glyph crop into a fixed size x size canvas, preserving aspect ratio."""
    crop = tight_crop(mask)
    h, w = crop.shape[:2]
    scale = inner / max(h, w, 1)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def glyph_features(mask: np.ndarray) -> np.ndarray:
    crop = tight_crop(mask)
    h, w = crop.shape[:2]
    norm = normalize_glyph(crop)
    hog = _HOG.compute(norm).ravel()
    zones = cv2.resize(norm.astype(np.float32) / 255.0, (7, 7), interpolation=cv2.INTER_AREA).ravel()
    aspect = np.array([np.log(max(w, 1) / max(h, 1))], dtype=np.float32)
    image = normalize_glyph(crop, CNN_SIZE, 26).astype(np.float32).ravel() / 255.0
    return np.concatenate([hog, zones, aspect, image]).astype(np.float32)
