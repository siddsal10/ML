"""Classical page -> lines -> symbols segmentation (OpenCV, no learning).

0. Illumination normalisation (optional, on by default at inference): the
   background is estimated with a grey-level morphological closing (which
   removes thin dark strokes) and divided out, so shadows and uneven phone
   lighting do not break the global threshold.
1. Otsu binarization (ink = 255).
2. Line segmentation from the horizontal projection profile.
3. Symbol segmentation by connected components within each line; components
   that overlap horizontally (the two bars of '=', a broken '5') are merged
   into one symbol.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MERGE_OVERLAP = 0.5


@dataclass
class SymbolCrop:
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray  # ink=255, tight crop containing only this symbol's pixels


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    k = max(15, (min(gray.shape[:2]) // 12) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.GaussianBlur(background, (0, 0), k / 4)
    return cv2.divide(gray, np.maximum(background, 1), scale=255)


def binarize(gray: np.ndarray, normalize: bool = False) -> np.ndarray:
    if normalize:
        gray = normalize_illumination(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def segment_lines(binary: np.ndarray) -> list[tuple[int, int]]:
    row_ink = np.count_nonzero(binary, axis=1)
    min_ink = max(1, int(0.002 * binary.shape[1]))
    rows = row_ink >= min_ink

    runs: list[list[int]] = []
    start = None
    for i, on in enumerate(rows):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(rows)])
    if not runs:
        return []

    # Merge runs separated by a small gap (e.g. a dot or stroke detached
    # from its line); gaps between real lines are much larger.
    median_h = float(np.median([b - a for a, b in runs]))
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] < 0.25 * median_h:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= max(4, 0.25 * median_h)]


def segment_symbols(line: np.ndarray) -> list[SymbolCrop]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(line, connectivity=8)
    line_h = line.shape[0]
    min_area = max(6, int(0.004 * line_h * line_h))

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comps.append({"labels": [i], "x0": x, "x1": x + w, "y0": y, "y1": y + h})
    comps.sort(key=lambda c: c["x0"])

    groups: list[dict] = []
    for c in comps:
        if groups:
            g = groups[-1]
            overlap = min(g["x1"], c["x1"]) - max(g["x0"], c["x0"])
            narrower = min(g["x1"] - g["x0"], c["x1"] - c["x0"])
            if narrower > 0 and overlap / narrower >= MERGE_OVERLAP:
                g["labels"] += c["labels"]
                g["x0"], g["x1"] = min(g["x0"], c["x0"]), max(g["x1"], c["x1"])
                g["y0"], g["y1"] = min(g["y0"], c["y0"]), max(g["y1"], c["y1"])
                continue
        groups.append(dict(c))

    crops = []
    for g in groups:
        region = labels[g["y0"] : g["y1"], g["x0"] : g["x1"]]
        mask = np.isin(region, g["labels"]).astype(np.uint8) * 255
        crops.append(SymbolCrop(int(g["x0"]), int(g["y0"]), int(g["x1"] - g["x0"]), int(g["y1"] - g["y0"]), mask))
    return crops


def segment_page(gray: np.ndarray, normalize: bool = True) -> list[list[SymbolCrop]]:
    """Returns the symbol crops of each text line, top to bottom, left to right."""
    binary = binarize(gray, normalize=normalize)
    result = []
    for y0, y1 in segment_lines(binary):
        line = binary[y0:y1]
        crops = segment_symbols(line)
        if not crops:
            continue
        for c in crops:
            c.y += y0
        result.append(crops)
    return result
