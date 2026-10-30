"""Classical page -> lines -> symbols segmentation (OpenCV, no learning).

1. Page binarization: divide out uneven lighting (phone photos), Otsu
   threshold (ink = 255), then erase ruled notebook lines.
2. Line segmentation from the horizontal projection profile.
3. Symbol segmentation by connected components within each line; components
   that overlap horizontally (the two bars of '=', a broken '5') are merged
   into one symbol.

Training crops are harvested with plain `binarize`; the photo steps in
`binarize_page` are identity on clean scans, so features are unchanged.
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


def binarize(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """Divide the image by an estimate of the paper's brightness, so shadows
    and lighting gradients become uniform white. A grey-level closing much
    wider than any pen stroke removes the ink and leaves the paper. Evenly
    lit images (scans, tablet renders) are returned unchanged."""
    k = max(31, min(gray.shape[:2]) // 6) | 1
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    if np.percentile(background, 99) - np.percentile(background, 1) < 25:
        return gray
    background = cv2.GaussianBlur(background, (k, k), 0).astype(np.float32)
    flat = gray.astype(np.float32) / np.maximum(background, 1.0) * 255.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def remove_ruled_lines(binary: np.ndarray) -> np.ndarray:
    """Erase notebook rules and margins: straight ink runs spanning at least
    half the page and several symbol heights. A minus or '=' bar is never
    that long relative to the symbols around it."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    h, w = binary.shape
    sym_h = float(np.median(stats[1:, cv2.CC_STAT_HEIGHT]))
    out = binary.copy()
    for kx, ky in ((max(w // 2, int(4 * sym_h)), 1), (1, max(h // 2, int(4 * sym_h)))):
        if kx > w or ky > h:
            continue
        rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
        # thicken slightly so the rule's anti-aliased edges go too
        out[cv2.dilate(rules, np.ones((3, 3), np.uint8)) > 0] = 0
    return out


def binarize_page(gray: np.ndarray) -> np.ndarray:
    return remove_ruled_lines(binarize(flatten_background(gray)))


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


def segment_page(gray: np.ndarray) -> list[list[SymbolCrop]]:
    """Returns the symbol crops of each text line, top to bottom, left to right."""
    binary = binarize_page(gray)
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
