"""Real handwritten glyphs from the CROHME *test* writers, for composing pages.

Symbols are harvested by the same count alignment as training
(app.ml.real_symbols), but kept as masks with their size and vertical
position relative to the digits of their expression. `compose_page` then
writes any line of working (e.g. a synthetic solution) in real handwriting,
so the whole pipeline - segmentation, recognizer, SymPy, mistake classifier -
can be run on it. The recognizer never trained on these writers.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from app.ml.real_symbols import TEST_FILES, TOKEN_TO_CLASS, _ink_band, _iter_parquet, _load_gray, is_linear
from app.ml.segmentation import binarize, segment_symbols


@dataclass
class Glyph:
    label: str
    mask: np.ndarray  # ink = 255, tight crop
    rel_h: float  # height / median digit height of its expression
    rel_cy: float  # (centre y - digits' centre y) / median digit height


def harvest_test_glyphs(years: tuple[str, ...]) -> dict[str, list[Glyph]]:
    pool: dict[str, list[Glyph]] = defaultdict(list)
    for year in years:
        for image_bytes, label in _iter_parquet(TEST_FILES[year]):
            tokens = label.split()
            if len(tokens) < 2 or not is_linear(tokens) or not all(t in TOKEN_TO_CLASS for t in tokens):
                continue
            crops = segment_symbols(_ink_band(binarize(_load_gray(image_bytes))))
            if len(crops) != len(tokens):
                continue
            digits = [c for c, t in zip(crops, tokens) if t.isdigit()]
            if not digits:
                continue
            ref_h = float(np.median([c.h for c in digits]))
            ref_cy = float(np.median([c.y + c.h / 2 for c in digits]))
            for c, t in zip(crops, tokens):
                pool[TOKEN_TO_CLASS[t]].append(Glyph(TOKEN_TO_CLASS[t], c.mask, c.h / ref_h, (c.y + c.h / 2 - ref_cy) / ref_h))
    return dict(pool)


def text_to_symbols(line: str) -> list[str] | None:
    """A line of linear notation as the symbols a student would write: '*' is
    the multiplication cross (class 'x'), and '3*x' is written '3x'. None if
    the line uses anything the recognizer has no class for."""
    line = line.replace(" ", "").replace("*x", "x").replace("*", "x")
    vocab = set("0123456789+-=x()/")
    return list(line) if line and set(line) <= vocab else None


def compose_page(lines: list[list[str]], pool: dict[str, list[Glyph]], rng: np.random.Generator,
                 x_height: int = 40) -> tuple[np.ndarray, list[int]]:
    """Write symbol lines in real handwriting on a white page.

    Returns the grayscale page and each line's baseline row (for drawing
    notebook rules under the writing)."""
    pitch = int(2.2 * x_height)
    margin = x_height
    placed, baselines = [], []
    width = 0
    for li, symbols in enumerate(lines):
        cy = margin + li * pitch + x_height // 2
        x = margin + int(rng.integers(0, x_height // 2))
        for s in symbols:
            g = pool[s][int(rng.integers(len(pool[s])))]
            h = max(2, int(round(x_height * g.rel_h * rng.uniform(0.92, 1.08))))
            w = max(2, int(round(g.mask.shape[1] * h / g.mask.shape[0])))
            mask = cv2.resize(g.mask, (w, h), interpolation=cv2.INTER_AREA if h < g.mask.shape[0] else cv2.INTER_LINEAR)
            y = int(round(cy + g.rel_cy * x_height - h / 2 + rng.normal(0, 0.04 * x_height)))
            placed.append((x, y, mask))
            x += w + int(round(x_height * rng.uniform(0.12, 0.4)))
        width = max(width, x)
        baselines.append(int(cy + x_height / 2))
    page = np.full((margin + len(lines) * pitch + margin // 2, width + margin), 255, np.uint8)
    for x, y, mask in placed:
        y0, y1 = max(0, y), min(page.shape[0], y + mask.shape[0])
        x1 = min(page.shape[1], x + mask.shape[1])
        region = mask[y0 - y: y1 - y, : x1 - x] > 127
        page[y0:y1, x:x1][region] = 0
    return page, baselines
