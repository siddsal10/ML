"""Simulated smartphone capture of a clean handwritten page.

A stand-in until real phone photos are collected (docs/REAL_WORLD_VALIDATION.md):
it shows *which* capture conditions break the classical reader, not how
often they occur. Each factor can be switched on alone or combined:

    paper        off-white paper with low-frequency texture, coloured pen ink
    rules        notebook rules under each line of writing (+ a margin line)
    gradient     lighting falls off across the page (fraction of brightness lost)
    shadow       a soft-edged cast shadow over part of the page
    tilt         perspective (corner shift as a fraction of size) + rotation (deg)
    blur         Gaussian defocus, sigma in pixels at x-height 40
    noise, jpeg  sensor noise (grey levels) and JPEG quality
    scale        lower capture resolution
"""
from __future__ import annotations

import cv2
import numpy as np

CONDITIONS: dict[str, dict] = {
    "clean": {},
    "paper_and_pen": {"paper": True},
    "ruled_paper": {"paper": True, "rules": True},
    "uneven_lighting": {"gradient": 0.45},
    "cast_shadow": {"shadow": 0.45},
    "tilt": {"tilt": (0.05, 3.0)},
    "blur": {"blur": 1.2},
    "noise_jpeg": {"noise": 8.0, "jpeg": 60},
    "low_resolution": {"scale": 0.6},
    "phone_mild": {"paper": True, "gradient": 0.25, "tilt": (0.02, 1.0), "blur": 0.6, "noise": 3.0, "jpeg": 85},
    "phone_moderate": {"paper": True, "rules": True, "gradient": 0.4, "shadow": 0.3, "tilt": (0.04, 2.5),
                       "blur": 1.0, "noise": 6.0, "jpeg": 70, "scale": 0.8},
    "phone_severe": {"paper": True, "rules": True, "gradient": 0.55, "shadow": 0.5, "tilt": (0.07, 5.0),
                     "blur": 1.6, "noise": 10.0, "jpeg": 50, "scale": 0.6},
}


def _soft_noise(shape, rng, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), sigma)


def capture(page: np.ndarray, rng: np.random.Generator, baselines: list[int] | None = None, *, paper: bool = False,
            rules: bool = False, gradient: float = 0.0, shadow: float = 0.0, tilt: tuple[float, float] = (0.0, 0.0),
            blur: float = 0.0, noise: float = 0.0, jpeg: int | None = None, scale: float = 1.0) -> np.ndarray:
    """Turn a clean page (ink 0 on 255) into a simulated phone photo (grayscale)."""
    h, w = page.shape
    ink = (page < 128).astype(np.float32)
    ink = cv2.GaussianBlur(ink, (0, 0), 0.7)  # pen edges are soft

    if paper:
        img = 238 + 4 * _soft_noise((h, w), rng, 25) + 1.5 * rng.normal(0, 1, (h, w))
        ink_level = float(rng.uniform(60, 110))  # blue / black ballpoint in grey
    else:
        img = np.full((h, w), 255.0, np.float32)
        ink_level = 0.0
    if rules:
        rule_level = float(rng.uniform(175, 200))
        pitch = int(np.median(np.diff(baselines))) if baselines and len(baselines) > 1 else 90
        start = (baselines[0] if baselines else pitch) % pitch
        for y in range(start, h, pitch):
            cv2.line(img, (0, y), (w, y), rule_level, 2)
        x_margin = int(rng.uniform(0.02, 0.05) * w)
        cv2.line(img, (x_margin, 0), (x_margin, h), rule_level - 20, 2)
    img = img * (1 - ink) + ink_level * ink

    light = np.ones((h, w), np.float32)
    if gradient:
        theta = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = (np.cos(theta) * xx / w + np.sin(theta) * yy / h)
        t = (t - t.min()) / max(np.ptp(t), 1e-6)
        light *= 1 - gradient * t
    if shadow:
        mask = np.zeros((h, w), np.float32)
        # a phone or hand shadow entering from one edge
        edge = int(rng.integers(4))
        depth = rng.uniform(0.3, 0.6)
        if edge == 0:
            mask[: int(depth * h)] = 1
        elif edge == 1:
            mask[int((1 - depth) * h):] = 1
        elif edge == 2:
            mask[:, : int(depth * w)] = 1
        else:
            mask[:, int((1 - depth) * w):] = 1
        mask = cv2.GaussianBlur(mask, (0, 0), 0.04 * max(h, w))
        light *= 1 - shadow * mask
    img = img * light

    shift, angle = tilt
    if shift or angle:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-shift, shift, (4, 2)).astype(np.float32) * np.float32([w, h])
        M = cv2.getPerspectiveTransform(src, dst)
        R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-angle, angle)), 1.0), [0, 0, 1]])
        img = cv2.warpPerspective(img, (R @ M).astype(np.float64), (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if scale != 1.0:
        img = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    if noise:
        img = img + rng.normal(0, noise, img.shape)
    out = np.clip(img, 0, 255).astype(np.uint8)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out
