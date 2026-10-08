"""Content-region analysis.

For the MVP (no large layout model such as LayoutLMv3 / DocLayout-YOLO
wired in), regions are approximated using classical contour-based
segmentation on the preprocessed (binarized) image. Each detected region is
classified heuristically as "text", "math", or "diagram" based on simple
shape/density features. This is a deliberately simple, explainable
approximation - not a learned layout model - and is documented as such.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_REGION_AREA = 400  # px^2, filters out noise specks


@dataclass
class DetectedRegion:
    region_type: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    order_index: int


def _classify_region(roi: np.ndarray, w: int, h: int) -> tuple[str, float]:
    """Heuristic region classifier.

    - Very high stroke density + roughly square aspect -> "diagram"
    - Wide, short, sparse regions with a mix of horizontal ink runs -> "text"
    - Regions rich in short disconnected strokes with digit-like aspect
      ratio (common in equations) -> "math"

    This is a coarse, deterministic heuristic intended for the MVP; it is
    documented as an approximation everywhere it is surfaced (API + UI).
    """
    ink = roi < 128
    density = float(np.mean(ink))
    aspect = w / max(h, 1)

    # Count connected components as a rough proxy for "many small symbols"
    # (favors math/text) vs "few large connected strokes" (favors diagrams).
    num_labels, _ = cv2.connectedComponents(ink.astype(np.uint8))
    component_density = num_labels / max(w * h, 1) * 10000

    if density > 0.35 and 0.5 < aspect < 2.0 and component_density < 2:
        return "diagram", 0.55
    if component_density >= 4 and aspect > 1.5:
        return "math", 0.6
    return "text", 0.55


def detect_regions(processed_image_path: str) -> list[DetectedRegion]:
    image = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []

    # Dilate to merge nearby strokes into coherent blocks before contouring.
    inverted = cv2.bitwise_not(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
    dilated = cv2.dilate(inverted, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < MIN_REGION_AREA:
            continue
        boxes.append((x, y, w, h))

    # Sort top-to-bottom, then left-to-right (natural reading order).
    boxes.sort(key=lambda b: (b[1] // 40, b[0]))

    regions: list[DetectedRegion] = []
    for idx, (x, y, w, h) in enumerate(boxes):
        roi = image[y : y + h, x : x + w]
        region_type, confidence = _classify_region(roi, w, h)
        regions.append(
            DetectedRegion(
                region_type=region_type,
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=confidence,
                order_index=idx,
            )
        )

    if not regions:
        # Whole image as a single "text" region fallback so downstream OCR
        # still has something to operate on.
        h, w = image.shape[:2]
        regions.append(
            DetectedRegion(region_type="text", x=0, y=0, width=w, height=h, confidence=0.3, order_index=0)
        )

    return regions
