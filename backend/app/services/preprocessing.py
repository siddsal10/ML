"""Image preprocessing pipeline.

Implements: orientation correction (best-effort via EXIF), grayscale
conversion, denoising, contrast enhancement, adaptive thresholding,
deskewing, and aspect-ratio-preserving resizing. Saves the processed image
to a separate path from the original.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


MAX_DIMENSION = 1600


def _correct_orientation(pil_image: Image.Image) -> Image.Image:
    """Apply EXIF orientation correction if present."""
    try:
        return ImageOps.exif_transpose(pil_image) or pil_image
    except Exception:
        return pil_image


def _resize_preserving_aspect(image: np.ndarray, max_dim: int = MAX_DIMENSION) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate and correct small rotational skew using minAreaRect on ink pixels."""
    inverted = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inverted > 30))
    if coords.shape[0] < 20:
        return gray  # not enough ink to estimate skew reliably

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Ignore implausible large "corrections" - likely a bad estimate.
    if abs(angle) > 15:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_image(input_path: str | Path, output_path: str | Path) -> dict:
    """Run the full preprocessing pipeline and save the result.

    Returns a small dict of metadata about what was done, useful for
    debugging and for surfacing warnings to the user.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []

    pil_image = Image.open(input_path)
    pil_image = _correct_orientation(pil_image)
    pil_image = pil_image.convert("RGB")

    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv_image = _resize_preserving_aspect(cv_image)

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # Deskew before thresholding (works better on grayscale ink mass)
    deskewed = _deskew(contrast_enhanced)

    # Adaptive thresholding
    thresholded = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10,
    )

    ink_ratio = float(np.mean(thresholded < 128))
    if ink_ratio < 0.002:
        warnings.append("Very little ink detected; image may be blank, blurry, or overexposed.")
    elif ink_ratio > 0.6:
        warnings.append("Unusually high ink coverage detected; check lighting/contrast.")

    cv2.imwrite(str(output_path), thresholded)

    return {
        "output_path": str(output_path),
        "width": thresholded.shape[1],
        "height": thresholded.shape[0],
        "ink_ratio": round(ink_ratio, 4),
        "warnings": warnings,
    }
