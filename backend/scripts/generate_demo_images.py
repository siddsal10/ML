"""Generates placeholder "handwritten-style" demo images for the seeded demo.

These are simple rendered-text PNGs (not real handwriting samples) used so
the deterministic demo provider (app/providers/demo_provider.py) has real
files to match against. This script renders each sample defined in
DEMO_SAMPLES and writes data/demo_samples/manifest.json mapping each
slug to its file path and actual sha256 hash. The demo provider reads that
manifest at runtime, so there is no need to predict image bytes in advance
- whatever this script actually writes is authoritative.

Run:
    python scripts/generate_demo_images.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.providers.demo_provider import DEMO_SAMPLES  # noqa: E402

OUTPUT_DIR = BACKEND_ROOT / "data" / "demo_samples"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def render_demo_image(slug: str, text: str) -> Path:
    width, height = 800, 300
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    except OSError:
        font = ImageFont.load_default(size=48)

    lines = text.split("\n")
    y = 60
    for line in lines:
        draw.text((60, y), line, fill=0, font=font)
        y += 70

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{slug}.png"
    image.save(out_path, format="PNG")
    return out_path


def main() -> None:
    print("Generating demo images...")
    manifest: dict[str, dict] = {}

    for slug, sample in DEMO_SAMPLES.items():
        path = render_demo_image(slug, sample["text"])
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[slug] = {
            "path": str(path.relative_to(BACKEND_ROOT)),
            "sha256": file_hash,
            "text": sample["text"],
        }
        print(f"  {slug:28s} -> {path.relative_to(BACKEND_ROOT)}  sha256={file_hash[:12]}...")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {MANIFEST_PATH.relative_to(BACKEND_ROOT)}")
    print("Restart the backend (or re-import app.providers.demo_provider) to pick up new hashes.")


if __name__ == "__main__":
    main()
