from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.preprocessing import preprocess_image


def _write_test_image(path: Path, with_text: bool = True) -> None:
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    if with_text:
        arr = np.array(img)
        arr[80:120, 50:350] = 0  # a black bar simulating ink
        img = Image.fromarray(arr)
    img.save(path)


def test_preprocess_image_produces_output_file():
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.png"
        output_path = Path(tmp) / "output.png"
        _write_test_image(input_path)

        meta = preprocess_image(input_path, output_path)

        assert output_path.exists()
        assert meta["width"] > 0
        assert meta["height"] > 0


def test_preprocess_image_flags_blank_image():
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "blank.png"
        output_path = Path(tmp) / "blank_out.png"
        _write_test_image(input_path, with_text=False)

        meta = preprocess_image(input_path, output_path)

        assert any("little ink" in w.lower() for w in meta["warnings"])


def test_preprocess_image_output_is_grayscale_binary():
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.png"
        output_path = Path(tmp) / "output.png"
        _write_test_image(input_path)

        preprocess_image(input_path, output_path)
        out_img = Image.open(output_path)
        assert out_img.mode in ("L", "1")
