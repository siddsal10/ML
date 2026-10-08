from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from app.services.benchmark_service import load_manifest, run_benchmark


def _make_manifest_and_image(tmp_dir: Path):
    img_path = tmp_dir / "sample1.png"
    Image.new("RGB", (100, 40), color="white").save(img_path)

    manifest_path = tmp_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "sample_id": "sample1",
                    "image_path": "sample1.png",
                    "reference_text": "12 + 15 = 27",
                    "expected_answer": "27",
                    "mistake_label": "correct_solution",
                }
            )
            + "\n"
        )
    return manifest_path


def test_load_manifest_jsonl_resolves_relative_paths():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest_path = _make_manifest_and_image(tmp_path)

        samples = load_manifest(manifest_path)

        assert len(samples) == 1
        assert samples[0].sample_id == "sample1"
        assert Path(samples[0].image_path).exists()


def test_run_benchmark_with_demo_provider_only():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest_path = _make_manifest_and_image(tmp_path)

        outcome = run_benchmark(manifest_path, provider_names=["demo"])

        assert "results" in outcome
        assert "summary" in outcome
        assert "demo" in outcome["summary"]
        assert outcome["summary"]["demo"]["available"] is True
        assert outcome["summary"]["demo"]["num_samples"] == 1


def test_run_benchmark_records_unavailable_provider_honestly(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest_path = _make_manifest_and_image(tmp_path)

        # vision_api has no key configured by default -> should be marked unavailable.
        outcome = run_benchmark(manifest_path, provider_names=["vision_api"])

        summary = outcome["summary"]["vision_api"]
        assert summary["available"] is False
        assert all(not r["available"] for r in outcome["results"])
