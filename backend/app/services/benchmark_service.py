"""Benchmark module.

Compares available OCR providers against a manifest of (image, reference
transcription, [expected mistake category]) samples using a common
interface. Providers that are unavailable are skipped gracefully and
recorded as such (never silently dropped). Supports CSV or JSONL manifests.

Manifest schema (also documented in docs/DATASET_GUIDE.md):
    image_path: str (relative to manifest's directory, or absolute)
    reference_text: str
    expected_answer: str (optional)
    mistake_label: str (optional, one of app.services.diagnosis.CATEGORIES)
    sample_id: str (optional, defaults to image_path)
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from app.providers.base import ProviderUnavailableError
from app.providers.registry import get_provider, list_providers
from app.services import diagnosis as diagnosis_service
from app.services import math_parser
from app.utils.text_metrics import (
    character_error_rate,
    exact_match,
    macro_f1,
    normalized_edit_distance,
    word_error_rate,
)


@dataclass
class ManifestSample:
    sample_id: str
    image_path: str
    reference_text: str
    expected_answer: str | None = None
    mistake_label: str | None = None


def load_manifest(manifest_path: str | Path) -> list[ManifestSample]:
    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent
    samples: list[ManifestSample] = []

    if manifest_path.suffix.lower() == ".jsonl":
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                samples.append(_row_to_sample(row, base_dir))
    elif manifest_path.suffix.lower() == ".csv":
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(_row_to_sample(row, base_dir))
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")

    return samples


def _row_to_sample(row: dict, base_dir: Path) -> ManifestSample:
    image_path = row["image_path"]
    if not Path(image_path).is_absolute():
        image_path = str((base_dir / image_path).resolve())
    return ManifestSample(
        sample_id=row.get("sample_id") or row["image_path"],
        image_path=image_path,
        reference_text=row.get("reference_text", ""),
        expected_answer=row.get("expected_answer") or None,
        mistake_label=row.get("mistake_label") or None,
    )


def run_benchmark(manifest_path: str | Path, provider_names: list[str] | None = None) -> dict:
    samples = load_manifest(manifest_path)
    providers = provider_names or [p.name for p in list_providers()]

    detailed_results: list[dict] = []
    per_provider_rows: dict[str, list[dict]] = {name: [] for name in providers}

    for provider_name in providers:
        provider = get_provider(provider_name)
        available, reason = provider.is_available()

        for sample in samples:
            if not available:
                detailed_results.append(
                    {
                        "provider": provider_name,
                        "sample_id": sample.sample_id,
                        "available": False,
                        "predicted_text": None,
                        "reference_text": sample.reference_text,
                        "cer": None,
                        "wer": None,
                        "exact_match": None,
                        "normalized_edit_distance": None,
                        "inference_time_ms": None,
                        "diagnosis_correct": None,
                        "predicted_mistake_category": None,
                        "reference_mistake_category": sample.mistake_label,
                        "skip_reason": reason,
                    }
                )
                continue

            try:
                if not Path(sample.image_path).exists():
                    raise FileNotFoundError(sample.image_path)
                ocr_result = provider.transcribe(sample.image_path)
            except (ProviderUnavailableError, FileNotFoundError) as exc:
                detailed_results.append(
                    {
                        "provider": provider_name,
                        "sample_id": sample.sample_id,
                        "available": False,
                        "predicted_text": None,
                        "reference_text": sample.reference_text,
                        "cer": None,
                        "wer": None,
                        "exact_match": None,
                        "normalized_edit_distance": None,
                        "inference_time_ms": None,
                        "diagnosis_correct": None,
                        "predicted_mistake_category": None,
                        "reference_mistake_category": sample.mistake_label,
                        "skip_reason": str(exc),
                    }
                )
                continue

            predicted_category = None
            diagnosis_correct = None
            if sample.mistake_label:
                math_analysis = math_parser.analyze_transcription(ocr_result.text)
                diag = diagnosis_service.diagnose(math_analysis, sample.expected_answer, ocr_result.confidence)
                predicted_category = diag.category
                diagnosis_correct = predicted_category == sample.mistake_label

            row = {
                "provider": provider_name,
                "sample_id": sample.sample_id,
                "available": True,
                "predicted_text": ocr_result.text,
                "reference_text": sample.reference_text,
                "cer": character_error_rate(sample.reference_text, ocr_result.text),
                "wer": word_error_rate(sample.reference_text, ocr_result.text),
                "exact_match": exact_match(sample.reference_text, ocr_result.text),
                "normalized_edit_distance": normalized_edit_distance(sample.reference_text, ocr_result.text),
                "inference_time_ms": round(ocr_result.inference_time_ms, 2),
                "diagnosis_correct": diagnosis_correct,
                "predicted_mistake_category": predicted_category,
                "reference_mistake_category": sample.mistake_label,
            }
            detailed_results.append(row)
            per_provider_rows[provider_name].append(row)

    summary = _summarize(providers, detailed_results)
    return {"results": detailed_results, "summary": summary}


def _summarize(providers: list[str], detailed_results: list[dict]) -> dict:
    summary = {}
    for provider_name in providers:
        rows = [r for r in detailed_results if r["provider"] == provider_name]
        available_rows = [r for r in rows if r["available"]]

        if not available_rows:
            summary[provider_name] = {
                "available": False,
                "num_samples": len(rows),
                "mean_cer": None,
                "mean_wer": None,
                "exact_match_accuracy": None,
                "mean_normalized_edit_distance": None,
                "mean_inference_time_ms": None,
                "diagnosis_accuracy": None,
                "mistake_category_macro_f1": None,
            }
            continue

        cers = [r["cer"] for r in available_rows]
        wers = [r["wer"] for r in available_rows]
        exacts = [1 if r["exact_match"] else 0 for r in available_rows]
        neds = [r["normalized_edit_distance"] for r in available_rows]
        times = [r["inference_time_ms"] for r in available_rows]

        diag_rows = [r for r in available_rows if r["diagnosis_correct"] is not None]
        diag_acc = (
            round(sum(1 for r in diag_rows if r["diagnosis_correct"]) / len(diag_rows), 4)
            if diag_rows
            else None
        )
        f1_pairs = [
            (r["predicted_mistake_category"], r["reference_mistake_category"])
            for r in diag_rows
            if r["predicted_mistake_category"] and r["reference_mistake_category"]
        ]
        f1 = macro_f1(f1_pairs) if f1_pairs else None

        summary[provider_name] = {
            "available": True,
            "num_samples": len(rows),
            "mean_cer": round(sum(cers) / len(cers), 4),
            "mean_wer": round(sum(wers) / len(wers), 4),
            "exact_match_accuracy": round(sum(exacts) / len(exacts), 4),
            "mean_normalized_edit_distance": round(sum(neds) / len(neds), 4),
            "mean_inference_time_ms": round(sum(times) / len(times), 2),
            "diagnosis_accuracy": diag_acc,
            "mistake_category_macro_f1": f1,
        }
    return summary


def export_results_csv(results: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        output_path.write_text("")
        return
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def export_summary_json(summary: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
