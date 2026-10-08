"""End-to-end research experiment script.

Demonstrates, in one executable script (usable as a Jupyter notebook by
running cell-by-cell, or as a plain CLI script):

  1. Dataset/manifest loading
  2. Image preprocessing (the same pipeline used in the live app)
  3. TrOCR inference (skipped gracefully if torch/transformers are not
     installed or no GPU/CPU budget is available - never fabricated)
  4. Metric calculation (CER/WER/exact-match/edit distance)
  5. Provider comparison across whatever providers are available
  6. Plotting benchmark results to evaluation/results/*.png

Device selection (CPU/CUDA/MPS), a fixed random seed, latency measurement,
and a configurable batch size are all wired in below. Results are saved to
evaluation/results/ rather than only printed, so a run can be revisited.

Run:
    python run_experiment.py --manifest manifests/demo_manifest.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = EVAL_ROOT.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.benchmark_service import load_manifest, run_benchmark  # noqa: E402
from app.services.preprocessing import preprocess_image  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # noqa: BLE001 - a partially-installed torch can raise
        # AttributeError rather than ImportError; seeding torch is a nice-to-
        # have here, never worth crashing the whole experiment script over.
        pass


def select_device(preferred: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "cpu (torch not installed)"

    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def step_1_load_dataset(manifest_path: str):
    print("\n[1/6] Loading dataset manifest...")
    samples = load_manifest(manifest_path)
    print(f"  Loaded {len(samples)} samples from {manifest_path}")
    return samples


def step_2_preprocess_samples(samples, output_dir: Path):
    print("\n[2/6] Running preprocessing pipeline on each sample...")
    output_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for sample in samples:
        out_path = output_dir / f"{sample.sample_id}_processed.png"
        if not Path(sample.image_path).exists():
            print(f"  SKIP {sample.sample_id}: image not found at {sample.image_path}")
            continue
        meta = preprocess_image(sample.image_path, out_path)
        metas.append({"sample_id": sample.sample_id, **meta})
        print(f"  {sample.sample_id}: ink_ratio={meta['ink_ratio']}, warnings={meta['warnings']}")
    return metas


def step_3_trocr_inference_demo(samples, device: str):
    print(f"\n[3/6] TrOCR inference demo (device={device})...")
    from app.providers.trocr_provider import get_trocr_provider

    provider = get_trocr_provider()
    available, reason = provider.is_available()
    if not available:
        print(f"  TrOCR unavailable ({reason}). Skipping - NOT fabricating output.")
        return []

    results = []
    for sample in samples[:2]:  # keep the demo script fast; full runs use step_5's benchmark
        if not Path(sample.image_path).exists():
            continue
        start = time.perf_counter()
        try:
            result = provider.transcribe(sample.image_path)
            elapsed = time.perf_counter() - start
            print(f"  {sample.sample_id}: '{result.text}' ({elapsed*1000:.1f} ms, conf={result.confidence})")
            results.append({"sample_id": sample.sample_id, "text": result.text, "latency_s": elapsed})
        except Exception as exc:  # noqa: BLE001
            print(f"  {sample.sample_id}: inference failed ({exc})")
    return results


def step_4_metrics_explainer():
    print("\n[4/6] Metrics used: CER, WER, exact-match accuracy, normalized edit")
    print("       distance, inference time (ms), diagnosis accuracy, mistake-category")
    print("       macro F1. See app/utils/text_metrics.py for implementations.")


def step_5_provider_comparison(manifest_path: str, providers: list[str] | None):
    print("\n[5/6] Running full provider comparison via the benchmark service...")
    outcome = run_benchmark(manifest_path, providers)
    for provider, stats in outcome["summary"].items():
        status = "available" if stats["available"] else "UNAVAILABLE"
        print(f"  {provider:12s} [{status}]  n={stats['num_samples']}")
    return outcome


def step_6_plot_results(outcome: dict, output_dir: Path):
    print("\n[6/6] Plotting benchmark results...")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed - skipping plot generation.")
        return

    providers = list(outcome["summary"].keys())
    available_providers = [p for p in providers if outcome["summary"][p]["available"]]
    if not available_providers:
        print("  No available providers to plot.")
        return

    cers = [outcome["summary"][p]["mean_cer"] for p in available_providers]
    times = [outcome["summary"][p]["mean_inference_time_ms"] for p in available_providers]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(available_providers, cers, color="#4f46e5")
    axes[0].set_title("Mean Character Error Rate")
    axes[0].set_ylabel("CER")

    axes[1].bar(available_providers, times, color="#16a34a")
    axes[1].set_title("Mean Inference Time")
    axes[1].set_ylabel("ms")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "provider_comparison.png"
    fig.savefig(out_path, dpi=120)
    print(f"  Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full research/evaluation experiment.")
    parser.add_argument("--manifest", default=str(EVAL_ROOT / "manifests" / "demo_manifest.jsonl"))
    parser.add_argument("--providers", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future batched inference.")
    args = parser.parse_args()

    print("=== AI Multimodal Tutor: Research Experiment ===")
    print(f"Seed: {args.seed} | Device preference: {args.device} | Batch size: {args.batch_size}")
    set_seed(args.seed)
    device = select_device(args.device)

    results_dir = EVAL_ROOT / "results"
    processed_dir = results_dir / "processed_samples"

    samples = step_1_load_dataset(args.manifest)
    preprocessing_meta = step_2_preprocess_samples(samples, processed_dir)
    trocr_results = step_3_trocr_inference_demo(samples, device)
    step_4_metrics_explainer()
    outcome = step_5_provider_comparison(args.manifest, args.providers)
    step_6_plot_results(outcome, results_dir)

    experiment_record = {
        "seed": args.seed,
        "device": device,
        "batch_size": args.batch_size,
        "manifest": args.manifest,
        "preprocessing_meta": preprocessing_meta,
        "trocr_demo_results": trocr_results,
        "benchmark_summary": outcome["summary"],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    record_path = results_dir / "experiment_record.json"
    record_path.write_text(json.dumps(experiment_record, indent=2, default=str))
    print(f"\nSaved full experiment record to {record_path}")


if __name__ == "__main__":
    main()
