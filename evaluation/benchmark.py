"""Standalone CLI for running the OCR/diagnosis benchmark outside the API.

Run:
    python benchmark.py --manifest manifests/demo_manifest.jsonl --providers demo trocr

Requires the backend's demo images to exist (run
backend/scripts/generate_demo_images.py first if using the bundled demo
manifest).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = EVAL_ROOT.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.benchmark_service import export_results_csv, export_summary_json, run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OCR/diagnosis benchmark from the command line.")
    parser.add_argument(
        "--manifest",
        default=str(EVAL_ROOT / "manifests" / "demo_manifest.jsonl"),
        help="Path to a CSV or JSONL manifest (default: bundled demo manifest).",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Provider names to test (default: all registered providers).",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(EVAL_ROOT / "results" / "run"),
        help="Prefix for output files (writes <prefix>.csv and <prefix>_summary.json).",
    )
    args = parser.parse_args()

    print(f"Loading manifest: {args.manifest}")
    outcome = run_benchmark(args.manifest, args.providers)

    print("\n=== Summary ===")
    for provider, stats in outcome["summary"].items():
        if not stats["available"]:
            print(f"  {provider:12s}  UNAVAILABLE")
            continue
        print(
            f"  {provider:12s}  n={stats['num_samples']:<4} "
            f"CER={stats['mean_cer']:<6} WER={stats['mean_wer']:<6} "
            f"exact_match={stats['exact_match_accuracy']:<6} "
            f"avg_ms={stats['mean_inference_time_ms']}"
        )

    output_csv = f"{args.output_prefix}.csv"
    output_json = f"{args.output_prefix}_summary.json"
    export_results_csv(outcome["results"], output_csv)
    export_summary_json(outcome["summary"], output_json)
    print(f"\nWrote detailed results to {output_csv}")
    print(f"Wrote summary to {output_json}")


if __name__ == "__main__":
    main()
