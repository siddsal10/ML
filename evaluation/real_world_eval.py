"""Evaluate StepWise on real photographed student work (see docs/REAL_WORLD_VALIDATION.md).

    python real_world_eval.py --manifest manifests/real_world.jsonl

Writes evaluation/results/real_world_<manifest name>.json and prints the summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_ROOT.parent / "backend"))

from app.ml.validation.real_world import evaluate_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tau", type=float, default=None, help="confirm-before-diagnose threshold (default: the app's)")
    args = parser.parse_args()

    report = evaluate_manifest(args.manifest, **({"tau": args.tau} if args.tau is not None else {}))
    out = EVAL_ROOT / "results" / f"real_world_{Path(args.manifest).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "pages"}, indent=2))
    print(f"\nfull report -> {out}")


if __name__ == "__main__":
    main()
