"""Run the validation studies on the trained (deployed) models and write reports.

    cd backend && python -m app.ml.validate                  # all studies (~25 min on a laptop CPU)
    python -m app.ml.validate --only cogtutor writers        # a subset

Studies (app/ml/validation/), each written to ml_models/reports/validation_<name>.json:
    propagation  misreads -> diagnosis on solutions written with real CROHME glyphs, under
                 simulated phone capture; chooses the confirm-before-diagnose threshold
    capture      the same pages read with plain Otsu vs the photo-robust page binarization
    reading      CROHME test expressions: error sources, constrained decoding, confidence
    writers      MNIST test digits: pen-on-paper scans by Census staff vs high-school students
    cogtutor     the mistake classifier on real Algebra I student errors (CogTutorBugs)
    skills       ASSISTments <-> StepWise skill mapping, with test metrics per mapped skill
    policy       practice-policy advantage under five learner simulators + study sample size

Needs the datasets from scripts/download_datasets.py (crohme, mnist, cogtutor, assistments).
"""
from __future__ import annotations

import argparse
import json
import time

from app.ml.model_cards import REPORT_DIR

STUDIES = ["propagation", "capture", "reading", "writers", "cogtutor", "skills", "policy"]


def _save(name: str, report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"validation_{name}.json").write_text(json.dumps(report, indent=2))
    print(f"          -> {REPORT_DIR / f'validation_{name}.json'}")


def _tau_from_report() -> float | None:
    path = REPORT_DIR / "validation_propagation.json"
    return json.loads(path.read_text())["tau"] if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=STUDIES, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    todo = args.only or STUDIES
    log = lambda m: print("          " + m)  # noqa: E731
    t0 = time.perf_counter()

    if "propagation" in todo:
        from app.ml.validation import propagation
        from app.ml.validation.crohme_glyphs import harvest_test_glyphs

        print("[propagation] composing solutions from CROHME test-writer glyphs ...")
        _save("propagation", propagation.run(harvest_test_glyphs(("2014", "2016", "2019")), seed=args.seed, log=log))

    if "capture" in todo:
        from app.ml.validation import propagation
        from app.ml.validation.crohme_glyphs import harvest_test_glyphs

        print("[capture] Otsu-only vs photo-robust binarization under simulated capture ...")
        _save("capture", propagation.binarization_ablation(harvest_test_glyphs(("2014", "2016", "2019")),
                                                           seed=args.seed, log=log))

    if "reading" in todo:
        from app.ml.validation import reading

        print("[reading] CROHME test expressions ...")
        _save("reading", reading.run(tau=_tau_from_report(), log=log))

    if "writers" in todo:
        from app.ml.validation import writers

        print("[writers] MNIST test digits by writer population ...")
        _save("writers", writers.run(log=log))

    if "cogtutor" in todo:
        from app.ml.validation import cogtutor

        print("[cogtutor] mistake classifier on real student errors ...")
        _save("cogtutor", cogtutor.run(seed=args.seed, log=log))

    if "skills" in todo:
        from app.ml.tracing.real_data import evaluate_assistments

        print("[skills] ASSISTments skill mapping (re-runs the real knowledge-tracing benchmark) ...")
        report, _ = evaluate_assistments(seed=args.seed)
        keys = ("dataset", "n_attempts", "n_learners", "n_skills", "split", "stepwise_skill_mapping", "stepwise_skill_match",
                "mapped_share_of_attempts", "test_metrics_on_mapped_skills", "stepwise_bkt_params_from_real_data", "models")
        _save("skills", {"task": "assistments_to_stepwise_skill_mapping", **{k: report[k] for k in keys}})

    if "policy" in todo:
        from app.ml.validation import policy

        print("[policy] practice policies under five learner simulators ...")
        _save("policy", policy.run(log=log))

    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
