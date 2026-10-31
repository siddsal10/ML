"""Train and evaluate both classical-ML models, then write a report.

    cd backend && python -m app.ml.train                 # full run (~1-2 min on a laptop CPU)
    python -m app.ml.train --quick                       # smaller data, for smoke tests
    python -m app.ml.train --extra-symbols path/to/dir   # add real labelled symbol crops
    python -m app.ml.train --only tracing                # train a single model group

Outputs:
    ml_models/{symbol_classifier,mistake_classifier,knowledge_tracing}.joblib (active versions)
    ml_models/reports/<model>.json (read by the ML Lab page) + confusion-matrix PNGs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.ml import real_benchmark, real_cnn, real_mistakes, real_symbols, registry
from app.ml.real_mistakes import train_real_mistake_classifier
from app.ml.artifacts import BACKEND_ROOT, MISTAKE_MODEL_PATH, SYMBOL_MODEL_PATH, load_bundle, save_bundle
from app.ml.irt.recommender import KT_MODEL_PATH
from app.ml.tracing.real_data import ASSISTMENTS_PATH, evaluate_assistments
from app.ml.clustering.cluster import evaluate_on_simulated_class
from app.ml.model_cards import REPORT_DIR
from app.ml.tracing.train import train_tracing
from app.ml.glyph_data import find_fonts, generate_dataset, render_expression
from app.ml.mistake_classifier import train_mistake_classifier
from app.ml.mistake_data import generate_samples
from app.ml.symbol_classifier import SymbolRecognizer, train_symbol_classifier
from app.utils.text_metrics import character_error_rate, exact_match


E2E_EXPRESSIONS = [
    "12 + 15 = 27", "-3 + 5 = -8", "2x + 4 = 10\nx = 3", "7 * 8 = 56", "3x - 6 = 9\n3x = 15\nx = 5",
    "(4 + 6) / 2 = 5", "45 - 19 = 26", "5x + 2 = 17\nx = 3", "9 - 14 = -5", "x / 3 = 4\nx = 12",
]


def end_to_end_ocr(recognizer: SymbolRecognizer, held_out_fonts: list[str], seed: int = 1) -> dict:
    """Render multi-symbol expressions in the held-out fonts and run full
    segmentation + recognition with the model that never saw those fonts."""
    font_paths = [p for p in find_fonts() if Path(p).name in held_out_fonts]
    rng = np.random.default_rng(seed)
    rows = []
    for fp in font_paths:
        for expr in E2E_EXPRESSIONS:
            page = render_expression(fp, expr, rng)
            hyp, conf, _ = recognizer.transcribe_gray(page)
            rows.append({"font": Path(fp).name, "reference": expr, "hypothesis": hyp,
                         "cer": character_error_rate(expr, hyp), "exact": exact_match(expr, hyp)})

    demo = []
    manifest = BACKEND_ROOT / "data" / "demo_samples" / "manifest.json"
    if manifest.exists():
        for slug, entry in json.loads(manifest.read_text()).items():
            hyp, conf, _ = recognizer.transcribe_file(BACKEND_ROOT / entry["path"])
            demo.append({"sample": slug, "reference": entry["text"], "hypothesis": hyp,
                         "cer": character_error_rate(entry["text"], hyp), "confidence": round(conf, 3)})

    return {
        "note": "Pages are rendered in fonts excluded from training. Fonts are a proxy for writers - real "
                "handwriting (IAM/CROHME/classroom manifests via the benchmark module) is still needed for a true estimate. "
                "demo_images use DejaVu Sans Bold, which is not in the training fonts.",
        "n_pages": len(rows),
        "mean_cer": round(float(np.mean([r["cer"] for r in rows])), 4) if rows else None,
        "exact_match": round(float(np.mean([r["exact"] for r in rows])), 4) if rows else None,
        "worst_examples": sorted(rows, key=lambda r: -r["cer"])[:5],
        "demo_images": demo,
    }


def _plot_confusion(cm: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(1 + 0.55 * len(labels), 1 + 0.5 * len(labels)))
    ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm_arr[i, j]:
                ax.text(j, i, cm_arr[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_report(name: str, report: dict) -> None:
    (REPORT_DIR / f"{name}.json").write_text(json.dumps(report, indent=2))


def _register(name: str, path: Path, headline: dict, notes: str) -> None:
    bundle = load_bundle(str(path))
    version = registry.register(name, bundle, headline, notes=notes, activate=True)
    print(f"      registered {name} {version} (active)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small datasets for a fast smoke run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-symbols", type=str, default=None, help="folder of real labelled symbol crops")
    parser.add_argument("--only", choices=["symbols", "real_symbols", "cnn_symbols", "mistakes", "real_mistakes", "tracing", "real_tracing", "clustering"], default=None)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    todo = [args.only] if args.only else ["symbols", "real_symbols", "cnn_symbols", "mistakes", "real_mistakes", "tracing", "real_tracing", "clustering"]
    have = real_symbols.available()
    have["assistments"] = ASSISTMENTS_PATH.exists()

    if "symbols" in todo:
        print("[symbols] generating glyph data ...")
        glyphs = generate_dataset(samples_per_font=8 if args.quick else 30, seed=args.seed, extra_dir=args.extra_symbols)
        print(f"          {len(glyphs.y)} symbols from {len(glyphs.group_names)} sources")
        report, held_out_recognizer, _ = train_symbol_classifier(glyphs, seed=args.seed)
        for name, m in report["models"].items():
            print(f"          {name:22s} acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")
        report["end_to_end_ocr"] = end_to_end_ocr(held_out_recognizer, report["held_out_fonts"], seed=args.seed + 1)
        print(f"          end-to-end CER {report['end_to_end_ocr']['mean_cer']}, exact-match {report['end_to_end_ocr']['exact_match']}")
        _save_report("symbol_classifier", report)
        _plot_confusion(report["confusion_matrix"], report["labels"], f"Symbols ({report['selected_model']}, held-out fonts)",
                        REPORT_DIR / "symbol_confusion.png")
        _register("symbol_classifier", SYMBOL_MODEL_PATH,
                  {"held_out_macro_f1": report["models"][report["selected_model"]]["macro_f1"], "algorithm": report["selected_model"], "n_corrections": 0},
                  "full training run (synthetic glyphs)")

    if "real_symbols" in todo:
        if not have["crohme"] or not have["hasy"]:
            print("[real_symbols] skipped - run scripts/download_datasets.py hasy crohme first")
        else:
            print("[real_symbols] harvesting CROHME + HASYv2, comparing training data and algorithms ...")
            report, bundle = real_benchmark.run(seed=args.seed, synthetic_per_font=8 if args.quick else 30,
                                                log=lambda m: print("          " + m))
            _save_report("symbol_classifier_real", report)
            best = report["algorithm_comparison"][report["selected_model"]]
            print(f"          selected {report['selected_model']} on {report['selected_training_data']}: "
                  f"CROHME test macro-F1 {best['test_symbols']['macro_f1']}, expressions exact {best['test_expressions']['exact_match']}")
            version = registry.register("symbol_classifier", bundle, {
                "held_out_macro_f1": best["test_symbols"]["macro_f1"], "benchmark": "CROHME 2014/2016/2019 test symbols",
                "algorithm": report["selected_model"], "n_corrections": 0,
            }, notes=f"real handwriting: {report['selected_training_data']}", activate=True)
            print(f"          registered symbol_classifier {version} (active)")

    if "cnn_symbols" in todo:
        if not have["crohme"] or not have["hasy"]:
            print("[cnn_symbols] skipped - run scripts/download_datasets.py hasy crohme mnist first")
        else:
            print("[cnn_symbols] CNN + recognition-driven segmentation on real handwriting ...")
            report, bundle = real_cnn.run(seed=args.seed, epochs=4 if args.quick else 20,
                                          log=lambda m: print("          " + m))
            _save_report("symbol_classifier_cnn", report)
            print(f"          validation symbols {report['symbols']['val_accuracy']}, "
                  f"expressions exact {report['expressions_test']['cnn_lattice_decoder']}")
            save_bundle(bundle, SYMBOL_MODEL_PATH)
            version = registry.register("symbol_classifier", bundle, {
                "held_out_macro_f1": report["symbols"]["val_accuracy"], "benchmark": "CROHME + HME100K validation symbols",
                "algorithm": "cnn", "n_corrections": 0}, notes="CNN + lattice decoding, real handwriting", activate=True)
            print(f"          registered symbol_classifier {version} (active)")

    if "mistakes" in todo:
        print("[mistakes] generating labelled student work ...")
        samples = generate_samples(n_per_label=80 if args.quick else 350, seed=args.seed)
        report = train_mistake_classifier(samples, seed=args.seed)
        for name, m in report["models"].items():
            cv = f"cv-F1={m['cv_macro_f1_mean']:.4f}  " if "cv_macro_f1_mean" in m else " " * 17
            print(f"          {name:22s} {cv}test macro-F1={m['test_macro_f1']:.4f}")
        _save_report("mistake_classifier", report)
        _plot_confusion(report["confusion_matrix"], report["labels"], f"Mistakes ({report['selected_model']})",
                        REPORT_DIR / "mistake_confusion.png")
        _plot_confusion(report["rule_based_confusion_matrix"], report["labels"], "Mistakes (rule-based baseline)",
                        REPORT_DIR / "mistake_confusion_rule_based.png")

    if "mistakes" in todo:
        from app.ml.validation.cogtutor import COGTUTOR_PATH, deploy_with_real_errors

        if COGTUTOR_PATH.exists():
            print("[mistakes] adding real CogTutor student errors (development templates) to training ...")
            real = deploy_with_real_errors(seed=args.seed, log=lambda m: print("          " + m))
            report["real_training"] = real
            _save_report("mistake_classifier", report)

    if "real_mistakes" in todo:
        if not real_mistakes.available():
            print("[real_mistakes] skipped - run scripts/download_datasets.py scratchmath first")
        else:
            print("[real_mistakes] ScratchMath (1,720 real student mistakes): error type + slip vs understanding ...")
            report = train_real_mistake_classifier(seed=args.seed)
            for task in ("seven_class", "slip_vs_understanding"):
                for name, m in report[task].items():
                    print(f"          {task[:5]} {name:40s} test acc={m['test_accuracy']:.3f} macro-F1={m['test_macro_f1']:.3f}")
            _save_report("real_mistake_classifier", report)

    if "tracing" in todo:
        print("[tracing] simulating learners, fitting BKT / PFA / Rasch, running policy simulation ...")
        report = train_tracing(seed=args.seed, n_learners=200 if args.quick else 600, quick=args.quick)
        for name, m in report["knowledge_tracing"].items():
            print(f"          {name:14s} AUC={m['auc']:.4f}  RMSE={m['rmse']:.4f}")
        for name, m in report["recommender_simulation"]["policies"].items():
            print(f"          policy {name:12s} skills mastered={m['skills_mastered_frac_mean']:.3f}")
        _save_report("knowledge_tracing", report)

    if "real_tracing" in todo:
        if not have["assistments"]:
            print("[real_tracing] skipped - run scripts/download_datasets.py assistments first")
        else:
            print("[real_tracing] ASSISTments 2009 (real students): EMA vs BKT vs PFA variants ...")
            report, real_params = evaluate_assistments(seed=args.seed, n_iter=15 if args.quick else 30)
            for name, m in report["models"].items():
                print(f"          {name:16s} AUC={m['auc']:.4f}  RMSE={m['rmse']:.4f}")
            _save_report("knowledge_tracing_real", report)
            kt = load_bundle(str(KT_MODEL_PATH))
            if kt is not None:
                kt["bkt_params"].update(real_params)
                kt["bkt_param_source"] = {s: ("ASSISTments 2009 (real)" if s in real_params else "simulation") for s in kt["bkt_params"]}
                save_bundle(kt, KT_MODEL_PATH)
                print(f"          deployed BKT parameters for {sorted(real_params)} now fitted on real students")

    if "clustering" in todo:
        print("[clustering] evaluating on a simulated class ...")
        report = evaluate_on_simulated_class(seed=args.seed)
        print(f"          k={report['k_selected']}  ARI vs planted groups={report['metrics']['ari_vs_planted_archetypes']}")
        _save_report("student_clustering", report)

    print(f"done in {time.perf_counter() - t0:.1f}s - reports in {REPORT_DIR}")


if __name__ == "__main__":
    main()
