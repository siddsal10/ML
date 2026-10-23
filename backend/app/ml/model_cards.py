"""Static descriptions of every ML model; metrics are always read from the
generated report files (ml_models/reports/<name>.json), never hardcoded."""
from __future__ import annotations

import json

from app.ml.artifacts import MODEL_DIR

REPORT_DIR = MODEL_DIR / "reports"

CARDS: dict[str, dict] = {
    "symbol_classifier": {
        "title": "Handwritten symbol recognizer",
        "stage": "Reading handwriting (OCR)",
        "kind": "Supervised classification · 17 classes",
        "input": "One segmented symbol (binary crop)",
        "output": "Symbol + probability for each class",
        "features": "HOG (324) + 7x7 zoning density (49) + aspect ratio (1) = 374 features",
        "algorithms": ["SVM (RBF)", "Random Forest", "k-Nearest Neighbours", "Logistic Regression"],
        "data": "REAL handwriting: CROHME training expressions (symbols harvested by aligning segmentation with LaTeX labels) "
                "+ HASYv2 (real symbols, ~150 writers) + synthetic font glyphs + student corrections",
        "split": "Choices on validation (held-out CROHME-train expressions + held-out fonts); results on the official "
                 "CROHME 2014/2016/2019 test sets (unseen writers)",
        "limitations": "CROHME is tablet handwriting rendered to images, not phone photos of paper; only linear "
                       "expressions are harvested; touching/cursive symbols are not segmented correctly.",
        "retrainable": True,
    },
    "mistake_classifier": {
        "title": "Mistake-type classifier",
        "stage": "Diagnosing the misconception",
        "kind": "Supervised classification · 9 classes",
        "input": "SymPy verification of every step + answer key + OCR confidence",
        "output": "Mistake category + probability + evidence",
        "features": "33 named features: sign-flip match, operator-swap match, term moved without sign change, first non-equivalent step, ...",
        "algorithms": ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree", "Rule-based baseline"],
        "data": "Synthetic student solutions with injected misconceptions (350 per class). Real-data counterpart: "
                "'Error type from real student mistakes' (ScratchMath)",
        "split": "Stratified 80/20; model chosen by 5-fold CV on the 80%",
        "limitations": "Trained on generated mistakes; real students combine errors in messier ways - the score is an upper bound.",
        "retrainable": False,
    },
    "real_mistake_classifier": {
        "title": "Error type from real student mistakes",
        "stage": "Diagnosing the misconception (second opinion)",
        "kind": "Supervised classification · 7 classes + slip vs understanding",
        "input": "Correct answer, student's wrong answer, question numbers, photo of the handwritten working",
        "output": "Error type (calculation, concept, strategy, copying, ...) and P(careless slip)",
        "features": "42 language-independent features: answer patterns (sign flip, decimal shift, one-digit slip, "
                    "intermediate value, swapped blanks ...), question statistics, scratchwork image statistics",
        "algorithms": ["Random Forest", "Gradient Boosting", "Logistic Regression", "Majority / random baselines"],
        "data": "REAL: ScratchMath - 1,720 wrong answers by Chinese primary & middle-school students with photos of "
                "their scratchwork and expert error labels (CC BY-NC-SA 4.0)",
        "split": "Stratified 80/20 hold-out; model chosen by 5-fold CV on the 80%",
        "limitations": "Small dataset with overlapping categories; Chinese word problems, so the deployed model uses only "
                       "language-independent features; only applied when the student's answer is wrong.",
        "retrainable": False,
    },
    "knowledge_tracing": {
        "title": "Knowledge tracing & item difficulty",
        "stage": "Tracking mastery and choosing practice",
        "kind": "Sequential prediction + IRT",
        "input": "A student's history of right/wrong answers per skill",
        "output": "P(skill mastered), P(next answer correct), item difficulty",
        "features": "BKT hidden state; PFA = per-skill success/failure counts + item difficulty; Rasch 1PL difficulty; ridge model on item features",
        "algorithms": ["Bayesian Knowledge Tracing (EM)", "Performance Factors Analysis (logistic regression)", "EMA baseline", "Rasch IRT"],
        "data": "REAL students: ASSISTments 2009-2010 (446k answers, 4,151 students, 110 skills) for the benchmark and "
                "for the deployed BKT parameters of 7 skills; simulated learners for the item bank / recommender study",
        "split": "Learner-disjoint 80/20",
        "limitations": "ASSISTments students and skills are close to, but not the same as, StepWise's; item difficulties "
                       "for StepWise's own question bank still come from simulation until real practice logs exist.",
        "retrainable": False,
    },
    "student_clustering": {
        "title": "Class misconception groups",
        "stage": "Teacher dashboard",
        "kind": "Unsupervised clustering",
        "input": "Per-student mistake rates + per-skill mastery",
        "output": "Named groups of students + 2-D map",
        "features": "6 mistake-rate features + 8 mastery features, standardized",
        "algorithms": ["K-Means (k by silhouette)", "Agglomerative (Ward)", "PCA for the 2-D map"],
        "data": "Simulated class with 4 planted archetypes (evaluation); live students (dashboard)",
        "split": "n/a (unsupervised) - evaluated by recovery of planted groups (ARI)",
        "limitations": "Planted archetypes are cleaner than real classes.",
        "retrainable": False,
    },
}


def load_report(name: str) -> dict | None:
    path = REPORT_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def headline(name: str, report: dict | None) -> dict | None:
    if not report:
        return None
    if name == "symbol_classifier":
        real = load_report("symbol_classifier_real")
        if real:
            m = real["algorithm_comparison"][real["selected_model"]]["test_symbols"]
            base = real["training_data_comparison"]["synthetic_only"]["test_symbols"]["macro_f1"]
            return {"label": "Macro-F1 on real handwriting (CROHME test)", "value": m["macro_f1"],
                    "model": real["selected_model"], "baseline": base, "baseline_label": "fonts-only"}
        m = report["models"][report["selected_model"]]
        return {"label": "Macro-F1 on unseen fonts", "value": m["macro_f1"], "model": report["selected_model"]}
    if name == "mistake_classifier":
        m = report["models"][report["selected_model"]]
        base = report["models"]["rule_based_baseline"]["test_macro_f1"]
        return {"label": "Test macro-F1 (synthetic)", "value": m["test_macro_f1"], "model": report["selected_model"],
                "baseline": base, "baseline_label": "rules"}
    if name == "knowledge_tracing":
        real = load_report("knowledge_tracing_real")
        if real:
            best = max(real["models"], key=lambda k: real["models"][k]["auc"])
            return {"label": "Next-answer AUC on real students", "value": real["models"][best]["auc"], "model": best,
                    "baseline": real["models"]["ema_baseline"]["auc"], "baseline_label": "EMA"}
        kt = report["knowledge_tracing"]
        return {"label": "Next-answer AUC (BKT)", "value": kt["bkt"]["auc"], "model": "bkt", "baseline": kt["ema_baseline"]["auc"]}
    if name == "real_mistake_classifier":
        b = report["slip_vs_understanding"][report["slip_vs_understanding_selected"]]
        return {"label": "Slip vs misunderstanding accuracy (real mistakes)", "value": b["test_accuracy"],
                "model": report["slip_vs_understanding_selected"],
                "baseline": report["slip_vs_understanding"]["majority_baseline"]["test_accuracy"], "baseline_label": "majority"}
    if name == "student_clustering":
        return {"label": "ARI vs planted groups", "value": report["metrics"]["ari_vs_planted_archetypes"], "model": "kmeans"}
    return None
