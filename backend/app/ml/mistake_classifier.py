"""Mistake-category classifier: model comparison against the rule-based baseline."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.ml.artifacts import MISTAKE_MODEL_PATH, save_bundle
from app.ml.mistake_data import LABELS, Sample
from app.ml.mistake_features import FEATURE_NAMES, FEATURE_VERSION, extract_features


def candidate_models(seed: int = 0) -> dict:
    return {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=1.0))]),
        "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed),
    }


def featurize(samples: list[Sample]) -> np.ndarray:
    return np.vstack([extract_features(s.text, s.expected_answer, s.ocr_confidence).vector for s in samples])


def _rule_based_predictions(samples: list[Sample]) -> list[str]:
    from app.services.diagnosis import diagnose_rule_based
    from app.services.math_parser import analyze_transcription

    return [
        diagnose_rule_based(analyze_transcription(s.text), s.expected_answer, s.ocr_confidence).category
        for s in samples
    ]


def train_mistake_classifier(samples: list[Sample], seed: int = 0, out_path: Path = MISTAKE_MODEL_PATH) -> dict:
    X = featurize(samples)
    y = np.array([s.label for s in samples])
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    results = {}
    for name, model in candidate_models(seed).items():
        cv_f1 = cross_val_score(model, X[idx_train], y[idx_train], cv=cv, scoring="f1_macro")
        t0 = time.perf_counter()
        model.fit(X[idx_train], y[idx_train])
        fit_s = time.perf_counter() - t0
        pred = model.predict(X[idx_test])
        results[name] = {
            "cv_macro_f1_mean": round(float(cv_f1.mean()), 4),
            "cv_macro_f1_std": round(float(cv_f1.std()), 4),
            "test_accuracy": round(float(accuracy_score(y[idx_test], pred)), 4),
            "test_macro_f1": round(float(f1_score(y[idx_test], pred, average="macro")), 4),
            "fit_seconds": round(fit_s, 2),
        }

    # Baseline: the hand-written decision tree that diagnosis used before ML.
    test_samples = [samples[i] for i in idx_test]
    rule_pred = _rule_based_predictions(test_samples)
    results["rule_based_baseline"] = {
        "test_accuracy": round(float(accuracy_score(y[idx_test], rule_pred)), 4),
        "test_macro_f1": round(float(f1_score(y[idx_test], rule_pred, average="macro", labels=LABELS)), 4),
    }

    # Model selection uses cross-validation on the training split only.
    best = max((n for n in results if "cv_macro_f1_mean" in results[n]), key=lambda n: results[n]["cv_macro_f1_mean"])
    best_model = candidate_models(seed)[best]
    best_model.fit(X[idx_train], y[idx_train])
    best_pred = best_model.predict(X[idx_test])

    rf = candidate_models(seed)["random_forest"].fit(X[idx_train], y[idx_train])
    importances = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])

    final = candidate_models(seed)[best]
    final.fit(X, y)
    report = {
        "task": "mistake_diagnosis",
        "split": "stratified 80/20; model selection by 5-fold CV on the 80%",
        "n_samples": int(len(y)),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "n_features": len(FEATURE_NAMES),
        "models": results,
        "selected_model": best,
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(y[idx_test], best_pred, labels=LABELS).tolist(),
        "rule_based_confusion_matrix": confusion_matrix(y[idx_test], rule_pred, labels=LABELS).tolist(),
        "per_class_f1": dict(zip(LABELS, [round(float(v), 4) for v in f1_score(y[idx_test], best_pred, labels=LABELS, average=None)])),
        "top_feature_importances": [[n, round(float(v), 4)] for n, v in importances[:15]],
    }
    save_bundle(
        {"model": final, "model_name": best, "classes": list(final.classes_), "feature_names": FEATURE_NAMES,
         "feature_version": FEATURE_VERSION, "metrics": report},
        out_path,
    )
    return report
