"""The mistake classifier on REAL student errors: CogTutorBugs.

CogTutorBugs (McNichols, Zhang & Lan, 2023; CC BY 4.0) holds 5,339 steps on
which Algebra I students using the Cognitive Tutor made an error the tutor
caught, each as (equation before, student's equation after, error label).
They are typed, not handwritten, so this tests diagnosis alone - exactly the
part trained only on synthetic data.

SymPy first splits the steps in two:
- non-equivalent steps (the new equation has different solutions): real
  mathematical errors, StepWise's job. Their labels are mapped to StepWise
  categories (LABEL_MAP) to score the fine-grained diagnosis;
- equivalent steps: valid but unhelpful moves (e.g. adding instead of
  subtracting to remove a term). StepWise gives no strategy feedback, so the
  right outcome is "this step is valid".

Each step is diagnosed as the two-line working "eq1 / eq2" with no answer
key (it is an intermediate step) and full confidence (typed text).

Problem templates (eq1 with numbers blanked) are split 30/70 into a
development part, used for error analysis when improving the classifier,
and a held-out part on which model versions are compared.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

import numpy as np
import sympy
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from app.ml.artifacts import BACKEND_ROOT, MISTAKE_MODEL_PATH, load_bundle
from app.ml.mistake_classifier import candidate_models, featurize
from app.ml.mistake_data import generate_samples
from app.ml.mistake_features import extract_features
from app.services.diagnosis import diagnose, diagnose_rule_based
from app.services.math_parser import X, UnsafeExpressionError, analyze_transcription, safe_parse

COGTUTOR_PATH = BACKEND_ROOT / "data" / "external" / "cogtutor" / "bug_full_5339.json"

# CogTutor error label -> StepWise category, for steps that are mathematically wrong.
LABEL_MAP = {
    "DIVIDE_MULTIPLY_OPP_SIGNS": "sign_error",  # "a negative times a negative is positive"
    "CHECK_SIGN": "sign_error",
    "FORGOT_NEGATIVE": "sign_error",
    "REVERSED_SIDES": "sign_error",  # computed b - a instead of a - b, so the sign flips
    "WRONG_OPERATION": "incorrect_operation",  # e.g. added where the step needed subtracting
    "EXPRESSION_NOT_EQUIVALENT": "arithmetic_error",  # simplified to a wrong value (variable -> manipulation)
    "FORGOT_VARIABLE": "algebraic_manipulation_error",
    "COMBINE_LIKE_TERMS": "algebraic_manipulation_error",
    "SIMPLIFY_LATER": "algebraic_manipulation_error",
}
CATEGORIES = ["sign_error", "incorrect_operation", "arithmetic_error", "algebraic_manipulation_error"]
ERROR_CATEGORIES = {"arithmetic_error", "sign_error", "incorrect_operation", "algebraic_manipulation_error",
                    "skipped_step", "final_answer_only_error", "formula_selection_error"}
_VAR = re.compile(r"[a-zA-Z]")


def _to_x(eq: str) -> str | None:
    """Rename the single variable to x (StepWise's variable); None if several letters."""
    letters = set(_VAR.findall(eq))
    if len({c.lower() for c in letters}) > 1:
        return None
    return _VAR.sub("x", eq)


def _solutions(eq_text: str):
    eq = safe_parse(eq_text)
    if not isinstance(eq, sympy.Eq):
        raise UnsafeExpressionError("not an equation")
    diff = sympy.simplify(eq.lhs - eq.rhs)
    if X not in diff.free_symbols:
        return "all" if diff == 0 else "none"
    return frozenset(round(complex(s).real, 6) for s in sympy.solve(diff, X) if abs(complex(s).imag) < 1e-9)


def template(eq: str) -> str:
    return re.sub(r"\d+(\.\d+)?", "#", eq)


def is_dev_template(eq1: str) -> bool:
    """Stable 30% of problem templates kept for development (error analysis)."""
    return int(hashlib.md5(template(eq1).encode()).hexdigest(), 16) % 10 < 3


def _in_scope(eq1: str, eq2: str) -> bool:
    """Close to StepWise's scope: integers only, x on one side, x never divided by."""
    if "." in eq1 + eq2 or "^" in eq1 + eq2 or re.search(r"/\s*\(?-?x", eq1):
        return False
    lhs, rhs = eq1.split("=")
    return ("x" in lhs) != ("x" in rhs)


def load_steps(path=COGTUTOR_PATH) -> tuple[list[dict], dict]:
    raw = json.loads(path.read_text())
    steps, skipped = [], Counter()
    for r in raw:
        eq1, eq2 = _to_x(r["eq1"] or ""), _to_x(r["eq2"] or "")
        if not eq1 or not eq2 or eq1.count("=") != 1 or eq2.count("=") != 1:
            skipped["not two single equations"] += 1
            continue
        try:
            equivalent = _solutions(eq1) == _solutions(eq2)
        except (UnsafeExpressionError, TypeError, ValueError, NotImplementedError):
            skipped["unparseable"] += 1
            continue
        label = r["Label Name"]
        category = LABEL_MAP.get(label) if not equivalent else None
        if category == "arithmetic_error" and _VAR.search(str(r["Input"])):
            category = "algebraic_manipulation_error"
        steps.append({"eq1": eq1, "eq2": eq2, "label": label, "equivalent": equivalent, "category": category,
                      "in_scope": _in_scope(eq1, eq2), "dev": is_dev_template(eq1)})
    return steps, {"n_records": len(raw), "n_used": len(steps), "skipped": dict(skipped)}


def _work(s: dict) -> str:
    return f"{s['eq1']}\n{s['eq2']}"


def _bootstrap_ci(correct: np.ndarray, seed: int = 0, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _classification(steps: list[dict], preds: list[str]) -> dict:
    gold = [s["category"] for s in steps]
    correct = np.array([p == g for p, g in zip(preds, gold)])
    return {
        "n": len(steps),
        "accuracy": round(float(correct.mean()), 4),
        "accuracy_95ci": _bootstrap_ci(correct),
        "macro_f1": round(float(f1_score(gold, preds, labels=CATEGORIES, average="macro", zero_division=0)), 4),
        "per_class_recall": {c: round(float(np.mean([p == c for p, g in zip(preds, gold) if g == c])), 4)
                             for c in CATEGORIES if c in gold},
        "confusion_rows_true_cols_pred": {
            "labels": CATEGORIES + ["other"],
            "matrix": confusion_matrix(gold, [p if p in CATEGORIES else "other" for p in preds],
                                       labels=CATEGORIES + ["other"]).tolist(),
        },
    }


def _detection(preds: list[str]) -> dict:
    n = len(preds)
    return {
        "n": n,
        "flagged_as_error": round(sum(p in ERROR_CATEGORIES for p in preds) / n, 4),
        "said_correct": round(sum(p == "correct_solution" for p in preds) / n, 4),
        "said_unreadable_or_insufficient": round(sum(p in ("insufficient_information", "transcription_uncertainty")
                                                     for p in preds) / n, 4),
    }


def _adaptation(mapped: list[dict], seed: int, log) -> dict:
    """Problem-grouped 5-fold CV on the real mapped errors: synthetic-only model vs
    synthetic + real training folds vs real-only (same Random Forest settings)."""
    synth = generate_samples(n_per_label=350, seed=seed)
    Xs, ys = featurize(synth), np.array([s.label for s in synth])
    Xr = np.vstack([extract_features(_work(s), None, 1.0).vector for s in mapped])
    yr = np.array([s["category"] for s in mapped])
    groups = np.array([template(s["eq1"]) for s in mapped])  # same problem template together
    deployed = load_bundle(str(MISTAKE_MODEL_PATH))["model"]
    fracs = (0.1, 0.25, 0.5, 1.0)
    pred = {"synthetic_only": np.empty(len(yr), dtype=object), "real_only": np.empty(len(yr), dtype=object),
            **{f"synthetic_plus_{int(f * 100)}pct_real": np.empty(len(yr), dtype=object) for f in fracs}}
    rng = np.random.default_rng(seed)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(Xr, yr, groups)):
        log(f"    fold {k + 1}/5")
        pred["synthetic_only"][te] = deployed.predict(Xr[te])
        pred["real_only"][te] = candidate_models(seed)["random_forest"].fit(Xr[tr], yr[tr]).predict(Xr[te])
        for f in fracs:
            sub = rng.permutation(tr)[: max(1, int(f * len(tr)))]
            m = candidate_models(seed)["random_forest"].fit(np.vstack([Xs, Xr[sub]]), np.concatenate([ys, yr[sub]]))
            pred[f"synthetic_plus_{int(f * 100)}pct_real"][te] = m.predict(Xr[te])
    return {name: {"accuracy": round(float(accuracy_score(yr, p)), 4),
                   "macro_f1": round(float(f1_score(yr, list(p), labels=CATEGORIES, average="macro", zero_division=0)), 4)}
            for name, p in pred.items()}


def run(seed: int = 0, log=print) -> dict:
    steps, stats = load_steps()
    log(f"  {stats}")
    ml = [diagnose(analyze_transcription(_work(s)), None, 1.0, transcription=_work(s)).category for s in steps]
    rules = [diagnose_rule_based(analyze_transcription(_work(s)), None, 1.0).category for s in steps]
    for s, m, r in zip(steps, ml, rules):
        s["ml"], s["rules"] = m, r

    wrong = [s for s in steps if not s["equivalent"]]
    valid = [s for s in steps if s["equivalent"]]
    mapped = [s for s in wrong if s["category"]]
    majority = Counter(s["category"] for s in mapped).most_common(1)[0][0]
    report = {
        "task": "mistake_diagnosis_on_real_student_errors",
        "dataset": "CogTutorBugs (McNichols, Zhang & Lan, 2023): Cognitive Tutor Algebra I, 95 students, CC BY 4.0",
        "data": stats,
        "input": "two-line working 'eq1 / eq2', variable renamed to x, no answer key, OCR confidence 1.0 (typed)",
        "label_map": LABEL_MAP,
        "n_mathematically_wrong_steps": len(wrong),
        "n_valid_but_unhelpful_steps": len(valid),
        "wrong_steps_by_label": dict(Counter(s["label"] for s in wrong).most_common()),
        "valid_steps_by_label": dict(Counter(s["label"] for s in valid).most_common()),
        "error_detection_on_wrong_steps": {"ml": _detection([s["ml"] for s in wrong]),
                                           "rule_based": _detection([s["rules"] for s in wrong])},
        "valid_steps_accepted": {"ml": _detection([s["ml"] for s in valid]),
                                 "rule_based": _detection([s["rules"] for s in valid])},
        "fine_grained": {
            "all": {"ml": _classification(mapped, [s["ml"] for s in mapped]),
                    "rule_based": _classification(mapped, [s["rules"] for s in mapped]),
                    "majority_class": {"class": majority,
                                       "accuracy": round(float(np.mean([s["category"] == majority for s in mapped])), 4)}},
        },
    }
    scoped = [s for s in mapped if s["in_scope"]]
    if scoped:
        report["fine_grained"]["in_scope"] = {
            "definition": "integers only, x on one side of eq1, x never in a denominator",
            "ml": _classification(scoped, [s["ml"] for s in scoped]),
            "rule_based": _classification(scoped, [s["rules"] for s in scoped]),
        }
    heldout = [s for s in mapped if not s["dev"]]
    report["fine_grained"]["heldout_templates"] = {
        "definition": "the 70% of problem templates not used for development",
        "ml": _classification(heldout, [s["ml"] for s in heldout]),
        "rule_based": _classification(heldout, [s["rules"] for s in heldout]),
        "majority_class_accuracy": round(float(np.mean([s["category"] == majority for s in heldout])), 4),
    }
    heldout_wrong = [s for s in wrong if not s["dev"]]
    report["error_detection_on_wrong_steps"]["ml_heldout_templates"] = _detection([s["ml"] for s in heldout_wrong])
    log("  domain adaptation (problem-grouped 5-fold CV) ...")
    report["adaptation_cv"] = _adaptation(mapped, seed, log)
    report["examples"] = [{k: s[k] for k in ("eq1", "eq2", "label", "category", "ml")} for s in mapped[:: max(1, len(mapped) // 12)]][:12]
    report["mistake_model"] = {k: load_bundle(str(MISTAKE_MODEL_PATH)).get(k) for k in ("model_name", "feature_version")}
    return report
