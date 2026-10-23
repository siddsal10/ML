"""Error-type classifier trained on REAL student mistakes: ScratchMath.

ScratchMath (songdj/ScratchMath, CC BY-NC-SA 4.0) has 1,720 real wrong answers
from Chinese primary/middle-school students. Each has the question, the correct
answer, a reference solution, the student's answer, a photo of the student's
handwritten scratchwork, and an expert-annotated error category (7 classes).
The expert's free-text explanation is NOT used as an input (it would leak the label).

Feature groups (all hand-engineered, classical ML):
- answer: how the wrong answer relates to the correct one - sign flip, decimal
  point shift, digit transposition, one-digit copy slip, missing multiply/divide
  by a number from the question, answer equals an intermediate value of the
  reference solution, swapped blanks, ...
- question: length and number statistics (language-independent)
- image: amount and layout of handwritten working (ink, strokes, lines)
- text (full model only): TF-IDF character n-grams of the Chinese question

Two models are trained:
- full: all groups (research benchmark on this dataset)
- portable: answer + question + image only, so it also runs on StepWise's
  English questions; this one is deployed.
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass

import cv2
import numpy as np
import sympy

from app.ml.artifacts import BACKEND_ROOT, MODEL_DIR
from app.services.math_parser import UnsafeExpressionError, safe_parse

SCRATCHMATH_DIR = BACKEND_ROOT / "data" / "external" / "scratchmath"
REAL_MISTAKE_MODEL_PATH = MODEL_DIR / "real_mistake_classifier.joblib"

CATEGORIES = [
    "calculation_error",        # 计算错误
    "question_misunderstanding",  # 题目理解错误
    "concept_error",            # 知识点错误
    "strategy_error",           # 答题技巧错误
    "copying_error",            # 手写誊抄错误
    "reasoning_error",          # 逻辑推理错误
    "attention_error",          # 注意力与细节错误
]
CATEGORY_LABELS = {
    "calculation_error": "Calculation error",
    "question_misunderstanding": "Misunderstood the question",
    "concept_error": "Concept / knowledge error",
    "strategy_error": "Solving-strategy error",
    "copying_error": "Copying / transcription error",
    "reasoning_error": "Logical reasoning error",
    "attention_error": "Attention / careless error",
}

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# --- parsing --------------------------------------------------------------------

def _latex_to_plain(s: str) -> str:
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "").replace("\\,", "")
    s = re.sub(r"\{\{", "{", s)
    s = re.sub(r"\}\}", "}", s)
    s = re.sub(r"\\[dt]?frac", r"\\frac", s)
    # mixed numbers 2\frac{3}{4} -> (2+(3)/(4))
    s = re.sub(r"(\d+)\s*\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1+(\2)/(\3))", s)
    for _ in range(4):
        s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/").replace("×", "*").replace("÷", "/")
    s = s.replace("%", "/100").replace("{", "(").replace("}", ")").replace("\\pi", "pi")
    return s.strip()


def parse_value(s: str):
    """Best-effort: a sympy number/expression for one answer string, or None."""
    s = (s or "").strip()
    if not s:
        return None
    if s in _CN_DIGITS:
        return sympy.Integer(_CN_DIGITS[s])
    plain = _latex_to_plain(s)
    plain = re.sub(r"[^\x00-\x7f]", "", plain).strip()  # drop units / Chinese words
    plain = plain.rstrip(".").replace(" ", "")
    if "=" in plain:  # "45-30=15" -> right-hand side
        plain = plain.split("=")[-1]
    if not plain:
        return None
    try:
        v = safe_parse(plain)
    except (UnsafeExpressionError, Exception):  # noqa: BLE001 - any parse failure means "not numeric"
        return None
    return v if isinstance(v, sympy.Expr) else None  # tuples ("1,2"), equations etc. are not single values


def split_blanks(s: str) -> list[str]:
    return [b.strip() for b in re.split(r"[;；]", s or "") if b.strip()]


def split_alternatives(s: str) -> list[str]:
    return [a.strip() for a in re.split(r"\n|或|,(?![^{]*\})|，", s or "") if a.strip()]


def _num(v) -> float | None:
    try:
        if v is not None and not v.free_symbols:
            f = complex(v.evalf())
            if abs(f.imag) < 1e-12 and math.isfinite(f.real):
                return f.real
    except Exception:  # noqa: BLE001
        pass
    return None


def _digits(x: float) -> str:
    s = f"{abs(x):.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "")


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


# --- features ---------------------------------------------------------------------

ANSWER_FEATURES = [
    "n_blanks", "n_wrong_blanks", "blanks_swapped", "student_empty", "student_numeric", "correct_numeric",
    "symbolic", "sym_negated", "log_abs_correct", "log_abs_student", "log_rel_error", "sign_flip",
    "decimal_shift", "int_ratio", "half_or_double", "off_by_one", "off_small", "digit_permutation",
    "one_digit_diff", "digit_edit_norm", "same_digit_count", "student_in_question", "student_in_solution",
    "diff_in_question", "ratio_in_question", "student_is_fraction", "correct_is_fraction", "student_has_text",
]
QUESTION_FEATURES = ["q_len", "q_n_numbers", "q_log_max_number", "q_has_fraction", "q_has_percent",
                     "q_has_equation", "q_has_decimal", "is_middle_school"]
IMAGE_FEATURES = ["img_ink", "img_components", "img_lines", "img_ink_bbox_fill", "img_aspect", "img_log_ink_px"]
PORTABLE_FEATURES = ANSWER_FEATURES + QUESTION_FEATURES + IMAGE_FEATURES


def answer_features(correct: str, student: str, question: str = "", solution: str = "") -> dict:
    f = {k: 0.0 for k in ANSWER_FEATURES}
    c_blanks, s_blanks = split_blanks(correct), split_blanks(student)
    f["n_blanks"] = float(len(c_blanks))
    f["student_empty"] = float(not s_blanks)
    f["student_has_text"] = float(bool(re.search(r"[^\x00-\x7f]", student or "")))
    q_nums = [float(x) for x in _NUM_RE.findall(question or "")]
    sol_nums = [float(x) for x in _NUM_RE.findall(solution or "")]

    # pair up blanks; find the first wrong one
    c_vals = [[_num(parse_value(a)) for a in split_alternatives(b)] for b in c_blanks]
    s_vals = [_num(parse_value(b)) for b in s_blanks]
    wrong = []
    for i, alts in enumerate(c_vals):
        sv = s_vals[i] if i < len(s_vals) else None
        if sv is None or not any(a is not None and _close(a, sv) for a in alts):
            wrong.append(i)
    f["n_wrong_blanks"] = float(len(wrong))
    flat_c = sorted(a[0] for a in c_vals if a and a[0] is not None)
    flat_s = sorted(v for v in s_vals if v is not None)
    f["blanks_swapped"] = float(len(c_blanks) > 1 and len(flat_c) == len(flat_s) > 1
                                and all(_close(a, b) for a, b in zip(flat_c, flat_s)))

    i = wrong[0] if wrong else 0
    c_raw = c_blanks[i] if i < len(c_blanks) else (correct or "")
    s_raw = s_blanks[i] if i < len(s_blanks) else (student or "")
    alts = [a for a in (c_vals[i] if i < len(c_vals) else []) if a is not None]
    s = s_vals[i] if i < len(s_vals) else None
    f["student_is_fraction"] = float("frac" in s_raw or "/" in s_raw)
    f["correct_is_fraction"] = float("frac" in c_raw or "/" in c_raw)

    if s is None or not alts:
        # symbolic answers (e.g. factorisations): compare up to sign
        cs, ss = parse_value(split_alternatives(c_raw)[0] if split_alternatives(c_raw) else c_raw), parse_value(s_raw)
        if cs is not None and ss is not None and (cs.free_symbols or ss.free_symbols):
            f["symbolic"] = 1.0
            try:
                f["sym_negated"] = float(sympy.simplify(sympy.expand(cs + ss)) == 0)
            except Exception:  # noqa: BLE001
                pass
        f["correct_numeric"] = float(bool(alts))
        return f

    c = min(alts, key=lambda a: abs(a - s))  # closest acceptable form
    f["student_numeric"] = f["correct_numeric"] = 1.0
    diff = s - c
    f["log_abs_correct"] = math.log10(1 + abs(c))
    f["log_abs_student"] = math.log10(1 + abs(s))
    f["log_rel_error"] = math.log10(1 + abs(diff) / max(abs(c), 1e-9))
    f["sign_flip"] = float(c != 0 and _close(s, -c))
    if c != 0 and s != 0:
        r = s / c
        f["decimal_shift"] = float(any(_close(r, 10.0 ** k) for k in (-3, -2, -1, 1, 2, 3)))
        f["half_or_double"] = float(_close(r, 2) or _close(r, 0.5))
        big = max(abs(r), 1 / abs(r))
        f["int_ratio"] = float(_close(big, round(big)) and 2 <= round(big) <= 12)
        f["ratio_in_question"] = float(any(_close(big, q) for q in q_nums if q not in (0, 1)))
    f["off_by_one"] = float(_close(abs(diff), 1))
    f["off_small"] = float(0 < abs(diff) <= max(2.0, 0.02 * abs(c)))
    ds, dc = _digits(s), _digits(c)
    f["digit_permutation"] = float(ds != dc and sorted(ds) == sorted(dc))
    f["same_digit_count"] = float(len(ds) == len(dc))
    f["one_digit_diff"] = float(len(ds) == len(dc) and sum(a != b for a, b in zip(ds, dc)) == 1)
    f["digit_edit_norm"] = _lev(ds, dc) / max(len(ds), len(dc), 1)
    f["student_in_question"] = float(any(_close(s, q) for q in q_nums))
    f["student_in_solution"] = float(any(_close(s, q) for q in sol_nums) and not _close(s, c))
    f["diff_in_question"] = float(any(_close(abs(diff), q) for q in q_nums if q != 0))
    return f


def question_features(question: str, level: str = "primary") -> dict:
    nums = [float(x) for x in _NUM_RE.findall(question or "")]
    return {
        "q_len": math.log1p(len(question or "")),
        "q_n_numbers": float(len(nums)),
        "q_log_max_number": math.log10(1 + max((abs(n) for n in nums), default=0)),
        "q_has_fraction": float("frac" in (question or "") or "/" in (question or "") or "分之" in (question or "")),
        "q_has_percent": float("%" in (question or "")),
        "q_has_equation": float(bool(re.search(r"[xy]\s*[=+\-]|=\s*[xy]|方程|解", question or ""))),
        "q_has_decimal": float(bool(re.search(r"\d\.\d", question or ""))),
        "is_middle_school": float(level == "middle"),
    }


def image_features(gray: np.ndarray | None) -> dict:
    if gray is None or gray.size == 0:
        return {k: 0.0 for k in IMAGE_FEATURES}
    h, w = gray.shape[:2]
    scale = 800 / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    ink = binary > 0
    n_comp = cv2.connectedComponentsWithStats(binary, connectivity=8)[0] - 1
    rows = ink.mean(axis=1) > 0.003
    lines = int(np.sum(rows[1:] & ~rows[:-1]) + rows[0])
    ys, xs = np.nonzero(ink)
    fill = float(ink.sum() / max(1, (ys.ptp() + 1) * (xs.ptp() + 1))) if len(ys) else 0.0
    return {
        "img_ink": float(ink.mean()),
        "img_components": math.log1p(n_comp),
        "img_lines": float(lines),
        "img_ink_bbox_fill": fill,
        "img_aspect": float(np.log(gray.shape[1] / gray.shape[0])),
        "img_log_ink_px": math.log1p(float(ink.sum())),
    }


def portable_vector(correct: str, student: str, question: str, solution: str, level: str, gray) -> np.ndarray:
    f = {**answer_features(correct, student, question, solution), **question_features(question, level), **image_features(gray)}
    return np.array([f[k] for k in PORTABLE_FEATURES], dtype=np.float64)


# --- dataset ----------------------------------------------------------------------

@dataclass
class ScratchMathData:
    X: np.ndarray
    y: np.ndarray
    questions: list[str]
    levels: list[str]


def load_scratchmath() -> ScratchMathData:
    import pyarrow.parquet as pq

    X, y, questions, levels = [], [], [], []
    for level in ("primary", "middle"):
        table = pq.read_table(SCRATCHMATH_DIR / f"{level}.parquet")
        for row in table.to_pylist():
            img = row["student_scratchwork"]
            gray = None
            if img and img.get("bytes"):
                gray = cv2.imdecode(np.frombuffer(img["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
            X.append(portable_vector(row["answer"], row["student_answer"], row["question"], row["solution"] or "", level, gray))
            y.append(CATEGORIES[int(row["error_category"])])
            questions.append(row["question"])
            levels.append(level)
    return ScratchMathData(np.vstack(X), np.array(y), questions, levels)


def available() -> bool:
    return all((SCRATCHMATH_DIR / f"{c}.parquet").exists() for c in ("primary", "middle"))


# --- training / evaluation ----------------------------------------------------------

SLIP_CATEGORIES = {"calculation_error", "copying_error", "attention_error"}


def to_slip_label(categories) -> np.ndarray:
    return np.array(["slip" if c in SLIP_CATEGORIES else "understanding" for c in categories])


def _portable_models(seed: int) -> dict:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "random_baseline": DummyClassifier(strategy="stratified", random_state=seed),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                                random_state=seed, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=seed),
    }


def train_real_mistake_classifier(seed: int = 0, out_path=REAL_MISTAKE_MODEL_PATH) -> dict:
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler

    from app.ml.artifacts import save_bundle

    data = load_scratchmath()
    X, y, q = data.X, data.y, np.array(data.questions)
    yb = to_slip_label(y)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    def evaluate(target, models):
        out = {}
        for name, m in models.items():
            cv_f1 = cross_val_score(m, X[tr], target[tr], cv=cv, scoring="f1_macro")
            m.fit(X[tr], target[tr])
            pred = m.predict(X[te])
            r = {"cv_macro_f1": round(float(cv_f1.mean()), 4), "test_accuracy": round(float(accuracy_score(target[te], pred)), 4),
                 "test_macro_f1": round(float(f1_score(target[te], pred, average="macro")), 4),
                 "test_balanced_accuracy": round(float(balanced_accuracy_score(target[te], pred)), 4)}
            if len(set(target)) == 2 and "baseline" not in name:
                r["test_auc"] = round(float(roc_auc_score(target[te] == "slip", m.predict_proba(X[te])[:, list(m.classes_).index("slip")])), 4)
            out[name] = r
        return out

    seven = evaluate(y, _portable_models(seed))
    binary = evaluate(yb, _portable_models(seed))

    # Research benchmark: + TF-IDF character n-grams of the (Chinese) question text.
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3, sublinear_tf=True)
    scaler = StandardScaler().fit(X[tr])
    Xf_tr = sp.hstack([tfidf.fit_transform(q[tr]), sp.csr_matrix(scaler.transform(X[tr]))]).tocsr()
    Xf_te = sp.hstack([tfidf.transform(q[te]), sp.csr_matrix(scaler.transform(X[te]))]).tocsr()
    full = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
    full_cv = cross_val_score(full, Xf_tr, y[tr], cv=cv, scoring="f1_macro")
    full.fit(Xf_tr, y[tr])
    fp = full.predict(Xf_te)
    seven["logistic_regression_with_question_text"] = {
        "cv_macro_f1": round(float(full_cv.mean()), 4), "test_accuracy": round(float(accuracy_score(y[te], fp)), 4),
        "test_macro_f1": round(float(f1_score(y[te], fp, average="macro")), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y[te], fp)), 4),
        "note": "not deployable in StepWise (Chinese question text)",
    }

    def pick(results):
        return max((n for n in results if "baseline" not in n and "with_question_text" not in n), key=lambda n: results[n]["cv_macro_f1"])

    best7, best2 = pick(seven), pick(binary)
    m7 = _portable_models(seed)[best7].fit(X[tr], y[tr])
    m2 = _portable_models(seed)[best2].fit(X[tr], yb[tr])
    pi = permutation_importance(m7, X[te], y[te], scoring="f1_macro", n_repeats=10, random_state=seed)
    order = np.argsort(-pi.importances_mean)

    report = {
        "task": "error_type_real_student_mistakes",
        "dataset": "ScratchMath (real Chinese K-9 students' wrong answers + handwritten scratchwork, expert-labelled)",
        "licence": "CC BY-NC-SA 4.0",
        "n_samples": int(len(y)),
        "class_counts": {c: int((y == c).sum()) for c in CATEGORIES},
        "split": "stratified 80/20 hold-out test; model chosen by 5-fold CV on the 80%",
        "features": {"portable": len(PORTABLE_FEATURES), "groups": ["answer pattern", "question statistics", "scratchwork image"]},
        "seven_class": seven,
        "seven_class_selected": best7,
        "seven_class_confusion_matrix": confusion_matrix(y[te], m7.predict(X[te]), labels=CATEGORIES).tolist(),
        "slip_vs_understanding": binary,
        "slip_vs_understanding_selected": best2,
        "slip_definition": "slip = calculation / copying / attention error; understanding = question, concept, strategy or reasoning error",
        "labels": CATEGORIES,
        "label_names": CATEGORY_LABELS,
        "top_feature_importances": [[PORTABLE_FEATURES[i], round(float(pi.importances_mean[i]), 4)] for i in order[:12]],
        "limitations": "Only 1,720 examples with fuzzy, overlapping categories; questions are Chinese word problems, so "
                       "StepWise uses the language-independent features only. Every sample is a wrong answer, so the model "
                       "is only applied when a student's answer is wrong.",
    }
    # deploy: refit the chosen models on all samples
    final7 = _portable_models(seed)[best7].fit(X, y)
    final2 = _portable_models(seed)[best2].fit(X, yb)
    save_bundle({"model": final7, "slip_model": final2, "features": PORTABLE_FEATURES, "categories": CATEGORIES,
                 "model_name": best7, "slip_model_name": best2, "metrics": report}, out_path)
    return report


def predict_error_type(bundle: dict, correct: str, student: str, question: str, gray, level: str) -> dict:
    x = portable_vector(correct, student, question, "", level, gray).reshape(1, -1)
    p7 = bundle["model"].predict_proba(x)[0]
    classes = list(bundle["model"].classes_)
    order = np.argsort(p7)[::-1]
    p2 = bundle["slip_model"].predict_proba(x)[0]
    slip_p = float(p2[list(bundle["slip_model"].classes_).index("slip")])
    return {
        "top": [{"category": classes[i], "label": CATEGORY_LABELS[classes[i]], "probability": round(float(p7[i]), 3)} for i in order[:3]],
        "slip_probability": round(slip_p, 3),
        "kind": "careless slip" if slip_p >= 0.5 else "gap in understanding",
    }
