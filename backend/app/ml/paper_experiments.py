"""Additional experiments for the paper: confidence intervals, significance
tests, feature ablations and robustness to phone-photo conditions.

    python -m app.ml.paper_experiments          # writes ml_models/reports/paper_experiments.json

Everything is evaluated on the same held-out data as the main results:
CROHME 2014/2016/2019 test sets (symbols and whole expressions), the
ScratchMath hold-out split, and ASSISTments test learners.
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from app.ml.artifacts import SYMBOL_MODEL_PATH, load_bundle
from app.ml.model_cards import REPORT_DIR

SEED = 0


# --- statistics ----------------------------------------------------------------

def bootstrap_ci(y, pred, metric, n: int = 1000, seed: int = SEED, groups=None) -> list[float]:
    """95% percentile bootstrap CI. With `groups`, whole groups (e.g. learners) are resampled."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    stats = []
    if groups is None:
        idx = np.arange(len(y))
        for _ in range(n):
            s = rng.choice(idx, len(idx))
            stats.append(metric(y[s], pred[s]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n):
            s = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
            stats.append(metric(y[s], pred[s]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def mcnemar(y, p1, p2) -> dict:
    """McNemar test with continuity correction on paired predictions."""
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    b = int(np.sum((p1 == y) & (p2 != y)))
    c = int(np.sum((p1 != y) & (p2 == y)))
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return {"model1_only_correct": b, "model2_only_correct": c, "chi2": round(float(stat), 3),
            "p_value": float(f"{chi2.sf(stat, 1):.3g}")}


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


# --- photo-like degradations -----------------------------------------------------------

def degrade(gray: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    g = gray.astype(np.float32)
    h, w = g.shape
    if kind == "clean":
        return gray
    if kind == "blur":
        g = cv2.GaussianBlur(g, (0, 0), 1.5)
    elif kind == "noise":
        g = g + rng.normal(0, 25, g.shape)
    elif kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 20])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    elif kind == "rotation":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.0)
        g = cv2.warpAffine(g, m, (w, h), borderValue=255)
    elif kind == "uneven_light":
        yy, xx = np.mgrid[0:h, 0:w]
        shade = 0.45 + 0.55 * (xx / w)  # dark on the left, bright on the right
        g = (g * 0.75 + 40) * shade
    elif kind == "photo_combined":
        yy, xx = np.mgrid[0:h, 0:w]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 3, 1.0)
        g = cv2.warpAffine(g, m, (w, h), borderValue=255)
        g = (g * 0.75 + 40) * (0.55 + 0.45 * (yy / h))
        g = cv2.GaussianBlur(g, (0, 0), 1.0) + rng.normal(0, 12, g.shape)
        g = np.clip(g, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 40])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return np.clip(g, 0, 255).astype(np.uint8)


DEGRADATIONS = ["clean", "blur", "noise", "jpeg", "rotation", "uneven_light", "photo_combined"]


# --- experiments ---------------------------------------------------------------------------

def symbol_experiments(log=print) -> dict:
    from sklearn.ensemble import RandomForestClassifier

    from app.ml.glyph_data import generate_dataset
    from app.ml.real_symbols import TEST_FILES, cached_real_sets, harvest_crohme
    from app.ml.symbol_classifier import SymbolRecognizer, candidate_models
    from app.utils.text_metrics import character_error_rate

    real = cached_real_sets()
    synth = generate_dataset(samples_per_font=30, seed=SEED)
    X_tr = np.vstack([synth.X, real["train_X"]])
    y_tr = np.concatenate([synth.y, real["train_y"]])
    X_te, y_te = real["test_X"], real["test_y"]
    out = {"n_test_symbols": int(len(y_te))}

    bundle = load_bundle(str(SYMBOL_MODEL_PATH))
    deployed = bundle["model"]
    p_svm = deployed.predict(X_te)
    out["deployed"] = {
        "model": bundle.get("model_name"),
        "accuracy": round(float(accuracy_score(y_te, p_svm)), 4),
        "accuracy_ci95": bootstrap_ci(y_te, p_svm, accuracy_score),
        "macro_f1": round(float(macro_f1(y_te, p_svm)), 4),
        "macro_f1_ci95": bootstrap_ci(y_te, p_svm, macro_f1),
    }
    log(f"  deployed: {out['deployed']}")

    log("  McNemar: deployed SVM vs random forest (same training data) ...")
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED).fit(X_tr, y_tr)
    p_rf = rf.predict(X_te)
    out["mcnemar_svm_vs_rf"] = {**mcnemar(y_te, p_svm, p_rf), "rf_accuracy": round(float(accuracy_score(y_te, p_rf)), 4)}

    log("  feature ablation (SVM) ...")
    groups = {"HOG only (324)": slice(0, 324), "zoning only (49)": slice(324, 373),
              "HOG + zoning (373)": slice(0, 373), "all features (374)": slice(0, 374)}
    abl = {}
    for name, sl in groups.items():
        m = candidate_models(SEED)["svm_rbf"].fit(X_tr[:, sl], y_tr)
        p = m.predict(X_te[:, sl])
        abl[name] = {"accuracy": round(float(accuracy_score(y_te, p)), 4), "macro_f1": round(float(macro_f1(y_te, p)), 4)}
        log(f"    {name}: {abl[name]}")
    out["feature_ablation"] = abl

    log("  robustness to photo-like degradations (whole test expressions) ...")
    _, exprs, _ = harvest_crohme(list(TEST_FILES.values()), group_offset=10_000_000, collect_expressions=True)
    rec = SymbolRecognizer(bundle)
    rob = {}
    for kind in DEGRADATIONS:
        row = {}
        for norm in (False, True):
            rng = np.random.default_rng(SEED)
            cers, exact = [], []
            for e in exprs:
                hyp, _, _ = rec.transcribe_gray(degrade(e.gray, kind, rng), normalize=norm)
                hyp = hyp.replace("\n", " ")
                cers.append(character_error_rate(e.target, hyp))
                exact.append(hyp == e.target)
            row["with_normalisation" if norm else "without_normalisation"] = {
                "exact_match": round(float(np.mean(exact)), 4), "mean_cer": round(float(np.mean(cers)), 4)}
        rob[kind] = row
        log(f"    {kind:15s} {row}")
    out["robustness"] = {"n_expressions": len(exprs), "results": rob}
    return out


def mistake_experiments(log=print) -> dict:
    from sklearn.model_selection import train_test_split

    from app.ml.real_mistakes import (ANSWER_FEATURES, IMAGE_FEATURES, PORTABLE_FEATURES, QUESTION_FEATURES,
                                      _portable_models, load_scratchmath, to_slip_label)

    data = load_scratchmath()
    X, y = data.X, data.y
    yb = to_slip_label(y)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    idx = {n: i for i, n in enumerate(PORTABLE_FEATURES)}
    cols = lambda names: [idx[n] for n in names]  # noqa: E731
    sets = {
        "answer only": cols(ANSWER_FEATURES),
        "answer + question": cols(ANSWER_FEATURES + QUESTION_FEATURES),
        "answer + image": cols(ANSWER_FEATURES + IMAGE_FEATURES),
        "question + image (no answer)": cols(QUESTION_FEATURES + IMAGE_FEATURES),
        "all (deployed)": cols(PORTABLE_FEATURES),
    }
    out = {"ablation": {}}
    for name, c in sets.items():
        m2 = _portable_models(SEED)["logistic_regression"].fit(X[tr][:, c], yb[tr])
        m7 = _portable_models(SEED)["random_forest"].fit(X[tr][:, c], y[tr])
        p2, p7 = m2.predict(X[te][:, c]), m7.predict(X[te][:, c])
        proba = m2.predict_proba(X[te][:, c])[:, list(m2.classes_).index("slip")]
        row = {"slip_accuracy": round(float(accuracy_score(yb[te], p2)), 4),
               "slip_auc": round(float(roc_auc_score(yb[te] == "slip", proba)), 4),
               "seven_accuracy": round(float(accuracy_score(y[te], p7)), 4),
               "seven_macro_f1": round(float(macro_f1(y[te], p7)), 4)}
        if name == "all (deployed)":
            row["slip_accuracy_ci95"] = bootstrap_ci(yb[te], p2, accuracy_score)
            row["slip_auc_ci95"] = bootstrap_ci(yb[te] == "slip", proba, roc_auc_score)
            row["seven_accuracy_ci95"] = bootstrap_ci(y[te], p7, accuracy_score)
            majority = np.full(len(te), "understanding")
            row["slip_vs_majority_mcnemar"] = mcnemar(yb[te], p2, majority)
        out["ablation"][name] = row
        log(f"    {name:30s} {row}")
    return out


def tracing_experiments(log=print) -> dict:
    from collections import defaultdict

    from sklearn.linear_model import LogisticRegression

    from app.ml.tracing.models import BKTParams, fit_bkt
    from app.ml.tracing.real_data import _fit_item_difficulty, _pfa_design, load_assistments

    report = json.loads((REPORT_DIR / "knowledge_tracing_real.json").read_text())
    attempts = load_assistments()
    learners = np.array(sorted({a.learner for a in attempts}))
    rng = np.random.default_rng(SEED)
    rng.shuffle(learners)  # same split as the main evaluation
    test_l = set(learners[: len(learners) // 5].tolist())
    train = [a for a in attempts if a.learner not in test_l]
    test = [a for a in attempts if a.learner in test_l]
    skills = sorted({a.skill for a in attempts})
    skill_index = {s: i for i, s in enumerate(skills)}

    item_b = _fit_item_difficulty(train, report["rasch_l2_selected"])
    X_tr, y_tr = _pfa_design(train, skill_index, item_b, recency=True)
    X_te, y_te = _pfa_design(test, skill_index, item_b, recency=True)
    p_best = LogisticRegression(C=1.0, max_iter=3000, fit_intercept=False).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

    # EMA and BKT in the same (time) order, so predictions are paired.
    seqs = defaultdict(list)
    for a in train:
        seqs[(a.learner, a.skill)].append(a.correct)
    by_skill = defaultdict(list)
    for (_, s), seq in seqs.items():
        by_skill[s].append(seq)
    bkt = {s: fit_bkt(v, n_iter=30) for s, v in by_skill.items()}
    ema, L = defaultdict(lambda: 0.5), {}
    p_ema, p_bkt = [], []
    for a in test:
        key = (a.learner, a.skill)
        p_ema.append(ema[key])
        ema[key] += 0.3 * (a.correct - ema[key])
        prm = bkt.get(a.skill, BKTParams(0.5, 0.1, 0.2, 0.1))
        l = L.get(key, prm.L0)
        p_bkt.append(l * (1 - prm.S) + (1 - l) * prm.G)
        post = l * (1 - prm.S) / (l * (1 - prm.S) + (1 - l) * prm.G) if a.correct else l * prm.S / (l * prm.S + (1 - l) * (1 - prm.G))
        L[key] = post + (1 - post) * prm.T
    groups = np.array([a.learner for a in test])
    out = {}
    for name, p in (("ema_baseline", p_ema), ("bkt", p_bkt), ("pfa_irt_recency", p_best)):
        out[name] = {"auc": round(float(roc_auc_score(y_te, p)), 4),
                     "auc_ci95_learner_bootstrap": bootstrap_ci(y_te, p, roc_auc_score, n=300, groups=groups)}
        log(f"    {name}: {out[name]}")
    # paired difference best - EMA, resampling learners
    p_best, p_ema = np.asarray(p_best), np.asarray(p_ema)
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(300):
        s = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        diffs.append(roc_auc_score(y_te[s], p_best[s]) - roc_auc_score(y_te[s], p_ema[s]))
    out["auc_difference_best_minus_ema"] = {"mean": round(float(np.mean(diffs)), 4),
                                            "ci95": [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])]}
    log(f"    difference: {out['auc_difference_best_minus_ema']}")
    return out


def dl_baseline_experiment(log=print, model_name: str = "microsoft/trocr-base-handwritten") -> dict:
    """Classical pipeline vs a deep-learning OCR baseline (TrOCR) on the same real
    CROHME test expressions: accuracy, latency per expression and model size.
    TrOCR output is normalised leniently (only maths symbols kept, canonical
    spacing) so the comparison favours the baseline rather than penalising
    formatting differences."""
    import os

    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    from app.ml.real_symbols import TEST_FILES, harvest_crohme
    from app.ml.symbol_classifier import SymbolRecognizer, format_symbols
    from app.utils.text_metrics import character_error_rate

    _, exprs, _ = harvest_crohme(list(TEST_FILES.values()), group_offset=10_000_000, collect_expressions=True)
    vocab = set("0123456789+-=x()/")

    def normalise(text: str) -> str:
        t = text.replace("×", "x").replace("X", "x").replace("÷", "/").replace("—", "-").replace("–", "-")
        return format_symbols([c for c in t if c in vocab])

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()
    n_params = sum(p.numel() for p in model.parameters())

    def run_trocr(gray):
        rows = np.where((gray < 128).any(axis=1))[0]
        cols = np.where((gray < 128).any(axis=0))[0]
        if len(rows):
            pad = 20
            gray = gray[max(rows.min() - pad, 0): rows.max() + pad, max(cols.min() - pad, 0): cols.max() + pad]
        pixel_values = processor(images=Image.fromarray(gray).convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(pixel_values, max_new_tokens=40)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    bundle = load_bundle(str(SYMBOL_MODEL_PATH))
    rec = SymbolRecognizer(bundle)
    rows = {"stepwise_svm": {"cer": [], "exact": [], "ms": []}, "trocr": {"cer": [], "exact": [], "ms": []}}
    examples = []
    run_trocr(exprs[0].gray)  # warm-up
    for e in exprs:
        t = time.perf_counter(); ours, _, _ = rec.transcribe_gray(e.gray); ours = ours.replace("\n", " ")
        rows["stepwise_svm"]["ms"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter(); raw = run_trocr(e.gray); theirs = normalise(raw)
        rows["trocr"]["ms"].append((time.perf_counter() - t) * 1000)
        for name, hyp in (("stepwise_svm", ours), ("trocr", theirs)):
            rows[name]["cer"].append(character_error_rate(e.target, hyp))
            rows[name]["exact"].append(hyp == e.target)
        if len(examples) < 8:
            examples.append({"target": e.target, "stepwise": ours, "trocr_raw": raw, "trocr_normalised": theirs})
    svm_bytes = os.path.getsize(SYMBOL_MODEL_PATH)
    out = {"n_expressions": len(exprs), "trocr_model": model_name, "examples": examples}
    for name, r in rows.items():
        out[name] = {"exact_match": round(float(np.mean(r["exact"])), 4),
                     "exact_match_ci95": bootstrap_ci(np.ones(len(r["exact"])), np.array(r["exact"], float),
                                                     lambda a, b: float(np.mean(b))),
                     "mean_cer": round(float(np.mean(r["cer"])), 4),
                     "median_ms_per_expression": round(float(np.median(r["ms"])), 1)}
    out["stepwise_svm"]["model_mb"] = round(svm_bytes / 1e6, 1)
    out["trocr"]["parameters_millions"] = round(n_params / 1e6, 1)
    out["trocr"]["model_mb_fp32"] = round(n_params * 4 / 1e6, 1)
    out["mcnemar_exact_match"] = mcnemar(np.ones(len(exprs), bool), np.array(rows["stepwise_svm"]["exact"]), np.array(rows["trocr"]["exact"]))
    log(f"    {json.dumps({k: out[k] for k in ('stepwise_svm', 'trocr')})}")
    return out


def main() -> None:
    t0 = time.perf_counter()
    report = {"seed": SEED}
    print("[symbols]")
    report["symbols"] = symbol_experiments()
    print("[real mistakes]")
    report["real_mistakes"] = mistake_experiments()
    print("[knowledge tracing]")
    report["knowledge_tracing"] = tracing_experiments()
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("[dl baseline] skipped - pip install -r requirements-dl.txt to compare against TrOCR")
    else:
        print("[dl baseline] TrOCR vs StepWise on CROHME test expressions ...")
        report["dl_baseline"] = dl_baseline_experiment()
    report["seconds"] = round(time.perf_counter() - t0, 1)
    (REPORT_DIR / "paper_experiments.json").write_text(json.dumps(report, indent=2))
    print(f"done in {report['seconds']}s")


if __name__ == "__main__":
    main()
