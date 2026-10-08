"""Train and evaluate the symbol recognizer on REAL handwriting.

Data
- train: CROHME training expressions (count-aligned symbols) + HASYv2 symbols
         (+ the synthetic font glyphs, depending on the configuration)
- validation (used ONLY for choices): 15% of CROHME training expressions + 25% of synthetic fonts.
  The selection score is the mean of the two macro-F1s, because the app must read both real
  handwriting and printed/typed work (demo sheets, worksheets).
- test: official CROHME 2014 / 2016 / 2019 test sets (different writers), never used for fitting
        or selection. Reported per symbol and end to end (whole expression read correctly).

Configurations compared (same SVM, same features): synthetic-only, real-only, synthetic+real.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from app.ml.glyph_data import generate_dataset
from app.ml.real_symbols import TEST_FILES, TRAIN_FILES, harvest_crohme, load_hasy
from app.ml.symbol_classifier import SymbolRecognizer, candidate_models
from app.ml.symbol_features import FEATURE_VERSION
from app.utils.text_metrics import character_error_rate


def _arr(s):
    return np.vstack(s.X), np.array(s.y), np.array(s.groups)


def _symbol_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    return {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 4)}


def _expression_metrics(model, expressions) -> dict:
    rec = SymbolRecognizer({"model": model, "model_name": "eval", "feature_version": FEATURE_VERSION})
    cers, exact = [], []
    for e in expressions:
        hyp, _, _ = rec.transcribe_gray(e.gray)
        hyp = hyp.replace("\n", " ")
        cers.append(character_error_rate(e.target, hyp))
        exact.append(hyp == e.target)
    return {"n": len(expressions), "exact_match": round(float(np.mean(exact)), 4) if exact else None,
            "mean_cer": round(float(np.mean(cers)), 4) if cers else None}


def run(seed: int = 0, synthetic_per_font: int = 30, log=print) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    log("harvesting CROHME train ...")
    crohme_train, _, train_stats = harvest_crohme(TRAIN_FILES)
    log(f"  {train_stats}")
    tests = {}
    for year, f in TEST_FILES.items():
        sym, exprs, st = harvest_crohme([f], group_offset=10_000_000, collect_expressions=True)
        tests[year] = (sym, exprs, st)
        log(f"  test {year}: {st}, in-vocabulary expressions {len(exprs)}")
    log("loading HASYv2 ...")
    hasy, hasy_stats = load_hasy()
    log(f"  {hasy_stats['symbols']} symbols from {hasy_stats['writers']} writers")
    log("generating synthetic glyphs ...")
    synth = generate_dataset(samples_per_font=synthetic_per_font, seed=seed)

    Xc, yc, gc = _arr(crohme_train)
    Xh, yh, _ = _arr(hasy)
    # validation = held-out CROHME training expressions + held-out fonts (for choices only)
    tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(Xc, yc, gc))
    s_tr, s_va = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(synth.X, synth.y, synth.groups))

    def validate(model) -> dict:
        real = _symbol_metrics(model, Xc[va_idx], yc[va_idx])
        printed = _symbol_metrics(model, synth.X[s_va], synth.y[s_va])
        return {"real_handwriting": real, "printed_fonts": printed,
                "selection_score": round((real["macro_f1"] + printed["macro_f1"]) / 2, 4)}

    X_test = np.vstack([np.vstack(tests[y][0].X) for y in tests])
    y_test = np.concatenate([np.array(tests[y][0].y) for y in tests])
    test_exprs = [e for y in tests for e in tests[y][1]]

    configs = {
        "synthetic_only": (synth.X[s_tr], synth.y[s_tr]),
        "real_only": (np.vstack([Xc[tr_idx], Xh]), np.concatenate([yc[tr_idx], yh])),
        "synthetic_plus_real": (np.vstack([synth.X[s_tr], Xc[tr_idx], Xh]), np.concatenate([synth.y[s_tr], yc[tr_idx], yh])),
    }

    # 1) Which training data? (SVM, evaluated on the official test years)
    data_results = {}
    for name, (X, y) in configs.items():
        log(f"training SVM on {name} ({len(y)} symbols) ...")
        m = candidate_models(seed)["svm_rbf"].fit(X, y)
        data_results[name] = {
            "n_train": int(len(y)),
            "validation": validate(m),
            "test_symbols": _symbol_metrics(m, X_test, y_test),
            "test_symbols_by_year": {yr: _symbol_metrics(m, np.vstack(tests[yr][0].X), np.array(tests[yr][0].y)) for yr in tests},
            "test_expressions": _expression_metrics(m, test_exprs),
        }
        log(f"  test symbols {data_results[name]['test_symbols']}  expressions {data_results[name]['test_expressions']}")

    best_data = max(data_results, key=lambda n: data_results[n]["validation"]["selection_score"])

    # 2) Which algorithm? (chosen on validation, reported on test)
    X, y = configs[best_data]
    algo_results, fitted = {}, {}
    for name, model in candidate_models(seed).items():
        log(f"algorithm {name} on {best_data} ...")
        t = time.perf_counter()
        model.fit(X, y)
        fitted[name] = model
        algo_results[name] = {"fit_seconds": round(time.perf_counter() - t, 1),
                              "validation": validate(model),
                              "test_symbols": _symbol_metrics(model, X_test, y_test)}
    best_algo = max(algo_results, key=lambda n: algo_results[n]["validation"]["selection_score"])
    best_model = fitted[best_algo]
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, best_model.predict(X_test), labels=labels)
    algo_results[best_algo]["test_expressions"] = _expression_metrics(best_model, test_exprs)

    # 3) Deployment model: best data + algorithm, refit including the validation data it uses.
    parts_X, parts_y = [X], [y]
    if best_data != "synthetic_only":
        parts_X.append(Xc[va_idx]); parts_y.append(yc[va_idx])
    if best_data != "real_only":
        parts_X.append(synth.X[s_va]); parts_y.append(synth.y[s_va])
    X_final, y_final = np.vstack(parts_X), np.concatenate(parts_y)
    final = candidate_models(seed)[best_algo].fit(X_final, y_final)

    report = {
        "task": "symbol_recognition_real_handwriting",
        "datasets": {
            "crohme_train": train_stats,
            "crohme_test": {yr: {**tests[yr][2], "in_vocab_expressions": len(tests[yr][1])} for yr in tests},
            "hasy": {"symbols": hasy_stats["symbols"], "writers": hasy_stats["writers"]},
            "synthetic": {"symbols": int(len(synth.y)), "fonts": len(synth.group_names)},
        },
        "protocol": "algorithm and training-data choice on validation = 15% held-out CROHME-train expressions + 25% "
                    "held-out fonts (mean macro-F1); all reported test "
                    "numbers are on the official CROHME 2014/2016/2019 test sets (unseen writers). Symbols are "
                    "harvested by count alignment of segmentation with LaTeX tokens.",
        "training_data_comparison": data_results,
        "selected_training_data": best_data,
        "algorithm_comparison": algo_results,
        "selected_model": best_algo,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    bundle = {"model": final, "model_name": best_algo, "classes": list(final.classes_),
              "feature_version": FEATURE_VERSION, "metrics": report,
              "training_data": f"{best_data} (+ validation data)"}
    return report, bundle
