"""Other writers, other capture: the recognizer on MNIST test digits.

MNIST was never used for training. Its digits were written with pen on paper
forms and scanned (not tablet ink like CROHME), by two populations. Per the
QMNIST reconstruction (Yadav & Bottou, NeurIPS 2019), the first 5,001 test
digits were written by US high-school students (NIST writers #2350-#2599) -
the closest public match to StepWise's users - and the last 4,999 by 48 US
Census Bureau employees. Each 28x28 digit is thresholded to a mask and
classified among all 17 classes, so reading a digit as an operator counts as
an error.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from app.ml.artifacts import BACKEND_ROOT, SYMBOL_MODEL_PATH, load_bundle
from app.ml.symbol_features import glyph_features

MNIST_PATH = BACKEND_ROOT / "data" / "external" / "mnist_784.arff"
POPULATIONS = {"high_school_students": slice(0, 5001), "census_employees": slice(5001, 10000)}


def load_mnist_test(path=MNIST_PATH) -> tuple[np.ndarray, np.ndarray]:
    """The last 10,000 rows of the OpenML ARFF are the MNIST test set, in order."""
    import pandas as pd

    with open(path) as f:
        n_header = next(i for i, line in enumerate(f) if line.strip().upper() == "@DATA") + 1
    df = pd.read_csv(path, skiprows=n_header, header=None, dtype=np.uint8)
    test = df.iloc[-10_000:].to_numpy()
    return test[:, :784].reshape(-1, 28, 28), test[:, 784].astype(str)


def run(log=print) -> dict:
    images, labels = load_mnist_test()
    model = load_bundle(str(SYMBOL_MODEL_PATH))["model"]
    feats = np.vstack([glyph_features(np.where(im > 127, 255, 0).astype(np.uint8)) for im in images])
    pred = model.predict(feats)
    out = {}
    for name, sl in POPULATIONS.items():
        y, p = labels[sl], pred[sl]
        digits = sorted(set(y))
        wrong = Counter((t, q) for t, q in zip(y, p) if t != q)
        out[name] = {
            "n": int(len(y)),
            "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1_digits": round(float(f1_score(y, p, labels=digits, average="macro", zero_division=0)), 4),
            "read_as_non_digit": round(float(np.mean([q not in digits for q in p])), 4),
            "per_digit_accuracy": {d: round(float(np.mean(p[y == d] == d)), 4) for d in digits},
            "top_confusions": [[t, q, c] for (t, q), c in wrong.most_common(6)],
        }
        log(f"  {name:26s} acc {out[name]['accuracy']:.4f}")
    return {
        "task": "symbol_recognition_unseen_writer_populations",
        "data": "MNIST test set (never used for training): scanned pen-on-paper digits; first 5,001 by ~250 "
                "high-school students, last 4,999 by 48 Census Bureau employees (QMNIST, Yadav & Bottou 2019)",
        "test_set_check": dict(sorted(Counter(labels).items())),
        "populations": out,
    }
