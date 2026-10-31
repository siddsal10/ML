"""Train and test the whole-line CTC recognizer on real handwriting.

    cd backend && python -m app.ml.real_ctc --hours 3

Data: every single-line linear expression (any tokens, not only StepWise's 17
classes) of CROHME train, HME100K train photos and MathWriting train inks, with
the same 1-in-7 validation split as app/ml/real_cnn.py. Tests: CROHME 2014/16/19
(127 lines), HME100K test photos (590), MathWriting test (34) - the in-vocabulary
expressions, scored as exact 17-class transcriptions exactly like the symbol CNN.
"""
from __future__ import annotations

import itertools
import json
import pickle
import time

import numpy as np

from app.ml.artifacts import save_bundle
from app.ml.line_ctc import LineCTCRecognizer, line_image
from app.ml.model_cards import REPORT_DIR
from app.ml.real_cnn import (CACHE_DIR, CANDIDATE_DIR, HME_DIR, HME_TRAIN, VAL_EVERY, _cached, _iter_mathwriting,
                             _iter_parquet, _target, crohme_test_expressions, hme_available, hme_test_expressions,
                             mw_available, previous_recognizer, render_ink)
from app.ml.real_symbols import CROHME_DIR, TOKEN_TO_CLASS, TRAIN_FILES, _load_gray, in_vocab, is_linear
from app.ml.segmentation import binarize, binarize_page, segment_lines
from app.ml.symbol_classifier import GRAMMAR_START, format_symbols, grammar_final, grammar_step
from app.ml.symbol_features import FEATURE_VERSION


def page_lines(gray: np.ndarray, photo: bool = True) -> list[np.ndarray]:
    binary = binarize_page(gray) if photo else binarize(gray)
    out = []
    for y0, y1 in segment_lines(binary):
        img = line_image(binary[y0:y1])
        if img is not None:
            out.append(img)
    return out


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.round(img * 255).astype(np.uint8)


def harvest_lines(rows, decode, photo: bool, name: str, log) -> dict:
    data = {"train": ([], []), "val": ([], [])}
    for idx, (raw, label) in enumerate(rows):
        tokens = label.split()
        if not tokens or not is_linear(tokens):
            continue
        lines = page_lines(decode(raw), photo)
        if len(lines) != 1:
            continue
        split = "val" if idx % VAL_EVERY == 0 else "train"
        data[split][0].append(_to_u8(lines[0]))
        data[split][1].append(tokens)
    log(f"  {name}: {len(data['train'][0])} training lines, {len(data['val'][0])} validation lines")
    return data


def _sources():
    yield "CROHME", lambda: itertools.chain(*(_iter_parquet(CROHME_DIR / f"{n}.parquet") for n in TRAIN_FILES)), _load_gray, False
    if hme_available():
        yield "HME100K", lambda: itertools.chain(*(_iter_parquet(HME_DIR / f"{n}.parquet") for n in HME_TRAIN)), _load_gray, True
    if mw_available():
        yield "MathWriting", lambda: _iter_mathwriting(("train",)), render_ink, False


EXTRA_CAP = 60_000


def harvest_extra(log, cap: int = EXTRA_CAP, seed: int = 0) -> dict:
    """Training-only lines from MathWriting's 'valid' and 'synthetic' splits (human-written symbol
    inks; 'synthetic' places them by bounding boxes), drawn with a random pen width. Linear
    expressions with at least one StepWise symbol, up to `cap` lines."""
    rng = np.random.default_rng(seed)
    imgs, labs = [], []
    for strokes, label in _iter_mathwriting(("valid", "synthetic")):
        tokens = label.split()
        if not tokens or not is_linear(tokens) or not any(t in TOKEN_TO_CLASS for t in tokens):
            continue
        lines = page_lines(render_ink(strokes, ink_height=int(rng.integers(50, 100)), thickness=int(rng.integers(2, 6))), False)
        if len(lines) != 1:
            continue
        imgs.append(_to_u8(lines[0])); labs.append(tokens)
        if len(imgs) >= cap:
            break
    log(f"  MathWriting valid+synthetic: {len(imgs)} extra training lines")
    return {"train": (imgs, labs), "val": ([], [])}


def build_lines(log, extra: bool = False) -> dict:
    out = {"train": ([], []), "val": ([], [])}
    sources = list(_sources())
    if extra and mw_available():
        sources.append(("MathWriting_extra", None, None, None))
    for name, rows, decode, photo in sources:
        path = CACHE_DIR / f"lines_{name.lower()}.pkl"
        if path.exists():
            d = pickle.loads(path.read_bytes())
        else:
            d = harvest_extra(log) if rows is None else harvest_lines(rows(), decode, photo, name, log)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(d))
        for k in out:
            out[k][0].extend(d[k][0]); out[k][1].extend(d[k][1])
    return out


def transcribe(model: LineCTCRecognizer, gray: np.ndarray, photo: bool = True) -> str | None:
    """17-class linear transcription (grammar-constrained decoding), or None if no line is accepted."""
    lines = []
    for img in page_lines(gray, photo):
        read = model.decode_constrained(img, TOKEN_TO_CLASS, grammar_step, grammar_final, GRAMMAR_START)
        if read is None:
            return None
        lines.append(format_symbols(read[0]))
    return " ".join(lines) if lines else None


def _exact(model, exprs, photo=True) -> float:
    return round(float(np.mean([transcribe(model, g, photo) == t for g, t in exprs])), 4) if exprs else None


def run(seed: int = 0, epochs: int = 20, hours: float | None = None, log=print, extra: bool = False,
        init: LineCTCRecognizer | None = None, lr: float = 2e-3) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    log("building line images (CROHME, HME100K, MathWriting) ...")
    d = build_lines(log, extra=extra)
    tr_imgs, tr_labels = d["train"]
    va_imgs, va_labels = d["val"]
    log(f"train {len(tr_imgs)} lines, val {len(va_imgs)} lines, "
        f"vocab {len({t for l in tr_labels for t in l})} tokens")
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(va_imgs), size=min(400, len(va_imgs)), replace=False)
    val_probe = ([va_imgs[i].astype(np.float32) / 255 for i in probe], [va_labels[i] for i in probe])
    model = LineCTCRecognizer(epochs=epochs, seed=seed, lr=lr)
    model.fit(tr_imgs, tr_labels, val=val_probe, log=lambda m: log("  " + m), max_hours=hours, init=init)

    va_exact_tokens = float(np.mean([model.decode(im.astype(np.float32) / 255)[0] == lab for im, lab in zip(va_imgs, va_labels)]))
    log("testing ...")
    old = previous_recognizer()
    by_year = crohme_test_expressions()
    crohme = [e for yr in by_year for e in by_year[yr]]
    hme = hme_test_expressions() if hme_available() else []
    mw = _cached("mathwriting_test", lambda: ({}, []))[1] if mw_available() else []
    report = {
        "task": "line_recognition_ctc_real_handwriting",
        "protocol": "trained on CROHME/HME100K/MathWriting training lines (6 of 7); validation = the 7th; tests are "
                    "the official CROHME 2014/2016/2019 sets, HME100K test photos and MathWriting test inks "
                    "(17-class linear expressions, exact match).",
        "n_train_lines": len(tr_imgs),
        "val_exact_all_tokens": round(va_exact_tokens, 4),
        "crohme_test": {"n": len(crohme), "line_ctc": _exact(model, crohme, photo=False),
                        "previous_hog_svm": round(float(np.mean([old(g) == t for g, t in crohme])), 4) if old else None},
        "crohme_test_by_year": {yr: {"n": len(ex), "line_ctc": _exact(model, ex, photo=False)} for yr, ex in by_year.items()},
        "real_photos_hme100k_test": {"n": len(hme), "line_ctc": _exact(model, hme)},
        "diverse_writers_mathwriting_test": {"n": len(mw), "line_ctc": _exact(model, mw, photo=False)},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    log(json.dumps({k: report[k] for k in ("crohme_test", "real_photos_hme100k_test", "diverse_writers_mathwriting_test")}))
    bundle = {"model_name": "line_ctc", "line_model": model, "feature_version": FEATURE_VERSION, "metrics": report,
              "training_data": "CROHME train + HME100K train photos + MathWriting train inks (whole lines)"}
    return report, bundle


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hours", type=float, default=None, help="stop training after this many hours")
    ap.add_argument("--tag", default="ctc1")
    ap.add_argument("--extra", action="store_true", help="add MathWriting valid+synthetic lines to training")
    ap.add_argument("--init", default=None, help="line model bundle to continue training from")
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    from app.ml.artifacts import load_bundle
    init = load_bundle(args.init)["line_model"] if args.init else None
    report, bundle = run(epochs=args.epochs, hours=args.hours, log=lambda m: print(m, flush=True), extra=args.extra,
                         init=init, lr=args.lr)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, CANDIDATE_DIR / f"line_{args.tag}.joblib")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"line_recognizer_{args.tag}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
