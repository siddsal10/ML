"""CNN recognizer + recognition-driven segmentation, trained on real handwriting.

    cd backend && python -m app.ml.real_cnn            # ~1.5 h on a laptop CPU (8 GB RAM is enough)

Data (all real handwriting except the font glyphs)
- train: CROHME training expressions (symbols by count alignment, 6 of every 7
         expressions), HASYv2 symbols, MNIST *training* digits (the MNIST test set
         stays unseen for validation/writers.py), synthetic font glyphs, a 'junk'
         class made of two neighbouring symbols merged into one crop, and
         HME100K training PHOTOS of real students' paper work (symbols by count
         alignment on the photo-robust binarization), same 6-of-7 split.
- validation (choices only: decoder weights): every 7th CROHME training expression
         and every 7th HME100K training photo - symbols and, end to end, whole lines.
- test:  official CROHME 2014 / 2016 / 2019 test sets (unseen writers), the same
         127 in-vocabulary expressions as the HOG + SVM results; and, separately,
         the HME100K test photos written only with the 17 classes (real phone
         photographs, unseen writers); and the MathWriting test split (writers
         disjoint from its training split) as a third, diverse-handwriting test.

MathWriting (230k human-written expressions, online inks) is streamed from its
.tgz and rendered to images with OpenCV; same count alignment and 6:1 split.

Self-training (--self-train BASE.joblib): the base model forced-aligns the known
transcription of every in-vocabulary training expression whose symbols could not
be count-aligned (mostly touching symbols), keeps confident alignments, and the
model is fine-tuned from BASE on the enlarged set.

Symbols are stored as 32x32 uint8 images (1 KB each) and parquet files are
streamed, so harvesting fits in a few hundred MB.
"""
from __future__ import annotations

import itertools
import json
import pickle
import re
import time

import numpy as np

from app.ml import registry
from app.ml.artifacts import SYMBOL_MODEL_PATH, save_bundle
from app.ml.glyph_data import generate_dataset
from app.ml.model_cards import REPORT_DIR
from app.ml.real_symbols import (CROHME_DIR, EXTERNAL, TEST_FILES, TOKEN_TO_CLASS, TRAIN_FILES, _ink_band,
                                 _load_gray, in_vocab, is_linear, load_hasy)
from app.ml.segmentation import binarize, binarize_page, page_lattices, segment_symbols, union_mask
from app.ml.symbol_classifier import (DEFAULT_DECODER, JUNK, SymbolRecognizer, decode_lattice,
                                      decode_lattice_argmax, decode_line, format_symbols)
from app.ml.symbol_cnn import CNNSymbolClassifier, compact_images
from app.ml.symbol_features import CNN_SIZE, FEATURE_VERSION, N_HANDCRAFTED, glyph_features, normalize_glyph

CACHE_DIR = EXTERNAL / "cache"
HME_DIR = EXTERNAL / "hme100k"
HME_TRAIN = ["train-00000-of-00002", "train-00001-of-00002"]
HME_TEST = "test-00000-of-00001"
MNIST_PATH = EXTERNAL / "mnist_784.arff"
MW_PATH = EXTERNAL / "mathwriting-2024.tgz"
CANDIDATE_DIR = SYMBOL_MODEL_PATH.parent / "candidates"
FORCED_MIN_P, FORCED_MEAN_P = 0.2, 0.6  # acceptance of a forced alignment
VAL_EVERY = 7
MAX_VAL_EXPRS = 600
MNIST_PER_DIGIT = 2500
SAMPLES_PER_EPOCH = 60_000


def glyph_image(mask: np.ndarray) -> np.ndarray:
    """The CNN's input as uint8 (identical to the image block of glyph_features)."""
    return normalize_glyph(mask, CNN_SIZE, 26).ravel()


def _iter_parquet(path, batch_size: int = 32):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["image", "label"]):
        for row in batch.to_pylist():
            yield row["image"]["bytes"], row["label"]


_LATEX_TOKEN = re.compile(r"\\[A-Za-z]+|\S")


def _iter_mathwriting(splits: tuple[str, ...]):
    """(strokes, label) for the human-written inks of the given splits, streamed from the .tgz."""
    import tarfile
    import xml.etree.ElementTree as ET

    with tarfile.open(MW_PATH, "r|gz") as tar:
        for m in tar:
            parts = m.name.split("/")
            if not m.isfile() or not m.name.endswith(".inkml") or len(parts) < 2 or parts[-2] not in splits:
                continue
            root = ET.fromstring(tar.extractfile(m).read())
            ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
            ann = {a.get("type"): (a.text or "") for a in root.iter(f"{ns}annotation")}
            label = ann.get("normalizedLabel") or ann.get("label")
            if not label:
                continue
            strokes = []
            for t in root.iter(f"{ns}trace"):
                pts = [pt.split()[:2] for pt in (t.text or "").strip().split(",") if pt.strip()]
                if pts:
                    strokes.append(np.array(pts, dtype=np.float32))
            if strokes:
                yield strokes, " ".join(_LATEX_TOKEN.findall(label))


def render_ink(strokes: list[np.ndarray], ink_height: int = 80, max_width: int = 860, thickness: int = 3) -> np.ndarray:
    """Draw pen strokes as dark ink on white, scaled to a fixed ink height."""
    import cv2

    allp = np.vstack(strokes)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    scale = ink_height / max(hi[1] - lo[1], 1e-3)
    scale = min(scale, max_width / max(hi[0] - lo[0], 1e-3))
    size = ((hi - lo) * scale).astype(int) + 40
    canvas = np.full((size[1], size[0]), 255, np.uint8)
    for st in strokes:
        pts = ((st - lo) * scale + 20).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, 0, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


def mw_available() -> bool:
    return MW_PATH.exists()


def mw_test_expressions() -> list:
    return [(render_ink(st), _target(lab.split())) for st, lab in _iter_mathwriting(("test",))
            if len(lab.split()) >= 2 and is_linear(lab.split()) and in_vocab(lab.split())]


def hme_available() -> bool:
    return all((HME_DIR / f"{n}.parquet").exists() for n in HME_TRAIN + [HME_TEST])


def _target(tokens: list[str]) -> str:
    return format_symbols([TOKEN_TO_CLASS[t] for t in tokens])


def harvest(paths, photo: bool, log, name: str, rows=None, decode=None) -> tuple[dict, list]:
    """Aligned symbols (+ junk pairs) as uint8 images, split 6:1 by expression,
    and the in-vocabulary validation expressions as whole images. `rows` /
    `decode` replace the parquet reader (MathWriting yields strokes)."""
    split_imgs = {k: [] for k in ("train", "val")}
    split_y = {k: [] for k in ("train", "val")}
    val_exprs, n_aligned = [], 0
    rows = rows if rows is not None else itertools.chain(*(_iter_parquet(p) for p in paths))
    decode = decode or _load_gray
    for idx, (image_bytes, label) in enumerate(rows):
        tokens = label.split()
        if len(tokens) < 2 or not is_linear(tokens) or not any(t in TOKEN_TO_CLASS for t in tokens):
            continue
        split = "val" if idx % VAL_EVERY == 0 else "train"
        gray = decode(image_bytes)
        if split == "val" and in_vocab(tokens) and len(val_exprs) < MAX_VAL_EXPRS:
            val_exprs.append((gray, _target(tokens)))
        crops = segment_symbols(_ink_band(binarize_page(gray) if photo else binarize(gray)))
        if len(crops) != len(tokens):
            continue
        n_aligned += 1
        for crop, tok in zip(crops, tokens):
            if tok in TOKEN_TO_CLASS:
                split_imgs[split].append(glyph_image(crop.mask))
                split_y[split].append(TOKEN_TO_CLASS[tok])
        for a, b in list(zip(crops, crops[1:]))[:: 2 if photo else 1]:
            split_imgs[split].append(glyph_image(union_mask([(a.x, a.y, a.mask), (b.x, b.y, b.mask)])[2]))
            split_y[split].append(JUNK)
    out = {}
    for k in split_imgs:
        out[f"{k}_X"] = np.array(split_imgs[k], dtype=np.uint8).reshape(-1, CNN_SIZE * CNN_SIZE)
        out[f"{k}_y"] = np.array(split_y[k])
    log(f"  {name}: {n_aligned} aligned expressions, train {len(out['train_y'])} crops "
        f"({int((out['train_y'] == JUNK).sum())} junk), val {len(out['val_y'])}, val expressions {len(val_exprs)}")
    return out, val_exprs


def _cached(name: str, build):
    path = CACHE_DIR / f"{name}_img.npz"
    if path.exists():
        d = dict(np.load(path, allow_pickle=False))
        with open(path.with_suffix(".val.pkl"), "rb") as f:
            return d, pickle.load(f)
    d, val = build()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)
    with open(path.with_suffix(".val.pkl"), "wb") as f:
        pickle.dump(val, f)
    return d, val


def _mnist_train(rng) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    with open(MNIST_PATH) as f:
        n_header = next(i for i, line in enumerate(f) if line.strip().upper() == "@DATA") + 1
    arr = pd.read_csv(MNIST_PATH, skiprows=n_header, header=None, dtype=np.uint8, nrows=60_000).to_numpy()
    imgs, labels = arr[:, :784].reshape(-1, 28, 28), arr[:, 784].astype(str)
    keep = np.concatenate([rng.permutation(np.flatnonzero(labels == d))[:MNIST_PER_DIGIT] for d in "0123456789"])
    X = np.vstack([glyph_image(np.where(imgs[i] > 127, 255, 0).astype(np.uint8)) for i in keep])
    return X, labels[keep]


def _single_symbols(seed: int, log) -> tuple[np.ndarray, np.ndarray]:
    def build():
        rng = np.random.default_rng(seed)
        hasy, st = load_hasy()
        log(f"  HASYv2: {st['symbols']} symbols from {st['writers']} writers")
        synth = generate_dataset(samples_per_font=30, seed=seed)
        X = [compact_images(np.vstack(hasy.X)), compact_images(synth.X)]
        y = [np.array(hasy.y), synth.y]
        if MNIST_PATH.exists():
            Xm, ym = _mnist_train(rng)
            log(f"  MNIST train: {len(ym)} digits")
            X.append(Xm); y.append(ym)
        return {"train_X": np.vstack(X), "train_y": np.concatenate(y)}, []
    d, _ = _cached("single_symbols", build)
    return d["train_X"], d["train_y"]


def crohme_test_expressions() -> dict[str, list]:
    out = {}
    for year, f in TEST_FILES.items():
        out[year] = [(_load_gray(b), _target(lab.split())) for b, lab in _iter_parquet(CROHME_DIR / f"{f}.parquet")
                     if len(lab.split()) >= 2 and is_linear(lab.split()) and in_vocab(lab.split())]
    return out


def hme_test_expressions() -> list:
    return [(_load_gray(b), _target(lab.split())) for b, lab in _iter_parquet(HME_DIR / f"{HME_TEST}.parquet")
            if len(lab.split()) >= 2 and is_linear(lab.split()) and in_vocab(lab.split())]


def forced_align(lines: list, classes: list[str], target: list[str]) -> list | None:
    """Segment the expression so that it reads exactly `target` (class names), maximising
    sum log p over the line lattices in order. Returns [(crop, p), ...] or None."""
    tgt = [classes.index(t) for t in target]
    T = len(tgt)
    carry = {0: (0.0, [])}  # target position -> (score, [(crop, p)])
    for lat, probs in lines:
        best = [dict() for _ in range(lat.n_nodes)]
        best[0] = carry
        by_start: dict[int, list[int]] = {}
        for e, (i, _, _) in enumerate(lat.edges):
            by_start.setdefault(i, []).append(e)
        for i in range(lat.n_nodes - 1):
            for pos, (score, path) in best[i].items():
                if pos >= T:
                    continue
                for e in by_start.get(i, []):
                    j, crop = lat.edges[e][1], lat.edges[e][2]
                    pr = float(probs[e, tgt[pos]])
                    sc = score + np.log(max(pr, 1e-9))
                    if pos + 1 not in best[j] or sc > best[j][pos + 1][0]:
                        best[j][pos + 1] = (sc, path + [(crop, pr)])
        carry = best[-1]
        if not carry:
            return None
    if T not in carry:
        return None
    path = carry[T][1]
    ps = np.array([p for _, p in path])
    return path if ps.min() >= FORCED_MIN_P and ps.mean() >= FORCED_MEAN_P else None


def self_train_harvest(model, log) -> dict:
    """Forced-align the in-vocabulary TRAINING expressions (never the 1-in-7 validation
    ones) that count alignment could not use."""
    classes = list(model.classes_)

    def sources():
        yield "CROHME", False, itertools.chain(*(_iter_parquet(CROHME_DIR / f"{n}.parquet") for n in TRAIN_FILES)), _load_gray
        if hme_available():
            yield "HME100K", True, itertools.chain(*(_iter_parquet(HME_DIR / f"{n}.parquet") for n in HME_TRAIN)), _load_gray
        if mw_available():
            yield "MathWriting", False, _iter_mathwriting(("train",)), render_ink

    X, y = [], []
    for name, photo, rows, decode in sources():
        tried = accepted = 0
        for idx, (raw, label) in enumerate(rows):
            tokens = label.split()
            if idx % VAL_EVERY == 0 or len(tokens) < 2 or not is_linear(tokens) or not in_vocab(tokens):
                continue
            gray = decode(raw)
            if len(segment_symbols(_ink_band(binarize_page(gray) if photo else binarize(gray)))) == len(tokens):
                continue  # already used by count alignment
            tried += 1
            path = forced_align(_precompute(model, [(gray, "")])[0][0], classes, [TOKEN_TO_CLASS[t] for t in tokens])
            if path is None:
                continue
            accepted += 1
            for crop, _ in path:
                X.append(glyph_image(crop.mask))
            y += [TOKEN_TO_CLASS[t] for t in tokens]
            for (a, _), (b, _) in list(zip(path, path[1:]))[::2]:
                X.append(glyph_image(union_mask([(a.x, a.y, a.mask), (b.x, b.y, b.mask)])[2]))
                y.append(JUNK)
        log(f"  {name}: forced alignment accepted {accepted} of {tried} expressions")
    return {"train_X": np.array(X, dtype=np.uint8).reshape(-1, CNN_SIZE * CNN_SIZE), "train_y": np.array(y)}


def _symbol_acc(model, X, y) -> float:
    real = y != JUNK
    return round(float(np.mean(model.predict(X[real]) == y[real])), 4) if real.any() else None


def _precompute(model, exprs) -> list:
    """Lattices and class probabilities per expression (independent of decoder weights)."""
    out = []
    for gray, target in exprs:
        lines = []
        for _, lat in page_lattices(gray):
            probs = model.predict_proba(np.vstack([glyph_features(c.mask) for _, _, c in lat.edges]))
            lines.append((lat, probs))
        out.append((lines, target))
    return out


def _exact(pre, classes, params) -> float:
    ok = []
    for lines, target in pre:
        texts = []
        for lat, probs in lines:
            path = (decode_lattice(lat, probs, classes, params) if params is not None else None) \
                or decode_lattice_argmax(lat, probs, classes)
            texts.append(format_symbols([classes[c] for _, c in path]))
        ok.append(" ".join(texts) == target)
    return round(float(np.mean(ok)), 4) if ok else None


def tune_decoder(pre, classes) -> tuple[dict, list]:
    grid = []
    for bonus, overlap, cut in itertools.product([-1.0, 0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 4.0], [0.0, 0.5, 1.0, 2.0, 4.0]):
        params = {"symbol_bonus": bonus, "overlap_cost": overlap, "cut_cost": cut}
        grid.append({**params, "val_exact": _exact(pre, classes, params)})
    best = max(grid, key=lambda g: g["val_exact"])
    return {k: best[k] for k in DEFAULT_DECODER}, grid


def previous_recognizer():
    """The HOG + SVM recognizer of the last release, read from git history; None if unavailable."""
    import io
    import subprocess

    import joblib

    try:
        raw = subprocess.run(["git", "show", "HEAD:backend/ml_models/symbol_classifier.joblib"], capture_output=True,
                             check=True, cwd=str(EXTERNAL.parents[1])).stdout
        bundle = joblib.load(io.BytesIO(raw))
    except Exception:  # noqa: BLE001 - no git / no file: the comparison is optional
        return None
    from app.ml.segmentation import segment_page

    model, classes = bundle["model"], list(bundle["model"].classes_)

    def transcribe(gray: np.ndarray) -> str:
        lines = []
        for crops in segment_page(gray):
            probs = model.predict_proba(np.vstack([glyph_features(c.mask)[:N_HANDCRAFTED] for c in crops]))
            lines.append(format_symbols([classes[i] for i in decode_line(probs, classes)]))
        return " ".join(lines)

    return transcribe


def run(seed: int = 0, epochs: int = 20, use_hme: bool = True, use_mw: bool = True, base=None,
        log=print) -> tuple[dict, dict]:
    """base: a fitted CNNSymbolClassifier to self-train from (forced alignment + fine-tuning)."""
    t0 = time.perf_counter()
    use_hme = use_hme and hme_available()
    use_mw = use_mw and mw_available()
    log("harvesting CROHME training expressions ...")
    crohme, val_exprs = _cached("crohme_train", lambda: harvest([CROHME_DIR / f"{n}.parquet" for n in TRAIN_FILES],
                                                                 False, log, "CROHME"))
    parts = [crohme]
    if use_hme:
        log("harvesting HME100K training photos ...")
        hme, hme_val = _cached("hme_train", lambda: harvest([HME_DIR / f"{n}.parquet" for n in HME_TRAIN],
                                                             True, log, "HME100K"))
        parts.append(hme)
        val_exprs = val_exprs + hme_val
    if use_mw:
        log("harvesting MathWriting training inks ...")
        mw, mw_val = _cached("mathwriting_train", lambda: harvest(None, False, log, "MathWriting",
                                                                  rows=_iter_mathwriting(("train",)), decode=render_ink))
        parts.append(mw)
        val_exprs = val_exprs + mw_val
    if base is not None:
        log("self-training: forced alignment with the base model ...")
        forced, _ = _cached(f"forced_{getattr(base, 'tag_', 'base')}", lambda: (self_train_harvest(base, log), []))
        parts.append({**forced, "val_X": np.zeros((0, CNN_SIZE * CNN_SIZE), np.uint8), "val_y": np.array([], dtype=str)})
    log("loading single-symbol sets (HASYv2, MNIST train, fonts) ...")
    Xs, ys = _single_symbols(seed, log)
    X_tr = np.vstack([p["train_X"] for p in parts] + [Xs])
    y_tr = np.concatenate([p["train_y"] for p in parts] + [ys])
    X_va = np.vstack([p["val_X"] for p in parts])
    y_va = np.concatenate([p["val_y"] for p in parts])
    del parts, Xs, ys
    log(f"train {len(y_tr)} crops ({int((y_tr == JUNK).sum())} junk), val {len(y_va)}, val expressions {len(val_exprs)}")

    model = CNNSymbolClassifier(epochs=epochs, seed=seed, verbose=True, samples_per_epoch=SAMPLES_PER_EPOCH,
                                lr=1e-3 if base is not None else 3e-3)
    model.fit(X_tr, y_tr, X_va, y_va, log=lambda m: log("  " + m), init=base)
    classes = list(model.classes_)
    del X_tr, y_tr

    log("tuning the lattice decoder on validation expressions ...")
    decoder, grid = tune_decoder(_precompute(model, val_exprs), classes)
    log(f"  best {decoder}: val exact {max(g['val_exact'] for g in grid)}")
    bundle = {"model": model, "model_name": "cnn", "classes": classes, "feature_version": FEATURE_VERSION,
              "decoder": decoder}

    log("testing on CROHME 2014/2016/2019 ...")
    old = previous_recognizer()
    by_year = crohme_test_expressions()
    test_exprs = [e for yr in by_year for e in by_year[yr]]
    pre_test = _precompute(model, test_exprs)
    plain = SymbolRecognizer(bundle, constrained=True, lattice=False)
    crohme_test = {
        "n": len(test_exprs),
        "previous_hog_svm": round(float(np.mean([old(g) == t for g, t in test_exprs])), 4) if old else None,
        "cnn_plain_segmentation_syntax": round(float(np.mean(
            [plain.transcribe_gray(g)[0].replace("\n", " ") == t for g, t in test_exprs])), 4),
        "cnn_lattice_argmax_grouping": _exact(pre_test, classes, None),
        "cnn_lattice_decoder": _exact(pre_test, classes, decoder),
    }
    log(f"  {crohme_test}")
    report = {
        "task": "symbol_recognition_real_handwriting_cnn",
        "protocol": "CNN settings fixed in advance; decoder weights chosen on validation expressions "
                    f"(every {VAL_EVERY}th CROHME / HME100K training expression, never trained on); test = official "
                    "CROHME 2014/2016/2019 test sets (unseen writers, 17-class expressions) and HME100K test photos.",
        "symbols": {"val_accuracy": _symbol_acc(model, X_va, y_va)},
        "expressions_test": crohme_test,
        "expressions_test_by_year": {yr: {"n": len(ex), "exact": _exact(_precompute(model, ex), classes, decoder)}
                                     for yr, ex in by_year.items()},
        "expressions_val": max(g["val_exact"] for g in grid),
        "decoder": decoder,
        "decoder_grid": grid,
    }
    if use_hme:
        log("testing on HME100K test photos ...")
        hme_test = hme_test_expressions()
        report["real_photos_hme100k_test"] = {
            "data": "HME100K test photographs (real students' paper work, phone-app uploads, unseen writers) "
                    "written only with the 17 classes",
            "n": len(hme_test),
            "previous_hog_svm": round(float(np.mean([old(g) == t for g, t in hme_test])), 4) if old else None,
            "cnn_lattice_decoder": _exact(_precompute(model, hme_test), classes, decoder),
        }
        log(f"  {report['real_photos_hme100k_test']}")
    if use_mw:
        log("testing on the MathWriting test split ...")
        mw_test = _cached("mathwriting_test", lambda: ({}, mw_test_expressions()))[1]
        report["diverse_writers_mathwriting_test"] = {
            "data": "MathWriting test inks (writers disjoint from training), rendered, 17-class linear expressions",
            "n": len(mw_test),
            "previous_hog_svm": round(float(np.mean([old(g) == t for g, t in mw_test])), 4) if old else None,
            "cnn_lattice_decoder": _exact(_precompute(model, mw_test), classes, decoder),
        }
        log(f"  {report['diverse_writers_mathwriting_test']}")
    report["self_trained_from_base"] = base is not None
    report["seconds"] = round(time.perf_counter() - t0, 1)
    bundle["metrics"] = report
    bundle["training_data"] = "CROHME train + HASYv2 + MNIST train + fonts + junk pairs" + \
        (" + HME100K train photos" if use_hme else "") + (" + MathWriting train inks" if use_mw else "") + \
        (" + forced alignments (self-training)" if base is not None else "")
    return report, bundle


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--no-deploy", action="store_true", help="report only, keep the active model")
    ap.add_argument("--no-hme", action="store_true", help="leave out the HME100K photos")
    ap.add_argument("--no-mathwriting", action="store_true", help="leave out MathWriting")
    ap.add_argument("--self-train", metavar="BASE", default=None, help="candidate .joblib to self-train from")
    ap.add_argument("--tag", default="r1", help="name of the saved candidate bundle")
    args = ap.parse_args()
    base = None
    if args.self_train:
        import joblib

        base = joblib.load(args.self_train)["model"]
        base.tag_ = args.self_train.replace("\\", "/").split("/")[-1].removesuffix(".joblib")
    report, bundle = run(seed=args.seed, epochs=args.epochs, use_hme=not args.no_hme, use_mw=not args.no_mathwriting,
                         base=base, log=lambda m: print(m, flush=True))
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    save_bundle(bundle, CANDIDATE_DIR / f"symbol_cnn_{args.tag}.joblib")
    (REPORT_DIR / f"symbol_classifier_cnn_{args.tag}.json").write_text(json.dumps(report, indent=2))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "symbol_classifier_cnn.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report.get(k) for k in ("symbols", "expressions_test", "expressions_test_by_year",
                                                 "real_photos_hme100k_test", "diverse_writers_mathwriting_test",
                                                 "decoder")}, indent=2))
    if not args.no_deploy:
        save_bundle(bundle, SYMBOL_MODEL_PATH)
        version = registry.register("symbol_classifier", bundle, {
            "held_out_macro_f1": report["symbols"]["val_accuracy"], "benchmark": "CROHME + HME100K validation symbols",
            "algorithm": "cnn", "n_corrections": 0}, notes="CNN + lattice decoding, real handwriting", activate=True)
        print(f"deployed symbol_classifier {version}")


if __name__ == "__main__":
    main()
