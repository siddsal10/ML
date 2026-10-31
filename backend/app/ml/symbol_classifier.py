"""Symbol classifier: model comparison, selection, and page transcription."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from app.ml.artifacts import SYMBOL_MODEL_PATH, load_bundle, save_bundle
from app.ml.glyph_data import GlyphDataset
from app.ml.segmentation import Lattice, binarize_page, line_lattice, page_lattices, segment_lines, segment_page
from app.ml.symbol_features import FEATURE_VERSION, N_HANDCRAFTED, glyph_features


def handcrafted(X: np.ndarray) -> np.ndarray:
    """HOG + zoning + aspect columns of a feature vector (what the classical models read)."""
    return np.asarray(X)[:, :N_HANDCRAFTED]


def candidate_models(seed: int = 0, with_cnn: bool = False, cnn_epochs: int = 18) -> dict:
    hc = ("handcrafted", FunctionTransformer(handcrafted))
    models = {
        "svm_rbf": Pipeline([hc, ("scale", StandardScaler()), ("clf", SVC(C=10, gamma="scale", probability=True, random_state=seed))]),
        "random_forest": Pipeline([hc, ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed))]),
        "knn": Pipeline([hc, ("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))]),
        "logistic_regression": Pipeline([hc, ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=0.5))]),
    }
    if with_cnn:
        from app.ml.symbol_cnn import CNNSymbolClassifier

        models["cnn"] = CNNSymbolClassifier(epochs=cnn_epochs, seed=seed)
    return models


def train_symbol_classifier(data: GlyphDataset, seed: int = 0, out_path: Path = SYMBOL_MODEL_PATH,
                            always_train_groups: set[int] | None = None) -> tuple[dict, "SymbolRecognizer", np.ndarray]:
    """Compare candidate models on a font-disjoint split, refit the best on all data, save it.

    Groups in `always_train_groups` (e.g. student corrections) never enter the
    test split. Returns the report, a recognizer built from the train-split
    model (never saw the held-out fonts) for honest end-to-end evaluation,
    and the test indices."""
    always = always_train_groups or set()
    eligible = np.array([i for i, g in enumerate(data.groups) if g not in always])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(splitter.split(data.X[eligible], data.y[eligible], data.groups[eligible]))
    test_idx = eligible[te]
    train_idx = np.setdiff1d(np.arange(len(data.y)), test_idx)
    held_out_fonts = sorted({data.group_names[g] for g in data.groups[test_idx]})

    results = {}
    for name, model in candidate_models(seed).items():
        t0 = time.perf_counter()
        model.fit(data.X[train_idx], data.y[train_idx])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = model.predict(data.X[test_idx])
        pred_ms = (time.perf_counter() - t0) * 1000 / len(test_idx)
        results[name] = {
            "accuracy": round(float(accuracy_score(data.y[test_idx], pred)), 4),
            "macro_f1": round(float(f1_score(data.y[test_idx], pred, average="macro")), 4),
            "fit_seconds": round(fit_s, 2),
            "predict_ms_per_symbol": round(pred_ms, 4),
        }

    best = max(results, key=lambda n: results[n]["macro_f1"])
    best_model = candidate_models(seed)[best]
    best_model.fit(data.X[train_idx], data.y[train_idx])
    labels = sorted(set(data.y))
    cm = confusion_matrix(data.y[test_idx], best_model.predict(data.X[test_idx]), labels=labels)

    # Final model is refit on every font so deployment sees all styles.
    final = candidate_models(seed)[best]
    final.fit(data.X, data.y)
    report = {
        "task": "symbol_recognition",
        "split": "font-disjoint (GroupShuffleSplit, 25% of fonts held out)",
        "n_samples": int(len(data.y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_fonts": len(data.group_names),
        "held_out_fonts": held_out_fonts,
        "models": results,
        "selected_model": best,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }
    save_bundle(
        {"model": final, "model_name": best, "classes": list(final.classes_), "feature_version": FEATURE_VERSION, "metrics": report},
        out_path,
    )
    held_out_recognizer = SymbolRecognizer({"model": best_model, "model_name": best, "feature_version": FEATURE_VERSION})
    return report, held_out_recognizer, test_idx


def format_symbols(symbols: list[str]) -> str:
    """Turn a recognized symbol sequence into canonical linear notation.

    - 'x' written between two numbers is the multiplication sign ('3 x 4').
    - Binary operators and '=' get one space on each side; a '-' at the start,
      after another operator, or after '(' is unary and stays attached.
    Handwritten gaps are too irregular to be trusted as word spaces.
    """
    syms = list(symbols)
    for i in range(1, len(syms) - 1):
        if syms[i] == "x" and (syms[i - 1].isdigit() or syms[i - 1] == ")") and (syms[i + 1].isdigit() or syms[i + 1] == "("):
            syms[i] = "*"
    out: list[str] = []
    prev = None
    for ch in syms:
        if ch in "+*/=" or (ch == "-" and prev is not None and prev not in "+-*/=("):
            out.append(f" {ch} ")
        else:
            out.append(ch)
        prev = ch
    return "".join(out).strip()


# Line grammar for syntax-constrained decoding: which kind of symbol may follow
# which in one line of arithmetic / linear-equation working. '+' and '-' can be
# signs ('+5', '3 * -2') as well as operators. 'x' is both the variable and the
# handwritten multiplication sign, so it may sit between digits.
_KIND = {**{d: "digit" for d in "0123456789"}, "x": "x", "+": "sign", "-": "sign", "/": "op",
         "=": "eq", "(": "lp", ")": "rp"}
_FOLLOWS = {
    None: {"digit", "x", "sign", "lp"},
    "digit": {"digit", "x", "op", "sign", "eq", "lp", "rp"},
    "x": {"digit", "x", "op", "sign", "eq", "lp", "rp"},
    "op": {"digit", "x", "sign", "lp"},
    "sign": {"digit", "x", "lp"},
    "eq": {"digit", "x", "sign", "lp"},
    "lp": {"digit", "x", "sign", "lp"},
    "rp": {"x", "op", "sign", "eq", "lp", "rp"},
}
_CAN_END = {"digit", "x", "rp"}
_MAX_DEPTH = 4


def is_well_formed(symbols: list[str]) -> bool:
    kind, depth = None, 0
    for s in symbols:
        k = _KIND[s]
        depth += (k == "lp") - (k == "rp")
        if k not in _FOLLOWS[kind] or not 0 <= depth <= _MAX_DEPTH:
            return False
        kind = k
    return kind in _CAN_END and depth == 0


GRAMMAR_START = (None, 0)


def grammar_step(state: tuple, symbol: str) -> tuple | None:
    """Next (kind, bracket depth) after `symbol`, or None if the line cannot continue with it."""
    kind, depth = state
    k = _KIND[symbol]
    d = depth + (k == "lp") - (k == "rp")
    return (k, d) if k in _FOLLOWS[kind] and 0 <= d <= _MAX_DEPTH else None


def grammar_final(state: tuple) -> bool:
    return state[0] in _CAN_END and state[1] == 0


def decode_line(probs: np.ndarray, classes: list[str], top_k: int = 3) -> list[int]:
    """Most probable class sequence (sum of log-probabilities over each symbol's
    top-k classes) that is a well-formed line, by Viterbi over (kind, bracket
    depth) states. Only syntax is used, never arithmetic truth, so a student's
    wrong answer is read as written. Falls back to per-symbol argmax when no
    well-formed reading exists."""
    argmax = [int(i) for i in np.argmax(probs, axis=1)]
    beams: dict[tuple, tuple[float, list[int]]] = {(None, 0): (0.0, [])}
    for p in probs:
        nxt: dict[tuple, tuple[float, list[int]]] = {}
        for (kind, depth), (score, path) in beams.items():
            for c in np.argsort(p)[::-1][:top_k]:
                k = _KIND[classes[c]]
                d = depth + (k == "lp") - (k == "rp")
                if k not in _FOLLOWS[kind] or not 0 <= d <= _MAX_DEPTH:
                    continue
                s = score + float(np.log(max(p[c], 1e-12)))
                if (k, d) not in nxt or s > nxt[(k, d)][0]:
                    nxt[(k, d)] = (s, path + [int(c)])
        if not nxt:
            return argmax
        beams = nxt
    ends = [v for (k, d), v in beams.items() if k in _CAN_END and d == 0]
    return max(ends, key=lambda v: v[0])[1] if ends else argmax


JUNK = "~"  # training-only class: two neighbouring symbols merged into one crop

# Lattice decoding weights, overridden by bundle["decoder"] (tuned on validation expressions).
DEFAULT_DECODER = {"symbol_bonus": 0.0, "overlap_cost": 2.0, "cut_cost": 1.0}


def decode_lattice(lat: Lattice, probs: np.ndarray, classes: list[str], params: dict,
                   top_k: int = 3) -> list[tuple[int, int]] | None:
    """Jointly choose the segmentation and the symbols of one line: the path of
    lattice edges and classes maximising sum(log p(class) + symbol_bonus) minus
    the cost of separating pieces that overlap (overlap_cost) or were one
    component (cut_cost), restricted to well-formed lines. Returns
    [(edge index, class index), ...] or None if no well-formed path exists."""
    sym = [c for c in range(len(classes)) if classes[c] in _KIND]
    sep = np.where(lat.boundary_cost == 1, params["overlap_cost"],
                   np.where(lat.boundary_cost == 2, params["cut_cost"], 0.0))
    out_edges: dict[int, list[int]] = {}
    for e, (i, _, _) in enumerate(lat.edges):
        out_edges.setdefault(i, []).append(e)
    cand = {e: sorted(sym, key=lambda c: -probs[e, c])[:top_k] for e in range(len(lat.edges))}
    best: list[dict] = [dict() for _ in range(lat.n_nodes)]
    best[0][(None, 0)] = (0.0, None)  # state -> (score, backpointer)
    for i in range(lat.n_nodes - 1):
        for state, (score, _) in best[i].items():
            kind, depth = state
            for e in out_edges.get(i, []):
                j = lat.edges[e][1]
                cost = sep[j] if j < lat.n_nodes - 1 else 0.0
                for c in cand[e]:
                    k = _KIND[classes[c]]
                    d = depth + (k == "lp") - (k == "rp")
                    if k not in _FOLLOWS[kind] or not 0 <= d <= _MAX_DEPTH:
                        continue
                    s2 = score + float(np.log(max(probs[e, c], 1e-12))) + params["symbol_bonus"] - cost
                    if (k, d) not in best[j] or s2 > best[j][(k, d)][0]:
                        best[j][(k, d)] = (s2, (i, state, e, c))
    ends = [(v[0], st) for st, v in best[-1].items() if st[0] in _CAN_END and st[1] == 0]
    if not ends:
        return None
    node, state = lat.n_nodes - 1, max(ends, key=lambda t: t[0])[1]
    path = []
    while node != 0:
        _, (i, prev, e, c) = best[node][state]
        path.append((e, c))
        node, state = i, prev
    return path[::-1]


def decode_lattice_argmax(lat: Lattice, probs: np.ndarray, classes: list[str]) -> list[tuple[int, int]]:
    """Fallback: the plain grouping, each symbol its most probable class."""
    sym = [c for c in range(len(classes)) if classes[c] in _KIND]
    index = {(i, j): e for e, (i, j, _) in enumerate(lat.edges)}
    p = lat.default_path
    out = []
    for a, b in zip(p, p[1:]):
        es = [index[(a, b)]] if (a, b) in index else [index[(k, k + 1)] for k in range(a, b)]
        out += [(e, max(sym, key=lambda c: probs[e, c])) for e in es]
    return out


class SymbolRecognizer:
    """Transcribes a page image using the trained symbol classifier."""

    def __init__(self, bundle: dict, constrained: bool = True, lattice: bool | None = None):
        if bundle.get("feature_version") != FEATURE_VERSION:
            raise ValueError("symbol model was trained with a different feature version; retrain it")
        self.model = bundle["model"]
        self.model_name = bundle.get("model_name", "unknown")
        self.classes = list(self.model.classes_)
        self.constrained = constrained
        self.decoder = {**DEFAULT_DECODER, **(bundle.get("decoder") or {})}
        # recognition-driven segmentation is on for bundles whose decoder was tuned for it
        self.lattice = bool(bundle.get("decoder")) if lattice is None else lattice
        # whole-line CTC reader, used per line when it beats the symbol reading by line_margin;
        # the symbol model also reads the lines it cannot (out-of-vocabulary or ill-formed)
        self.line_model = bundle.get("line_model")
        self.line_margin = float(bundle.get("line_margin", 0.06))

    @classmethod
    def load(cls, path: Path = SYMBOL_MODEL_PATH) -> "SymbolRecognizer | None":
        bundle = load_bundle(str(path))
        return cls(bundle) if bundle is not None else None

    def _region(self, li: int, crop, p: np.ndarray, c: int, with_masks: bool) -> dict:
        others = [i for i in np.argsort(p)[::-1] if i != c and self.classes[i] != JUNK][:2]
        return {
            "line": li,
            "bbox": [crop.x, crop.y, crop.w, crop.h],
            "symbol": self.classes[c],
            "confidence": round(float(p[c]), 3),
            "alternatives": [[self.classes[i], round(float(p[i]), 3)] for i in others],
            **({"mask": crop.mask} if with_masks else {}),
        }

    def _read_lattice(self, li: int, lat: Lattice, with_masks: bool) -> tuple[list[str], list[float], list[dict]]:
        probs = self.model.predict_proba(np.vstack([glyph_features(c.mask) for _, _, c in lat.edges]))
        path = (decode_lattice(lat, probs, self.classes, self.decoder) if self.constrained else None) \
            or decode_lattice_argmax(lat, probs, self.classes)
        symbols, confs, regions = [], [], []
        for e, c in path:
            symbols.append(self.classes[c])
            confs.append(float(probs[e, c]))
            regions.append(self._region(li, lat.edges[e][2], probs[e], c, with_masks))
        return symbols, confs, regions

    def _transcribe_lattice(self, gray: np.ndarray, with_masks: bool, normalize: bool) -> tuple[str, float, list[dict]]:
        lines_out, confidences, regions = [], [], []
        for li, (_, lat) in enumerate(page_lattices(gray, normalize=normalize)):
            symbols, confs, regs = self._read_lattice(li, lat, with_masks)
            confidences += confs
            regions += regs
            lines_out.append(format_symbols(symbols))
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return "\n".join(lines_out), confidence, regions

    def _read_line_ctc(self, li: int, band: np.ndarray, y0: int, with_masks: bool):
        """Read one binary line with the CTC model; None if it read something StepWise cannot use."""
        from app.ml.line_ctc import frame_columns, line_image_with_box
        from app.ml.real_symbols import TOKEN_TO_CLASS

        r = line_image_with_box(band)
        if r is None:
            return None
        img, col0, scale = r
        if self.constrained:  # only StepWise's symbols, only well-formed lines (syntax, never arithmetic)
            read = self.line_model.decode_constrained(img, TOKEN_TO_CLASS, grammar_step, grammar_final, GRAMMAR_START)
            if read is None:
                return None
            symbols, spans = read
        else:
            tokens, spans = self.line_model.decode(img)
            symbols = [TOKEN_TO_CLASS.get(t) for t in tokens]
            if not symbols or None in symbols:
                return None
        confs, regions = [], []
        for sym, sp in zip(symbols, spans):
            a, b = frame_columns(sp["f0"], sp["f1"], col0, scale)
            sub = band[:, max(0, a):max(a + 1, b)]
            rows, cols = np.flatnonzero(sub.any(axis=1)), np.flatnonzero(sub.any(axis=0))
            if len(rows) == 0:
                rows, cols = np.array([0, band.shape[0] - 1]), np.array([0, max(0, sub.shape[1] - 1)])
            mask = sub[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
            x, y = max(0, a) + int(cols[0]), y0 + int(rows[0])
            p = self.model.predict_proba(glyph_features(mask).reshape(1, -1))[0] if mask.any() \
                else np.zeros(len(self.classes))
            others = [i for i in np.argsort(p)[::-1] if self.classes[i] not in (sym, JUNK)][:2]
            confs.append(sp["conf"])
            # the line model is overconfident off its training distribution, so a symbol is only as
            # sure as the symbol model agrees with it; this keeps confirm-before-diagnose highlighting it
            agree = float(p[self.classes.index(sym)]) if sym in self.classes else 0.0
            regions.append({
                "line": li, "bbox": [x, y, int(mask.shape[1]), int(mask.shape[0])], "symbol": sym,
                "confidence": round(min(float(sp["conf"]), agree), 3),
                "alternatives": [[self.classes[i], round(float(p[i]), 3)] for i in others],
                **({"mask": mask} if with_masks else {}),
            })
        return symbols, confs, regions

    def _transcribe_ctc(self, gray: np.ndarray, with_masks: bool, normalize: bool) -> tuple[str, float, list[dict]]:
        binary = binarize_page(gray, normalize=normalize)
        lines_out, confidences, regions = [], [], []
        for li, (y0, y1) in enumerate(segment_lines(binary)):
            band = binary[y0:y1]
            ctc = self._read_line_ctc(li, band, y0, with_masks)
            lat = line_lattice(band)
            for _, _, c in lat.edges:
                c.y += y0
            read = self._read_lattice(li, lat, with_masks) if lat.edges else None
            # the line reading replaces the symbol reading only when it is clearly more confident
            # (margin tuned on validation expressions in app/ml/real_ctc.py)
            if ctc is not None and (read is None or ctc[0] != read[0] and
                                    np.mean(ctc[1]) - np.mean(read[1]) > self.line_margin):
                read = ctc
            if read is None:
                continue
            symbols, confs, regs = read
            confidences += confs
            regions += regs
            lines_out.append(format_symbols(symbols))
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return "\n".join(lines_out), confidence, regions

    def transcribe_gray(self, gray: np.ndarray, with_masks: bool = False, normalize: bool = True) -> tuple[str, float, list[dict]]:
        if self.line_model is not None:
            return self._transcribe_ctc(gray, with_masks, normalize)
        if self.lattice:
            return self._transcribe_lattice(gray, with_masks, normalize)
        lines_out, confidences, regions = [], [], []
        for li, crops in enumerate(segment_page(gray, normalize=normalize)):
            feats = np.vstack([glyph_features(c.mask) for c in crops])
            probs = self.model.predict_proba(feats)
            probs = probs * np.array([c != JUNK for c in self.classes])
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            chosen = decode_line(probs, self.classes) if self.constrained else np.argmax(probs, axis=1)
            symbols = []
            for crop, p, c in zip(crops, probs, chosen):
                symbols.append(self.classes[c])
                confidences.append(float(p[c]))
                regions.append(self._region(li, crop, p, c, with_masks))
            lines_out.append(format_symbols(symbols))
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return "\n".join(lines_out), confidence, regions

    def transcribe_file(self, image_path: str | Path, with_masks: bool = False) -> tuple[str, float, list[dict]]:
        gray = load_gray(image_path)
        if gray is None:
            raise ValueError(f"could not read image: {image_path}")
        return self.transcribe_gray(gray, with_masks=with_masks)


def load_gray(image_path: str | Path) -> np.ndarray | None:
    """Read an image as grayscale, honouring EXIF orientation and flattening
    transparency onto white (canvas drawings are transparent PNGs)."""
    from PIL import Image, ImageOps

    try:
        img = ImageOps.exif_transpose(Image.open(image_path))
    except (OSError, ValueError):
        return None
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return np.array(img.convert("L"))
