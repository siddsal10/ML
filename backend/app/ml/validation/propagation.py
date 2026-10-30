"""From symbols to diagnosis: how much do misreads cost, and does confirmation help?

Fresh synthetic solutions (a seed the mistake classifier never trained on) are
written in real handwriting from the CROHME test writers, optionally put
through a simulated phone capture, read by the deployed recognizer and
diagnosed by the deployed mistake classifier. The same solution diagnosed
from its true transcription is the reference, so any difference is caused by
reading alone.

Confirm-before-diagnose: symbols read with confidence below tau are
highlighted and the diagnosis waits until the student has checked them (the
app's correction flow). Simulated student: corrects every highlighted symbol,
retypes a line whose symbol count is wrong if it has a highlight, and does
not notice unhighlighted misreads. tau is chosen on separate validation
pages, never on the pages reported.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

from app.ml.mistake_data import generate_samples
from app.ml.symbol_classifier import SymbolRecognizer, format_symbols
from app.ml.validation.crohme_glyphs import compose_page, text_to_symbols
from app.ml.validation.photo import CONDITIONS, capture
from app.services.diagnosis import diagnose
from app.services.math_parser import analyze_transcription

# Categories whose work can be written with the recognizer's 17 symbols
# (the other two are garbled or non-math text).
LABELS = ["correct_solution", "arithmetic_error", "sign_error", "incorrect_operation",
          "algebraic_manipulation_error", "skipped_step", "final_answer_only_error"]
ERROR_LABELS = set(LABELS) - {"correct_solution"}
TAU_GRID = [round(float(t), 2) for t in np.arange(0.30, 0.91, 0.05)] + [0.92, 0.94, 0.96, 0.98]
MAX_SILENT = 0.05  # tau target: <=5% of pages diagnosed differently from their true transcription
CURVE_CONDITIONS = ("clean", "phone_mild", "phone_moderate")


class _Diagnoser:
    """Cached diagnosis category of a transcription (texts repeat across tau values)."""

    def __init__(self):
        self.cache: dict = {}

    def __call__(self, text: str, expected: str | None, conf: float) -> str:
        key = (text, expected, round(conf, 3))
        if key not in self.cache:
            self.cache[key] = diagnose(analyze_transcription(text), expected, conf, transcription=text).category
        return self.cache[key]


def make_solutions(n_per_label: int, seed: int, diag: _Diagnoser) -> list[dict]:
    out, counts = [], {lab: 0 for lab in LABELS}
    for s in generate_samples(n_per_label=3 * n_per_label, seed=seed):
        if s.label not in counts or counts[s.label] >= n_per_label:
            continue
        lines = [text_to_symbols(ln) for ln in s.text.splitlines()]
        if not lines or any(ln is None for ln in lines):
            continue
        counts[s.label] += 1
        reference = "\n".join(format_symbols(ln) for ln in lines)
        out.append({"label": s.label, "expected": s.expected_answer, "lines": lines, "reference": reference,
                    "reference_category": diag(reference, s.expected_answer, 1.0)})
    return out


def read_pages(solutions: list[dict], pool, rec: SymbolRecognizer, condition: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for sol in solutions:
        page, baselines = compose_page(sol["lines"], pool, rng)
        img = capture(page, rng, baselines, **CONDITIONS[condition])
        text, conf, regions = rec.transcribe_gray(img)
        by_line = defaultdict(list)
        for r in regions:
            by_line[r["line"]].append((r["symbol"], r["confidence"]))
        rows.append({**sol, "text": text, "mean_conf": conf, "read_lines": [by_line[i] for i in sorted(by_line)]})
    return rows


def confirm(row: dict, tau: float | None) -> tuple[str, int, int]:
    """Transcription after the simulated student has checked the highlighted
    symbols, how many symbols were highlighted and how many lines retyped."""
    read, ref = row["read_lines"], row["lines"]
    flags = [[tau is not None and c < tau for _, c in ln] for ln in read]
    n_flagged = sum(map(sum, flags))
    if n_flagged == 0:
        return row["text"], 0, 0
    if len(read) != len(ref):  # lines split or merged: the student retypes the page
        return row["reference"], n_flagged, len(ref)
    lines, retyped = [], 0
    for rl, fl, true in zip(read, flags, ref):
        if len(rl) == len(true):
            lines.append([t if f else s for (s, _), f, t in zip(rl, fl, true)])
        elif any(fl):
            lines.append(true)
            retyped += 1
        else:
            lines.append([s for s, _ in rl])
    return "\n".join(format_symbols(ln) for ln in lines), n_flagged, retyped


def outcome(rows: list[dict], tau: float | None, diag: _Diagnoser) -> dict:
    final, flagged, retyped, silent = [], [], [], []
    for r in rows:
        text, n, n_retyped = confirm(r, tau)
        cat = diag(text, r["expected"], 1.0 if n else r["mean_conf"])
        final.append(cat)
        flagged.append(n)
        retyped.append(n_retyped)
        silent.append(cat != r["reference_category"])
    gold = [r["label"] for r in rows]
    correct_work = [i for i, g in enumerate(gold) if g == "correct_solution"]
    return {
        "accuracy": round(float(np.mean([f == g for f, g in zip(final, gold)])), 4),
        "macro_f1": round(float(f1_score(gold, final, labels=LABELS, average="macro", zero_division=0)), 4),
        "pages_diagnosed_differently_from_true_text": round(float(np.mean(silent)), 4),
        "correct_work_marked_wrong": round(float(np.mean([final[i] in ERROR_LABELS for i in correct_work])), 4),
        "symbols_to_check_per_page": round(float(np.mean(flagged)), 2),
        "pages_with_a_highlight": round(float(np.mean([n > 0 for n in flagged])), 4),
        "lines_to_retype_per_page": round(float(np.mean(retyped)), 2),
    }


def choose_tau(rows: list[dict], diag: _Diagnoser) -> tuple[float, bool]:
    """Smallest tau meeting the target on validation pages (and whether any did)."""
    for tau in TAU_GRID:
        if outcome(rows, tau, diag)["pages_diagnosed_differently_from_true_text"] <= MAX_SILENT:
            return tau, True
    return TAU_GRID[-1], False


def run(pool, seed: int = 0, n_per_label: int = 60, n_val_per_label: int = 30, log=print) -> dict:
    rec = SymbolRecognizer.load()
    diag = _Diagnoser()
    log("validation pages (choose tau) ...")
    val_rows = read_pages(make_solutions(n_val_per_label, seed + 201, diag), pool, rec, "clean", seed=seed + 202)
    tau, target_met = choose_tau(val_rows, diag)
    log(f"  tau = {tau} (target met on validation: {target_met})")

    test = make_solutions(n_per_label, seed + 101, diag)
    results, curves = {}, {}
    for condition in CONDITIONS:
        rows = read_pages(test, pool, rec, condition, seed=seed + 102)
        results[condition] = {
            "transcription_exact": round(float(np.mean([r["text"] == r["reference"] for r in rows])), 4),
            "no_confirmation": outcome(rows, None, diag),
            "confirm_below_tau": outcome(rows, tau, diag),
        }
        if condition in CURVE_CONDITIONS:
            curves[condition] = [{"tau": t, **outcome(rows, t, diag)} for t in TAU_GRID]
        r = results[condition]
        log(f"  {condition:16s} exact={r['transcription_exact']:.3f}  acc={r['no_confirmation']['accuracy']:.3f} -> "
            f"{r['confirm_below_tau']['accuracy']:.3f}  checks/page={r['confirm_below_tau']['symbols_to_check_per_page']}")

    return {
        "task": "misread_propagation_to_diagnosis",
        "setup": "synthetic solutions (seed unseen by the mistake classifier) written with real CROHME 2014/16/19 "
                 "test-writer glyphs, read by the deployed recognizer and diagnosed by the deployed mistake classifier",
        "n_test_pages": len(test),
        "n_validation_pages": len(val_rows),
        "mean_symbols_per_page": round(float(np.mean([sum(map(len, s["lines"])) for s in test])), 1),
        "per_label": {lab: sum(s["label"] == lab for s in test) for lab in LABELS},
        "reference_accuracy_true_transcription": round(float(np.mean([s["reference_category"] == s["label"] for s in test])), 4),
        "tau": tau,
        "tau_target_met_on_validation": target_met,
        "validation_at_tau": outcome(val_rows, tau, diag),
        "tau_rule": f"smallest tau on validation pages (clean) with <= {MAX_SILENT:.0%} of pages diagnosed differently "
                    "from their true transcription after confirmation",
        "conditions": results,
        "tau_curves": curves,
        "simulated_student": "corrects every highlighted symbol; retypes a line whose symbol count is wrong if it has a "
                             "highlight; does not notice unhighlighted misreads",
    }


def binarization_ablation(pool, seed: int = 0, n_per_label: int = 60, log=print) -> dict:
    """Reading the same test pages under each capture condition with the plain Otsu
    binarization the recognizer used before Review-3 vs the photo-robust page
    binarization (lighting flattening + ruled-line removal)."""
    from app.ml import segmentation
    from app.utils.text_metrics import character_error_rate

    rec = SymbolRecognizer.load()
    test = make_solutions(n_per_label, seed + 101, _Diagnoser())
    photo_robust = segmentation.binarize_page
    out: dict = {}
    try:
        for name, fn in (("otsu_only", segmentation.binarize), ("photo_robust", photo_robust)):
            segmentation.binarize_page = fn
            for condition in CONDITIONS:
                rows = read_pages(test, pool, rec, condition, seed=seed + 102)
                out.setdefault(condition, {})[name] = {
                    "page_exact": round(float(np.mean([r["text"] == r["reference"] for r in rows])), 4),
                    "mean_cer": round(float(np.mean([character_error_rate(r["reference"], r["text"]) for r in rows])), 4),
                }
            log(f"  {name}: " + ", ".join(f"{c} CER {v[name]['mean_cer']:.3f}" for c, v in out.items()))
    finally:
        segmentation.binarize_page = photo_robust
    return {"task": "capture_binarization_ablation", "n_pages": len(test),
            "pages": "the propagation study's test pages (same seeds)", "conditions": out}
