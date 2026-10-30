"""Evaluate StepWise on a hand-labelled set of real photographed student work.

The protocol for collecting the set is in docs/REAL_WORLD_VALIDATION.md; run
it with evaluation/real_world_eval.py. Manifest (JSONL, one page per line):

    sample_id, image_path, reference_text      required (reference_text = true transcription)
    expected_answer                             answer key, if the task has one
    mistake_label                               annotator A's category (diagnosis.CATEGORIES)
    mistake_label_b                             annotator B's category, for agreement
    writer_id, device, paper, lighting, grade   optional, for breakdowns

Reported: reading (exact match, CER) overall and per device / paper /
lighting / writer; diagnosis from the true transcription (the mistake
classifier alone); diagnosis end to end from the photo, without and with
confirm-before-diagnose; inter-annotator agreement (Cohen's kappa).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from app.ml.symbol_classifier import SymbolRecognizer, format_symbols, load_gray
from app.ml.validation.crohme_glyphs import text_to_symbols
from app.ml.validation.propagation import confirm
from app.services.diagnosis import CONFIRM_BELOW, diagnose, diagnose_rule_based
from app.services.math_parser import analyze_transcription
from app.utils.text_metrics import character_error_rate

BREAKDOWNS = ("device", "paper", "lighting", "writer_id", "grade")


def load_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if not Path(row["image_path"]).is_absolute():
                row["image_path"] = str((path.parent / row["image_path"]).resolve())
            rows.append(row)
    return rows


def _ci(values: list[bool], seed: int = 0, n: int = 2000) -> list[float] | None:
    if not values:
        return None
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _scores(gold: list[str], pred: list[str]) -> dict:
    hits = [g == p for g, p in zip(gold, pred)]
    return {"n": len(gold), "accuracy": round(float(np.mean(hits)), 4), "accuracy_95ci": _ci(hits),
            "macro_f1": round(float(f1_score(gold, pred, average="macro", zero_division=0)), 4)}


def _category(text: str, expected: str | None, conf: float) -> str:
    return diagnose(analyze_transcription(text), expected, conf, transcription=text).category


def evaluate_manifest(path: str | Path, tau: float = CONFIRM_BELOW) -> dict:
    rec = SymbolRecognizer.load()
    rows = []
    for s in load_manifest(path):
        gray = load_gray(s["image_path"])
        if gray is None:
            rows.append({**s, "missing": True})
            continue
        text, conf, regions = rec.transcribe_gray(gray)
        by_line = defaultdict(list)
        for r in regions:
            by_line[r["line"]].append((r["symbol"], r["confidence"]))
        ref_lines = [text_to_symbols(ln) for ln in s["reference_text"].splitlines() if ln.strip()]
        # compare in the recognizer's canonical spacing, whatever spacing the annotator typed
        reference = ("\n".join(format_symbols(ln) for ln in ref_lines) if all(ln is not None for ln in ref_lines)
                     else s["reference_text"].strip())
        row = {**s, "text": text, "exact": text == reference, "cer": character_error_rate(reference, text)}
        if s.get("mistake_label"):
            expected = s.get("expected_answer")
            row["from_true_text"] = _category(s["reference_text"], expected, 1.0)
            row["rule_based_true_text"] = diagnose_rule_based(analyze_transcription(s["reference_text"]), expected, 1.0).category
            row["end_to_end"] = _category(text, expected, conf)
            if all(ln is not None for ln in ref_lines):  # symbol-level confirmation needs a symbol reference
                confirmed, n_checked, _ = confirm({"read_lines": [by_line[i] for i in sorted(by_line)], "lines": ref_lines,
                                                   "text": text, "reference": reference}, tau)
                row["confirmed"] = _category(confirmed, expected, 1.0 if n_checked else conf)
                row["symbols_checked"] = n_checked
        rows.append(row)

    read = [r for r in rows if not r.get("missing")]
    report = {"manifest": str(path), "n_pages": len(rows), "n_missing_images": len(rows) - len(read), "tau": tau,
              "reading": {"exact_match": round(float(np.mean([r["exact"] for r in read])), 4) if read else None,
                          "exact_match_95ci": _ci([r["exact"] for r in read]),
                          "mean_cer": round(float(np.mean([r["cer"] for r in read])), 4) if read else None}}
    for key in BREAKDOWNS:
        groups = defaultdict(list)
        for r in read:
            if r.get(key) is not None:
                groups[str(r[key])].append(r)
        if groups:
            report["reading"][f"by_{key}"] = {g: {"n": len(v), "exact_match": round(float(np.mean([r["exact"] for r in v])), 4),
                                                  "mean_cer": round(float(np.mean([r["cer"] for r in v])), 4)}
                                              for g, v in sorted(groups.items())}
    labelled = [r for r in read if r.get("mistake_label")]
    if labelled:
        gold = [r["mistake_label"] for r in labelled]
        report["diagnosis"] = {
            "from_true_transcription": _scores(gold, [r["from_true_text"] for r in labelled]),
            "rule_based_from_true_transcription": _scores(gold, [r["rule_based_true_text"] for r in labelled]),
            "end_to_end_no_confirmation": _scores(gold, [r["end_to_end"] for r in labelled]),
        }
        conf_rows = [r for r in labelled if "confirmed" in r]
        if conf_rows:
            report["diagnosis"]["end_to_end_confirm_before_diagnose"] = {
                **_scores([r["mistake_label"] for r in conf_rows], [r["confirmed"] for r in conf_rows]),
                "symbols_checked_per_page": round(float(np.mean([r["symbols_checked"] for r in conf_rows])), 2),
                "note": "simulated: the student corrects exactly the highlighted symbols",
            }
    pairs = [(r["mistake_label"], r["mistake_label_b"]) for r in rows if r.get("mistake_label") and r.get("mistake_label_b")]
    if pairs:
        a, b = zip(*pairs)
        kappa = float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else float("nan")  # undefined for one label
        report["annotator_agreement"] = {"n": len(pairs), "raw": round(float(np.mean([x == y for x, y in pairs])), 4),
                                         "cohen_kappa": None if np.isnan(kappa) else round(kappa, 4)}
    report["pages"] = [{k: r.get(k) for k in ("sample_id", "reference_text", "text", "exact", "mistake_label",
                                              "from_true_text", "end_to_end", "confirmed")} for r in rows]
    return report
