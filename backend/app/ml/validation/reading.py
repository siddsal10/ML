"""Why 92.6% of symbols but only ~57% of expressions? CROHME test expressions.

For the deployed recognizer on the 127 in-vocabulary CROHME 2014/16/19 test
expressions: where whole-line errors come from (symbol substitution vs
segmentation), how exact match falls with line length (and what independent
symbol errors would predict), what syntax-constrained decoding recovers, and
how well the minimum symbol confidence separates right from wrong reads.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from app.ml.artifacts import SYMBOL_MODEL_PATH, load_bundle
from app.ml.real_symbols import TEST_FILES, TOKEN_TO_CLASS, harvest_crohme
from app.ml.segmentation import segment_page
from app.ml.symbol_classifier import SymbolRecognizer, is_well_formed
from app.ml.validation.propagation import TAU_GRID

LENGTH_BUCKETS = [(1, 5), (6, 10), (11, 99)]


def _error_kind(rec_regions: list[dict], n_lines: int, n_tokens: int, exact: bool) -> str:
    if exact:
        return "exact"
    if n_lines != 1:
        return "line_split"
    if len(rec_regions) < n_tokens:
        return "under_segmentation"
    if len(rec_regions) > n_tokens:
        return "over_segmentation"
    return "symbol_substitution"


def run(tau: float | None = None, log=print) -> dict:
    bundle = load_bundle(str(SYMBOL_MODEL_PATH))
    test_syms, exprs, _ = harvest_crohme(list(TEST_FILES.values()), group_offset=10_000_000, collect_expressions=True)
    symbol_acc = float(np.mean(bundle["model"].predict(np.vstack(test_syms.X)) == np.array(test_syms.y)))
    log(f"  {len(exprs)} expressions, deployed symbol accuracy {symbol_acc:.4f}")

    decoders = {}
    for name, constrained in (("argmax", False), ("syntax_constrained", True)):
        rec = SymbolRecognizer(bundle, constrained=constrained)
        rows = []
        for e in exprs:
            text, _, regions = rec.transcribe_gray(e.gray)
            hyp = text.replace("\n", " ")
            exact = hyp == e.target
            symbols = [TOKEN_TO_CLASS[t] for t in e.tokens]
            rows.append({"exact": exact, "n": len(symbols), "well_formed": is_well_formed(symbols),
                         "min_conf": min((r["confidence"] for r in regions), default=0.0),
                         "kind": _error_kind(regions, len(segment_page(e.gray)), len(symbols), exact)})
        wf = [r for r in rows if r["well_formed"]]
        decoders[name] = {
            "exact_match": round(float(np.mean([r["exact"] for r in rows])), 4),
            "exact_match_well_formed_lines": round(float(np.mean([r["exact"] for r in wf])), 4),
            "n_well_formed": len(wf),
            "error_kinds": dict(Counter(r["kind"] for r in rows)),
            "by_length": {f"{a}-{b}" if b < 99 else f">{a - 1}": {
                "n": len(sub), "exact_match": round(float(np.mean([r["exact"] for r in sub])), 4),
                "independent_errors_predict": round(float(np.mean([symbol_acc ** r["n"] for r in sub])), 4)}
                for a, b in LENGTH_BUCKETS if (sub := [r for r in rows if a <= r["n"] <= b])},
            "_rows": rows,
        }
        log(f"  {name:20s} exact {decoders[name]['exact_match']:.4f}  well-formed {decoders[name]['exact_match_well_formed_lines']:.4f}")

    rows = decoders["syntax_constrained"].pop("_rows")
    decoders["argmax"].pop("_rows")
    ok = np.array([r["exact"] for r in rows])
    minc = np.array([r["min_conf"] for r in rows])
    curve = [{"tau": t, "coverage": round(float((minc >= t).mean()), 4),
              "exact_among_accepted": round(float(ok[minc >= t].mean()), 4) if (minc >= t).any() else None}
             for t in TAU_GRID]
    return {
        "task": "expression_reading_analysis",
        "data": "CROHME 2014/2016/2019 test expressions written only with the 17 recognizer classes",
        "n_expressions": len(exprs),
        "mean_symbols_per_expression": round(float(np.mean([r["n"] for r in rows])), 2),
        "deployed_symbol_accuracy": round(symbol_acc, 4),
        "independent_errors_predict_exact": round(float(np.mean([symbol_acc ** r["n"] for r in rows])), 4),
        "decoders": decoders,
        "confidence_curve_syntax_constrained": curve,
        "gate_tau": tau,
        "gate_at_tau": None if tau is None else next((c for c in curve if abs(c["tau"] - tau) < 1e-9), None),
        "note": "Well-formed = could be a line of student working; the others are CROHME fragments such as '+ 7 8'.",
    }
