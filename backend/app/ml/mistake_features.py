"""Feature engineering for the mistake-diagnosis classifier.

SymPy does the verifiable part (is a step true? which x solves this
equation?); these features then describe *how* a wrong step is wrong, e.g.
"the written result equals what you'd get by flipping the sign of one
term", and the classifier learns which patterns map to which category.

Every feature is deterministic and named (FEATURE_NAMES), so a prediction
can be explained by listing the features that fired.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import sympy
from sympy import Eq

from app.services.math_parser import (
    X,
    MathAnalysis,
    UnsafeExpressionError,
    analyze_transcription,
    compare_final_answer,
    safe_parse,
    safe_parse_unevaluated,
)

FEATURE_VERSION = 1

FEATURE_NAMES = [
    # transcription quality
    "ocr_confidence",
    "frac_nonmath_chars",
    # structure
    "n_lines",
    "n_equations",
    "n_expressions",
    "frac_unparseable",
    "has_variable",
    # numeric (arithmetic) step checks
    "n_invalid_numeric",
    "first_invalid_pos",
    "has_negative_operand",
    "rel_error",
    "off_by_small",
    "off_by_place_value",
    "sign_flip_whole",
    "sign_flip_term",
    "op_swap_add_sub",
    "op_swap_cross",
    # algebra (linear in x) step checks
    "is_linear_problem",
    "n_algebra_steps",
    "nonequiv_found",
    "nonequiv_pos",
    "jumped_to_answer",
    "cand_transpose_no_sign_change",
    "cand_forgot_divide",
    "cand_forgot_subtract",
    "cand_divide_one_term",
    "cand_multiply_instead",
    "cand_negated_answer",
    "cand_off_small",
    # final answer
    "expected_known",
    "final_known",
    "final_matches_expected",
    "work_consistent",
]

# Human-readable phrasing for features used as evidence.
EVIDENCE_TEXT = {
    "sign_flip_whole": "The written result is the correct value with the opposite sign.",
    "sign_flip_term": "The written result matches flipping the sign of one term.",
    "op_swap_add_sub": "The written result matches swapping + and -.",
    "op_swap_cross": "The written result matches using a different operation (e.g. + instead of x).",
    "off_by_small": "The written result is off by only 1-2 from the correct value.",
    "off_by_place_value": "The written result is off by a power of ten (carry / place-value slip).",
    "cand_transpose_no_sign_change": "A term was moved across '=' without changing its sign.",
    "cand_forgot_divide": "The coefficient of x was never divided out.",
    "cand_forgot_subtract": "The constant term was never moved across '='.",
    "cand_divide_one_term": "Only one term was divided by the coefficient.",
    "cand_multiply_instead": "The coefficient was multiplied instead of divided.",
    "cand_negated_answer": "The answer has the opposite sign of the true solution.",
    "cand_off_small": "The solution is off by a small amount (arithmetic slip).",
    "jumped_to_answer": "No intermediate working was shown between problem and answer.",
}

_NONMATH = re.compile(r"[^0-9a-zA-Z\s\+\-\*/\^=\.\(\)\,]")
_NEG_OPERAND = re.compile(r"(^|[+\-*/(=])\s*-\s*\d")
_BINARY_OP = re.compile(r"(?<=[0-9x\)])\s*([+\-*/])")


@dataclass
class ExtractedFeatures:
    vector: np.ndarray
    values: dict[str, float]
    analysis: MathAnalysis
    suspected_step: str | None


def _value(expr) -> sympy.Expr | None:
    try:
        if expr.free_symbols:
            return None
        v = sympy.nsimplify(expr)
        return v if v.is_number and v.is_real and v.is_finite else None
    except Exception:  # noqa: BLE001
        return None


def _eq(a, b) -> bool:
    return a is not None and b is not None and bool(sympy.simplify(a - b) == 0)


def _numeric_step_features(line: str, f: dict) -> None:
    lhs_s, rhs_s = line.split("=")
    try:
        lhs_u = safe_parse_unevaluated(lhs_s)
        lhs = _value(safe_parse(lhs_s))
        rhs = _value(safe_parse(rhs_s))
    except UnsafeExpressionError:
        return
    if lhs is None or rhs is None:
        return

    diff = abs(float(rhs - lhs))
    f["has_negative_operand"] = float(bool(_NEG_OPERAND.search(lhs_s.strip())))
    f["rel_error"] = min(diff / max(1.0, abs(float(lhs))), 10.0)
    f["off_by_small"] = float(0 < diff <= 2)
    f["off_by_place_value"] = float(diff in (10.0, 100.0, 1000.0))
    f["sign_flip_whole"] = float(lhs != 0 and _eq(rhs, -lhs))

    if isinstance(lhs_u, sympy.Add):
        terms = [_value(t) for t in lhs_u.args]
        if all(t is not None for t in terms):
            total = sum(terms)
            f["sign_flip_term"] = float(any(_eq(rhs, total - 2 * t) for t in terms if t != 0))

    # Re-evaluate the left side with each binary operator replaced.
    lhs_clean = lhs_s.strip()
    for m in _BINARY_OP.finditer(lhs_clean):
        op = m.group(1)
        for alt in "+-*/":
            if alt == op:
                continue
            swapped = lhs_clean[: m.start(1)] + alt + lhs_clean[m.end(1) :]
            try:
                val = _value(safe_parse(swapped))
            except (UnsafeExpressionError, ZeroDivisionError):
                continue
            if _eq(rhs, val):
                key = "op_swap_add_sub" if {op, alt} == {"+", "-"} else "op_swap_cross"
                f[key] = 1.0


def _linear_parts(eq: Eq) -> tuple | None:
    """For an equation a*x + b = c (no x on the right), return (a, b, c)."""
    if X in eq.rhs.free_symbols:
        return None
    poly = sympy.Poly(sympy.expand(eq.lhs), X) if X in eq.lhs.free_symbols else None
    if poly is None or poly.degree() != 1:
        return None
    a, b = poly.coeff_monomial(X), poly.coeff_monomial(1)
    c = _value(eq.rhs)
    if c is None or a == 0:
        return None
    return a, b, c


def _solve(eq: Eq):
    try:
        sols = sympy.solve(eq, X)
    except Exception:  # noqa: BLE001
        return None
    return _value(sols[0]) if len(sols) == 1 else None


def _algebra_features(analysis: MathAnalysis, f: dict) -> int | None:
    """Returns the index (into analysis.steps) of the first non-equivalent step, if any."""
    eqs = []
    for i, step in enumerate(analysis.steps):
        if step.kind != "equation":
            continue
        try:
            parsed = safe_parse(step.normalized_text)
        except UnsafeExpressionError:
            continue
        if isinstance(parsed, Eq) and X in (parsed.lhs.free_symbols | parsed.rhs.free_symbols):
            eqs.append((i, parsed, _solve(parsed)))

    if not eqs or eqs[0][2] is None:
        return None
    f["is_linear_problem"] = 1.0
    f["n_algebra_steps"] = float(len(eqs) - 1)

    bad = None
    for k in range(1, len(eqs)):
        if not _eq(eqs[k][2], eqs[k - 1][2]):
            bad = k
            break
    f["work_consistent"] = float(bad is None)
    if bad is None:
        return None

    f["nonequiv_found"] = 1.0
    f["nonequiv_pos"] = bad / max(1, len(eqs) - 1)
    f["jumped_to_answer"] = float(bad == 1 and len(eqs) == 2)

    v = eqs[bad][2]
    s = eqs[bad - 1][2]
    if v is None:
        return eqs[bad][0]
    parts = _linear_parts(eqs[bad - 1][1])
    if parts is not None:
        a, b, c = parts
        f["cand_transpose_no_sign_change"] = float(b != 0 and _eq(v, (c + b) / a))
        f["cand_forgot_divide"] = float(a != 1 and _eq(v, c - b))
        f["cand_forgot_subtract"] = float(b != 0 and _eq(v, c / a))
        f["cand_divide_one_term"] = float(a != 1 and b != 0 and _eq(v, c / a - b))
        f["cand_multiply_instead"] = float(a not in (1, -1) and _eq(v, (c - b) * a))
    f["cand_negated_answer"] = float(s != 0 and _eq(v, -s))
    f["cand_off_small"] = float(0 < abs(float(v - s)) <= 2)
    return eqs[bad][0]


def extract_features(text: str, expected_answer: str | None, ocr_confidence: float,
                     analysis: MathAnalysis | None = None) -> ExtractedFeatures:
    analysis = analysis or analyze_transcription(text or "")
    f = {name: 0.0 for name in FEATURE_NAMES}
    f["first_invalid_pos"] = -1.0
    f["nonequiv_pos"] = -1.0
    f["final_matches_expected"] = -1.0

    raw = text or ""
    stripped = re.sub(r"\s", "", raw)
    f["ocr_confidence"] = float(ocr_confidence)
    f["frac_nonmath_chars"] = len(_NONMATH.findall(stripped)) / max(1, len(stripped))

    steps = analysis.steps
    n = len(steps)
    f["n_lines"] = float(n)
    f["n_equations"] = float(sum(s.kind == "equation" for s in steps))
    f["n_expressions"] = float(sum(s.kind == "expression" for s in steps))
    f["frac_unparseable"] = sum(s.kind == "unparseable" for s in steps) / max(1, n)
    f["has_variable"] = float(any("x" in s.normalized_text for s in steps if s.kind != "unparseable"))

    suspected = None
    invalid = [i for i, s in enumerate(steps) if s.kind == "equation" and s.is_valid is False]
    f["n_invalid_numeric"] = float(len(invalid))
    if invalid:
        f["first_invalid_pos"] = invalid[0] / max(1, n - 1)
        suspected = steps[invalid[0]].raw_text
        _numeric_step_features(steps[invalid[0]].normalized_text, f)
    elif all(s.kind != "unparseable" for s in steps):
        f["work_consistent"] = 1.0

    bad_idx = _algebra_features(analysis, f)
    if suspected is None and bad_idx is not None:
        suspected = steps[bad_idx].raw_text
    if f["n_invalid_numeric"] > 0:
        f["work_consistent"] = 0.0

    f["expected_known"] = float(expected_answer is not None)
    f["final_known"] = float(analysis.final_value is not None)
    match = compare_final_answer(analysis.final_value, expected_answer)
    if match is not None:
        f["final_matches_expected"] = float(match)

    vector = np.array([f[name] for name in FEATURE_NAMES], dtype=np.float64)
    return ExtractedFeatures(vector=vector, values=f, analysis=analysis, suspected_step=suspected)
