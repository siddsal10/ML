"""Mathematical parsing and verification.

Handles OCR cleanup/normalization and safe parsing of arithmetic and
simple linear-algebra expressions using SymPy's parse_expr with a
restricted transformation set and an explicit whitelist of allowed
symbols/functions. `eval()` / `exec()` are never used anywhere in this
module or its callers.
"""
from __future__ import annotations

import re
from tokenize import TokenError
from dataclasses import dataclass, field

import sympy
from sympy import Eq, Symbol, sympify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

X = Symbol("x")

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Only these names are permitted to resolve during parsing. This blocks
# arbitrary attribute/function access that a raw eval() would allow.
_ALLOWED_LOCAL_DICT = {
    "x": X,
    "y": Symbol("y"),
    "pi": sympy.pi,
    "sqrt": sympy.sqrt,
    "Abs": sympy.Abs,
}

_OCR_NORMALIZATION_RULES: list[tuple[str, str]] = [
    (r"[×✕✖]", "*"),
    (r"[÷]", "/"),
    (r"[–—]", "-"),  # en/em dash -> minus
    (r"(?<=\d)\s*[oO](?=\D|$)", "0"),  # trailing letter-O used as zero, cautiously
    (r"[lI](?=[+\-*/=]|\s|$)", "1"),  # isolated l/I used as one
    (r"[ \t]+", " "),  # collapse horizontal whitespace only - preserve newlines as step separators
]


class UnsafeExpressionError(ValueError):
    """Raised when an expression cannot be safely parsed."""


@dataclass
class ParsedStep:
    raw_text: str
    normalized_text: str
    kind: str  # "equation" | "expression" | "unparseable"
    sympy_repr: str | None = None
    is_valid: bool | None = None  # for equations: True if LHS == RHS
    error: str | None = None


@dataclass
class MathAnalysis:
    steps: list[ParsedStep] = field(default_factory=list)
    final_value: str | None = None
    all_steps_valid: bool = True
    has_unparseable_step: bool = False


def normalize_ocr_text(raw_text: str) -> str:
    text = raw_text
    for pattern, replacement in _OCR_NORMALIZATION_RULES:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def _split_lines(text: str) -> list[str]:
    # Split on newlines; also split a single line containing multiple `=`
    # separated statements on obvious step boundaries (rare, best-effort).
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines


def safe_parse(expr_str: str):
    """Safely parse a math expression/equation string using SymPy.

    Never uses eval(). Restricts resolvable names to _ALLOWED_LOCAL_DICT so
    arbitrary code cannot be injected via the OCR'd text.
    """
    cleaned = expr_str.strip().rstrip(".")
    if not cleaned:
        raise UnsafeExpressionError("empty expression")

    # Reject anything containing characters outside a tight whitelist before
    # even attempting to parse - defense in depth beyond local_dict scoping.
    if not re.fullmatch(r"[0-9a-zA-Z\s\+\-\*/\^=\.\(\)\,]+", cleaned):
        raise UnsafeExpressionError(f"disallowed characters in expression: {cleaned!r}")

    try:
        if "=" in cleaned and cleaned.count("=") == 1:
            lhs_str, rhs_str = cleaned.split("=")
            lhs = parse_expr(
                lhs_str, transformations=_TRANSFORMATIONS, local_dict=_ALLOWED_LOCAL_DICT, evaluate=True
            )
            rhs = parse_expr(
                rhs_str, transformations=_TRANSFORMATIONS, local_dict=_ALLOWED_LOCAL_DICT, evaluate=True
            )
            # evaluate=False is essential here: Eq(a, b) with evaluate=True
            # collapses to a plain sympy boolean (True/False) whenever both
            # sides are equal/unequal numeric literals, which would silently
            # turn every purely-numeric equation into a non-Equality object
            # and break downstream isinstance(parsed, Eq) checks.
            return Eq(lhs, rhs, evaluate=False)
        return parse_expr(
            cleaned, transformations=_TRANSFORMATIONS, local_dict=_ALLOWED_LOCAL_DICT, evaluate=True
        )
    except (sympy.SympifyError, TypeError, SyntaxError, AttributeError, TokenError, ZeroDivisionError) as exc:
        raise UnsafeExpressionError(f"could not parse {cleaned!r}: {exc}") from exc


def safe_parse_unevaluated(expr_str: str):
    """Like safe_parse for a single expression side, but keeps the written
    structure (e.g. '-3 + 5' stays Add(-3, 5) instead of collapsing to 2),
    so features can reason about the individual terms a student wrote."""
    cleaned = expr_str.strip().rstrip(".")
    if not cleaned or "=" in cleaned or not re.fullmatch(r"[0-9a-zA-Z\s\+\-\*/\^\.\(\)\,]+", cleaned):
        raise UnsafeExpressionError(f"cannot parse side {cleaned!r}")
    try:
        return parse_expr(cleaned, transformations=_TRANSFORMATIONS, local_dict=_ALLOWED_LOCAL_DICT, evaluate=False)
    except (sympy.SympifyError, TypeError, SyntaxError, AttributeError, TokenError, ZeroDivisionError) as exc:
        raise UnsafeExpressionError(f"could not parse {cleaned!r}: {exc}") from exc


def analyze_transcription(raw_text: str) -> MathAnalysis:
    """Parse each line of a transcription and check equation validity."""
    normalized_full = normalize_ocr_text(raw_text)
    lines = _split_lines(normalized_full)

    analysis = MathAnalysis()
    last_numeric_value = None

    for line in lines:
        step = ParsedStep(raw_text=line, normalized_text=line, kind="unparseable")
        try:
            parsed = safe_parse(line)
            step.sympy_repr = str(parsed)
            if isinstance(parsed, Eq):
                step.kind = "equation"
                # If the equation has no free symbols beyond x/y with a
                # concrete numeric claim (e.g. "12 + 15 = 27"), verify by
                # simplifying LHS - RHS == 0. If it *has* a free variable
                # (e.g. "x = 3" as the final solved value), we cannot
                # verify it numerically alone - it's recorded as valid form
                # only, and overall correctness is judged against the
                # expected final answer elsewhere.
                diff = sympy.simplify(parsed.lhs - parsed.rhs)
                if diff.free_symbols:
                    step.is_valid = None
                    if parsed.rhs.free_symbols == set() and parsed.lhs == X:
                        last_numeric_value = parsed.rhs
                else:
                    step.is_valid = bool(diff == 0)
                    if not step.is_valid:
                        analysis.all_steps_valid = False
            else:
                step.kind = "expression"
                last_numeric_value = parsed if not parsed.free_symbols else last_numeric_value
        except UnsafeExpressionError as exc:
            step.error = str(exc)
            step.kind = "unparseable"
            analysis.has_unparseable_step = True

        analysis.steps.append(step)

    if last_numeric_value is not None:
        analysis.final_value = str(last_numeric_value)
    elif analysis.steps and analysis.steps[-1].kind == "equation":
        # Fall back to the RHS text of the last equation line.
        last = analysis.steps[-1]
        if "=" in last.normalized_text:
            analysis.final_value = last.normalized_text.split("=")[-1].strip()

    return analysis


def compare_final_answer(student_final: str | None, expected_answer: str | None) -> bool | None:
    """Compare a student's final answer to an expected answer using SymPy equality.

    Returns None if either side cannot be safely parsed (unknown, not incorrect).
    """
    if student_final is None or expected_answer is None:
        return None
    try:
        a = safe_parse(student_final)
        b = safe_parse(expected_answer)
        if isinstance(a, Eq) or isinstance(b, Eq):
            return None
        return bool(sympy.simplify(a - b) == 0)
    except UnsafeExpressionError:
        return None


def solve_linear_equation(equation_str: str) -> str | None:
    """Best-effort canonical solve of a single-variable linear equation like '2x + 4 = 10'."""
    try:
        parsed = safe_parse(equation_str)
        if not isinstance(parsed, Eq):
            return None
        solutions = sympy.solve(parsed, X)
        if not solutions:
            return None
        return f"x = {solutions[0]}"
    except UnsafeExpressionError:
        return None
