"""Generates a canonical, step-by-step solution for supported question types.

Supported question types (school-level arithmetic + simple linear algebra):
  - "a OP b" arithmetic (+, -, *, /)
  - "a*x + b = c" linear equations

Anything outside these patterns returns an empty list rather than a
fabricated solution - the architecture is documented as extensible to more
STEM topics, not comprehensive today.
"""
from __future__ import annotations

import re

import sympy

from app.services.math_parser import X, safe_parse, UnsafeExpressionError

_ARITHMETIC_RE = re.compile(r"^\s*([\-]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([\-]?\d+(?:\.\d+)?)\s*(=.*)?$")
_LINEAR_RE = re.compile(
    r"^\s*([\-]?\d*)\s*x\s*([+\-]\s*\d+(?:\.\d+)?)?\s*=\s*([\-]?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)


def generate_canonical_solution(question_or_first_step: str, expected_answer: str | None) -> list[str]:
    text = question_or_first_step.strip()

    m = _ARITHMETIC_RE.match(text)
    if m:
        a, op, b, _ = m.groups()
        try:
            a_val, b_val = sympy.Rational(a), sympy.Rational(b)
        except (ValueError, TypeError):
            return []
        op_map = {"+": a_val + b_val, "-": a_val - b_val, "*": a_val * b_val, "/": a_val / b_val if b_val != 0 else None}
        result = op_map.get(op)
        if result is None:
            return ["Division by zero is undefined."]
        return [f"{a} {op} {b} = {result}"]

    m = _LINEAR_RE.match(text)
    if m:
        coeff_str, const_str, rhs_str = m.groups()
        coeff = coeff_str if coeff_str not in ("", "-", None) else ("1" if coeff_str != "-" else "-1")
        const_str = const_str.replace(" ", "") if const_str else "+0"
        try:
            equation_text = f"{coeff}*x{const_str}={rhs_str}"
            parsed = safe_parse(equation_text)
            solutions = sympy.solve(parsed, X)
        except UnsafeExpressionError:
            return []
        if not solutions:
            return []
        steps = [f"Start: {coeff}x{const_str} = {rhs_str}"]
        if const_str not in ("+0", "-0"):
            sign = const_str[0]
            magnitude = const_str[1:]
            inverse = "-" if sign == "+" else "+"
            steps.append(f"Subtract {magnitude} from both sides: {coeff}x = {rhs_str} {inverse} {magnitude}")
        steps.append(f"Divide both sides by {coeff}: x = {solutions[0]}")
        return steps

    return []
