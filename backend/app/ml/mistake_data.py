"""Synthetic labelled student work for training the mistake classifier.

Each generator produces a transcription in the same linear notation the OCR
emits, an expected answer (or None when the answer key is unknown), an OCR
confidence, and a gold mistake category. Error types follow common
misconception taxonomies for early algebra: sign handling with negative
numbers, wrong operation, moving a term across '=' without changing its
sign, dividing only one term, and omitting a step.

`formula_selection_error` is not generated - it needs word problems / named
formulas, which are outside this build's arithmetic + linear-equation scope.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np
from sympy import Rational

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}

LABELS = [
    "correct_solution",
    "arithmetic_error",
    "sign_error",
    "incorrect_operation",
    "algebraic_manipulation_error",
    "skipped_step",
    "final_answer_only_error",
    "transcription_uncertainty",
    "insufficient_information",
]


@dataclass
class Sample:
    text: str
    expected_answer: str | None
    ocr_confidence: float
    label: str


def _fmt_num(v) -> str:
    v = Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def _fmt_operand(v: int, first: bool, rng) -> str:
    if v < 0 and not first:
        return f"({v})" if rng.random() < 0.5 else str(v)
    return str(v)


def _arith_line(a: int, op: str, b: int, result, rng) -> str:
    sp = " " if rng.random() < 0.7 else ""
    return f"{_fmt_operand(a, True, rng)}{sp}{op}{sp}{_fmt_operand(b, False, rng)}{sp}={sp}{_fmt_num(result)}"


def _linear_lhs(a: int, b: int, rng) -> str:
    sp = " " if rng.random() < 0.7 else ""
    coef = "" if a == 1 else ("-" if a == -1 else str(a))
    if rng.random() < 0.15 and a not in (1, -1):
        coef += "*"
    sign = "+" if b > 0 else "-"
    return f"{coef}x{sp}{sign}{sp}{abs(b)}"


def _conf(rng) -> float:
    return 1.0 if rng.random() < 0.3 else float(rng.uniform(0.5, 1.0))


def _maybe_expected(value, rng, p_unknown: float = 0.2) -> str | None:
    return None if rng.random() < p_unknown else _fmt_num(value)


# --- arithmetic ---------------------------------------------------------------

def _alternatives(a: int, op: str, b: int) -> set:
    alts = {f(a, b) for f in OPS.values()}
    true = OPS[op](a, b)
    alts |= {-true, OPS[op](-a, b), OPS[op](a, -b)}
    return alts


def gen_arith_correct(rng) -> Sample:
    op = str(rng.choice(list(OPS)))
    a, b = int(rng.integers(-20, 60)), int(rng.integers(-20, 60))
    r = OPS[op](a, b)
    return Sample(_arith_line(a, op, b, r, rng), _maybe_expected(r, rng), _conf(rng), "correct_solution")


def gen_arith_error(rng) -> Sample:
    while True:
        op = str(rng.choice(list(OPS)))
        a, b = int(rng.integers(2, 99)), int(rng.integers(2, 99))
        true = OPS[op](a, b)
        r = true + int(rng.choice([-2, -1, 1, 2, 10, -10, 3, -3]))
        if r not in _alternatives(a, op, b):
            break
    return Sample(_arith_line(a, op, b, r, rng), _maybe_expected(true, rng), _conf(rng), "arithmetic_error")


def gen_arith_sign_error(rng) -> Sample:
    while True:
        op = str(rng.choice(list(OPS)))
        a, b = int(rng.integers(-30, 30)), int(rng.integers(-30, 30))
        if a >= 0 and b >= 0 or a == 0 or b == 0:
            continue
        true = OPS[op](a, b)
        wrong = {-true, OPS[op](-a, b), OPS[op](a, -b)} - {true}
        if wrong:
            r = int(rng.choice(sorted(wrong)))
            break
    return Sample(_arith_line(a, op, b, r, rng), _maybe_expected(true, rng), _conf(rng), "sign_error")


def gen_arith_wrong_op(rng) -> Sample:
    while True:
        op = str(rng.choice(list(OPS)))
        a, b = int(rng.integers(2, 40)), int(rng.integers(2, 20))
        true = OPS[op](a, b)
        r = OPS[str(rng.choice([o for o in OPS if o != op]))](a, b)
        if r != true:
            break
    return Sample(_arith_line(a, op, b, r, rng), _maybe_expected(true, rng), _conf(rng), "incorrect_operation")


# --- linear equations a*x + b = c ---------------------------------------------

def _linear_problem(rng) -> tuple[int, int, int, int]:
    a = int(rng.choice([2, 3, 4, 5, 6, 7, 8, 9, -2, -3]))
    s = int(rng.integers(-9, 13))
    b = int(rng.choice([v for v in range(-15, 16) if v != 0]))
    return a, b, a * s + b, s


def _algebra(lines: list[str], expected, rng, label: str) -> Sample:
    sp = " " if rng.random() < 0.7 else ""
    text = "\n".join(ln.replace(" = ", f"{sp}={sp}") for ln in lines)
    return Sample(text, _maybe_expected(expected, rng), _conf(rng), label)


def gen_alg_correct(rng) -> Sample:
    a, b, c, s = _linear_problem(rng)
    lines = [f"{_linear_lhs(a, b, rng)} = {c}"]
    if rng.random() < 0.6:
        lines.append(f"{a}x = {c - b}")
    lines.append(f"x = {s}")
    return _algebra(lines, s, rng, "correct_solution")


def gen_alg_manipulation(rng) -> Sample:
    a, b, c, s = _linear_problem(rng)
    kind = rng.choice(["transpose", "one_term", "multiply"])
    lines = [f"{_linear_lhs(a, b, rng)} = {c}"]
    if kind == "transpose":
        if rng.random() < 0.6:
            lines.append(f"{a}x = {c + b}")
        v = Rational(c + b, a)
    elif kind == "one_term":
        v = Rational(c, a) - b
    else:
        if rng.random() < 0.5:
            lines.append(f"{a}x = {c - b}")
        v = Rational(c - b) * a
    if v == s:
        return gen_alg_manipulation(rng)
    lines.append(f"x = {_fmt_num(v)}")
    return _algebra(lines, s, rng, "algebraic_manipulation_error")


def gen_alg_skipped(rng) -> Sample:
    a, b, c, s = _linear_problem(rng)
    v = Rational(c - b) if rng.random() < 0.6 else Rational(c, a)
    if v == s:
        return gen_alg_skipped(rng)
    return _algebra([f"{_linear_lhs(a, b, rng)} = {c}", f"x = {_fmt_num(v)}"], s, rng, "skipped_step")


def gen_alg_arith_slip(rng) -> Sample:
    a, b, c, s = _linear_problem(rng)
    delta = int(rng.choice([-2, -1, 1, 2]))
    mid = c - b + delta
    return _algebra(
        [f"{_linear_lhs(a, b, rng)} = {c}", f"{a}x = {mid}", f"x = {_fmt_num(Rational(mid, a))}"],
        s, rng, "arithmetic_error",
    )


def gen_alg_sign(rng) -> Sample:
    a, b, c, s = _linear_problem(rng)
    if s == 0:
        return gen_alg_sign(rng)
    lines = [f"{_linear_lhs(a, b, rng)} = {c}"]
    if rng.random() < 0.5:
        lines.append(f"{a}x = {c - b}")
    lines.append(f"x = {-s}")
    return _algebra(lines, s, rng, "sign_error")


# --- in-place simplification inside an equation -----------------------------------
# Real tutor logs (CogTutorBugs) show students writing the computation first and
# slipping when simplifying it: '3x = 12 - 4' -> '3x = 16', or 'x = 8/(-4)' -> 'x = 2'.

def _alg_inplace(rng, slip: str, label: str) -> Sample:
    while True:
        a, b, c, s = _linear_problem(rng)
        lines = [f"{_linear_lhs(a, b, rng)} = {c}"]
        if slip == "wrong_op" or rng.random() < 0.5:  # slip while simplifying c - b
            true = c - b
            wrong = {"wrong_op": c + b, "sign": -true, "arith": true + int(rng.choice([-2, -1, 1, 2, 10, -10]))}[slip]
            lines += [f"{a}x = {c} - {b}" if b > 0 else f"{a}x = {c} + {-b}", f"{a}x = {wrong}"]
            if rng.random() < 0.7:
                lines.append(f"x = {_fmt_num(Rational(wrong, a))}")
        else:  # slip while dividing by the coefficient
            true = s
            wrong = {"sign": -s, "arith": s + int(rng.choice([-2, -1, 1, 2]))}[slip]
            lines += [f"{a}x = {c - b}", f"x = {c - b}/{f'({a})' if a < 0 else a}", f"x = {wrong}"]
        if wrong != true and true != 0 and c != 0:
            return _algebra(lines, s, rng, label)


def gen_alg_inplace_wrong_op(rng) -> Sample:
    return _alg_inplace(rng, "wrong_op", "incorrect_operation")


def gen_alg_inplace_sign(rng) -> Sample:
    return _alg_inplace(rng, "sign", "sign_error")


def gen_alg_inplace_arith(rng) -> Sample:
    return _alg_inplace(rng, "arith", "arithmetic_error")


# --- other categories -----------------------------------------------------------

def gen_final_answer_only(rng) -> Sample:
    base = gen_arith_correct(rng) if rng.random() < 0.5 else gen_alg_correct(rng)
    true_val = base.text.splitlines()[-1].split("=")[-1].strip()
    wrong = Rational(true_val) + int(rng.choice([-5, -3, -2, -1, 1, 2, 3, 7, 10]))
    return Sample(base.text, _fmt_num(wrong), base.ocr_confidence, "final_answer_only_error")


_GARBLE = list("?#~S$&@lIoO%")


def gen_transcription_uncertainty(rng) -> Sample:
    base = [gen_arith_correct, gen_alg_correct, gen_arith_error, gen_alg_manipulation][int(rng.integers(0, 4))](rng)
    chars = list(base.text)
    for i, ch in enumerate(chars):
        if ch not in "\n " and rng.random() < 0.3:
            chars[i] = str(rng.choice(_GARBLE)) if rng.random() < 0.8 else ""
    return Sample("".join(chars), base.expected_answer, float(rng.uniform(0.2, 0.55)), "transcription_uncertainty")


_FILLER = ["see above", "idk", "ans", "the answer is", "work", "???", "same as q3", "x", "= =", "+", "12 +"]


def gen_insufficient(rng) -> Sample:
    choice = rng.random()
    if choice < 0.4:
        text = str(rng.choice(_FILLER))
    elif choice < 0.7:
        a, b = int(rng.integers(1, 50)), int(rng.integers(1, 50))
        text = f"{a} + {b}"  # expression with no claimed result
    else:
        text = "".join(rng.choice(list("abcdefgh !#")) for _ in range(int(rng.integers(3, 12))))
    return Sample(text, None if rng.random() < 0.5 else str(rng.integers(1, 99)), _conf(rng), "insufficient_information")


GENERATORS = {
    "correct_solution": [gen_arith_correct, gen_alg_correct],
    "arithmetic_error": [gen_arith_error, gen_alg_arith_slip, gen_alg_inplace_arith],
    "sign_error": [gen_arith_sign_error, gen_alg_sign, gen_alg_inplace_sign],
    "incorrect_operation": [gen_arith_wrong_op, gen_alg_inplace_wrong_op],
    "algebraic_manipulation_error": [gen_alg_manipulation],
    "skipped_step": [gen_alg_skipped],
    "final_answer_only_error": [gen_final_answer_only],
    "transcription_uncertainty": [gen_transcription_uncertainty],
    "insufficient_information": [gen_insufficient],
}


def generate_samples(n_per_label: int = 300, seed: int = 0) -> list[Sample]:
    rng = np.random.default_rng(seed)
    samples = []
    for label in LABELS:
        gens = GENERATORS[label]
        for i in range(n_per_label):
            samples.append(gens[i % len(gens)](rng))
    return samples
