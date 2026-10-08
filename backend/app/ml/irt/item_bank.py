"""Deterministic practice-item bank (~40 items per skill).

Each item carries hand-designed difficulty features (operand magnitude,
negatives, number of steps, ...). IRT (app.ml.irt.rasch) estimates each
item's difficulty from responses; a ridge regression on these features
predicts difficulty for items that have little or no response data.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache

from app.ml.skills import SKILL_IDS

ITEMS_PER_SKILL = 40
FEATURE_NAMES = ["log_magnitude", "n_negatives", "n_steps", "has_division", "has_brackets", "answer_negative", "level"]


@dataclass
class Item:
    id: str
    skill: str
    prompt: str
    answer: str
    explanation: str
    level: int  # 0 easy, 1 medium, 2 hard (template level, used as a feature and for UI)
    features: dict = field(default_factory=dict)


def _feats(nums: list[int], n_steps: int, division=False, brackets=False, answer=0, level=0) -> dict:
    import math

    return {
        "log_magnitude": math.log10(1 + max(abs(n) for n in nums)),
        "n_negatives": float(sum(n < 0 for n in nums)),
        "n_steps": float(n_steps),
        "has_division": float(division),
        "has_brackets": float(brackets),
        "answer_negative": float(answer < 0),
        "level": float(level),
    }


def _s1(r: random.Random, lvl: int) -> tuple:
    hi = [10, 60, 400][lvl]
    a, b = r.randint(1, hi), r.randint(1, hi)
    if r.random() < 0.5:
        a, b = max(a, b), min(a, b)
        return (f"{a} - {b}", a - b, f"{a} - {b} = {a - b}. Subtract the smaller number from the larger.", [a, b], 1, {})
    return (f"{a} + {b}", a + b, f"{a} + {b} = {a + b}. Add the ones, then the tens, carrying where needed.", [a, b], 1, {})


def _s2(r: random.Random, lvl: int) -> tuple:
    hi = [6, 12, 25][lvl]
    a, b = r.randint(2, hi), r.randint(2, hi)
    if r.random() < 0.4:
        return (f"{a * b} / {a}", b, f"{a * b} / {a} = {b}, because {a} x {b} = {a * b}.", [a * b, a], 1, {"division": True})
    return (f"{a} * {b}", a * b, f"{a} x {b} = {a * b}.", [a, b], 1, {})


def _s3(r: random.Random, lvl: int) -> tuple:
    hi = [10, 30, 90][lvl]
    a, b = -r.randint(1, hi), r.randint(1, hi)
    kind = r.choice(["+", "-", "*"] if lvl else ["+", "-"])
    if kind == "+":
        return (f"{a} + {b}", a + b, f"Start at {a} on the number line and move {b} to the right: {a + b}.", [a, b], 1, {})
    if kind == "-":
        c = -r.randint(1, hi)
        return (f"{b} - ({c})", b - c, f"Subtracting a negative is adding: {b} - ({c}) = {b} + {-c} = {b - c}.", [b, c], 1, {"brackets": True})
    return (f"{a} * {b}", a * b, f"Negative x positive is negative: {a} x {b} = {a * b}.", [a, b], 1, {})


def _s4(r: random.Random, lvl: int) -> tuple:
    a, b, c = r.randint(2, 9), r.randint(2, 9), r.randint(2, 9 + 5 * lvl)
    if lvl == 0:
        return (f"{c} + {a} * {b}", c + a * b, f"Multiply first: {a} x {b} = {a * b}, then add {c}: {c + a * b}.", [a, b, c], 2, {})
    if lvl == 1:
        return (f"({c} + {a}) * {b}", (c + a) * b, f"Brackets first: {c} + {a} = {c + a}, then x {b} = {(c + a) * b}.", [a, b, c], 2, {"brackets": True})
    d = r.randint(2, 6)
    return (f"({c} - {a}) * {b} + {d}", (c - a) * b + d, f"Brackets: {c - a}; times {b}: {(c - a) * b}; plus {d}: {(c - a) * b + d}.", [a, b, c, d], 3, {"brackets": True})


def _s5(r: random.Random, lvl: int) -> tuple:
    s = r.randint(1, [9, 20, 40][lvl])
    if r.random() < 0.5:
        b = r.randint(1, [10, 30, 90][lvl])
        return (f"x + {b} = {s + b}", s, f"Subtract {b} from both sides: x = {s + b} - {b} = {s}.", [b, s + b], 1, {"eq": True})
    a = r.randint(2, [5, 9, 15][lvl])
    return (f"{a}x = {a * s}", s, f"Divide both sides by {a}: x = {a * s} / {a} = {s}.", [a, a * s], 1, {"eq": True, "division": True})


def _two_step(r: random.Random, lvl: int, neg: bool) -> tuple:
    a = r.randint(2, [5, 9, 12][lvl])
    s = r.randint(-9, 12) if neg else r.randint(1, 12)
    b = r.randint(1, [10, 25, 60][lvl]) * (r.choice([-1, 1]) if neg else 1)
    if neg and r.random() < 0.4:
        a = -a
    c = a * s + b
    sign = "+" if b > 0 else "-"
    coef = "-" if a == -1 else str(a)
    prompt = f"{coef}x {sign} {abs(b)} = {c}"
    expl = (f"{'Subtract' if b > 0 else 'Add'} {abs(b)} on both sides: {a}x = {c - b}. "
            f"Divide both sides by {a}: x = {s}.")
    return (prompt, s, expl, [a, b, c], 2, {"eq": True, "division": True})


def _s6(r, lvl):
    return _two_step(r, lvl, neg=False)


def _s7(r: random.Random, lvl: int) -> tuple:
    a = r.randint(2, [5, 9, 12][lvl])
    s = r.randint(1, 12)
    b = r.randint(1, [10, 25, 60][lvl])
    c = a * s - b
    expl = (f"The {b} is subtracted on the left, so ADD {b} to both sides (its sign changes when it crosses '='): "
            f"{a}x = {c + b}, then x = {s}.")
    return (f"{a}x - {b} = {c}", s, expl, [a, b, c], 2, {"eq": True, "division": True})


def _s8(r, lvl):
    return _two_step(r, lvl, neg=True)


_GENERATORS = {"S1": _s1, "S2": _s2, "S3": _s3, "S4": _s4, "S5": _s5, "S6": _s6, "S7": _s7, "S8": _s8}


@lru_cache(maxsize=1)
def build_item_bank(seed: int = 0) -> tuple[Item, ...]:
    items = []
    for skill in SKILL_IDS:
        r = random.Random(f"{seed}-{skill}")
        seen = set()
        i = 0
        while len([it for it in items if it.skill == skill]) < ITEMS_PER_SKILL:
            lvl = i % 3
            i += 1
            prompt, ans, expl, nums, steps, flags = _GENERATORS[skill](r, lvl)
            if prompt in seen:
                continue
            seen.add(prompt)
            is_eq = flags.get("eq", False)
            text = f"Solve for x: {prompt}" if is_eq else f"What is {prompt}?"
            n = len([it for it in items if it.skill == skill])
            items.append(Item(
                id=f"{skill}-{n:03d}", skill=skill, prompt=text, answer=str(ans), explanation=expl, level=lvl,
                features=_feats(nums, steps, flags.get("division", False), flags.get("brackets", False), int(ans), lvl),
            ))
    return tuple(items)


def item_by_id() -> dict[str, Item]:
    return {it.id: it for it in build_item_bank()}


def feature_matrix(items) -> "list[list[float]]":
    return [[it.features[f] for f in FEATURE_NAMES] for it in items]
