"""Skill taxonomy shared by diagnosis, knowledge tracing, the item bank and
the teacher dashboard. Every submission and practice item maps to exactly
one skill."""
from __future__ import annotations

import re

SKILLS: list[dict] = [
    {"id": "S1", "name": "Whole-number addition & subtraction", "short": "Add / subtract"},
    {"id": "S2", "name": "Multiplication & division", "short": "Multiply / divide"},
    {"id": "S3", "name": "Integer operations with negatives", "short": "Negatives"},
    {"id": "S4", "name": "Order of operations & brackets", "short": "Brackets"},
    {"id": "S5", "name": "One-step linear equations", "short": "One-step eq."},
    {"id": "S6", "name": "Two-step linear equations", "short": "Two-step eq."},
    {"id": "S7", "name": "Moving terms across '=' (inverse operations)", "short": "Moving terms"},
    {"id": "S8", "name": "Equations with negative numbers", "short": "Neg. equations"},
]
SKILL_IDS = [s["id"] for s in SKILLS]
SKILL_BY_ID = {s["id"]: s for s in SKILLS}

_NEG = re.compile(r"(^|[+\-*/(=])\s*-\s*\d|\(\s*-")
_LINEAR_TWO_STEP = re.compile(r"\d\s*\*?\s*x\s*[+\-]\s*\d")


def infer_skill(text: str, category: str | None = None) -> str:
    """Deterministically map a piece of student work (+ diagnosed category) to a skill."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    has_x = "x" in (text or "")
    has_neg = bool(_NEG.search(first))

    if has_x:
        if category == "algebraic_manipulation_error":
            return "S7"
        if category == "sign_error" or has_neg:
            return "S8"
        return "S6" if _LINEAR_TWO_STEP.search(first) else "S5"
    if category == "sign_error" or has_neg:
        return "S3"
    if "(" in first:
        return "S4"
    if "*" in first or "/" in first:
        return "S2"
    if category == "incorrect_operation" and ("+" in first or "-" in first):
        return "S1"
    return "S1"
