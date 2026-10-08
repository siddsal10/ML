"""Character/word error rate and edit-distance metrics.

Implemented from scratch (standard Levenshtein DP) so the benchmark module
has no hard dependency on an external metrics package.
"""
from __future__ import annotations


def _levenshtein(a: list, b: list) -> int:
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    distance = _levenshtein(ref_chars, hyp_chars)
    return round(distance / len(ref_chars), 4)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = _levenshtein(ref_words, hyp_words)
    return round(distance / len(ref_words), 4)


def normalized_edit_distance(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 0.0
    distance = _levenshtein(list(reference), list(hypothesis))
    return round(distance / max(len(reference), len(hypothesis), 1), 4)


def exact_match(reference: str, hypothesis: str) -> bool:
    return reference.strip() == hypothesis.strip()


def macro_f1(pairs: list[tuple[str, str]]) -> float | None:
    """pairs = [(predicted_category, true_category), ...]. Macro-averaged F1."""
    if not pairs:
        return None
    categories = sorted({p for p, _ in pairs} | {t for _, t in pairs})
    f1_scores = []
    for cat in categories:
        tp = sum(1 for p, t in pairs if p == cat and t == cat)
        fp = sum(1 for p, t in pairs if p == cat and t != cat)
        fn = sum(1 for p, t in pairs if p != cat and t == cat)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return round(sum(f1_scores) / len(f1_scores), 4)
