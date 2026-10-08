"""Unsupervised grouping of students by misconception profile (teacher view).

Profile per student = rate of each mistake category over their diagnosed
submissions + BKT mastery per skill. KMeans (k picked by silhouette) is
compared with agglomerative (Ward) clustering; PCA gives 2-D coordinates
for the dashboard scatter. Each cluster is named after the profile
features most elevated relative to the class mean.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from app.ml.mistake_data import LABELS
from app.ml.skills import SKILL_BY_ID, SKILL_IDS

ERROR_LABELS = [l for l in LABELS if l not in ("correct_solution", "insufficient_information", "transcription_uncertainty")]
PROFILE_FEATURES = [f"rate:{l}" for l in ERROR_LABELS] + [f"mastery:{s}" for s in SKILL_IDS]

_PRETTY = {
    "rate:arithmetic_error": "arithmetic slips",
    "rate:sign_error": "sign errors",
    "rate:incorrect_operation": "wrong operation",
    "rate:algebraic_manipulation_error": "moving terms",
    "rate:skipped_step": "skipped steps",
    "rate:final_answer_only_error": "final-answer slips",
}


def build_profile(categories: list[str], mastery: dict[str, float]) -> np.ndarray:
    counts = Counter(categories)
    n = max(1, sum(counts.values()))
    return np.array([counts[l] / n for l in ERROR_LABELS] + [mastery.get(s, 0.0) for s in SKILL_IDS])


def _name(center: np.ndarray, overall_mean: np.ndarray, scale: np.ndarray) -> str:
    # Features that barely vary across the class carry no signal for naming.
    z = np.where(scale > 0.02, (center - overall_mean) / np.where(scale > 0, scale, 1), 0.0)
    n_err = len(ERROR_LABELS)
    err_z, mastery_z = z[:n_err], z[n_err:]
    if err_z.max() < 0.1 and mastery_z.mean() > 0:
        return "Strong & consistent"
    parts = []
    for i in np.argsort(err_z)[::-1][:2]:
        if err_z[i] > 0.4:
            parts.append(_PRETTY[PROFILE_FEATURES[i]])
    if not parts:
        weakest = SKILL_IDS[int(np.argmin(mastery_z))]
        return f"Needs work: {SKILL_BY_ID[weakest]['short'].lower()}"
    return "Struggling with " + " & ".join(parts)


def cluster_students(profiles: np.ndarray, k_range=range(2, 7), seed: int = 0) -> dict:
    """Returns labels, names, 2-D coordinates and quality metrics."""
    n = len(profiles)
    single = {"labels": [0] * n, "names": {0: "All students"}, "coords": [[0.0, 0.0]] * n, "k": 1, "metrics": {}}
    if n < 4:
        return single
    X = StandardScaler().fit_transform(profiles)
    n_distinct = len(np.unique(X.round(8), axis=0))
    best = None
    for k in k_range:
        if k >= n or k > n_distinct:
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(set(labels)) < 2 or len(set(labels)) >= n:
            continue
        sil = silhouette_score(X, labels)
        if best is None or sil > best[1]:
            best = (k, sil, labels)
    if best is None:  # every profile identical: nothing to separate
        return single
    k, sil, labels = best
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    coords = PCA(n_components=2, random_state=seed).fit_transform(X)
    mean, std = profiles.mean(axis=0), profiles.std(axis=0)
    names = {int(c): _name(profiles[labels == c].mean(axis=0), mean, std) for c in sorted(set(labels))}
    return {
        "labels": [int(l) for l in labels],
        "names": names,
        "coords": coords.round(3).tolist(),
        "k": int(k),
        "metrics": {
            "kmeans_silhouette": round(float(sil), 4),
            "kmeans_davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
            "agglomerative_silhouette": round(float(silhouette_score(X, agg)), 4),
            "agreement_ari_kmeans_vs_agglomerative": round(float(adjusted_rand_score(labels, agg)), 4),
        },
    }


def evaluate_on_simulated_class(seed: int = 0, n_students: int = 120) -> dict:
    """Plant archetypes in simulated students and check that clustering recovers them."""
    from app.ml.irt.item_bank import build_item_bank
    from app.ml.tracing.models import BKTParams, bkt_trace
    from app.ml.tracing.simulate import TRUE_BKT, SimLearner, true_item_difficulty

    rng = np.random.default_rng(seed)
    archetype_errors = {
        "negatives": {"sign_error": 0.6, "arithmetic_error": 0.1},
        "moving_terms": {"algebraic_manipulation_error": 0.45, "skipped_step": 0.25},
        "foundations": {"arithmetic_error": 0.4, "incorrect_operation": 0.3},
        "strong": {"arithmetic_error": 0.05},
    }
    names = list(archetype_errors)
    b_true = true_item_difficulty(seed)
    bank = build_item_bank()
    profiles, truth = [], []
    for i in range(n_students):
        arch = names[i % len(names)]
        cats = []
        for _ in range(12):
            r, acc = rng.random(), 0.0
            label = "correct_solution"
            for cat, p in archetype_errors[arch].items():
                acc += p
                if r < acc:
                    label = cat
                    break
            cats.append(label)
        learner = SimLearner(i, np.random.default_rng(seed * 7919 + i), arch)
        mastery = {}
        for s in SKILL_IDS:
            items = [it for it in bank if it.skill == s]
            seq = [learner.answer(s, b_true[items[int(rng.integers(len(items)))].id]) for _ in range(8)]
            _, hist = bkt_trace(BKTParams(*TRUE_BKT[s]), seq)
            mastery[s] = hist[-1]
        profiles.append(build_profile(cats, mastery))
        truth.append(names.index(arch))
    result = cluster_students(np.array(profiles), seed=seed)
    result["metrics"]["ari_vs_planted_archetypes"] = round(float(adjusted_rand_score(truth, result["labels"])), 4)
    return {
        "task": "student_clustering",
        "data": f"{n_students} simulated students in 4 planted archetypes ({', '.join(names)})",
        "k_selected": result["k"],
        "metrics": result["metrics"],
        "cluster_names": {str(k): v for k, v in result["names"].items()},
        "limitations": "Simulated archetypes are cleanly separated; real classes will overlap more.",
    }
