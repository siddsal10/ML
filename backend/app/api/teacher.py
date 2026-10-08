from __future__ import annotations

from collections import Counter

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.ml.clustering.cluster import build_profile, cluster_students
from app.ml.irt.recommender import LearnerModel
from app.ml.skills import SKILLS
from app.models.student import Student
from app.services.learner_service import learner_state, mastery_vector

router = APIRouter(prefix="/api/teacher", tags=["teacher"])

SUGGESTIONS = {
    "sign errors": "Number-line warm-up: 10 minutes of adding and subtracting negatives with a physical or drawn number line.",
    "moving terms": "Balance-scale activity: model equations as scales and practise doing the same thing to both sides.",
    "skipped steps": "Ask the group to write one operation per line; peer-check that every line is equivalent to the previous one.",
    "wrong operation": "Keyword sort: match word cues and symbols to operations before computing.",
    "arithmetic slips": "Estimation first: predict the size of the answer before calculating, then check.",
    "final-answer slips": "Answer-checking routine: substitute the answer back into the original problem.",
    "Strong": "Extension problems: multi-step equations and creating their own error-spotting puzzles for peers.",
}


def _suggestion(name: str) -> str:
    for key, text in SUGGESTIONS.items():
        if key in name:
            return text
    return "Targeted practice on the weakest skill using the recommended items."


@router.get("/class")
def class_overview(db: Session = Depends(get_db)) -> dict:
    model = LearnerModel()
    students = db.query(Student).order_by(Student.name).all()
    rows, profiles, all_categories = [], [], Counter()
    for st in students:
        states, events, _ = learner_state(db, st.id, model)
        categories = []
        for sub in st.submissions:
            if sub.analyses:
                m = sorted(sub.analyses, key=lambda a: a.created_at)[-1].mistake
                if m:
                    categories.append(m.category)
        errors = [c for c in categories if c not in ("correct_solution", "insufficient_information", "transcription_uncertainty")]
        all_categories.update(errors)
        mastery = mastery_vector(states)
        profiles.append(build_profile(categories, mastery))
        rows.append({
            "id": st.id, "name": st.name, "mastery": mastery,
            "n_submissions": len(st.submissions), "n_practice": sum(e["source"] == "practice" for e in events),
            "top_mistake": Counter(errors).most_common(1)[0][0] if errors else None,
        })

    clusters = {}
    metrics = {}
    if rows:
        result = cluster_students(np.array(profiles))
        metrics = result["metrics"]
        for row, label, (x, y) in zip(rows, result["labels"], result["coords"]):
            row.update(cluster=label, x=x, y=y)
        for label, name in result["names"].items():
            members = [r for r in rows if r["cluster"] == label]
            clusters[str(label)] = {"name": name, "size": len(members), "suggestion": _suggestion(name),
                                    "avg_mastery": round(float(np.mean([np.mean(list(r["mastery"].values())) for r in members])), 3)}

    return {
        "skills": SKILLS,
        "students": rows,
        "clusters": clusters,
        "cluster_metrics": metrics,
        "misconceptions": [{"category": c, "count": n} for c, n in all_categories.most_common()],
    }
