"""Builds each student's per-skill observation history (practice answers +
diagnosed submissions) and runs the learner model over it."""
from __future__ import annotations

from collections import defaultdict
from datetime import timezone

from sqlalchemy.orm import Session

from app.ml.irt.recommender import LearnerModel, SkillState
from app.ml.skills import SKILL_IDS
from app.models.learning import PracticeAttempt
from app.models.submission import Submission

# Submissions with these categories say nothing about the student's skill.
NON_EVIDENCE = {"insufficient_information", "transcription_uncertainty"}


def _aware(dt):
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


def events_for(db: Session, student_id: str) -> list[dict]:
    events = []
    for a in db.query(PracticeAttempt).filter(PracticeAttempt.student_id == student_id):
        events.append({"time": _aware(a.created_at), "skill": a.skill, "correct": int(a.correct), "source": "practice", "ref": a.item_id})
    for sub in db.query(Submission).filter(Submission.student_id == student_id):
        if not sub.skill or not sub.analyses:
            continue
        latest = sorted(sub.analyses, key=lambda a: a.created_at)[-1]
        if latest.mistake is None or latest.mistake.category in NON_EVIDENCE:
            continue
        events.append({"time": _aware(sub.created_at), "skill": sub.skill,
                       "correct": int(latest.mistake.category == "correct_solution"), "source": "handwritten", "ref": sub.id})
    events.sort(key=lambda e: e["time"])
    return events


def learner_state(db: Session, student_id: str, model: LearnerModel | None = None) -> tuple[dict[str, SkillState], list[dict], LearnerModel]:
    model = model or LearnerModel()
    events = events_for(db, student_id)
    outcomes: dict[str, list[int]] = defaultdict(list)
    for e in events:
        outcomes[e["skill"]].append(e["correct"])
    states = model.skill_states(outcomes)
    # attach mastery-after-event to the timeline
    seen: dict[str, int] = defaultdict(int)
    for e in events:
        e["mastery"] = round(states[e["skill"]].history[seen[e["skill"]]], 4)
        seen[e["skill"]] += 1
    return states, events, model


def mastery_vector(states: dict[str, SkillState]) -> dict[str, float]:
    return {s: round(states[s].mastery, 4) for s in SKILL_IDS}
