from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.ml.irt.item_bank import item_by_id
from app.ml.skills import SKILL_BY_ID, SKILL_IDS, SKILLS
from app.models.learning import PracticeAttempt
from app.models.student import Student
from app.models.submission import Submission
from app.services.learner_service import learner_state, mastery_vector
from app.services.quiz_service import grade_answer

router = APIRouter(tags=["practice"])


def _student(db: Session, student_id: str) -> Student:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@router.get("/api/skills")
def list_skills() -> list[dict]:
    return SKILLS


@router.get("/api/students/{student_id}/skills")
def student_skills(student_id: str, db: Session = Depends(get_db)) -> dict:
    _student(db, student_id)
    states, events, model = learner_state(db, student_id)
    return {
        "model": "bkt" if model.trained else "bkt-default-params",
        "skills": [
            {**SKILL_BY_ID[s], "mastery": round(states[s].mastery, 4), "attempts": states[s].attempts,
             "successes": states[s].successes, "mastered": states[s].mastered}
            for s in SKILL_IDS
        ],
        "timeline": [
            {"time": e["time"].isoformat(), "skill": e["skill"], "correct": bool(e["correct"]), "source": e["source"], "mastery": e["mastery"]}
            for e in events
        ],
    }


@router.get("/api/practice/{student_id}/next")
def next_items(student_id: str, skill: str | None = Query(None), n: int = Query(3, ge=1, le=10),
               db: Session = Depends(get_db)) -> dict:
    _student(db, student_id)
    if skill is not None and skill not in SKILL_BY_ID:
        raise HTTPException(status_code=422, detail="Unknown skill")
    states, _, model = learner_state(db, student_id)

    reason = "requested"
    if skill is None:
        latest = (db.query(Submission).filter(Submission.student_id == student_id, Submission.skill.isnot(None))
                  .order_by(Submission.created_at.desc()).first())
        diagnosed = None
        if latest and latest.analyses:
            mistake = sorted(latest.analyses, key=lambda a: a.created_at)[-1].mistake
            if mistake and mistake.category not in ("correct_solution", "insufficient_information", "transcription_uncertainty"):
                diagnosed = latest.skill
        skill = model.target_skill(states, diagnosed)
        reason = "latest_mistake" if skill == diagnosed else "lowest_mastery"

    solved = {a.item_id for a in db.query(PracticeAttempt).filter(PracticeAttempt.student_id == student_id, PracticeAttempt.correct.is_(True))}
    recs = model.recommend(skill, states[skill], n=n, exclude=solved)
    return {
        "skill": SKILL_BY_ID[skill],
        "reason": reason,
        "mastery": round(states[skill].mastery, 4),
        "items": [{"id": it.id, "prompt": it.prompt, "level": ["easy", "medium", "hard"][it.level],
                   "predicted_p": round(p, 3)} for it, p in recs],
    }


class AnswerIn(BaseModel):
    student_id: str
    item_id: str
    answer: str = Field(..., min_length=1, max_length=100)


@router.post("/api/practice/answer")
def answer_item(payload: AnswerIn, db: Session = Depends(get_db)) -> dict:
    _student(db, payload.student_id)
    item = item_by_id().get(payload.item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Unknown item")
    states, _, model = learner_state(db, payload.student_id)
    before = states[item.skill]
    predicted = model.p_correct(item, before)
    correct = grade_answer(payload.answer, item.answer)

    db.add(PracticeAttempt(student_id=payload.student_id, item_id=item.id, skill=item.skill, answer=payload.answer.strip(),
                           correct=correct, predicted_p=round(predicted, 4)))
    db.commit()
    after_states, _, _ = learner_state(db, payload.student_id, model)
    after = after_states[item.skill]
    return {
        "correct": correct,
        "expected_answer": item.answer,
        "explanation": item.explanation,
        "predicted_p": round(predicted, 3),
        "skill": SKILL_BY_ID[item.skill],
        "mastery_before": round(before.mastery, 4),
        "mastery_after": round(after.mastery, 4),
        "mastered": after.mastered,
        "all_mastery": mastery_vector(after_states),
    }
