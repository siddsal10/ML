from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.student import Student
from app.schemas.progress import (
    HistoryOut,
    MasteryOut,
    MistakeCategoryCount,
    ProgressOut,
    RecentQuizOut,
    RecentSubmissionOut,
)
from app.services import mastery_service

router = APIRouter(prefix="/api/students", tags=["progress"])


def _require_student(student_id: str, db: Session) -> Student:
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@router.get("/{student_id}/progress", response_model=ProgressOut)
def get_progress(student_id: str, db: Session = Depends(get_db)) -> ProgressOut:
    _require_student(student_id, db)

    topic_mastery = mastery_service.get_topic_mastery(db, student_id)
    overall = mastery_service.get_overall_mastery(db, student_id)
    mistake_counts = mastery_service.get_mistake_category_counts(db, student_id)
    recent_quizzes = mastery_service.get_recent_quizzes(db, student_id, limit=10)

    return ProgressOut(
        student_id=student_id,
        overall_mastery=overall,
        topic_mastery=[MasteryOut.model_validate(r) for r in topic_mastery],
        mistake_category_counts=[MistakeCategoryCount(category=c, count=n) for c, n in mistake_counts],
        recent_quizzes=[
            RecentQuizOut(quiz_id=q.id, topic=q.topic, created_at=q.created_at, score=q.score)
            for q in recent_quizzes
        ],
    )


@router.get("/{student_id}/history", response_model=HistoryOut)
def get_history(student_id: str, db: Session = Depends(get_db)) -> HistoryOut:
    _require_student(student_id, db)

    submissions = mastery_service.get_recent_submissions(db, student_id, limit=30)
    quizzes = mastery_service.get_recent_quizzes(db, student_id, limit=30)

    submission_rows = []
    for s in submissions:
        latest = sorted(s.analyses, key=lambda a: a.created_at)[-1] if s.analyses else None
        submission_rows.append(
            RecentSubmissionOut(
                submission_id=s.id,
                topic=s.topic,
                created_at=s.created_at,
                is_correct=latest.is_correct if latest else None,
                mistake_category=(latest.mistake.category if latest and latest.mistake else None),
                question=s.original_question,
                skill=s.skill,
            )
        )

    return HistoryOut(
        student_id=student_id,
        submissions=submission_rows,
        quizzes=[
            RecentQuizOut(quiz_id=q.id, topic=q.topic, created_at=q.created_at, score=q.score) for q in quizzes
        ],
    )
