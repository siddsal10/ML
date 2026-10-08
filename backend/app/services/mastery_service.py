"""DB-backed mastery tracking and progress aggregation."""
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.analysis import AnalysisResult
from app.models.mastery import MasteryRecord
from app.models.mistake import Mistake
from app.models.quiz import Quiz
from app.models.submission import Submission
from app.services.quiz_service import update_mastery


def get_or_create_mastery(db: Session, student_id: str, topic: str) -> MasteryRecord:
    record = (
        db.query(MasteryRecord)
        .filter(MasteryRecord.student_id == student_id, MasteryRecord.topic == topic)
        .one_or_none()
    )
    if record is None:
        record = MasteryRecord(student_id=student_id, topic=topic, mastery_score=0.0, attempts_count=0)
        db.add(record)
        db.flush()
    return record


def apply_quiz_result(db: Session, student_id: str, topic: str, score_fraction: float) -> MasteryRecord:
    record = get_or_create_mastery(db, student_id, topic)
    record.mastery_score = update_mastery(record.mastery_score, score_fraction)
    record.attempts_count += 1
    db.commit()
    db.refresh(record)
    return record


def get_topic_mastery(db: Session, student_id: str) -> list[MasteryRecord]:
    return db.query(MasteryRecord).filter(MasteryRecord.student_id == student_id).all()


def get_overall_mastery(db: Session, student_id: str) -> float:
    records = get_topic_mastery(db, student_id)
    if not records:
        return 0.0
    return round(sum(r.mastery_score for r in records) / len(records), 4)


def get_mistake_category_counts(db: Session, student_id: str) -> list[tuple[str, int]]:
    rows = (
        db.query(Mistake.category, func.count(Mistake.id))
        .join(AnalysisResult, Mistake.analysis_id == AnalysisResult.id)
        .join(Submission, AnalysisResult.submission_id == Submission.id)
        .filter(Submission.student_id == student_id)
        .group_by(Mistake.category)
        .all()
    )
    return [(category, count) for category, count in rows]


def get_recent_submissions(db: Session, student_id: str, limit: int = 20) -> list[Submission]:
    return (
        db.query(Submission)
        .filter(Submission.student_id == student_id)
        .order_by(Submission.created_at.desc())
        .limit(limit)
        .all()
    )


def get_recent_quizzes(db: Session, student_id: str, limit: int = 20) -> list[Quiz]:
    return (
        db.query(Quiz)
        .filter(Quiz.student_id == student_id)
        .order_by(Quiz.created_at.desc())
        .limit(limit)
        .all()
    )
