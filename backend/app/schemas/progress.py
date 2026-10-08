from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class MasteryOut(BaseModel):
    topic: str
    mastery_score: float
    attempts_count: int
    last_updated: datetime


class MistakeCategoryCount(BaseModel):
    category: str
    count: int


class RecentSubmissionOut(BaseModel):
    submission_id: str
    topic: str
    created_at: datetime
    is_correct: bool | None
    mistake_category: str | None
    question: str | None = None
    skill: str | None = None


class RecentQuizOut(BaseModel):
    quiz_id: str
    topic: str
    created_at: datetime
    score: float | None


class ProgressOut(BaseModel):
    student_id: str
    overall_mastery: float
    topic_mastery: list[MasteryOut]
    mistake_category_counts: list[MistakeCategoryCount]
    recent_quizzes: list[RecentQuizOut]


class HistoryOut(BaseModel):
    student_id: str
    submissions: list[RecentSubmissionOut]
    quizzes: list[RecentQuizOut]
