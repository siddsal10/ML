from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QuizGenerateRequest(BaseModel):
    student_id: str
    submission_id: str | None = None
    topic: str
    weak_concept: str | None = None


class QuizQuestionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    order_index: int
    difficulty: str
    prompt: str
    source: str
    # expected_answer / explanation deliberately excluded until graded


class QuizOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    student_id: str
    topic: str
    weak_concept: str | None
    created_at: datetime
    score: float | None
    submitted_at: datetime | None
    questions: list[QuizQuestionOut]


class QuizAnswerIn(BaseModel):
    question_id: str
    submitted_answer: str = Field(..., min_length=1, max_length=300)


class QuizSubmitRequest(BaseModel):
    answers: list[QuizAnswerIn]


class QuizGradedQuestionOut(BaseModel):
    question_id: str
    prompt: str
    difficulty: str
    submitted_answer: str
    expected_answer: str
    is_correct: bool
    explanation: str


class QuizSubmitResponse(BaseModel):
    quiz_id: str
    score: float
    total_questions: int
    correct_count: int
    graded_questions: list[QuizGradedQuestionOut]
    updated_mastery: float
