from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.mistake import Mistake
from app.models.quiz import Quiz, QuizAttempt, QuizQuestion
from app.models.student import Student
from app.models.submission import Submission
from app.schemas.quiz import (
    QuizGenerateRequest,
    QuizGradedQuestionOut,
    QuizOut,
    QuizSubmitRequest,
    QuizSubmitResponse,
)
from app.services import quiz_service
from app.services.mastery_service import apply_quiz_result

router = APIRouter(prefix="/api/quizzes", tags=["quizzes"])


@router.post("/generate", response_model=QuizOut)
def generate_quiz(payload: QuizGenerateRequest, db: Session = Depends(get_db)) -> Quiz:
    student = db.get(Student, payload.student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")

    weak_concept = payload.weak_concept
    if weak_concept is None and payload.submission_id:
        submission = db.get(Submission, payload.submission_id)
        if submission and submission.analyses:
            latest = sorted(submission.analyses, key=lambda a: a.created_at)[-1]
            if latest.mistake:
                weak_concept = latest.mistake.category

    quiz = Quiz(
        student_id=payload.student_id,
        submission_id=payload.submission_id,
        topic=payload.topic,
        weak_concept=weak_concept,
    )
    db.add(quiz)
    db.flush()

    generated = quiz_service.generate_quiz_questions(payload.topic, weak_concept)
    for idx, q in enumerate(generated):
        db.add(
            QuizQuestion(
                quiz_id=quiz.id,
                order_index=idx,
                difficulty=q.difficulty,
                prompt=q.prompt,
                expected_answer=q.expected_answer,
                explanation=q.explanation,
                source=q.source,
            )
        )
    db.commit()
    db.refresh(quiz)
    return quiz


@router.get("/{quiz_id}", response_model=QuizOut)
def get_quiz(quiz_id: str, db: Session = Depends(get_db)) -> Quiz:
    quiz = db.get(Quiz, quiz_id)
    if quiz is None:
        raise HTTPException(status_code=404, detail="Quiz not found")
    return quiz


@router.post("/{quiz_id}/submit", response_model=QuizSubmitResponse)
def submit_quiz(quiz_id: str, payload: QuizSubmitRequest, db: Session = Depends(get_db)) -> QuizSubmitResponse:
    quiz = db.get(Quiz, quiz_id)
    if quiz is None:
        raise HTTPException(status_code=404, detail="Quiz not found")
    if quiz.submitted_at is not None:
        raise HTTPException(status_code=400, detail="Quiz has already been submitted")

    questions_by_id = {q.id: q for q in quiz.questions}
    answers_by_question = {a.question_id: a for a in payload.answers}

    graded: list[QuizGradedQuestionOut] = []
    correct_count = 0

    for question in quiz.questions:
        answer = answers_by_question.get(question.id)
        submitted_text = answer.submitted_answer if answer else ""
        is_correct = quiz_service.grade_answer(submitted_text, question.expected_answer) if answer else False
        if is_correct:
            correct_count += 1

        db.add(
            QuizAttempt(
                question_id=question.id,
                submitted_answer=submitted_text,
                is_correct=is_correct,
            )
        )
        graded.append(
            QuizGradedQuestionOut(
                question_id=question.id,
                prompt=question.prompt,
                difficulty=question.difficulty,
                submitted_answer=submitted_text,
                expected_answer=question.expected_answer,
                is_correct=is_correct,
                explanation=question.explanation,
            )
        )

    total = len(quiz.questions)
    score = round(correct_count / total, 4) if total else 0.0

    quiz.score = score
    quiz.submitted_at = datetime.now(timezone.utc)
    db.commit()

    mastery_record = apply_quiz_result(db, quiz.student_id, quiz.topic, score)

    return QuizSubmitResponse(
        quiz_id=quiz.id,
        score=score,
        total_questions=total,
        correct_count=correct_count,
        graded_questions=graded,
        updated_mastery=mastery_record.mastery_score,
    )
