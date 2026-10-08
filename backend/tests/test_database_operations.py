from __future__ import annotations

from app.models.mastery import MasteryRecord
from app.models.quiz import Quiz, QuizQuestion
from app.models.student import Student
from app.models.submission import Submission


def test_create_student_and_query(db_session):
    student = Student(name="Test Student", grade_level="Grade 8")
    db_session.add(student)
    db_session.commit()

    fetched = db_session.get(Student, student.id)
    assert fetched is not None
    assert fetched.name == "Test Student"


def test_submission_cascade_delete_removes_regions(db_session):
    from app.models.region import ExtractedRegion

    student = Student(name="Cascade Test")
    db_session.add(student)
    db_session.commit()

    submission = Submission(
        student_id=student.id,
        subject="math",
        topic="arithmetic",
        original_question="1+1?",
        original_image_path="/tmp/fake.png",
    )
    db_session.add(submission)
    db_session.commit()

    region = ExtractedRegion(
        submission_id=submission.id, region_type="text", x=0, y=0, width=10, height=10, confidence=0.5
    )
    db_session.add(region)
    db_session.commit()

    db_session.delete(submission)
    db_session.commit()

    remaining = db_session.query(ExtractedRegion).filter_by(submission_id=submission.id).all()
    assert remaining == []


def test_student_deletion_cascades_to_submissions(db_session):
    student = Student(name="ToDelete")
    db_session.add(student)
    db_session.commit()

    submission = Submission(
        student_id=student.id,
        subject="math",
        topic="arithmetic",
        original_question="1+1?",
        original_image_path="/tmp/fake.png",
    )
    db_session.add(submission)
    db_session.commit()
    submission_id = submission.id

    db_session.delete(student)
    db_session.commit()

    assert db_session.get(Submission, submission_id) is None


def test_mastery_record_unique_per_student_topic(db_session):
    student = Student(name="Mastery Test")
    db_session.add(student)
    db_session.commit()

    record = MasteryRecord(student_id=student.id, topic="algebra", mastery_score=0.5, attempts_count=1)
    db_session.add(record)
    db_session.commit()

    fetched = (
        db_session.query(MasteryRecord)
        .filter_by(student_id=student.id, topic="algebra")
        .one()
    )
    assert fetched.mastery_score == 0.5


def test_quiz_question_ordering(db_session):
    student = Student(name="Order Test")
    db_session.add(student)
    db_session.commit()

    quiz = Quiz(student_id=student.id, topic="arithmetic")
    db_session.add(quiz)
    db_session.flush()

    for idx in (2, 0, 1):
        db_session.add(
            QuizQuestion(
                quiz_id=quiz.id,
                order_index=idx,
                difficulty="easy",
                prompt=f"Q{idx}",
                expected_answer="1",
                explanation="because",
            )
        )
    db_session.commit()
    db_session.refresh(quiz)

    ordered_indices = [q.order_index for q in quiz.questions]
    assert ordered_indices == [0, 1, 2]
