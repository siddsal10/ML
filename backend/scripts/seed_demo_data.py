"""Seeds a demo student plus example correct/incorrect submissions.

Requires scripts/generate_demo_images.py to have been run first so the
demo image files (and their manifest) exist on disk.

Run:
    python scripts/seed_demo_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.database import SessionLocal, init_db  # noqa: E402
from app.models.student import Student  # noqa: E402
from app.models.submission import Submission  # noqa: E402
from app.services.submission_service import run_analysis_pipeline  # noqa: E402

DEMO_SAMPLES_DIR = BACKEND_ROOT / "data" / "demo_samples"

SEED_SUBMISSIONS = [
    {
        "slug": "demo_correct_arithmetic",
        "topic": "arithmetic",
        "question": "What is 12 + 15?",
        "expected_answer": "27",
    },
    {
        "slug": "demo_sign_error",
        "topic": "arithmetic",
        "question": "What is -3 + 5?",
        "expected_answer": "2",
    },
    {
        "slug": "demo_correct_algebra",
        "topic": "algebra",
        "question": "Solve for x: 2x + 4 = 10",
        "expected_answer": "3",
    },
    {
        "slug": "demo_algebra_error",
        "topic": "algebra",
        "question": "Solve for x: 2x + 4 = 10",
        "expected_answer": "3",
    },
]


def main() -> None:
    if not (DEMO_SAMPLES_DIR / "manifest.json").exists():
        print("ERROR: demo image manifest not found. Run scripts/generate_demo_images.py first.")
        sys.exit(1)

    init_db()
    db = SessionLocal()
    try:
        existing = db.query(Student).filter(Student.name == "Demo Student").one_or_none()
        if existing:
            print(f"Demo student already exists: {existing.id}. Skipping seed to avoid duplicates.")
            return

        student = Student(name="Demo Student", grade_level="Grade 7")
        db.add(student)
        db.commit()
        db.refresh(student)
        print(f"Created demo student: {student.id}")

        for item in SEED_SUBMISSIONS:
            image_path = DEMO_SAMPLES_DIR / f"{item['slug']}.png"
            if not image_path.exists():
                print(f"  WARNING: missing image for {item['slug']}, skipping.")
                continue

            submission = Submission(
                student_id=student.id,
                subject="math",
                topic=item["topic"],
                original_question=item["question"],
                expected_answer=item["expected_answer"],
                provider_requested="ml",
                original_image_path=str(image_path),
            )
            db.add(submission)
            db.commit()
            db.refresh(submission)

            analysis = run_analysis_pipeline(db, submission, "ml")
            category = analysis.mistake.category if analysis.mistake else "unknown"
            print(f"  Seeded submission {submission.id} ({item['slug']}) -> diagnosed as '{category}'")

        print("\nDemo data seeded successfully.")
        print(f"Demo student id: {student.id}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
