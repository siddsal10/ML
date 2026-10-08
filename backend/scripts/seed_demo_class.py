"""Seeds a simulated class so the teacher dashboard, clustering and progress
pages have realistic data.

Each simulated student belongs to a hidden archetype (e.g. "negatives").
Their handwritten work is generated with the mistake generators, rendered
in a handwriting-style font, and sent through the REAL pipeline (ML OCR ->
SymPy -> ML diagnosis), so OCR misreads and misdiagnoses happen just as they
would for real uploads. Practice answers come from the learner simulator.

Run (after training):  python scripts/seed_demo_class.py [--students 24]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.database import SessionLocal, init_db  # noqa: E402
from app.ml.glyph_data import find_fonts, render_expression  # noqa: E402
from app.ml.irt.item_bank import build_item_bank  # noqa: E402
from app.ml.mistake_data import GENERATORS  # noqa: E402
from app.ml.tracing.simulate import SimLearner, true_item_difficulty  # noqa: E402
from app.models.learning import PracticeAttempt  # noqa: E402
from app.models.student import Student  # noqa: E402
from app.models.submission import Submission  # noqa: E402
from app.services.submission_service import run_analysis_pipeline  # noqa: E402

NAMES = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Ishaan", "Saanvi", "Vihaan", "Kiara", "Arjun", "Myra",
         "Reyansh", "Aadhya", "Ayaan", "Navya", "Dhruv", "Riya", "Krish", "Tara", "Aditya", "Zara", "Neel", "Ira",
         "Om", "Pari", "Yash", "Siya", "Dev", "Anika"]

ARCHETYPE_ERRORS = {
    "negatives": {"sign_error": 0.55, "arithmetic_error": 0.1},
    "moving_terms": {"algebraic_manipulation_error": 0.4, "skipped_step": 0.25},
    "foundations": {"arithmetic_error": 0.35, "incorrect_operation": 0.3},
    "strong": {"arithmetic_error": 0.05},
}
HANDWRITING_FONTS = ["Bradley Hand Bold.ttf", "Noteworthy.ttc", "Chalkboard.ttc", "Comic Sans MS.ttf", "MarkerFelt.ttc"]


def _sample_label(rng, arch: str) -> str:
    r, acc = rng.random(), 0.0
    for cat, p in ARCHETYPE_ERRORS[arch].items():
        acc += p
        if r < acc:
            return cat
    return "correct_solution"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", type=int, default=24)
    parser.add_argument("--submissions", type=int, default=6)
    parser.add_argument("--practice", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    settings = get_settings()
    rng = np.random.default_rng(args.seed)
    fonts = [f for f in find_fonts() if Path(f).name in HANDWRITING_FONTS] or [None]
    upload_dir = Path(settings.UPLOAD_DIR) / "seed"
    upload_dir.mkdir(parents=True, exist_ok=True)
    bank = build_item_bank()
    b_true = true_item_difficulty(0)
    archetypes = list(ARCHETYPE_ERRORS)
    now = datetime.now(timezone.utc)

    try:
        if db.query(Student).filter(Student.grade_level == "Demo class").count():
            print("Demo class already seeded - skipping.")
            return
        for i in range(args.students):
            arch = archetypes[i % len(archetypes)]
            student = Student(name=NAMES[i % len(NAMES)], grade_level="Demo class")
            db.add(student)
            db.commit()
            font = fonts[i % len(fonts)]
            diagnosed = []
            for j in range(args.submissions):
                label = _sample_label(rng, arch)
                if label == "correct_solution" and rng.random() < 0.5:
                    gen = GENERATORS["correct_solution"][1]  # algebra
                else:
                    gen = GENERATORS[label][int(rng.integers(len(GENERATORS[label])))]
                sample = gen(rng)
                page = render_expression(font, sample.text.replace("*", "x"), rng)
                path = upload_dir / f"{student.id}_{j}.png"
                cv2.imwrite(str(path), page)
                question = sample.text.splitlines()[0].split("=")[0].strip() if "x" not in sample.text else sample.text.splitlines()[0]
                sub = Submission(student_id=student.id, subject="math", topic="algebra" if "x" in sample.text else "arithmetic",
                                 original_question=f"Solve: {question}", expected_answer=sample.expected_answer,
                                 provider_requested="ml", original_image_path=str(path))
                db.add(sub)
                db.commit()
                analysis = run_analysis_pipeline(db, sub, "ml")
                sub.created_at = now - timedelta(days=21 - 3 * j, hours=int(rng.integers(0, 8)))
                db.commit()
                diagnosed.append(analysis.mistake.category if analysis.mistake else "?")

            learner = SimLearner(i, np.random.default_rng(args.seed * 1000 + i), arch)
            for k in range(args.practice):
                item = bank[int(rng.integers(len(bank)))]
                correct = learner.answer(item.skill, b_true[item.id])
                db.add(PracticeAttempt(student_id=student.id, item_id=item.id, skill=item.skill,
                                       answer=item.answer if correct else "0", correct=bool(correct),
                                       created_at=now - timedelta(days=20 - k, hours=1)))
            db.commit()
            print(f"  {student.name:8s} ({arch:12s}) diagnosed: {', '.join(diagnosed)}")
        print("Demo class seeded.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
