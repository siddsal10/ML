from __future__ import annotations

import io

from PIL import Image


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), color="white").save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_and_get_student(client):
    resp = client.post("/api/students", json={"name": "Alice", "grade_level": "Grade 6"})
    assert resp.status_code == 201
    student = resp.json()
    assert student["name"] == "Alice"

    resp2 = client.get(f"/api/students/{student['id']}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == student["id"]


def test_get_nonexistent_student_returns_404(client):
    resp = client.get("/api/students/does-not-exist")
    assert resp.status_code == 404


def test_list_students(client):
    client.post("/api/students", json={"name": "Bob"})
    resp = client.get("/api/students")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) >= 1


def test_list_providers(client):
    resp = client.get("/api/providers")
    assert resp.status_code == 200
    names = {p["name"] for p in resp.json()}
    assert {"ml", "demo", "trocr", "donut", "vision_api"}.issubset(names)


def test_analyze_submission_end_to_end_demo_mode(client):
    student_resp = client.post("/api/students", json={"name": "Carol"})
    student_id = student_resp.json()["id"]

    files = {"image": ("test.png", _png_bytes(), "image/png")}
    data = {
        "student_id": student_id,
        "subject": "math",
        "topic": "arithmetic",
        "original_question": "What is 12 + 15?",
        "expected_answer": "27",
        "provider": "demo",
    }
    resp = client.post("/api/submissions/analyze", data=data, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_demo_mode"] is True
    assert "submission_id" in body
    assert body["provider_used"] == "demo"
    # Unknown image in demo mode -> honest warning, no fabricated confidence.
    assert body["ocr_confidence"] < 0.5


def test_analyze_submission_rejects_bad_extension(client):
    student_resp = client.post("/api/students", json={"name": "Dan"})
    student_id = student_resp.json()["id"]

    files = {"image": ("test.exe", b"not an image", "application/octet-stream")}
    data = {
        "student_id": student_id,
        "topic": "arithmetic",
        "original_question": "What is 1+1?",
    }
    resp = client.post("/api/submissions/analyze", data=data, files=files)
    assert resp.status_code == 422


def test_analyze_submission_unknown_student_returns_404(client):
    files = {"image": ("test.png", _png_bytes(), "image/png")}
    data = {
        "student_id": "does-not-exist",
        "topic": "arithmetic",
        "original_question": "What is 1+1?",
    }
    resp = client.post("/api/submissions/analyze", data=data, files=files)
    assert resp.status_code == 404


def test_reanalyze_with_corrected_transcription(client):
    student_resp = client.post("/api/students", json={"name": "Eve"})
    student_id = student_resp.json()["id"]

    files = {"image": ("test.png", _png_bytes(), "image/png")}
    data = {
        "student_id": student_id,
        "topic": "arithmetic",
        "original_question": "What is 12 + 15?",
        "expected_answer": "27",
        "provider": "demo",
    }
    analyze_resp = client.post("/api/submissions/analyze", data=data, files=files)
    submission_id = analyze_resp.json()["submission_id"]

    reanalyze_resp = client.post(
        f"/api/submissions/{submission_id}/reanalyze",
        json={"corrected_transcription": "12 + 15 = 27"},
    )
    assert reanalyze_resp.status_code == 200
    body = reanalyze_resp.json()
    assert body["correctness"] is True
    assert body["mistake_category"] == "correct_solution"


def test_quiz_generate_and_submit_flow(client):
    student_resp = client.post("/api/students", json={"name": "Frank"})
    student_id = student_resp.json()["id"]

    gen_resp = client.post(
        "/api/quizzes/generate",
        json={"student_id": student_id, "topic": "arithmetic", "weak_concept": None},
    )
    assert gen_resp.status_code == 200
    quiz = gen_resp.json()
    assert len(quiz["questions"]) == 3

    # Fetch full quiz (with expected answers not exposed) then submit garbage
    # answers - grading endpoint should still succeed and return a score.
    answers = [{"question_id": q["id"], "submitted_answer": "0"} for q in quiz["questions"]]
    submit_resp = client.post(f"/api/quizzes/{quiz['id']}/submit", json={"answers": answers})
    assert submit_resp.status_code == 200
    result = submit_resp.json()
    assert result["total_questions"] == 3
    assert 0.0 <= result["score"] <= 1.0


def test_quiz_cannot_be_submitted_twice(client):
    student_resp = client.post("/api/students", json={"name": "Grace"})
    student_id = student_resp.json()["id"]
    gen_resp = client.post(
        "/api/quizzes/generate", json={"student_id": student_id, "topic": "arithmetic"}
    )
    quiz = gen_resp.json()
    answers = [{"question_id": q["id"], "submitted_answer": "0"} for q in quiz["questions"]]
    client.post(f"/api/quizzes/{quiz['id']}/submit", json={"answers": answers})

    second_resp = client.post(f"/api/quizzes/{quiz['id']}/submit", json={"answers": answers})
    assert second_resp.status_code == 400


def test_progress_endpoint_returns_defaults_for_new_student(client):
    student_resp = client.post("/api/students", json={"name": "Henry"})
    student_id = student_resp.json()["id"]

    resp = client.get(f"/api/students/{student_id}/progress")
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_mastery"] == 0.0
    assert body["topic_mastery"] == []


def test_history_endpoint(client):
    student_resp = client.post("/api/students", json={"name": "Ivy"})
    student_id = student_resp.json()["id"]

    resp = client.get(f"/api/students/{student_id}/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["submissions"] == []
    assert body["quizzes"] == []
