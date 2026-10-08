from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.ml.artifacts import SYMBOL_MODEL_PATH

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEMO = BACKEND_ROOT / "data" / "demo_samples"
needs_symbol_model = pytest.mark.skipif(not SYMBOL_MODEL_PATH.exists(), reason="run python -m app.ml.train")


def _student(client, name="Ada"):
    return client.post("/api/students", json={"name": name}).json()["id"]


def _png(text: str) -> bytes:
    img = Image.new("L", (520, 130), 255)
    ImageDraw.Draw(img).text((20, 25), text, fill=0, font=ImageFont.load_default(size=64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _analyze(client, sid, png: bytes, expected="3", question="Solve 2x + 4 = 10"):
    return client.post(
        "/api/submissions/analyze",
        data={"student_id": sid, "topic": "algebra", "original_question": question, "expected_answer": expected, "provider": "ml"},
        files={"image": ("w.png", png, "image/png")},
    )


def test_skills_endpoint_lists_eight_skills(client):
    assert len(client.get("/api/skills").json()) == 8


def test_practice_flow_updates_mastery(client):
    sid = _student(client)
    nxt = client.get(f"/api/practice/{sid}/next", params={"skill": "S3"}).json()
    assert nxt["skill"]["id"] == "S3" and len(nxt["items"]) == 3
    assert all("answer" not in it for it in nxt["items"])  # answers never leak

    from app.ml.irt.item_bank import item_by_id

    item = nxt["items"][0]
    right = client.post("/api/practice/answer", json={"student_id": sid, "item_id": item["id"], "answer": item_by_id()[item["id"]].answer}).json()
    assert right["correct"] is True and right["mastery_after"] > right["mastery_before"]
    wrong = client.post("/api/practice/answer", json={"student_id": sid, "item_id": nxt["items"][1]["id"], "answer": "99999"}).json()
    assert wrong["correct"] is False and wrong["mastery_after"] < right["mastery_after"]

    skills = client.get(f"/api/students/{sid}/skills").json()
    s3 = next(s for s in skills["skills"] if s["id"] == "S3")
    assert s3["attempts"] == 2 and len(skills["timeline"]) == 2

    again = client.get(f"/api/practice/{sid}/next", params={"skill": "S3"}).json()
    assert item["id"] not in {it["id"] for it in again["items"]}  # solved item excluded


def test_practice_rejects_unknown_skill_and_item(client):
    sid = _student(client)
    assert client.get(f"/api/practice/{sid}/next", params={"skill": "S99"}).status_code == 422
    assert client.post("/api/practice/answer", json={"student_id": sid, "item_id": "nope", "answer": "1"}).status_code == 404


@needs_symbol_model
def test_analysis_stores_symbols_and_corrections_reanalyze(client):
    sid = _student(client)
    res = _analyze(client, sid, (DEMO / "demo_algebra_error.png").read_bytes()
                   if (DEMO / "demo_algebra_error.png").exists() else _png("x = 7"))
    assert res.status_code == 200
    body = client.get(f"/api/submissions/{res.json()['submission_id']}/analysis").json()
    assert body["provider_used"] == "ml" and body["symbols"] and body["image_width"]
    assert {"ocr", "verify", "diagnose"} <= set(body["stage_timings"])
    assert body["skill"] in {"S5", "S6", "S7"}

    # Change the last symbol (the final answer digit) to "3": the work becomes correct.
    last = max(body["symbols"], key=lambda s: s["idx"])
    fixed = client.post(f"/api/submissions/{body['submission_id']}/symbols/corrections",
                        json={"corrections": [{"idx": last["idx"], "label": "3"}]}).json()
    assert fixed["extracted_text"].endswith("= 3")
    assert fixed["mistake_category"] == "correct_solution"
    assert any(s["corrected_label"] == "3" for s in fixed["symbols"])

    from app.ml import active_learning
    assert sum(active_learning.correction_counts().values()) == 1


@needs_symbol_model
def test_symbol_corrections_validate_input(client):
    sid = _student(client)
    sub = _analyze(client, sid, _png("1 + 1 = 2"), expected="2").json()["submission_id"]
    assert client.post(f"/api/submissions/{sub}/symbols/corrections", json={"corrections": [{"idx": 999, "label": "1"}]}).status_code == 422
    assert client.post(f"/api/submissions/{sub}/symbols/corrections", json={"corrections": [{"idx": 0, "label": "#"}]}).status_code == 422


def test_teacher_overview(client):
    for name in ("A", "B", "C", "D", "E"):
        sid = _student(client, name)
        client.post("/api/practice/answer", json={"student_id": sid, "item_id": "S1-000", "answer": "0"})
    data = client.get("/api/teacher/class").json()
    assert len(data["students"]) == 5 and len(data["skills"]) == 8
    assert all({"cluster", "x", "y", "mastery"} <= set(s) for s in data["students"])
    assert data["clusters"]


def test_lab_models_and_details(client):
    models = client.get("/api/lab/models").json()
    assert {m["name"] for m in models} == {"symbol_classifier", "mistake_classifier", "knowledge_tracing", "student_clustering"}
    detail = client.get("/api/lab/models/mistake_classifier").json()
    assert detail["title"] and "limitations" in detail
    assert client.get("/api/lab/models/nope").status_code == 404


@needs_symbol_model
def test_lab_playground_predicts_symbols(client):
    res = client.post("/api/lab/symbols/predict", files={"image": ("d.png", _png("7 + 8"), "image/png")}).json()
    assert len(res["symbols"]) == 3
    assert all(len(s["top"]) == 5 and s["model_view"].startswith("data:image/png") for s in res["symbols"])


def test_lab_dataset_preview(client):
    data = client.get("/api/lab/dataset").json()
    assert len(data["classes"]) == 17 and data["grid"].startswith("data:image/png")


def test_registry_register_and_activate(tmp_path, monkeypatch):
    from app.ml import artifacts, registry

    monkeypatch.setattr(registry, "MODEL_DIR", tmp_path)
    v1 = registry.register("toy", {"model": 1}, {"m": 0.5}, activate=True)
    v2 = registry.register("toy", {"model": 2}, {"m": 0.7})
    assert (v1, v2) == ("v1", "v2")
    assert registry.active_version("toy") == "v1"
    registry.activate_version("toy", "v2")
    assert artifacts.load_bundle(str(tmp_path / "toy.joblib"))["model"] == 2
    with pytest.raises(ValueError):
        registry.activate_version("toy", "v9")


@needs_symbol_model
def test_retrain_with_corrections_registers_inactive_version(tmp_path):
    import cv2
    import numpy as np

    from app.ml import active_learning, registry

    crop = np.full((40, 30), 255, np.uint8)
    cv2.line(crop, (15, 4), (15, 36), 0, 4)
    src = tmp_path / "crop.png"
    cv2.imwrite(str(src), crop)
    assert active_learning.save_correction(str(src), "1")
    assert active_learning.save_correction(str(src), "not-a-symbol") is None

    result = active_learning.retrain_symbol_model(samples_per_font=2, use_real=False)
    assert result["version"].startswith("v")
    assert result["comparison"]["corrections_in_sample"]["n"] == 1
    versions = registry.list_versions("symbol_classifier")
    assert versions["active"] != result["version"]  # never auto-activated
