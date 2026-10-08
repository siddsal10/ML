from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def temp_env(monkeypatch):
    """Point the app at a fresh temporary SQLite DB + upload dirs per test."""
    tmp_dir = tempfile.mkdtemp(prefix="ai_tutor_test_")
    db_path = Path(tmp_dir) / "test.db"
    upload_dir = Path(tmp_dir) / "uploads"
    processed_dir = Path(tmp_dir) / "processed"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("PROCESSED_DIR", str(processed_dir))
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("DEFAULT_OCR_PROVIDER", "demo")

    from app.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    settings.ensure_dirs()

    yield {"tmp_dir": tmp_dir, "db_path": db_path}

    get_settings.cache_clear()


@pytest.fixture(scope="function")
def db_session(temp_env):
    """Yields a fresh SQLAlchemy session backed by an isolated temp SQLite DB."""
    from app.db.database import get_session_factory, init_db

    init_db()
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(temp_env):
    """FastAPI TestClient wired to the isolated temp DB."""
    from app.db.database import init_db

    init_db()

    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """A tiny valid PNG for upload-validation tests."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (50, 50), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def isolate_ml_side_effects(monkeypatch, tmp_path):
    """Keep tests from writing symbol crops, corrections or registry files into the repo."""
    from app.ml import active_learning, registry

    monkeypatch.setattr(active_learning, "CROP_DIR", tmp_path / "symbol_crops")
    monkeypatch.setattr(active_learning, "LABELLED_DIR", tmp_path / "symbol_crops" / "labelled")
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(registry, "VERSIONS_DIR", tmp_path / "versions")
