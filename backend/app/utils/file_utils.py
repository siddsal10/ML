"""Upload validation and safe filesystem helpers."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings


class UploadValidationError(ValueError):
    pass


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_filename(filename: str) -> str:
    """Strip any path components and disallowed characters from a filename."""
    name = Path(filename).name  # drops any directory components -> prevents path traversal
    name = _SAFE_NAME_RE.sub("_", name)
    return name or "upload"


def validate_upload(file: UploadFile, content: bytes) -> None:
    settings = get_settings()

    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file extension '{ext}'. Allowed: {settings.ALLOWED_UPLOAD_EXTENSIONS}"
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise UploadValidationError(f"File too large (max {settings.MAX_UPLOAD_SIZE_MB} MB).")

    if len(content) == 0:
        raise UploadValidationError("Uploaded file is empty.")

    # Lightweight magic-byte sniffing to catch mislabeled files.
    signatures = {
        b"\xff\xd8\xff": ".jpg",
        b"\x89PNG\r\n\x1a\n": ".png",
        b"RIFF": ".webp",  # WEBP starts with RIFF....WEBP
    }
    if not any(content.startswith(sig) for sig in signatures):
        raise UploadValidationError("File content does not match a supported image format.")


def build_upload_path(student_id: str, original_filename: str) -> Path:
    settings = get_settings()
    safe_name = sanitize_filename(original_filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    # student_id is a UUID we generate ourselves, but sanitize defensively anyway.
    safe_student_id = _SAFE_NAME_RE.sub("_", student_id)
    directory = Path(settings.UPLOAD_DIR) / safe_student_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / unique_name


def build_processed_path(upload_path: Path) -> Path:
    settings = get_settings()
    relative = upload_path.relative_to(Path(get_settings().UPLOAD_DIR))
    processed = Path(settings.PROCESSED_DIR) / relative
    processed.parent.mkdir(parents=True, exist_ok=True)
    return processed.with_suffix(".png")
