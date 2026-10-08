from __future__ import annotations

import io

import pytest
from fastapi import UploadFile

from app.utils.file_utils import UploadValidationError, sanitize_filename, validate_upload


def _make_upload(filename: str) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(b""))


def test_sanitize_filename_strips_path_traversal():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("../../../evil.png") == "evil.png"


def test_sanitize_filename_strips_unsafe_characters():
    result = sanitize_filename("my photo!@#.png")
    assert " " not in result
    assert "!" not in result


def test_validate_upload_rejects_bad_extension(temp_env, sample_image_bytes):
    upload = _make_upload("malware.exe")
    with pytest.raises(UploadValidationError):
        validate_upload(upload, sample_image_bytes)


def test_validate_upload_rejects_empty_file(temp_env):
    upload = _make_upload("photo.png")
    with pytest.raises(UploadValidationError):
        validate_upload(upload, b"")


def test_validate_upload_rejects_mismatched_content(temp_env):
    upload = _make_upload("photo.png")
    with pytest.raises(UploadValidationError):
        validate_upload(upload, b"not actually a png")


def test_validate_upload_accepts_valid_png(temp_env, sample_image_bytes):
    upload = _make_upload("photo.png")
    validate_upload(upload, sample_image_bytes)  # should not raise


def test_validate_upload_rejects_oversized_file(temp_env, monkeypatch):
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
    get_settings.cache_clear()

    upload = _make_upload("photo.png")
    big_content = b"\x89PNG\r\n\x1a\n" + b"0" * (2 * 1024 * 1024)
    with pytest.raises(UploadValidationError):
        validate_upload(upload, big_content)
