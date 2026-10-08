"""Tables for the ML loop: recognized symbols (active learning) and practice attempts (knowledge tracing)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RecognizedSymbol(Base):
    """One symbol found by the ML recognizer, in original-image pixel coordinates."""

    __tablename__ = "recognized_symbols"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    submission_id: Mapped[str] = mapped_column(String(36), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False, index=True)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    line: Mapped[int] = mapped_column(Integer, nullable=False)
    x: Mapped[int] = mapped_column(Integer, nullable=False)
    y: Mapped[int] = mapped_column(Integer, nullable=False)
    w: Mapped[int] = mapped_column(Integer, nullable=False)
    h: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted: Mapped[str] = mapped_column(String(4), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    alternatives: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    corrected_label: Mapped[str | None] = mapped_column(String(4), nullable=True)
    crop_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(20), nullable=True)

    @property
    def label(self) -> str:
        return self.corrected_label or self.predicted


class PracticeAttempt(Base):
    __tablename__ = "practice_attempts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    student_id: Mapped[str] = mapped_column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    item_id: Mapped[str] = mapped_column(String(20), nullable=False)
    skill: Mapped[str] = mapped_column(String(4), nullable=False)
    answer: Mapped[str] = mapped_column(String(100), nullable=False)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    predicted_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
