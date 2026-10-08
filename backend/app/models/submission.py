from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )

    subject: Mapped[str] = mapped_column(String(80), nullable=False, default="math")
    topic: Mapped[str] = mapped_column(String(120), nullable=False)
    original_question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    provider_requested: Mapped[str | None] = mapped_column(String(40), nullable=True)
    provider_used: Mapped[str | None] = mapped_column(String(40), nullable=True)

    original_image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    processed_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    raw_transcription: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_transcription: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(nullable=True)

    is_demo_mode: Mapped[bool] = mapped_column(default=False)
    skill: Mapped[str | None] = mapped_column(String(4), nullable=True)
    stage_timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    student: Mapped["Student"] = relationship("Student", back_populates="submissions")
    regions: Mapped[list["ExtractedRegion"]] = relationship(
        "ExtractedRegion", back_populates="submission", cascade="all, delete-orphan"
    )
    analyses: Mapped[list["AnalysisResult"]] = relationship(
        "AnalysisResult", back_populates="submission", cascade="all, delete-orphan"
    )
