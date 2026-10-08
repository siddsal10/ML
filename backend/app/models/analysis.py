from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, ForeignKey, Text, Boolean, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisResult(Base):
    """Result of running the math-understanding + diagnosis pipeline once.

    A submission can be re-analyzed (e.g. after the student edits the OCR
    transcription), so a submission may have multiple AnalysisResult rows;
    the most recent one is treated as authoritative.
    """

    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    submission_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False
    )

    interpreted_steps: Mapped[str] = mapped_column(JSON, nullable=False, default=list)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    canonical_solution: Mapped[str] = mapped_column(JSON, nullable=False, default=list)

    corrected_solution_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    hint_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    prerequisite_concept: Mapped[str | None] = mapped_column(String(200), nullable=True)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    warnings: Mapped[str] = mapped_column(JSON, nullable=False, default=list)
    recommended_next_action: Mapped[str | None] = mapped_column(String(200), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    submission: Mapped["Submission"] = relationship("Submission", back_populates="analyses")
    mistake: Mapped["Mistake | None"] = relationship(
        "Mistake", back_populates="analysis", uselist=False, cascade="all, delete-orphan"
    )
