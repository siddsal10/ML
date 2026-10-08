from __future__ import annotations

import uuid

from sqlalchemy import String, ForeignKey, Text, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Mistake(Base):
    """A diagnosed mistake category for a given analysis result."""

    __tablename__ = "mistakes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_results.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    category: Mapped[str] = mapped_column(String(60), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    evidence: Mapped[str] = mapped_column(JSON, nullable=False, default=list)
    suspected_step: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(
        String(30), nullable=False, default="deterministic"
    )  # deterministic | llm

    analysis: Mapped["AnalysisResult"] = relationship("AnalysisResult", back_populates="mistake")
