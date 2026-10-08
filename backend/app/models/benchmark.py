from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, ForeignKey, Float, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    manifest_path: Mapped[str] = mapped_column(String(500), nullable=False)
    providers: Mapped[str] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="completed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    summary: Mapped[str] = mapped_column(JSON, nullable=False, default=dict)

    results: Mapped[list["BenchmarkResult"]] = relationship(
        "BenchmarkResult", back_populates="run", cascade="all, delete-orphan"
    )


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    sample_id: Mapped[str] = mapped_column(String(200), nullable=False)
    available: Mapped[bool] = mapped_column(default=True)
    predicted_text: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    reference_text: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    cer: Mapped[float | None] = mapped_column(Float, nullable=True)
    wer: Mapped[float | None] = mapped_column(Float, nullable=True)
    exact_match: Mapped[bool | None] = mapped_column(nullable=True)
    normalized_edit_distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    diagnosis_correct: Mapped[bool | None] = mapped_column(nullable=True)
    predicted_mistake_category: Mapped[str | None] = mapped_column(String(60), nullable=True)
    reference_mistake_category: Mapped[str | None] = mapped_column(String(60), nullable=True)

    run: Mapped["BenchmarkRun"] = relationship("BenchmarkRun", back_populates="results")
