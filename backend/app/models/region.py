from __future__ import annotations

import uuid

from sqlalchemy import String, ForeignKey, Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class ExtractedRegion(Base):
    """A detected content region within a submission image.

    Bounding boxes are stored as pixel coordinates in the *processed* image
    coordinate space: (x, y, width, height).
    """

    __tablename__ = "extracted_regions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    submission_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False
    )

    region_type: Mapped[str] = mapped_column(String(40), nullable=False)  # text | math | diagram
    x: Mapped[int] = mapped_column(Integer, nullable=False)
    y: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    extracted_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    submission: Mapped["Submission"] = relationship("Submission", back_populates="regions")
