import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class InspectionSource(str, enum.Enum):
    upload = "upload"
    mvtec_ad = "mvtec_ad"


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False, index=True)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    source: Mapped[InspectionSource] = mapped_column(
        Enum(InspectionSource, name="inspection_source"),
        nullable=False,
        server_default=InspectionSource.upload.value,
    )
    inspection_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # AI prediction (Phase 7) - populated by app.inspections.service.run_ai_inference from
    # app.ai.inference.predict_image. Independent of `status`/MVTec ground truth: NULL means
    # "no AI result available yet" (old record, unsupported category, or inference failure),
    # never a stand-in for a ground-truth or workflow value.
    ai_prediction: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    ai_reconstruction_error: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ai_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ai_model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    product: Mapped["Product"] = relationship(back_populates="inspections")
    defects: Mapped[list["Defect"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan"
    )
