from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False)
    product_code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    # The product's MVTec AD category (one of app.dataset.categories.MVTEC_CATEGORIES), which
    # selects the AI model for this product's UPLOADED images. NULL means "no category": uploads
    # for the product get no AI prediction. MVTec dataset imports ignore it - their category
    # always comes from the dataset path. Validated in the API layer, not by a DB constraint.
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    inspections: Mapped[list["Inspection"]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
