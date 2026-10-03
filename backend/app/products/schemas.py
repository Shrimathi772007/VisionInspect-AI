from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.dataset.categories import MvtecCategory


class ProductCreate(BaseModel):
    product_name: str = Field(min_length=1, max_length=255)
    product_code: str = Field(min_length=1, max_length=100)
    # Optional MVTec category (exact lowercase dataset folder name, e.g. "metal_nut"); any
    # other value is rejected with 422. Omitted -> NULL, i.e. no AI prediction for uploads.
    category: MvtecCategory | None = None


class ProductCategoryUpdate(BaseModel):
    # Required key; null clears the category. Only affects inspections created afterwards.
    category: MvtecCategory | None


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    product_name: str
    product_code: str
    category: str | None = None
    created_at: datetime
