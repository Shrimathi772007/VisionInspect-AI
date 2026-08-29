from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ProductCreate(BaseModel):
    product_name: str = Field(min_length=1, max_length=255)
    product_code: str = Field(min_length=1, max_length=100)


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    product_name: str
    product_code: str
    created_at: datetime
