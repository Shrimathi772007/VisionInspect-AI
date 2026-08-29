from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.inspection import InspectionSource


class InspectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    product_id: int
    status: str
    source: InspectionSource
    inspection_date: datetime
    created_at: datetime
