from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator

from app.models.inspection import InspectionSource


class InspectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    product_id: int
    status: str
    source: InspectionSource
    dataset_category: Optional[str] = None
    dataset_split: Optional[str] = None
    dataset_defect_type: Optional[str] = None
    dataset_filename: Optional[str] = None
    inspection_date: datetime
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def extract_dataset_metadata(cls, data):
        """Derive safe dataset labels for the API response; the underlying
        filesystem path (data.image_path) is never serialized."""
        if isinstance(data, dict):
            return data

        fields = {
            "id": data.id,
            "product_id": data.product_id,
            "status": data.status,
            "source": data.source,
            "inspection_date": data.inspection_date,
            "created_at": data.created_at,
        }

        if data.source == InspectionSource.mvtec_ad and data.image_path:
            parts = data.image_path.split("/")
            if len(parts) == 4:
                category, split, defect_type, filename = parts
                fields.update(
                    dataset_category=category,
                    dataset_split=split,
                    dataset_defect_type=defect_type,
                    dataset_filename=filename,
                )

        return fields


class DatasetImportRequest(BaseModel):
    product_id: int
    category: str
    split: str
    defect_type: str
    filename: str
