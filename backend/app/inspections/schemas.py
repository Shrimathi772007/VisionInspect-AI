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

    # Defect categorization (Milestone 3 Phase 1) - the known defect category/type, kept
    # strictly separate from `status` (business ground truth) and `ai_prediction` (the
    # anomaly detector's guess, which cannot classify defect types). NULL/unknown when no
    # category is known, e.g. a generic user upload.
    defect_category: Optional[str] = None

    # AI prediction (Phase 7) - from app.ai.inference.predict_image, kept strictly separate
    # from `status`/dataset_defect_type (MVTec ground truth). NULL when no AI result is
    # available (old record, unsupported category, or inference failure).
    ai_prediction: Optional[str] = None
    ai_reconstruction_error: Optional[float] = None
    ai_threshold: Optional[float] = None
    ai_model_name: Optional[str] = None

    # Severity scoring / quality risk assessment (Milestone 3 Phase 2) - from
    # app.inspections.severity, kept strictly separate from `status`, `defect_category`,
    # and `ai_prediction`. NULL/"Not assessed" when current evidence is insufficient -
    # never a fabricated score (see app.inspections.severity for why).
    severity_score: Optional[float] = None
    severity_level: Optional[str] = None
    quality_risk: Optional[str] = None

    # Quality assessment (Milestone 3 Phase 3) - from app.inspections.quality, a fourth
    # conclusion derived from (not a copy of) status/ai_prediction/defect_category/
    # severity_level. NULL only for rows created before this phase existed.
    quality_decision: Optional[str] = None
    quality_assessment: Optional[str] = None
    quality_recommendation: Optional[str] = None

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
            "defect_category": data.defect_category,
            "ai_prediction": data.ai_prediction,
            "ai_reconstruction_error": data.ai_reconstruction_error,
            "ai_threshold": data.ai_threshold,
            "ai_model_name": data.ai_model_name,
            "severity_score": data.severity_score,
            "severity_level": data.severity_level,
            "quality_risk": data.quality_risk,
            "quality_decision": data.quality_decision,
            "quality_assessment": data.quality_assessment,
            "quality_recommendation": data.quality_recommendation,
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
