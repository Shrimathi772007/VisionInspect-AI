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

    # Inspection performance instrumentation (Milestone 4) - measured wall-clock milliseconds
    # (see app.models.inspection for exactly what each one covers). NULL means "not
    # measured", e.g. every inspection recorded before this was added - never zero/estimated.
    processing_time_ms: Optional[float] = None
    ai_inference_time_ms: Optional[float] = None

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
            "processing_time_ms": data.processing_time_ms,
            "ai_inference_time_ms": data.ai_inference_time_ms,
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


# ---------------------------------------------------------------------------
# Production quality report (Milestone 3 Phase 4) - a read-only, structured presentation
# of one inspection's existing Phase 1-3 evidence. See app.inspections.report for the
# builder that populates these; these models only define the response shape. No new
# business logic or decision algorithm lives here - report.quality.decision and
# report.report_summary.overall_result are always exactly Inspection.quality_decision
# (Phase 3), never recomputed.
# ---------------------------------------------------------------------------


class InspectionReportSection(BaseModel):
    id: int
    product_id: int
    product_name: str
    status: str
    source: InspectionSource
    inspection_date: datetime
    created_at: datetime


class DatasetReportSection(BaseModel):
    """MVTec dataset reference for this inspection - present only for mvtec_ad inspections,
    None for generic uploads (never fabricated)."""

    category: str
    split: str
    defect_type: str
    filename: str


class DefectReportSection(BaseModel):
    # `category` is Phase 1 MVTec ground truth, never AI output - kept in its own field,
    # never labeled or presented as an "AI classification".
    category: Optional[str] = None
    ai_prediction: Optional[str] = None
    ai_reconstruction_error: Optional[float] = None
    ai_threshold: Optional[float] = None
    ai_model_name: Optional[str] = None


class SeverityReportSection(BaseModel):
    score: Optional[float] = None
    level: Optional[str] = None
    quality_risk: Optional[str] = None


class QualityReportSection(BaseModel):
    # decision is never NULL in the report even if Inspection.quality_decision is (a
    # pre-Phase-3 row) - see app.inspections.report.build_production_quality_report.
    decision: str
    assessment: Optional[str] = None
    recommendation: Optional[str] = None


class ReportSummary(BaseModel):
    # Always identical to `quality.decision` above - restated here only for a convenient
    # top-level summary, never an independently computed value.
    overall_result: str
    # Report completeness/availability, NOT a product quality outcome (see Phase 4 spec):
    # "COMPLETE" once a real quality assessment has run, "PARTIAL" for a pre-Phase-3 row
    # with no quality_decision yet.
    report_status: str
    summary: str


class ProductionQualityReport(BaseModel):
    inspection: InspectionReportSection
    dataset: Optional[DatasetReportSection] = None
    defect: DefectReportSection
    severity: SeverityReportSection
    quality: QualityReportSection
    report_summary: ReportSummary
