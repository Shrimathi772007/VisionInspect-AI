from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ai.inference.localization import reliability_level
from app.ai.inference.serving import SERVING_CONFIGS
from app.inspections.defect_form import defect_form_label
from app.inspections.severity import recommended_action_for_level
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

    # The product's CURRENT MVTec category (read-only, from the related product - not stored on
    # the inspection). Lets a client tell "no category set" (None) apart from "category set but
    # no AI result" (e.g. no served model for it, or inference failed). Because it reflects the
    # product now, it can differ from the category that was in effect when this inspection was
    # scored.
    product_category: Optional[str] = None

    # Margin-based confidence, manual-review rule and anomaly-based defect localization (see
    # app.ai.inference.localization). NULL/None means "not computed" (no AI result, an older row,
    # or a best-effort failure).
    ai_confidence: Optional[float] = Field(
        default=None,
        description="sigmoid(10 * |ln(score / threshold)|), in [0.5, 1): how far the anomaly score is from "
        "the model's decision threshold. A margin-based heuristic, NOT a calibrated probability.",
    )
    ai_reliability: Optional[str] = Field(
        default=None,
        description='Display band of ai_confidence: "high" (>= 0.95), "medium" (0.70 to < 0.95), '
        '"low" (< 0.70, manual review).',
    )
    review_required: Optional[bool] = None
    review_reason: Optional[str] = None
    localization: Optional[dict[str, Any]] = Field(
        default=None,
        description='Anomaly-map localization ("anomaly_map_threshold_v1"): boxes derived from the anomaly '
        "heatmap (not an object detector), area_pct and centroid, in original-image coordinates normalised "
        "to 0..1 (origin top-left). Empty boxes for an AI \"good\" prediction.",
    )
    has_heatmap: bool = False
    defect_form: Optional[str] = Field(
        default=None,
        description="defect_form_v1 (multiple_regions / large_area / linear / small_spot / localized_patch): a "
        "shape-based form derived from the anomaly region; not a trained defect-type classifier. Only for AI-"
        "defective inspections with regions; never written into defect_category (ground truth).",
    )
    defect_form_label: Optional[str] = None
    severity_action: Optional[str] = Field(
        default=None,
        description="The specification's recommended action for severity_level (Critical: reject product and "
        "trigger quality inspection workflow; High: repair or rework; Medium: inspection review; Low: minor).",
    )
    model_gate: Optional[str] = Field(
        default=None,
        description="Static evidence gate (EXCELLENT / GOOD / ACCEPTABLE / NOT_PRODUCTION_READY) of the "
        "registered model that produced ai_prediction; None when there is no AI result or that model is no "
        "longer the category's registered model.",
    )

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
            "product_category": data.product.category if data.product is not None else None,
            "ai_confidence": data.ai_confidence,
            "ai_reliability": reliability_level(data.ai_confidence),
            "review_required": data.review_required,
            "review_reason": data.review_reason,
            "localization": data.localization,
            # heatmap_path itself is internal and never serialized.
            "has_heatmap": bool(data.heatmap_path),
            **defect_form_fields(data.localization),
            "severity_action": recommended_action_for_level(data.severity_level),
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

        fields["model_gate"] = model_gate_for(
            fields.get("dataset_category") if data.source == InspectionSource.mvtec_ad else fields["product_category"],
            data.ai_model_name,
        )
        return fields


def defect_form_fields(localization) -> dict:
    """defect_form / defect_form_label from a stored localization dict (both None when absent)."""
    form = localization.get("defect_form") if isinstance(localization, dict) else None
    return {"defect_form": form, "defect_form_label": defect_form_label(form)}


def model_gate_for(category: Optional[str], ai_model_name: Optional[str]) -> Optional[str]:
    """The registry gate of `category`'s served model, only if that model is the one that produced
    the stored prediction (`ai_model_name`); None otherwise (no AI result, no category, or the
    category is now served by a different model / the product's category changed)."""
    if not category or not ai_model_name:
        return None
    config = SERVING_CONFIGS.get(category)
    if config is None or config.model_name != ai_model_name:
        return None
    return config.gate


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
    # Same values and meaning as the InspectionOut fields of the same names (ai_confidence is a
    # margin-based heuristic, NOT a calibrated probability; boxes come from the anomaly heatmap).
    ai_confidence: Optional[float] = None
    ai_reliability: Optional[str] = None
    review_required: Optional[bool] = None
    review_reason: Optional[str] = None
    localization: Optional[dict[str, Any]] = None
    has_heatmap: bool = False
    model_gate: Optional[str] = None
    # Shape-based form of the anomaly region (defect_form_v1) - not a defect-type classification.
    defect_form: Optional[str] = None
    defect_form_label: Optional[str] = None


class SeverityReportSection(BaseModel):
    score: Optional[float] = None
    level: Optional[str] = None
    quality_risk: Optional[str] = None
    recommended_action: Optional[str] = None


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


class BatchUploadItem(BaseModel):
    """One file of POST /inspections/batch: its inspection (exactly as POST /upload returns it) or a short
    error message - never both, never a server path."""

    filename: str
    inspection: Optional[InspectionOut] = None
    error: Optional[str] = None


class BatchUploadOut(BaseModel):
    total: int
    succeeded: int
    failed: int
    items: list[BatchUploadItem]
