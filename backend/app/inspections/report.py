"""Milestone 3 Phase 4: production quality reports.

Composes ONE already-persisted Inspection + its Product into a structured, read-only
production quality report. This is purely a presentation/reporting layer over existing
Phase 1-3 evidence - it implements no new business logic and no second quality-decision
algorithm. The authoritative decision remains exactly Inspection.quality_decision (Phase
3, app.inspections.quality); this module only ever reads it, never recomputes it.

Kept deliberately framework/DB-free (same convention as app.ai.inference/
app.inspections.severity/app.inspections.quality): the builder below takes plain ORM
objects the caller has already fetched and returns a plain Pydantic model (see
app.inspections.schemas for the response shape) - it does not query the database itself,
so it never fetches more than the one inspection/product the caller already has.
"""

from app.inspections.quality import NOT_ASSESSED
from app.inspections.schemas import (
    DatasetReportSection,
    DefectReportSection,
    InspectionReportSection,
    ProductionQualityReport,
    QualityReportSection,
    ReportSummary,
    SeverityReportSection,
)
from app.models.inspection import Inspection, InspectionSource
from app.models.product import Product

# Shown only when quality_assessment is NULL (a pre-Phase-3 row) - states plainly that no
# assessment has run, never a guessed summary of what it might have said.
_NOT_ASSESSED_SUMMARY = "Quality assessment has not been run for this inspection."


def _build_dataset_section(inspection: Inspection) -> DatasetReportSection | None:
    """The MVTec dataset reference for `inspection`, if derivable - same image_path
    parsing convention already used by app.inspections.schemas.InspectionOut and
    app.inspections.service. None for uploads (never fabricated)."""
    if inspection.source != InspectionSource.mvtec_ad:
        return None

    parts = inspection.image_path.split("/")
    if len(parts) != 4:
        return None

    category, split, defect_type, filename = parts
    return DatasetReportSection(category=category, split=split, defect_type=defect_type, filename=filename)


def build_production_quality_report(inspection: Inspection, product: Product) -> ProductionQualityReport:
    """Compose the existing Phase 1-3 evidence for one inspection into a structured report.

    `overall_result` is always exactly `inspection.quality_decision`, falling back to the
    existing NOT_ASSESSED vocabulary (never a new value) only when that field is NULL -
    e.g. an inspection created before Phase 3 existed, or one for which quality assessment
    was otherwise never run.
    """
    decision = inspection.quality_decision or NOT_ASSESSED
    report_status = "COMPLETE" if inspection.quality_decision is not None else "PARTIAL"

    return ProductionQualityReport(
        inspection=InspectionReportSection(
            id=inspection.id,
            product_id=inspection.product_id,
            product_name=product.product_name,
            status=inspection.status,
            source=inspection.source,
            inspection_date=inspection.inspection_date,
            created_at=inspection.created_at,
        ),
        dataset=_build_dataset_section(inspection),
        defect=DefectReportSection(
            category=inspection.defect_category,
            ai_prediction=inspection.ai_prediction,
            ai_reconstruction_error=inspection.ai_reconstruction_error,
            ai_threshold=inspection.ai_threshold,
            ai_model_name=inspection.ai_model_name,
        ),
        severity=SeverityReportSection(
            score=inspection.severity_score,
            level=inspection.severity_level,
            quality_risk=inspection.quality_risk,
        ),
        quality=QualityReportSection(
            decision=decision,
            assessment=inspection.quality_assessment,
            recommendation=inspection.quality_recommendation,
        ),
        report_summary=ReportSummary(
            overall_result=decision,
            report_status=report_status,
            summary=inspection.quality_assessment or _NOT_ASSESSED_SUMMARY,
        ),
    )
