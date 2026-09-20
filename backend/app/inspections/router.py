import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user, require_role
from app.database import get_db
from app.dataset.service import build_dataset_relative_path
from app.inspections.analytics import (
    ALLOWED_WINDOW_DAYS,
    TREND_WINDOW_DAYS,
    InspectionAnalyticsSummary,
    get_inspection_analytics_summary,
)
from app.inspections.report import build_production_quality_report
from app.inspections.schemas import DatasetImportRequest, InspectionOut, ProductionQualityReport
from app.inspections.service import (
    apply_quality_assessment,
    apply_severity_assessment,
    record_processing_time,
    run_ai_inference,
)
from app.inspections.storage import (
    DATASET_ROOT,
    STORAGE_ROOT,
    delete_upload_file,
    resolve_image_path,
    save_upload_file,
)
from app.models.inspection import Inspection, InspectionSource
from app.models.product import Product
from app.models.user import User, UserRole

router = APIRouter(prefix="/inspections", tags=["inspections"])


@router.get("", response_model=list[InspectionOut])
def list_inspections(
    limit: int | None = Query(
        default=None,
        ge=1,
        le=1000,
        description="Return only the newest `limit` inspections (applied as a SQL LIMIT). "
        "Omit to return every inspection, as before.",
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = select(Inspection).order_by(Inspection.created_at.desc())
    if limit is not None:
        query = query.limit(limit)
    return db.execute(query).scalars().all()


@router.get("/analytics/summary", response_model=InspectionAnalyticsSummary)
def get_analytics_summary(
    days: int = Query(
        default=TREND_WINDOW_DAYS,
        description="Trailing window, in days, for the time-windowed sections (trend monitoring, "
        f"per-product recent trend, performance metrics). One of {list(ALLOWED_WINDOW_DAYS)}.",
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Database-aggregated inspection/AI metrics for the monitoring dashboard.

    Placed before the /{inspection_id} routes below so "analytics" is never
    mistaken for an inspection id.
    """
    try:
        return get_inspection_analytics_summary(db, window_days=days)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))


@router.post("/upload", response_model=InspectionOut, status_code=status.HTTP_201_CREATED)
async def upload_inspection(
    product_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.quality_engineer)),
):
    # Start of the server-side handling time recorded as Inspection.processing_time_ms (see
    # app.models.inspection for exactly what it does and does not cover).
    started_at = time.perf_counter()

    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    relative_path, absolute_path = await save_upload_file(product_id, file)

    try:
        inspection = Inspection(
            product_id=product_id,
            image_path=relative_path,
            source=InspectionSource.upload,
        )
        db.add(inspection)
        db.commit()
        db.refresh(inspection)
    except Exception:
        db.rollback()
        absolute_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create inspection record",
        )

    # Best-effort AI prediction, after the inspection is safely persisted - never allowed
    # to fail or roll back the creation above (see app.inspections.service.run_ai_inference).
    run_ai_inference(inspection, db)

    # Severity assessment runs after AI inference so it can see ai_prediction/defect_category
    # once those are settled - today it uses only defect_category (see app.inspections.severity
    # for why the other three factors are currently unavailable) and stays honestly
    # "not assessed" for uploads, which never have a defect_category.
    apply_severity_assessment(inspection, db)

    # Quality assessment runs last so it can reason about the settled ai_prediction/
    # severity_level - see app.inspections.quality for the evidence-precedence rules.
    apply_quality_assessment(inspection, db)

    # Last, so the recorded time covers everything above (see record_processing_time).
    record_processing_time(inspection, db, started_at)

    return inspection


@router.post("/import", response_model=InspectionOut, status_code=status.HTTP_201_CREATED)
def import_dataset_inspection(
    payload: DatasetImportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.quality_engineer)),
):
    started_at = time.perf_counter()  # see upload_inspection

    product = db.get(Product, payload.product_id)
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    # Validates the dataset reference and resolves it to an internal, storage-relative
    # path; the original file under DATASET_ROOT is never copied or modified.
    relative_path = build_dataset_relative_path(
        payload.category, payload.split, payload.defect_type, payload.filename
    )
    inspection_status = "good" if payload.defect_type == "good" else "defective"

    inspection = Inspection(
        product_id=payload.product_id,
        image_path=relative_path,
        source=InspectionSource.mvtec_ad,
        status=inspection_status,
        # MVTec ground-truth defect category, taken verbatim from the dataset's own
        # defect_type directory name (already validated by build_dataset_relative_path
        # above) - works for any MVTec category, not just bottle.
        defect_category=payload.defect_type,
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    # Best-effort AI prediction, after the inspection is safely persisted - never allowed
    # to fail or roll back the creation above (see app.inspections.service.run_ai_inference).
    run_ai_inference(inspection, db)

    # Severity assessment - see app.inspections.severity for the current all-four-required
    # policy; MVTec imports have a real defect_category but that alone is not sufficient.
    apply_severity_assessment(inspection, db)

    # Quality assessment runs last so it can reason about the settled ai_prediction/
    # severity_level - see app.inspections.quality for the evidence-precedence rules.
    apply_quality_assessment(inspection, db)

    # Last, so the recorded time covers everything above (see record_processing_time).
    record_processing_time(inspection, db, started_at)

    return inspection


@router.get("/{inspection_id}", response_model=InspectionOut)
def get_inspection(
    inspection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inspection not found")
    return inspection


@router.get("/{inspection_id}/report", response_model=ProductionQualityReport)
def get_inspection_report(
    inspection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Structured production quality report for one inspection (Milestone 3 Phase 4).

    Same authorization as GET /{inspection_id} - any authenticated user (both
    quality_engineer and factory_supervisor) may read it. Composes only this inspection's
    already-persisted data (see app.inspections.report) - no new query beyond the
    inspection and its related product, and no new quality-decision algorithm.
    """
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inspection not found")
    return build_production_quality_report(inspection, inspection.product)


@router.get("/{inspection_id}/image")
def get_inspection_image(
    inspection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inspection not found")

    source_root = STORAGE_ROOT if inspection.source == InspectionSource.upload else DATASET_ROOT
    image_path = resolve_image_path(source_root, inspection.image_path)

    if not image_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    return FileResponse(image_path)


@router.delete("/{inspection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_inspection(
    inspection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.quality_engineer)),
):
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inspection not found")

    # Only uploaded images live under STORAGE_ROOT and are owned by the app;
    # mvtec_ad inspections reference DATASET_ROOT and must never be touched.
    if inspection.source == InspectionSource.upload:
        delete_upload_file(inspection.image_path)

    db.delete(inspection)
    db.commit()
