import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.auth.dependencies import get_current_user, require_role
from app.database import get_db
from app.dataset.service import build_dataset_relative_path
from app.inspections.analytics import (
    ALLOWED_WINDOW_DAYS,
    TREND_WINDOW_DAYS,
    CategoryAnalytics,
    InspectionAnalyticsSummary,
    get_category_analytics,
    get_inspection_analytics_summary,
)
from app.inspections.report import build_production_quality_report
from app.ai.analytics import enhancement
from app.ai.preprocessing.patchcore_preprocess import decode_image
from app.inspections.schemas import (
    BatchUploadItem,
    BatchUploadOut,
    DatasetImportRequest,
    InspectionOut,
    ProductionQualityReport,
)
from app.inspections.service import (
    apply_quality_assessment,
    apply_severity_assessment,
    record_processing_time,
    run_ai_inference,
)
from app.inspections.storage import (
    DATASET_ROOT,
    HEATMAP_ROOT,
    STORAGE_ROOT,
    delete_heatmap_file,
    delete_upload_file,
    resolve_image_path,
    save_upload_file,
)
from app.models.inspection import Inspection, InspectionSource
from app.models.product import Product
from app.models.user import User, UserRole

logger = logging.getLogger(__name__)

# POST /batch limits (each file is also held to the single-upload size limit).
MAX_BATCH_FILES = 20
MAX_BATCH_BYTES = 50 * 1024 * 1024

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
    # selectinload: InspectionOut.product_category reads each inspection's product; load them
    # all in one extra query instead of one lazy load per inspection.
    query = select(Inspection).options(selectinload(Inspection.product)).order_by(Inspection.created_at.desc())
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


@router.get("/analytics/by-category", response_model=CategoryAnalytics)
def get_analytics_by_category(
    days: int = Query(
        default=TREND_WINDOW_DAYS,
        description=f"Trailing window, in days. One of {list(ALLOWED_WINDOW_DAYS)} (same as the summary).",
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Per-category counts for all 15 MVTec categories over the last `days` days: inspections, AI-analysed,
    AI defective/good, defect rate, review_required and quality decisions.

    Read-only. Derived from AI predictions, not ground truth: defect_rate is the share of AI-analysed
    inspections the AI predicted defective. Categories are resolved like the AI path (imports from the
    dataset path, uploads from the product's category). Placed before the /{inspection_id} routes.
    """
    try:
        return get_category_analytics(db, window_days=days)
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

    return await _process_upload(db, product_id, file, started_at)


async def _process_upload(db: Session, product_id: int, file: UploadFile, started_at: float) -> Inspection:
    """The single-image upload pipeline, shared by POST /upload and POST /batch: validate + store the file,
    create the row, then best-effort AI inference (with localization), severity, quality and timing.
    Raises HTTPException (400/413 for a bad file, 500 if the row cannot be created) - never with a path."""
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

    # Best-effort AI prediction above uses the product's MVTec category (None -> no prediction);
    # see app.inspections.service._resolve_category.
    #
    # Severity assessment runs after AI inference so it can see ai_prediction/localization/
    # ai_confidence once those are settled - an upload the AI found defective and localized gets
    # severity_v1 (see app.inspections.severity); anything else stays honestly "not assessed".
    apply_severity_assessment(inspection, db)

    # Quality assessment runs last so it can reason about the settled ai_prediction/
    # severity_level - see app.inspections.quality for the evidence-precedence rules.
    apply_quality_assessment(inspection, db)

    # Last, so the recorded time covers everything above (see record_processing_time).
    record_processing_time(inspection, db, started_at)

    return inspection


@router.post("/batch", response_model=BatchUploadOut)
async def batch_upload_inspections(
    product_id: int = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.quality_engineer)),
):
    """Upload up to 20 images (50 MB in total) for one product. Each file goes, one after another, through
    exactly the single-upload pipeline (validation, storage, AI with localization, severity, quality). A bad
    file is reported in its own item and never stops the others; errors are short and never contain paths."""
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files: at most {MAX_BATCH_FILES} per batch.",
        )
    if sum(_upload_size(f) for f in files) > MAX_BATCH_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Batch exceeds the maximum total size of {MAX_BATCH_BYTES // (1024 * 1024)} MB.",
        )
    if db.get(Product, product_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    items = []
    for upload_file in files:
        filename = Path(upload_file.filename or "").name[:255] or "unnamed"
        try:
            inspection = await _process_upload(db, product_id, upload_file, time.perf_counter())
            items.append(BatchUploadItem(filename=filename, inspection=InspectionOut.model_validate(inspection)))
        except HTTPException as exc:
            message = exc.detail if isinstance(exc.detail, str) else "Upload failed."
            items.append(BatchUploadItem(filename=filename, error=message))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            db.rollback()
            logger.warning("Batch upload item failed for product %s: %s", product_id, type(exc).__name__)
            items.append(BatchUploadItem(filename=filename, error="Upload failed."))

    succeeded = sum(1 for item in items if item.inspection is not None)
    return BatchUploadOut(total=len(items), succeeded=succeeded, failed=len(items) - succeeded, items=items)


def _upload_size(upload_file: UploadFile) -> int:
    if upload_file.size is not None:
        return upload_file.size
    handle = upload_file.file
    position = handle.tell()
    handle.seek(0, 2)
    size = handle.tell()
    handle.seek(position)
    return size


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


def _load_inspection_image(inspection_id: int, db: Session):
    """The inspection's original image (BGR), resolved exactly like GET /{id}/image (containment-checked).
    404 when the inspection or its file is missing or the file cannot be decoded."""
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inspection not found")
    source_root = STORAGE_ROOT if inspection.source == InspectionSource.upload else DATASET_ROOT
    image_path = resolve_image_path(source_root, inspection.image_path)
    if not image_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    try:
        return decode_image(image_path)
    except Exception:  # noqa: BLE001 - unreadable file: same answer as a missing one, no details
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")


@router.get("/{inspection_id}/enhanced")
def get_inspection_enhanced_preview(
    inspection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Denoised + contrast-enhanced PNG preview of the inspection image (long side <= 640 px). Preview
    only. The AI models do not use the enhanced image; nothing is stored."""
    image = _load_inspection_image(inspection_id, db)
    return Response(content=enhancement.encode_png(enhancement.enhance(image)), media_type="image/png")


@router.get("/{inspection_id}/image-quality")
def get_inspection_image_quality(
    inspection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Before/after image-quality metrics of the enhancement preview (sharpness, contrast, noise estimate,
    brightness) and the resolution. Preview only. The AI models do not use the enhanced image."""
    return enhancement.quality_report(_load_inspection_image(inspection_id, db))


@router.get("/{inspection_id}/heatmap")
def get_inspection_heatmap(
    inspection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """The anomaly heatmap of one inspection: an RGBA PNG in the original image's geometry (at most
    320 px on the long side), to be stretched over the image. Any authenticated user; 404 when the
    inspection has no heatmap. The stored path is resolved inside storage/heatmaps only."""
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inspection not found")
    if not inspection.heatmap_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Heatmap not found")

    heatmap_path = resolve_image_path(HEATMAP_ROOT, inspection.heatmap_path)
    if not heatmap_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Heatmap not found")

    return FileResponse(heatmap_path, media_type="image/png")


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

    heatmap_path = inspection.heatmap_path
    db.delete(inspection)
    db.commit()

    # Best effort, after the row is gone: the heatmap is app-owned and only ever resolved inside
    # storage/heatmaps (for imports too - the dataset image itself is never touched).
    try:
        delete_heatmap_file(heatmap_path)
    except Exception as exc:  # noqa: BLE001 - a leftover heatmap must never fail the delete
        logger.warning("Could not delete heatmap of inspection %s: %s", inspection_id, exc)
