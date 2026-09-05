from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user, require_role
from app.database import get_db
from app.dataset.service import build_dataset_relative_path
from app.inspections.analytics import InspectionAnalyticsSummary, get_inspection_analytics_summary
from app.inspections.schemas import DatasetImportRequest, InspectionOut
from app.inspections.service import run_ai_inference
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.execute(select(Inspection).order_by(Inspection.created_at.desc())).scalars().all()


@router.get("/analytics/summary", response_model=InspectionAnalyticsSummary)
def get_analytics_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Database-aggregated inspection/AI metrics for the monitoring dashboard.

    Placed before the /{inspection_id} routes below so "analytics" is never
    mistaken for an inspection id.
    """
    return get_inspection_analytics_summary(db)


@router.post("/upload", response_model=InspectionOut, status_code=status.HTTP_201_CREATED)
async def upload_inspection(
    product_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.quality_engineer)),
):
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

    return inspection


@router.post("/import", response_model=InspectionOut, status_code=status.HTTP_201_CREATED)
def import_dataset_inspection(
    payload: DatasetImportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.quality_engineer)),
):
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
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    # Best-effort AI prediction, after the inspection is safely persisted - never allowed
    # to fail or roll back the creation above (see app.inspections.service.run_ai_inference).
    run_ai_inference(inspection, db)

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
