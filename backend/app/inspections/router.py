from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user, require_role
from app.database import get_db
from app.inspections.schemas import InspectionOut
from app.inspections.storage import DATASET_ROOT, STORAGE_ROOT, resolve_image_path, save_upload_file
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
