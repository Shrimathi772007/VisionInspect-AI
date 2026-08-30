from fastapi import APIRouter, Depends, Query, Response

from app.auth.dependencies import get_current_user
from app.dataset import service
from app.dataset.preprocessing import generate_preview_bytes
from app.dataset.schemas import CategoryDetail, CategoryImages, SplitSummary
from app.models.user import User

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/categories", response_model=list[str])
def get_categories(current_user: User = Depends(get_current_user)):
    return service.list_categories()


@router.get("/categories/{category}", response_model=CategoryDetail)
def get_category_detail(category: str, current_user: User = Depends(get_current_user)):
    splits = [
        SplitSummary(split=split, defect_types=service.list_defect_types(category, split))
        for split in service.VALID_SPLITS
    ]
    return CategoryDetail(category=category, splits=splits)


@router.get("/categories/{category}/images", response_model=CategoryImages)
def get_category_images(
    category: str,
    split: str = Query(...),
    defect_type: str = Query(...),
    current_user: User = Depends(get_current_user),
):
    filenames = service.list_images(category, split, defect_type)
    return CategoryImages(category=category, split=split, defect_type=defect_type, filenames=filenames)


@router.get("/categories/{category}/preview")
def get_preview(
    category: str,
    split: str = Query(...),
    defect_type: str = Query(...),
    filename: str = Query(...),
    max_dim: int = Query(320, ge=64, le=800),
    current_user: User = Depends(get_current_user),
):
    image_path = service.resolve_dataset_image(category, split, defect_type, filename)
    data = generate_preview_bytes(image_path, max_dim)
    return Response(content=data, media_type="image/jpeg")
