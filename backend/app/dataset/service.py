import re
from pathlib import Path

from fastapi import HTTPException, status

from app.inspections.storage import DATASET_ROOT, resolve_image_path

VALID_SPLITS = ("train", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
EXCLUDED_CATEGORY_ENTRIES = {"license.txt", "readme.txt"}
SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_segment(value: str, field_name: str) -> None:
    if not value or ".." in value or not SEGMENT_PATTERN.match(value):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {field_name}.")


def list_categories() -> list[str]:
    """Live directory scan of DATASET_ROOT — never hardcoded."""
    if not DATASET_ROOT.is_dir():
        return []
    categories = []
    for entry in sorted(DATASET_ROOT.iterdir()):
        if not entry.is_dir() or entry.name in EXCLUDED_CATEGORY_ENTRIES:
            continue
        if (entry / "train").is_dir() and (entry / "test").is_dir():
            categories.append(entry.name)
    return categories


def get_category_dir(category: str) -> Path:
    _validate_segment(category, "category")
    if category not in list_categories():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found")
    return DATASET_ROOT / category


def get_split_dir(category: str, split: str) -> Path:
    if split not in VALID_SPLITS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Split must be 'train' or 'test'.")
    split_dir = get_category_dir(category) / split
    if not split_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Split not found")
    return split_dir


def list_defect_types(category: str, split: str) -> list[dict]:
    split_dir = get_split_dir(category, split)
    results = []
    for entry in sorted(split_dir.iterdir()):
        if entry.is_dir():
            count = sum(1 for f in entry.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            results.append({"defect_type": entry.name, "count": count})
    return results


def get_defect_type_dir(category: str, split: str, defect_type: str) -> Path:
    _validate_segment(defect_type, "defect_type")
    defect_dir = get_split_dir(category, split) / defect_type
    if not defect_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Defect type not found")
    return defect_dir


def list_images(category: str, split: str, defect_type: str) -> list[str]:
    defect_dir = get_defect_type_dir(category, split, defect_type)
    return sorted(f.name for f in defect_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_dataset_image(category: str, split: str, defect_type: str, filename: str) -> Path:
    """Resolve a dataset image path, validating every segment and containment under DATASET_ROOT."""
    _validate_segment(filename, "filename")
    get_defect_type_dir(category, split, defect_type)  # validates category/split/defect_type exist

    relative_path = f"{category}/{split}/{defect_type}/{filename}"
    resolved = resolve_image_path(DATASET_ROOT, relative_path)
    if not resolved.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return resolved


def build_dataset_relative_path(category: str, split: str, defect_type: str, filename: str) -> str:
    """Validate the requested dataset image and return its internal storage-relative path."""
    resolve_dataset_image(category, split, defect_type, filename)
    return f"{category}/{split}/{defect_type}/{filename}"
