import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BACKEND_DIR / ".env")

STORAGE_ROOT = (BACKEND_DIR / os.getenv("UPLOAD_STORAGE_ROOT", "storage/uploads")).resolve()
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", str(BACKEND_DIR.parent / "dataset"))).resolve()

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}

CHUNK_SIZE = 1024 * 1024


async def save_upload_file(product_id: int, upload_file: UploadFile) -> tuple[str, Path]:
    original_name = upload_file.filename or ""
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Allowed formats: JPEG, PNG.",
        )

    product_dir = STORAGE_ROOT / str(product_id)
    product_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{extension}"
    absolute_path = product_dir / filename

    size = 0
    try:
        with open(absolute_path, "wb") as buffer:
            while True:
                chunk = await upload_file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"File exceeds maximum size of {MAX_UPLOAD_SIZE_MB} MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        absolute_path.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()

    if size == 0:
        absolute_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    if not _is_valid_image(absolute_path):
        absolute_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid JPEG or PNG image.",
        )

    relative_path = f"{product_id}/{filename}"
    return relative_path, absolute_path


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.format in ALLOWED_PIL_FORMATS
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def resolve_image_path(source_root: Path, relative_path: str) -> Path:
    resolved = (source_root / relative_path).resolve()
    if not resolved.is_relative_to(source_root):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return resolved
