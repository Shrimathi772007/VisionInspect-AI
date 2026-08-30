import io
from pathlib import Path

from PIL import Image

DEFAULT_MAX_DIM = 320
MIN_MAX_DIM = 64
MAX_MAX_DIM = 800


def generate_preview_bytes(path: Path, max_dim: int = DEFAULT_MAX_DIM) -> bytes:
    """Basic preprocessing for display: RGB-normalize and bound the image to max_dim, as a JPEG."""
    clamped_dim = max(MIN_MAX_DIM, min(max_dim, MAX_MAX_DIM))
    with Image.open(path) as img:
        rgb_image = img.convert("RGB")
        rgb_image.thumbnail((clamped_dim, clamped_dim))
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
