"""Inspection business logic that sits between the router and the AI/storage layers.

Currently home to one thing: best-effort AI inference for an already-created,
already-committed inspection. Kept separate from app.ai.inference so that
package stays framework/DB-free, and separate from the router so the same
logic is not duplicated across the upload and import endpoints.
"""

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.ai.inference import predict_image
from app.inspections.storage import DATASET_ROOT, STORAGE_ROOT, resolve_image_path
from app.models.inspection import Inspection, InspectionSource

logger = logging.getLogger(__name__)


def _resolve_category(inspection: Inspection) -> str | None:
    """The MVTec category for `inspection`, if one is safely derivable.

    Only mvtec_ad inspections carry a recoverable category - their image_path
    is "<category>/<split>/<defect_type>/<filename>" by construction (see
    app.dataset.service.build_dataset_relative_path). Uploaded images have no
    such structure and no other field ties them to an MVTec category, so
    there is nothing to safely derive - returning None there is intentional,
    not a bug: this project deliberately does not guess or fabricate one.
    """
    if inspection.source != InspectionSource.mvtec_ad:
        return None

    parts = inspection.image_path.split("/")
    if len(parts) != 4:
        return None

    category = parts[0]
    return category or None


def _resolve_absolute_path(inspection: Inspection) -> Path:
    source_root = STORAGE_ROOT if inspection.source == InspectionSource.upload else DATASET_ROOT
    return resolve_image_path(source_root, inspection.image_path)


def run_ai_inference(inspection: Inspection, db: Session) -> None:
    """Best-effort AI prediction for one already-persisted inspection.

    Never raises: a missing category, missing model artifact, or any other
    inference problem is logged and left as a no-op with ai_* fields
    untouched (NULL, if they were already NULL). Never touches `status` or
    any MVTec ground-truth field - those are separate concepts and this
    function only ever writes to the dedicated ai_* columns.
    """
    category = _resolve_category(inspection)
    if category is None:
        return

    try:
        image_path = _resolve_absolute_path(inspection)
        result = predict_image(image_path, category)
    except Exception as exc:  # noqa: BLE001 - inference must never break inspection creation
        logger.warning(
            "AI inference skipped for inspection %s (category=%r): %s",
            inspection.id,
            category,
            exc,
        )
        return

    inspection.ai_prediction = result.prediction
    inspection.ai_reconstruction_error = result.reconstruction_error
    inspection.ai_threshold = result.threshold
    inspection.ai_model_name = result.model_name
    db.commit()
    db.refresh(inspection)
