"""Inspection business logic that sits between the router and the AI/storage layers.

Home to best-effort AI inference and severity assessment for an already-created,
already-committed inspection. Kept separate from app.ai.inference/app.inspections.severity
so those packages stay framework/DB-free, and separate from the router so the same logic
is not duplicated across the upload and import endpoints.
"""

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.ai.inference import predict_image
from app.dataset.ground_truth import compute_defect_area_ratio
from app.inspections import severity
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


def _resolve_mvtec_parts(inspection: Inspection) -> tuple[str, str, str] | None:
    """(category, defect_type, filename) for `inspection`, if safely derivable.

    Same reasoning/structure as _resolve_category above, extended to the two extra parts
    needed to look up this inspection's MVTec ground-truth mask (see
    app.dataset.ground_truth.compute_defect_area_ratio). Uploads have no such structure -
    returning None there is intentional, not a bug.
    """
    if inspection.source != InspectionSource.mvtec_ad:
        return None

    parts = inspection.image_path.split("/")
    if len(parts) != 4:
        return None

    category, _split, defect_type, filename = parts
    if not category or not defect_type or not filename:
        return None
    return category, defect_type, filename


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


def apply_severity_assessment(inspection: Inspection, db: Session) -> None:
    """Severity scoring / quality risk assessment for one already-persisted inspection.

    Resolves the one piece of real, filesystem-backed evidence this app currently has -
    the MVTec ground-truth defect mask area ratio (app.dataset.ground_truth), when this
    inspection has a derivable MVTec category/defect_type/filename - then hands plain
    values to app.inspections.severity.assess_severity, which stays pure and framework/DB-
    free. Never touches `status`, `ai_prediction`, or `defect_category` - those are
    separate concepts this function only ever reads, never writes.

    Never raises: like run_ai_inference, a missing/unreadable mask is treated as "no
    evidence" (see compute_defect_area_ratio) rather than a failure, so severity assessment
    can never block inspection creation.
    """
    defect_area_ratio = None
    mvtec_parts = _resolve_mvtec_parts(inspection)
    if mvtec_parts is not None:
        category, defect_type, filename = mvtec_parts
        try:
            defect_area_ratio = compute_defect_area_ratio(category, defect_type, filename)
        except Exception as exc:  # noqa: BLE001 - severity assessment must never break inspection creation
            logger.warning(
                "Defect area ratio unavailable for inspection %s (category=%r, defect_type=%r): %s",
                inspection.id,
                category,
                defect_type,
                exc,
            )

    result = severity.assess_severity(inspection.defect_category, defect_area_ratio=defect_area_ratio)
    inspection.severity_score = result.score
    inspection.severity_level = result.level
    inspection.quality_risk = result.quality_risk
    db.commit()
    db.refresh(inspection)
