"""Inspection business logic that sits between the router and the AI/storage layers.

Home to best-effort AI inference, severity assessment, and quality assessment for an
already-created, already-committed inspection. Kept separate from app.ai.inference/
app.inspections.severity/app.inspections.quality so those packages stay framework/DB-free,
and separate from the router so the same logic is not duplicated across the upload and
import endpoints.
"""

import logging
import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.ai.inference import DEFECTIVE_PREDICTION, GOOD_PREDICTION, localization, predict_image
from app.ai.inference.serving import SERVING_CONFIGS
from app.ai.preprocessing.patchcore_preprocess import decode_image
from app.dataset.ground_truth import compute_defect_area_ratio
from app.inspections import quality, severity
from app.inspections.defect_form import add_defect_form
from app.inspections.storage import (
    DATASET_ROOT,
    HEATMAP_ROOT,
    STORAGE_ROOT,
    delete_heatmap_file,
    heatmap_absolute_path,
    heatmap_relative_path,
    resolve_image_path,
)
from app.models.inspection import Inspection, InspectionSource

logger = logging.getLogger(__name__)


def _resolve_category(inspection: Inspection) -> str | None:
    """The MVTec category whose AI model should score `inspection`, if there is one.

    - mvtec_ad imports: derived from image_path, which is
      "<category>/<split>/<defect_type>/<filename>" by construction (see
      app.dataset.service.build_dataset_relative_path). The dataset path always wins over
      the product's category - an import is scored by the model of the category it came from.
    - uploads: the category a quality engineer set on the inspection's product (validated
      against app.dataset.categories.MVTEC_CATEGORIES when it was set), or None when the
      product has none. Nothing is guessed from the image itself, so the result is only as
      right as that product setting: an upload of a different kind of object is still scored
      by the product's category model.

    The rule itself is resolve_category (also used by the by-category analytics, so both
    always agree); this wrapper only reads the product's category, and only for uploads.
    """
    product_category = None
    if inspection.source == InspectionSource.upload and inspection.product is not None:
        product_category = inspection.product.category
    return resolve_category(inspection.source, inspection.image_path, product_category)


def resolve_category(source: InspectionSource, image_path: str, product_category: str | None) -> str | None:
    """_resolve_category's rule on plain values: uploads -> `product_category`; mvtec_ad imports ->
    the first segment of a 4-part "<category>/<split>/<defect_type>/<filename>" image_path; else None."""
    if source == InspectionSource.upload:
        return product_category

    if source != InspectionSource.mvtec_ad:
        return None

    parts = image_path.split("/")
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
        result = predict_image(image_path, category, return_patch_scores=True)
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
    # The time predict_image itself measured for this call (model load + preprocessing +
    # inference + comparison with the category's configured locked threshold, which is looked
    # up, not derived) - recorded only on this success path, so it stays
    # NULL whenever no AI prediction actually ran.
    inspection.ai_inference_time_ms = result.processing_time_ms
    db.commit()
    db.refresh(inspection)

    # Committed above first, so nothing below can lose the AI result.
    apply_reliability_and_localization(inspection, db, result, category, image_path)


def _gate_for(category: str, model_name: str) -> str | None:
    """The static evidence gate of the registry entry that served this prediction (None when the
    category is not registered, or its entry is not the model that produced the result)."""
    config = SERVING_CONFIGS.get(category)
    if config is None or config.model_name != model_name:
        return None
    return config.gate


def _localize_and_render(inspection: Inspection, result, image_path: Path) -> tuple[dict | None, str | None]:
    """(localization, heatmap relative path) for a patch-model result; (None, None) when the result
    carries no patch grid (e.g. a ConvAE model). Writes storage/heatmaps/<inspection_id>.png."""
    if result.patch_scores is None or result.input_mode is None:
        return None, None
    height, width = decode_image(image_path).shape[:2]  # the same validated decode the model input came from
    amap = localization.anomaly_map(result.patch_scores, result.input_size[0])
    region = localization.map_region(result.input_mode, width, height)
    if result.prediction == DEFECTIVE_PREDICTION:
        located = localization.localize(amap, result.threshold, region)
    else:
        located = localization.empty_localization()
    located["analysed_region"] = [round(v, 6) for v in region]
    located["image_size"] = [width, height]
    if result.prediction == DEFECTIVE_PREDICTION:
        try:
            # Shape-based form of the region (defect_form_v1) - not a defect-type classifier, and never
            # written into defect_category (ground truth).
            add_defect_form(located, width, height)
        except Exception as exc:  # noqa: BLE001 - best effort, the localization itself is kept
            logger.warning("Defect form skipped for inspection %s: %s", inspection.id, exc)

    rgba = localization.render_heatmap(amap, result.threshold, width, height, region)
    HEATMAP_ROOT.mkdir(parents=True, exist_ok=True)
    relative_path = heatmap_relative_path(inspection.id)
    localization.write_png_atomic(rgba, heatmap_absolute_path(relative_path))
    return located, relative_path


def apply_reliability_and_localization(inspection: Inspection, db: Session, result, category: str,
                                       image_path: Path) -> None:
    """Best-effort margin confidence, manual-review flag, anomaly-map localization and heatmap for
    an inspection whose AI result was just stored (app.ai.inference.localization).

    Never raises and never touches ai_* / status: on any failure the affected new fields stay NULL
    (confidence/review and localization/heatmap fail independently) and the inspection is kept.
    """
    try:
        confidence = localization.margin_confidence(result.reconstruction_error, result.threshold)
        review = localization.review_rule(confidence, _gate_for(category, result.model_name))
        inspection.ai_confidence = confidence
        inspection.review_required = review.required
        inspection.review_reason = review.reason
    except Exception as exc:  # noqa: BLE001 - must never break inspection creation
        logger.warning("Confidence/review skipped for inspection %s: %s", inspection.id, exc)

    written = None
    try:
        located, written = _localize_and_render(inspection, result, image_path)
        inspection.localization = located
        inspection.heatmap_path = written
    except Exception as exc:  # noqa: BLE001 - must never break inspection creation
        logger.warning("Localization skipped for inspection %s: %s", inspection.id, exc)

    try:
        db.commit()
        db.refresh(inspection)
    except Exception as exc:  # noqa: BLE001 - must never break inspection creation
        db.rollback()
        logger.warning("Could not store localization for inspection %s: %s", inspection.id, exc)
        if written:
            try:
                delete_heatmap_file(written)
            except Exception:  # noqa: BLE001
                pass


def _ai_only_severity(inspection: Inspection) -> severity.SeverityAssessment | None:
    """severity_v1 (app.inspections.severity.assess_ai_severity) for an inspection WITHOUT ground truth
    whose AI prediction is defective and whose localization has regions and a defect form; None otherwise
    (imports with ground truth keep the ground-truth path, good predictions get no severity). Never raises."""
    if inspection.status in (GOOD_PREDICTION, DEFECTIVE_PREDICTION) or inspection.ai_prediction != DEFECTIVE_PREDICTION:
        return None
    located = inspection.localization if isinstance(inspection.localization, dict) else None
    if not located or not located.get("boxes") or located.get("defect_form_score") is None:
        return None
    try:
        return severity.assess_ai_severity(
            located.get("area_pct"), located.get("centroid"), located.get("defect_form_score"), inspection.ai_confidence
        )
    except Exception as exc:  # noqa: BLE001 - severity assessment must never break inspection creation
        logger.warning("AI severity skipped for inspection %s: %s", inspection.id, exc)
        return None


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

    Inspections WITHOUT ground truth (uploads) that the AI found defective and localized are instead
    scored by severity_v1 from their localization, defect form and confidence (see _ai_only_severity);
    for everything else this is exactly the ground-truth path above.
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
    ai_result = _ai_only_severity(inspection)
    if ai_result is not None:
        result = ai_result
    inspection.severity_score = result.score
    inspection.severity_level = result.level
    inspection.quality_risk = result.quality_risk
    db.commit()
    db.refresh(inspection)


def apply_quality_assessment(inspection: Inspection, db: Session) -> None:
    """Quality decision / assessment / recommendation for one already-persisted inspection.

    Must run after run_ai_inference and apply_severity_assessment so it can reason about
    their settled results (see app.inspections.quality's evidence-precedence rules) -
    reads `status`, `ai_prediction`, `defect_category`, and `severity_level`, but never
    writes to any of them: quality_decision/quality_assessment/quality_recommendation are
    a separate, fourth conclusion, not a restatement of any single existing field.

    Pure computation (app.inspections.quality.assess_quality never touches the filesystem,
    a model, or the database), so like apply_severity_assessment there is nothing here
    that can fail.
    """
    result = quality.assess_quality(
        status=inspection.status,
        ai_prediction=inspection.ai_prediction,
        defect_category=inspection.defect_category,
        severity_level=inspection.severity_level,
        review_required=inspection.review_required,
        severity_score=inspection.severity_score,
    )
    inspection.quality_decision = result.decision
    inspection.quality_assessment = result.assessment
    inspection.quality_recommendation = result.recommendation
    db.commit()
    db.refresh(inspection)


def record_processing_time(inspection: Inspection, db: Session, started_at: float) -> None:
    """Persist the measured server-side handling time of one inspection request.

    `started_at` is a time.perf_counter() reading taken by the endpoint handler when it
    began working on the inspection; the elapsed time to "now" is stored in milliseconds as
    Inspection.processing_time_ms. Called last, after quality assessment, so the figure
    covers everything the handler did for this inspection (see app.models.inspection for
    exactly what is and is not included).

    Never raises: like the other post-persist steps, a failure to record a timing is logged
    and swallowed rather than allowed to fail an inspection that is already safely saved -
    the column simply stays NULL ("not measured") rather than getting a made-up value.
    """
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    try:
        inspection.processing_time_ms = elapsed_ms
        db.commit()
        db.refresh(inspection)
    except Exception as exc:  # noqa: BLE001 - timing persistence must never break inspection creation
        db.rollback()
        logger.warning("Could not record processing time for inspection %s: %s", inspection.id, exc)
