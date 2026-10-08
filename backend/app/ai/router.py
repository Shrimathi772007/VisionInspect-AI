"""Read-only view of the AI models served per MVTec category (GET /ai/models).

Exposes only what a user needs to judge an AI prediction: which model serves a category, its input mode,
its locked threshold, its evidence gate and its static final-test metrics. Artifact paths, provenance and
hashes stay server-side.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.ai.box_eval import box_eval_for
from app.ai.inference.serving import SERVING_CONFIGS, CategoryServingConfig
from app.auth.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/ai", tags=["ai"])


class ServedModelOut(BaseModel):
    category: str
    model_name: str
    family: str
    # WRN-50: crop224 / full256 / full320. Other families: "resize" (whole image resized to input_size).
    input_mode: str
    input_size: tuple[int, int]
    threshold: float
    gate: str | None
    final_test_recall: float | None
    final_test_fpr: float | None
    final_test_auroc: float | None
    final_test_average_precision: float | None
    # Localization evaluation addendum (app/ai/box_eval_addendum.json): a second scoring of the final test set for
    # localization only, with anomaly-map boxes (not a trained detector). None when the file or category is missing.
    box_ap50: float | None
    box_ap50_merged: float | None
    pixel_auroc: float | None


def _served_model(config: CategoryServingConfig) -> ServedModelOut:
    return ServedModelOut(
        category=config.category,
        model_name=config.model_name,
        family=config.model_family,
        input_mode=config.input_mode or "resize",
        input_size=tuple(config.input_size),
        threshold=config.threshold,
        gate=config.gate,
        final_test_recall=config.final_test_recall,
        final_test_fpr=config.final_test_fpr,
        final_test_auroc=config.final_test_auroc,
        final_test_average_precision=config.final_test_average_precision,
        **box_eval_for(config.category),
    )


@router.get("/models", response_model=list[ServedModelOut])
def list_served_models(current_user: User = Depends(get_current_user)):
    """Every category with a registered model, alphabetically. Static metadata only - loads no model."""
    return [_served_model(SERVING_CONFIGS[category]) for category in sorted(SERVING_CONFIGS)]
