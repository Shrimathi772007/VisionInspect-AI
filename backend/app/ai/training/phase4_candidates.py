"""Milestone 4 Phase 4: predefined, staged optimization candidate configurations.

Small, fixed candidate set (not a grid search) - see the module docstring in
scripts/train_and_evaluate_bottle_phase4.py for the full staged design:
    Stage 1 (input size):    128 (baseline) / 160 / 192
    Stage 2 (epochs):        15 (baseline) / 25 / 35, at the Stage 1 winner's input size
    Stage 3 (learning rate): 1e-3 (baseline) / 5e-4, at the Stage 1+2 winning config

All candidates share: architecture=conv_autoencoder, batch_size=8, seed=42,
and are trained on the exact Phase 3 167-image training subset - only the
parameter named in each stage varies. This module only builds
configuration objects; it does not train anything and does not import
anything related to the final test set.
"""

from dataclasses import dataclass

from app.ai.training.schemas import TrainingConfig

CATEGORY = "bottle"
BASELINE_IMAGE_SIZE = (128, 128)
BASELINE_EPOCHS = 15
BASELINE_LEARNING_RATE = 1e-3
BASELINE_BATCH_SIZE = 8
BASELINE_SEED = 42


@dataclass(frozen=True)
class CandidateConfig:
    """One Phase 4 candidate's full, explicit configuration."""

    candidate_id: str
    stage: str
    training_config: TrainingConfig
    notes: str
    reuse_phase3_artifact: bool = False  # True only for the exact Phase 3 configuration, during stage exploration


def stage1_candidates() -> list[CandidateConfig]:
    """Input-size experiments; all other parameters fixed at the Phase 3 baseline."""
    return [
        CandidateConfig(
            candidate_id="candidate_input128",
            stage="stage1_input_size",
            training_config=TrainingConfig(
                category=CATEGORY, image_size=(128, 128), batch_size=BASELINE_BATCH_SIZE,
                epochs=BASELINE_EPOCHS, learning_rate=BASELINE_LEARNING_RATE, seed=BASELINE_SEED,
            ),
            notes=(
                "Identical configuration to the protected Phase 3 baseline. During stage exploration "
                "this reuses the Phase 3 artifact instead of retraining (Phase 3 already proved "
                "bit-identical reproducibility for this exact config) - if this candidate wins overall, "
                "it is retrained fresh, twice, for Phase 4's own reproducibility evidence."
            ),
            reuse_phase3_artifact=True,
        ),
        CandidateConfig(
            candidate_id="candidate_input160",
            stage="stage1_input_size",
            training_config=TrainingConfig(
                category=CATEGORY, image_size=(160, 160), batch_size=BASELINE_BATCH_SIZE,
                epochs=BASELINE_EPOCHS, learning_rate=BASELINE_LEARNING_RATE, seed=BASELINE_SEED,
            ),
            notes="Larger input resolution; same architecture/epochs/learning rate.",
        ),
        CandidateConfig(
            candidate_id="candidate_input192",
            stage="stage1_input_size",
            training_config=TrainingConfig(
                category=CATEGORY, image_size=(192, 192), batch_size=BASELINE_BATCH_SIZE,
                epochs=BASELINE_EPOCHS, learning_rate=BASELINE_LEARNING_RATE, seed=BASELINE_SEED,
            ),
            notes="Even larger input resolution; same architecture/epochs/learning rate.",
        ),
    ]


def stage2_candidates(best_image_size: tuple[int, int]) -> list[CandidateConfig]:
    """Epoch-count experiments at the Stage 1 winning input size."""
    return [
        CandidateConfig(
            candidate_id="candidate_epochs25",
            stage="stage2_epochs",
            training_config=TrainingConfig(
                category=CATEGORY, image_size=best_image_size, batch_size=BASELINE_BATCH_SIZE,
                epochs=25, learning_rate=BASELINE_LEARNING_RATE, seed=BASELINE_SEED,
            ),
            notes=f"Stage 1 winning input size {best_image_size}, +10 epochs over baseline.",
        ),
        CandidateConfig(
            candidate_id="candidate_epochs35",
            stage="stage2_epochs",
            training_config=TrainingConfig(
                category=CATEGORY, image_size=best_image_size, batch_size=BASELINE_BATCH_SIZE,
                epochs=35, learning_rate=BASELINE_LEARNING_RATE, seed=BASELINE_SEED,
            ),
            notes=f"Stage 1 winning input size {best_image_size}, +20 epochs over baseline.",
        ),
    ]


def stage3_candidate(best_image_size: tuple[int, int], best_epochs: int) -> CandidateConfig:
    """Learning-rate experiment at the Stage 1+2 winning configuration."""
    return CandidateConfig(
        candidate_id="candidate_lr5e4",
        stage="stage3_learning_rate",
        training_config=TrainingConfig(
            category=CATEGORY, image_size=best_image_size, batch_size=BASELINE_BATCH_SIZE,
            epochs=best_epochs, learning_rate=5e-4, seed=BASELINE_SEED,
        ),
        notes=f"Stage 1+2 winning config (image_size={best_image_size}, epochs={best_epochs}), halved learning rate.",
    )
