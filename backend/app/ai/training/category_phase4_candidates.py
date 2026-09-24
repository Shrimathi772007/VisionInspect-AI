"""Predefined, staged optimization candidates for one category (Milestone 4 Phase 4 protocol).

Category-parameterized counterpart of app.ai.training.phase4_candidates (which is hard-wired to
Bottle and is left untouched). Same staged design, fixed before any final-test access:

    Stage 1 (input size):    128 (baseline) / 160 / 192     epochs 15, lr 1e-3
    Stage 2 (epochs):        15 (baseline) / 25 / 35        at the Stage 1 winner's size, lr 1e-3
    Stage 3 (learning rate): 1e-3 (baseline) / 5e-4         at the Stage 1+2 winning configuration

Everything else is frozen at the Phase 3 baseline: ConvAutoencoder (latent 128), batch 8, seed 42,
Adam + MSE, CPU, no augmentation. Only the parameter named in each stage varies.

A candidate is identified by its (input size, epochs, learning rate) alone, so the same
configuration reached from two stages is one candidate and is trained at most once. This module
only builds configuration objects: it trains nothing and imports nothing about test data.
"""

from app.ai.training.phase4_candidates import CandidateConfig
from app.ai.training.schemas import TrainingConfig

BASELINE_IMAGE_SIZE = (128, 128)
BASELINE_EPOCHS = 15
BASELINE_LEARNING_RATE = 1e-3
BASELINE_BATCH_SIZE = 8
BASELINE_SEED = 42

STAGE1_IMAGE_SIZES = ((128, 128), (160, 160), (192, 192))
STAGE2_EPOCHS = (15, 25, 35)
STAGE3_LEARNING_RATES = (1e-3, 5e-4)


def candidate_id(image_size: tuple[int, int], epochs: int, learning_rate: float) -> str:
    return f"input{image_size[0]}_epochs{epochs}_lr{learning_rate:g}"


def is_baseline(image_size: tuple[int, int], epochs: int, learning_rate: float) -> bool:
    """True only for the exact Phase 3 configuration (128x128, 15 epochs, lr 1e-3)."""
    return (tuple(image_size), epochs, learning_rate) == (
        BASELINE_IMAGE_SIZE,
        BASELINE_EPOCHS,
        BASELINE_LEARNING_RATE,
    )


def make_candidate(category: str, stage: str, image_size: tuple[int, int], epochs: int, learning_rate: float,
                   notes: str = "") -> CandidateConfig:
    baseline = is_baseline(image_size, epochs, learning_rate)
    return CandidateConfig(
        candidate_id=candidate_id(image_size, epochs, learning_rate),
        stage=stage,
        training_config=TrainingConfig(
            category=category,
            image_size=tuple(image_size),
            batch_size=BASELINE_BATCH_SIZE,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=BASELINE_SEED,
            device="cpu",
        ),
        notes=notes or ("Identical to the protected Phase 3 baseline." if baseline else ""),
        # Reuse of the Phase 3 artifact is allowed for the exact baseline configuration and for
        # nothing else.
        reuse_phase3_artifact=baseline,
    )


def stage1_candidates(category: str) -> list[CandidateConfig]:
    return [
        make_candidate(category, "stage1_input_size", size, BASELINE_EPOCHS, BASELINE_LEARNING_RATE)
        for size in STAGE1_IMAGE_SIZES
    ]


def stage2_candidates(category: str, image_size: tuple[int, int]) -> list[CandidateConfig]:
    return [
        make_candidate(category, "stage2_epochs", image_size, epochs, BASELINE_LEARNING_RATE)
        for epochs in STAGE2_EPOCHS
    ]


def stage3_candidates(category: str, image_size: tuple[int, int], epochs: int) -> list[CandidateConfig]:
    return [
        make_candidate(category, "stage3_learning_rate", image_size, epochs, lr) for lr in STAGE3_LEARNING_RATES
    ]
