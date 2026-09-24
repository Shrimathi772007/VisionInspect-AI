"""Milestone 4 Phase 3 methodology as a reusable, category-parameterized runner.

The validated Bottle Phase 3 baseline (scripts/train_and_evaluate_bottle_phase3.py) is a
one-off script with `CATEGORY = "bottle"`, Bottle's image counts and Bottle's Phase 1/2 model
hard-coded. That script and its artifacts stay untouched (they are Bottle's historical
reproducibility record). This module re-expresses the SAME protocol for any MVTec category,
built from the very same building blocks Bottle used - no re-implemented methodology:

    discover_train_samples(category)                      train/good only, filename-sorted
      -> split_train_validation(0.8, seed=42)             numpy default_rng permutation
      -> train_anomaly_model_on_samples(training subset)  ConvAutoencoder, MSE, Adam
      -> compute_errors_for_samples(validation subset)    validation reconstruction errors
      -> generate_candidates(validation errors)           mean+K*std (K=3,2.5,2,1.5,1), percentiles
      -> select_threshold_from_validation(candidates,     rule: lowest threshold whose
                                          validation errors)   validation FP rate <= 10%
      ==== SELECTION LOCKED ====
      -> discover_test_samples(category)                  only NOW are test images touched
      -> compute_errors_for_samples(test) once, score every candidate (reporting only)

Nothing here trains a model or picks a threshold from anything other than validation errors:
`select_threshold_from_validation` has no parameter through which test errors or labels
could reach it, and test discovery happens strictly after the selection is locked (recorded in
`CategoryPhase3Run.events` and verified by `audit_leakage`).

Deliberately does NOT register anything for serving (app.ai.inference.serving) - training and
validating a category never makes it production-active.
"""

import hashlib
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from torch import nn

from app.ai.evaluation.metrics import compute_classification_metrics
from app.ai.evaluation.phase3_report import default_phase3_report_path
from app.ai.evaluation.phase3_threshold_selection import Phase3SelectionResult, select_threshold_from_validation
from app.ai.evaluation.threshold_experiments import (
    CandidateTestResult,
    ThresholdCandidate,
    compute_errors_for_samples,
    evaluate_candidate_on_final_test,
    generate_candidates,
)
from app.ai.training.artifacts import get_model_path
from app.ai.training.dataset import discover_test_samples, discover_train_samples
from app.ai.training.phase3_train import train_anomaly_model_on_samples
from app.ai.training.schemas import DatasetSample, TrainingConfig, TrainingResult
from app.ai.training.validation_split import TrainValidationSplit, split_train_validation

PHASE3_MODEL_NAME = "phase3_validation/autoencoder"  # artifact name, as used for Bottle
PHASE3_TRAINING_FRACTION = 0.8
PHASE3_SPLIT_SEED = 42

# Ordered, machine-checkable record of what happened in a run (see audit_leakage).
EVENT_SPLIT = "split"
EVENT_TRAINED = "trained_on_training_subset"
EVENT_VALIDATION_ERRORS = "validation_errors_computed"
EVENT_SELECTION_LOCKED = "selection_locked"
EVENT_TEST_DISCOVERED = "test_samples_discovered"
EVENT_TEST_SCORED = "final_test_scored"


def phase3_training_config(category: str) -> TrainingConfig:
    """The exact Bottle Phase 3 training configuration, for `category`.

    128x128, batch 8, 15 epochs, lr 1e-3, seed 42, ConvAutoencoder(latent 128). CPU is
    requested explicitly - nothing here assumes a GPU.
    """
    return TrainingConfig(
        category=category,
        image_size=(128, 128),
        batch_size=8,
        epochs=15,
        learning_rate=1e-3,
        seed=42,
        device="cpu",
    )


@dataclass(frozen=True)
class CategoryOutputs:
    """Where a category's Phase 3 artifacts live - derived from the category name alone."""

    category: str
    model_path: Path
    report_path: Path


def resolve_outputs(category: str) -> CategoryOutputs:
    return CategoryOutputs(
        category=category,
        model_path=get_model_path(category, model_name=PHASE3_MODEL_NAME),
        report_path=default_phase3_report_path(category),
    )


def ensure_no_overwrite(outputs: CategoryOutputs) -> None:
    """Refuse to run if this category already has a Phase 3 artifact or report.

    A validated model (Bottle's, or any later one) must never be silently overwritten by a
    re-run; re-training a category is an explicit, manual decision.
    """
    existing = [p for p in (outputs.model_path, outputs.report_path) if p.exists()]
    if existing:
        raise FileExistsError(
            f"Phase 3 output already exists for category '{outputs.category}': "
            + ", ".join(str(p) for p in existing)
            + ". Refusing to overwrite."
        )


@dataclass
class ScoredModel:
    """Everything derived from a trained model + a split: validation errors, the locked
    threshold selection, and the (single) final-test scoring."""

    validation_errors: list[float]
    validation_mean: float
    validation_std: float
    candidates: list[ThresholdCandidate]
    selection: Phase3SelectionResult
    test_samples: list[DatasetSample]
    test_labels: list[int]
    test_errors: list[float]
    candidate_results: list[CandidateTestResult]
    validation_inference_ms: float
    final_test_inference_ms: float
    events: list[str] = field(default_factory=list)


def score_model(
    category: str,
    model: nn.Module,
    split: TrainValidationSplit,
    image_size: tuple[int, int],
    events: list[str] | None = None,
) -> ScoredModel:
    """Validation errors -> candidates -> LOCKED selection -> only then final-test scoring.

    Pure evaluation: never trains or modifies `model`. This is the stage shared by the
    trained-model run and by any later re-scoring of an already-saved model.
    """
    events = events if events is not None else []
    model.eval()

    start = time.perf_counter()
    validation_errors = compute_errors_for_samples(model, split.validation, image_size)
    validation_inference_ms = (time.perf_counter() - start) * 1000
    events.append(EVENT_VALIDATION_ERRORS)

    validation_mean = statistics.fmean(validation_errors)
    validation_std = (sum((e - validation_mean) ** 2 for e in validation_errors) / len(validation_errors)) ** 0.5

    candidates = generate_candidates(validation_errors)
    selection = select_threshold_from_validation(candidates, validation_errors)  # validation data only
    events.append(EVENT_SELECTION_LOCKED)

    # --- The final test set is first touched here, after the selection above is locked. ---
    test_samples = discover_test_samples(category)
    events.append(EVENT_TEST_DISCOVERED)
    test_labels = [s.label for s in test_samples]

    start = time.perf_counter()
    test_errors = compute_errors_for_samples(model, test_samples, image_size)
    final_test_inference_ms = (time.perf_counter() - start) * 1000
    events.append(EVENT_TEST_SCORED)

    candidate_results = [evaluate_candidate_on_final_test(c, test_labels, test_errors) for c in candidates]

    return ScoredModel(
        validation_errors=validation_errors,
        validation_mean=validation_mean,
        validation_std=validation_std,
        candidates=candidates,
        selection=selection,
        test_samples=test_samples,
        test_labels=test_labels,
        test_errors=test_errors,
        candidate_results=candidate_results,
        validation_inference_ms=validation_inference_ms,
        final_test_inference_ms=final_test_inference_ms,
        events=events,
    )


@dataclass
class CategoryPhase3Run:
    category: str
    config: TrainingConfig
    split: TrainValidationSplit
    training_result: TrainingResult
    model_md5: str
    scored: ScoredModel
    events: list[str]
    total_ms: float

    @property
    def model_path(self) -> Path:
        return self.training_result.model_path


def _file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_category_phase3(category: str, model_path: Path, config: TrainingConfig | None = None) -> CategoryPhase3Run:
    """One complete Phase 3 pass for `category`: split, train, validate, lock, final-test.

    Saves the trained model to `model_path` (callers choose canonical vs scratch location).
    """
    config = config or phase3_training_config(category)
    started = time.perf_counter()
    events: list[str] = []

    train_samples = discover_train_samples(category)
    split = split_train_validation(train_samples, training_fraction=PHASE3_TRAINING_FRACTION, seed=PHASE3_SPLIT_SEED)
    events.append(EVENT_SPLIT)
    if {s.path for s in split.training} & {s.path for s in split.validation}:
        raise AssertionError("training/validation overlap detected")

    training_result, model = train_anomaly_model_on_samples(split.training, config, model_path)
    events.append(EVENT_TRAINED)
    model_md5 = _file_md5(training_result.model_path)

    scored = score_model(category, model, split, config.image_size, events=events)

    return CategoryPhase3Run(
        category=category,
        config=config,
        split=split,
        training_result=training_result,
        model_md5=model_md5,
        scored=scored,
        events=events,
        total_ms=(time.perf_counter() - started) * 1000,
    )


# ---------------------------------------------------------------------------
# Selected-threshold predictions and final metrics
# ---------------------------------------------------------------------------

def predictions_at_threshold(test_samples: list[DatasetSample], test_errors: list[float], threshold: float) -> list[dict]:
    """Per-image prediction (error > threshold -> defective) for already-computed test errors."""
    return [
        {
            "file": f"{s.defect_type}/{s.path.name}",
            "defect_type": s.defect_type,
            "label": s.label,
            "reconstruction_error": error,
            "predicted_label": 1 if error > threshold else 0,
        }
        for s, error in zip(test_samples, test_errors)
    ]


def defect_type_recall(predictions: list[dict]) -> dict[str, dict]:
    """Detected/total per defect type (defective images only)."""
    breakdown: dict[str, dict] = {}
    for p in predictions:
        if p["label"] != 1:
            continue
        entry = breakdown.setdefault(p["defect_type"], {"total": 0, "detected": 0})
        entry["total"] += 1
        entry["detected"] += p["predicted_label"]
    for entry in breakdown.values():
        entry["recall"] = entry["detected"] / entry["total"]
    return breakdown


def metrics_from_predictions(predictions: list[dict]) -> dict:
    metrics = compute_classification_metrics(
        [p["label"] for p in predictions], [p["predicted_label"] for p in predictions]
    )
    negatives = metrics.true_negatives + metrics.false_positives
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "true_negatives": metrics.true_negatives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "true_positives": metrics.true_positives,
        "confusion_matrix": [
            [metrics.true_negatives, metrics.false_positives],
            [metrics.false_negatives, metrics.true_positives],
        ],
        # Existing project definitions (see the Bottle Phase 4 report's manufacturing_metrics):
        "defect_identification_accuracy": metrics.recall,
        "false_defect_detection_rate": metrics.false_positives / negatives if negatives else 0.0,
        "total_test_samples": len(predictions),
    }


def selected_predictions(scored: ScoredModel) -> list[dict] | None:
    """Per-test-image prediction at the locked threshold, or None if nothing was selected."""
    if scored.selection.selected is None:
        return None
    return predictions_at_threshold(scored.test_samples, scored.test_errors, scored.selection.selected.threshold)


def per_defect_type_recall(scored: ScoredModel) -> dict[str, dict] | None:
    """Detected/total per defect type at the locked threshold (reporting only)."""
    predictions = selected_predictions(scored)
    return None if predictions is None else defect_type_recall(predictions)


def selected_final_metrics(scored: ScoredModel) -> dict | None:
    predictions = selected_predictions(scored)
    return None if predictions is None else metrics_from_predictions(predictions)


# ---------------------------------------------------------------------------
# Reproducibility comparison
# ---------------------------------------------------------------------------

def compare_runs(a: CategoryPhase3Run, b: CategoryPhase3Run) -> dict:
    """Field-by-field comparison of two independent runs. Reports what differs; hides nothing."""
    sa, sb = a.scored, b.scored

    def _sel(s: ScoredModel):
        x = s.selection.selected
        return (s.selection.status, None if x is None else (x.method, x.parameter, x.threshold))

    pa, pb = selected_predictions(sa), selected_predictions(sb)
    result = {
        "split_identical": (
            [s.path.name for s in a.split.training] == [s.path.name for s in b.split.training]
            and [s.path.name for s in a.split.validation] == [s.path.name for s in b.split.validation]
        ),
        "model_hash_identical": a.model_md5 == b.model_md5,
        "validation_errors_identical": sa.validation_errors == sb.validation_errors,
        "candidate_thresholds_identical": [c.threshold for c in sa.candidates] == [c.threshold for c in sb.candidates],
        "selection_identical": _sel(sa) == _sel(sb),
        "final_test_errors_identical": sa.test_errors == sb.test_errors,
        "final_predictions_identical": pa == pb,
        "final_metrics_identical": selected_final_metrics(sa) == selected_final_metrics(sb),
        "all_candidate_confusion_matrices_identical": [
            (r.true_negatives, r.false_positives, r.false_negatives, r.true_positives) for r in sa.candidate_results
        ]
        == [(r.true_negatives, r.false_positives, r.false_negatives, r.true_positives) for r in sb.candidate_results],
    }
    result["scientifically_reproducible"] = all(
        result[k]
        for k in (
            "split_identical",
            "validation_errors_identical",
            "candidate_thresholds_identical",
            "selection_identical",
            "final_predictions_identical",
            "final_metrics_identical",
            "all_candidate_confusion_matrices_identical",
        )
    )
    result["bit_identical"] = result["scientifically_reproducible"] and result["model_hash_identical"] and (
        result["final_test_errors_identical"]
    )
    return result


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_leakage_checks(category: str, split: TrainValidationSplit, test_samples: list[DatasetSample]) -> dict:
    """Split/test-set independence checks shared by every runner (Phase 3 and Phase 4).

    Path checks are backed by a content-hash check so a test image that is a byte-for-byte copy
    of a training/validation image would also be caught.
    """
    training = {s.path for s in split.training}
    validation = {s.path for s in split.validation}
    test = {s.path for s in test_samples}
    all_train_good = {s.path for s in discover_train_samples(category)}

    training_hashes = {_content_hash(p) for p in training}
    validation_hashes = {_content_hash(p) for p in validation}
    test_hashes = {_content_hash(p) for p in test}

    rerun = split_train_validation(
        discover_train_samples(category), training_fraction=PHASE3_TRAINING_FRACTION, seed=PHASE3_SPLIT_SEED
    )
    return {
        "training_and_validation_disjoint": not (training & validation),
        "training_plus_validation_equals_all_train_good": (training | validation) == all_train_good
        and len(training) + len(validation) == len(all_train_good),
        "training_only_from_train_good": all(s.split == "train" and s.defect_type == "good" for s in split.training),
        "validation_only_from_train_good": all(s.split == "train" and s.defect_type == "good" for s in split.validation),
        "test_only_from_test_split": all(s.split == "test" for s in test_samples),
        "test_disjoint_from_training_by_path": not (test & training),
        "test_disjoint_from_validation_by_path": not (test & validation),
        "test_disjoint_from_training_by_content": not (test_hashes & training_hashes),
        "test_disjoint_from_validation_by_content": not (test_hashes & validation_hashes),
        "split_is_deterministic": [s.path for s in rerun.training] == [s.path for s in split.training]
        and [s.path for s in rerun.validation] == [s.path for s in split.validation],
    }


def audit_leakage(run: CategoryPhase3Run) -> dict:
    """Independent verification of the Phase 3 protections for one completed run.

    Every check is a boolean derived from the run's actual data, not an assertion of intent.
    """
    split, scored = run.split, run.scored
    order = {event: i for i, event in enumerate(run.events)}
    checks = split_leakage_checks(run.category, split, scored.test_samples)
    checks.update(
        {
            "model_trained_on_exactly_the_training_subset": run.training_result.num_training_images
            == len(split.training),
            "selection_locked_before_test_touched": order.get(EVENT_SELECTION_LOCKED, 10**9)
            < order.get(EVENT_TEST_DISCOVERED, -1)
            < order.get(EVENT_TEST_SCORED, -1),
            "validation_errors_computed_before_selection": order.get(EVENT_VALIDATION_ERRORS, 10**9)
            < order.get(EVENT_SELECTION_LOCKED, -1),
            "final_test_scored_exactly_once": run.events.count(EVENT_TEST_SCORED) == 1,
            "selection_uses_validation_only": _selection_accepts_only_validation_inputs(),
        }
    )
    checks["all_passed"] = all(checks.values())
    return checks


def _selection_accepts_only_validation_inputs() -> bool:
    """select_threshold_from_validation must have no parameter that could carry test data."""
    import inspect

    parameters = set(inspect.signature(select_threshold_from_validation).parameters)
    return parameters == {"candidates", "validation_errors", "max_validation_false_positive_rate"}


def dataset_counts(category: str) -> dict:
    """Real dataset counts for `category` (discovery only; nothing is decoded)."""
    train = discover_train_samples(category)
    test = discover_test_samples(category)
    by_type: dict[str, int] = {}
    for s in test:
        by_type[s.defect_type] = by_type.get(s.defect_type, 0) + 1
    return {
        "train_good": len(train),
        "test_total": len(test),
        "test_good": by_type.get("good", 0),
        "test_defective": len(test) - by_type.get("good", 0),
        "test_defect_type_counts": dict(sorted(by_type.items())),
    }


__all__ = [
    "PHASE3_MODEL_NAME",
    "CategoryOutputs",
    "CategoryPhase3Run",
    "ScoredModel",
    "audit_leakage",
    "compare_runs",
    "dataset_counts",
    "defect_type_recall",
    "metrics_from_predictions",
    "predictions_at_threshold",
    "split_leakage_checks",
    "ensure_no_overwrite",
    "per_defect_type_recall",
    "phase3_training_config",
    "resolve_outputs",
    "run_category_phase3",
    "score_model",
    "selected_final_metrics",
    "selected_predictions",
]
