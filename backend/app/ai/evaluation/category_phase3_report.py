"""Build the per-category Phase 3 evaluation report (JSON) for the parameterized runner.

Same directory/file convention as Bottle's Phase 3 report
(backend/ai_models/<category>/evaluation_reports/phase3_validated_model_report.json, Git-ignored
with the rest of backend/ai_models/), but a category-generic layout: no Phase 1 model, Phase 1
baseline or Phase 2 comparison sections (those exist only for Bottle's history), and every count in
the limitations text comes from the run itself rather than being hard-coded.

The report distinguishes two things that must not be conflated:
  - `validation_protocol_status`: did every Phase 3 protection pass (leakage audit on both runs,
    reproducibility, a threshold selected from validation alone)?
  - the final-test metrics: how well the model performs. Passing the protocol says the numbers
    are trustworthy, not that they are good.
"""

import platform
from dataclasses import asdict
from datetime import UTC, datetime

from app.ai.evaluation.category_phase3 import (
    CategoryPhase3Run,
    dataset_counts,
    per_defect_type_recall,
    selected_final_metrics,
    selected_predictions,
)
from app.ai.evaluation.model_metadata import ModelMetadata
from app.ai.evaluation.phase3_threshold_selection import MAX_VALIDATION_FALSE_POSITIVE_RATE, STATUS_SELECTED

PROTOCOL_PASSED = "PHASE 3 PROTOCOL PASSED"
PROTOCOL_NOT_VALIDATED = "NOT VALIDATED"


def protocol_status(run: CategoryPhase3Run, leakage_runs: list[dict], reproducibility: dict) -> tuple[str, list[str]]:
    problems = []
    if run.scored.selection.status != STATUS_SELECTED:
        problems.append("no threshold candidate met the validation false-positive ceiling (no threshold selected)")
    if not all(check["all_passed"] for check in leakage_runs):
        problems.append("a leakage check failed")
    if not reproducibility["scientifically_reproducible"]:
        problems.append("the two runs were not scientifically reproducible")
    return (PROTOCOL_PASSED if not problems else PROTOCOL_NOT_VALIDATED, problems)


def _candidate_rows(run: CategoryPhase3Run) -> list[dict]:
    stability = {(s.candidate.method, s.candidate.parameter): s for s in run.scored.selection.stability_by_candidate}
    rows = []
    for result in run.scored.candidate_results:
        c = result.candidate
        s = stability[(c.method, c.parameter)]
        rows.append(
            {
                "method": c.method,
                "parameter": c.parameter,
                "threshold": c.threshold,
                "validation_false_positive_count": s.validation_false_positive_count,
                "validation_false_positive_rate": s.validation_false_positive_rate,
                "passes_validation_stability_check": s.passes_stability_check,
                # Reporting only - never an input to selection (selection was locked first).
                "final_test_reporting_only": {
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                    "confusion_matrix": [
                        [result.true_negatives, result.false_positives],
                        [result.false_negatives, result.true_positives],
                    ],
                    "false_positive_rate": result.false_positive_rate,
                },
            }
        )
    return rows


def build_category_phase3_report(
    *,
    run: CategoryPhase3Run,
    model_metadata: ModelMetadata,
    git_branch: str,
    git_head: str,
    leakage_runs: list[dict],
    reproducibility: dict,
    run2_total_ms: float,
) -> dict:
    scored = run.scored
    selected = scored.selection.selected
    status, problems = protocol_status(run, leakage_runs, reproducibility)
    counts = dataset_counts(run.category)
    val_n = len(scored.validation_errors)

    return {
        "report_type": "category_phase3_validated_model",
        "category": run.category,
        "generated_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "git": {"branch": git_branch, "head": git_head},
        "validation_protocol_status": status,
        "validation_protocol_problems": problems,
        "serving_status": "NOT registered for production serving (app.ai.inference.serving) - evaluation artifact only.",
        "model": asdict(model_metadata),
        "dataset": counts,
        "training_config": {
            "architecture": run.config.model.architecture,
            "latent_channels": run.config.model.latent_channels,
            "image_size": list(run.config.image_size),
            "batch_size": run.config.batch_size,
            "epochs": run.config.epochs,
            "learning_rate": run.config.learning_rate,
            "seed": run.config.seed,
            "device": run.config.device,
            "optimizer": "Adam",
            "loss": "MSE",
            "preprocessing": "existing app.ai.preprocessing pipeline (decode -> RGB -> INTER_AREA resize -> /255)",
            "augmentation": "none",
        },
        "split": {
            "training_fraction": 0.8,
            "seed": 42,
            "training_count": len(run.split.training),
            "validation_count": val_n,
            "final_test_count": len(scored.test_labels),
        },
        "training_result": {
            "duration_seconds": run.training_result.duration_seconds,
            "final_loss": run.training_result.final_loss,
            "num_training_images": run.training_result.num_training_images,
            "loss_history": run.training_result.loss_history,
        },
        "validation_statistics": {
            "sample_count": val_n,
            "mean_error": scored.validation_mean,
            "std_error": scored.validation_std,
            "min_error": min(scored.validation_errors),
            "max_error": max(scored.validation_errors),
        },
        "threshold_selection": {
            "status": scored.selection.status,
            "rule": (
                "Reject candidates whose validation false-positive rate exceeds "
                f"{MAX_VALIDATION_FALSE_POSITIVE_RATE:.0%}; among the rest, pick the lowest threshold. "
                "Candidates: mean+K*std for K in (3, 2.5, 2, 1.5, 1) and percentiles (95, 97, 98, 99) of "
                "the validation errors. The 10% ceiling is an engineering constraint inherited from "
                "Phase 2, not a specification value."
            ),
            "selected": None if selected is None else {"method": selected.method, "parameter": selected.parameter},
            "selected_threshold": None if selected is None else selected.threshold,
            "reasoning": scored.selection.reasoning,
            "candidates": _candidate_rows(run),
        },
        "final_test_at_selected_threshold": selected_final_metrics(scored),
        "per_defect_type_recall_at_selected_threshold": per_defect_type_recall(scored),
        "final_test_predictions_at_selected_threshold": selected_predictions(scored),
        "reproducibility": {"run_count": 2, **reproducibility},
        "leakage_verification": {"run1": leakage_runs[0], "run2": leakage_runs[1]},
        "timing": {
            "training_duration_s": run.training_result.duration_seconds,
            "validation_evaluation_ms": scored.validation_inference_ms,
            "validation_mean_ms_per_image": scored.validation_inference_ms / val_n,
            "final_test_evaluation_ms": scored.final_test_inference_ms,
            "final_test_mean_ms_per_image": scored.final_test_inference_ms / len(scored.test_labels),
            "run1_total_ms": run.total_ms,
            "run2_total_ms": run2_total_ms,
            "note": "CPU-only wall-clock on this machine; measured, not extrapolated to other categories.",
        },
        "mAP": "Not applicable: reconstruction-error classifier with no localization/detection output.",
        "limitations": [
            "The validation set contains only good images (MVTec provides no labeled-defective "
            "validation split); it can establish false-positive behavior only, never recall or F1.",
            f"The final test set has {counts['test_total']} images ({counts['test_good']} good, "
            f"{counts['test_defective']} defective); differences of a few images between operating "
            "points are suggestive, not statistically definitive.",
            f"With {val_n} validation images, each misclassified validation image shifts the validation "
            f"false-positive rate by ~{100 / val_n:.1f} percentage points.",
            "The 10% validation false-positive ceiling is an engineering operating constraint inherited "
            "from Phase 2, not a project-specification value.",
            "Inputs are downscaled to 128x128 from the original resolution; small defects may be "
            "under-represented.",
            "Passing the Phase 3 protocol certifies how the numbers were obtained, not that the model "
            "is good enough for production.",
        ],
    }
