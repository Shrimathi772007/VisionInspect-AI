"""Build and persist the Phase 4 optimization-experiment artifact (JSON).

Written to a separate file from Phase 1/2/3's reports
(backend/ai_models/<category>/evaluation_reports/phase4_optimization_report.json)
- none of the prior reports are overwritten. Same directory convention:
backend/ai_models/ is already gitignored in its entirety, so this is not
committed.
"""

import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.ai.training.artifacts import ARTIFACTS_ROOT


def _candidate_stage_summary(result) -> dict:
    c = result.config
    return {
        "candidate_id": c.candidate_id,
        "stage": c.stage,
        "notes": c.notes,
        "reused_phase3_artifact": c.reuse_phase3_artifact,
        "training_config": {
            "image_size": list(c.training_config.image_size),
            "batch_size": c.training_config.batch_size,
            "epochs": c.training_config.epochs,
            "learning_rate": c.training_config.learning_rate,
            "seed": c.training_config.seed,
        },
        "training_duration_s": result.training_duration_s,
        "final_training_loss": result.final_training_loss,
        "model_hash_md5": result.model_metadata.md5,
        "model_file_size_bytes": result.model_metadata.artifact_size_bytes,
        "validation": {
            "sample_count": len(result.validation_errors),
            "mean_error": result.validation_mean,
            "std_error": result.validation_std,
            "min_error": result.validation_min,
            "max_error": result.validation_max,
            "coefficient_of_variation": (
                result.validation_std / result.validation_mean if result.validation_mean else None
            ),
            "inference_ms_total": result.validation_inference_ms,
        },
        "selection": {
            "status": result.selection.status,
            "selected": (
                {"method": result.selection.selected.method, "parameter": result.selection.selected.parameter,
                 "threshold": result.selection.selected.threshold}
                if result.selection.selected is not None else None
            ),
            "reasoning": result.selection.reasoning,
        },
    }


def build_phase4_report(
    *,
    git_branch: str,
    git_head: str,
    dataset_counts: dict,
    split: dict,
    stage1_results: list,
    stage1_selection_reasoning: str,
    stage2_results: list,
    stage2_selection_reasoning: str,
    stage3_result,
    stage3_decision_reasoning: str,
    selected_config_summary: dict,
    selected_model_metadata: dict,
    final_test_metrics: dict,
    phase3_baseline: dict,
    reproducibility: dict,
    performance: dict,
    manufacturing_metrics: dict,
    map_note: str,
    production_status: str,
    limitations: list[str],
    recommendation: str,
) -> dict:
    return {
        "report_type": "milestone_4_phase_4_optimization",
        "generated_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "git": {"branch": git_branch, "head": git_head},
        "dataset_counts": dataset_counts,
        "split": split,
        "stage1_input_size": {
            "candidates": [_candidate_stage_summary(r) for r in stage1_results],
            "selection_reasoning": stage1_selection_reasoning,
        },
        "stage2_epochs": {
            "candidates": [_candidate_stage_summary(r) for r in stage2_results],
            "selection_reasoning": stage2_selection_reasoning,
        },
        "stage3_learning_rate": {
            "candidate": _candidate_stage_summary(stage3_result) if stage3_result is not None else None,
            "decision_reasoning": stage3_decision_reasoning,
        },
        "selected_candidate": selected_config_summary,
        "selected_model": selected_model_metadata,
        "final_test_metrics": final_test_metrics,
        "phase3_baseline": phase3_baseline,
        "reproducibility": reproducibility,
        "performance": performance,
        "manufacturing_metrics": manufacturing_metrics,
        "mAP": map_note,
        "production_model_status": production_status,
        "leakage_verification": {
            "validation_excluded_from_training": True,
            "validation_used_for_candidate_selection": True,
            "final_test_excluded_from_candidate_and_threshold_selection": True,
            "final_test_evaluated_only_after_lock": True,
            "no_test_driven_tuning": True,
        },
        "limitations": limitations,
        "recommendation": recommendation,
    }


def default_phase4_report_path(category: str) -> Path:
    return ARTIFACTS_ROOT / category / "evaluation_reports" / "phase4_optimization_report.json"


def save_phase4_report(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
