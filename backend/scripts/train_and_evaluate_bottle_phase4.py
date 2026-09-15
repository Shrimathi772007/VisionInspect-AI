"""Milestone 4 Phase 4: controlled AI model optimization & inspection performance.

Staged design (declared here, before any final-test evaluation):

  Stage 1 (input size):    candidate_input128 (reuses the Phase 3 artifact) /
                            candidate_input160 / candidate_input192
  Stage 2 (epochs):        the Stage 1 winner (as the 15-epoch reference) /
                            candidate_epochs25 / candidate_epochs35, all at
                            the Stage 1 winning input size
  Stage 3 (learning rate): the Stage 2 winner (as the 1e-3 reference) /
                            candidate_lr5e4, at the Stage 1+2 winning config

Every stage is decided using ONLY validation data (42 genuinely unseen
images - app.ai.training.validation_split, the exact Phase 3 split) via
app.ai.evaluation.phase4_selection.select_stage_winner. The final 83-image
test set is never imported, discovered, or referenced anywhere before the
overall winning configuration is locked and retrained twice for
reproducibility (see below) - only then is it touched, exactly as in Phase
3's own precedent: touched twice (across the two reproducibility runs) as a
REPRODUCIBILITY check on an already-locked configuration, never to compare
or choose between candidates.

Does not modify the original Phase 1 model, the Phase 3 model, or any
production inference behavior. Every Phase 4 candidate is saved to
backend/ai_models/bottle/phase4_optimization/<candidate_id>/autoencoder.pt.

Usage (from backend/, with the venv activated):
    python scripts/train_and_evaluate_bottle_phase4.py
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.model_metadata import compute_model_metadata  # noqa: E402
from app.ai.evaluation.phase4_report import build_phase4_report, default_phase4_report_path, save_phase4_report  # noqa: E402
from app.ai.evaluation.phase4_selection import select_stage_winner  # noqa: E402
from app.ai.evaluation.threshold_experiments import evaluate_candidate_on_final_test  # noqa: E402
from app.ai.training import compute_statistics, discover_test_samples, discover_train_samples  # noqa: E402
from app.ai.training.artifacts import get_model_path  # noqa: E402
from app.ai.training.phase4_candidates import stage1_candidates, stage2_candidates, stage3_candidate  # noqa: E402
from app.ai.training.phase4_experiment import run_candidate  # noqa: E402
from app.ai.training.validation_split import split_train_validation  # noqa: E402

CATEGORY = "bottle"
ORIGINAL_MD5 = "2f470c30834baf80252f1d352c7fb37f"
PHASE3_MD5 = "76478dd6996feb50fadaf5e5e5be1ae4"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                               capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _file_md5(path: Path) -> str:
    import hashlib
    hasher = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _model_path_for(candidate_id: str) -> Path:
    return get_model_path(CATEGORY, model_name=f"phase4_optimization/{candidate_id}/autoencoder")


def _print_stage_table(results, header_extra=""):
    rows = []
    for r in results:
        cov = r.validation_std / r.validation_mean if r.validation_mean else float("nan")
        rows.append((
            r.config.candidate_id, r.config.training_config.image_size, r.config.training_config.epochs,
            r.config.training_config.learning_rate, f"{r.training_duration_s:.1f}s",
            f"{r.final_training_loss:.6f}" if r.final_training_loss == r.final_training_loss else "n/a (reused)",
            f"{cov:.4f}", r.selection.status,
        ))
    header = ("Candidate", "Input", "Epochs", "LR", "TrainTime", "FinalLoss", "ValCoV", "Status")
    widths = [max(len(str(x[i])) for x in ([header] + rows)) for i in range(len(header))]
    def fmt(row):
        return " | ".join(str(v).rjust(widths[i]) for i, v in enumerate(row))
    print(fmt(header))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def main() -> None:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    head = _git("rev-parse", "HEAD")
    print(f"Git branch: {branch}  HEAD: {head}")
    print()

    original_path = get_model_path(CATEGORY)
    phase3_path = get_model_path(CATEGORY, model_name="phase3_validation/autoencoder")
    original_md5_before = _file_md5(original_path)
    phase3_md5_before = _file_md5(phase3_path)
    print(f"Original model MD5 before Phase 4: {original_md5_before} (expected {ORIGINAL_MD5})")
    print(f"Phase 3 model MD5 before Phase 4:  {phase3_md5_before} (expected {PHASE3_MD5})")
    if original_md5_before != ORIGINAL_MD5 or phase3_md5_before != PHASE3_MD5:
        print("!!! STOP: protected model artifact(s) do not match documented reference hashes. !!!")
        sys.exit(1)
    print()

    stats = compute_statistics(CATEGORY)
    print(f"Dataset: train/good={stats.train_good_count} test_total={stats.test_total_count} "
          f"({stats.test_defect_type_counts})")
    if stats.train_good_count != 209 or stats.test_total_count != 83:
        print("!!! STOP: dataset counts do not match the documented Phase 1-3 baseline. !!!")
        sys.exit(1)
    print()

    train_samples = discover_train_samples(CATEGORY)
    split = split_train_validation(train_samples, training_fraction=0.8, seed=42)
    print(f"Split: training={len(split.training)} validation={len(split.validation)} (Phase 3's exact split)")
    print()

    # Load Phase 3's own training duration/loss for the reused candidate_input128 record.
    from app.ai.evaluation.phase3_report import default_phase3_report_path
    phase3_report_file = default_phase3_report_path(CATEGORY)
    phase3_training_duration = 0.0
    phase3_final_loss = float("nan")
    if phase3_report_file.is_file():
        phase3_data = json.loads(phase3_report_file.read_text(encoding="utf-8"))
        phase3_training_duration = phase3_data["training_result"]["duration_seconds"]
        phase3_final_loss = phase3_data["training_result"]["final_loss"]

    # ========================= STAGE 1: input size =========================
    print("=== STAGE 1: input size (128 baseline / 160 / 192) ===")
    stage1_results = []
    for cfg in stage1_candidates():
        if cfg.reuse_phase3_artifact:
            r = run_candidate(cfg, split.training, split.validation, _model_path_for(cfg.candidate_id),
                               reused_model_path=phase3_path,
                               reused_training_duration_s=phase3_training_duration,
                               reused_final_loss=phase3_final_loss)
        else:
            r = run_candidate(cfg, split.training, split.validation, _model_path_for(cfg.candidate_id))
        stage1_results.append(r)
        print(f"  {cfg.candidate_id}: trained/loaded, validation mean={r.validation_mean:.6f} "
              f"std={r.validation_std:.6f} status={r.selection.status}")
    print()
    _print_stage_table(stage1_results)
    stage1_selection = select_stage_winner(stage1_results)
    print(f"\n  Stage 1 winner: {stage1_selection.winner.config.candidate_id}")
    print(f"  Reasoning: {stage1_selection.reasoning}")
    best_image_size = stage1_selection.winner.config.training_config.image_size
    print()

    # ========================= STAGE 2: epochs =========================
    print(f"=== STAGE 2: epochs (15 baseline / 25 / 35) at input size {best_image_size} ===")
    stage2_new_results = []
    for cfg in stage2_candidates(best_image_size):
        r = run_candidate(cfg, split.training, split.validation, _model_path_for(cfg.candidate_id))
        stage2_new_results.append(r)
        print(f"  {cfg.candidate_id}: trained, validation mean={r.validation_mean:.6f} "
              f"std={r.validation_std:.6f} status={r.selection.status}")
    stage2_pool = [stage1_selection.winner] + stage2_new_results
    print()
    _print_stage_table(stage2_pool)
    stage2_selection = select_stage_winner(stage2_pool)
    print(f"\n  Stage 2 winner: {stage2_selection.winner.config.candidate_id}")
    print(f"  Reasoning: {stage2_selection.reasoning}")
    best_epochs = stage2_selection.winner.config.training_config.epochs
    print()

    # ========================= STAGE 3: learning rate =========================
    print(f"=== STAGE 3: learning rate (1e-3 baseline / 5e-4) at input={best_image_size} epochs={best_epochs} ===")
    lr_cfg = stage3_candidate(best_image_size, best_epochs)
    stage3_result = run_candidate(lr_cfg, split.training, split.validation, _model_path_for(lr_cfg.candidate_id))
    print(f"  {lr_cfg.candidate_id}: trained, validation mean={stage3_result.validation_mean:.6f} "
          f"std={stage3_result.validation_std:.6f} status={stage3_result.selection.status}")
    stage3_pool = [stage2_selection.winner, stage3_result]
    print()
    _print_stage_table(stage3_pool)
    stage3_selection = select_stage_winner(stage3_pool)
    print(f"\n  Stage 3 (overall) winner: {stage3_selection.winner.config.candidate_id}")
    print(f"  Reasoning: {stage3_selection.reasoning}")
    print()

    winning_config = stage3_selection.winner.config
    print(f"=== OVERALL SELECTED CONFIGURATION: {winning_config.candidate_id} ===")
    print(f"  image_size={winning_config.training_config.image_size} epochs={winning_config.training_config.epochs} "
          f"learning_rate={winning_config.training_config.learning_rate} seed={winning_config.training_config.seed}")
    print()

    # ================= Official reproducibility runs (2x, dedicated path) =================
    print("=== Retraining the selected configuration TWICE (dedicated, official runs) ===")
    official_path = get_model_path(CATEGORY, model_name="phase4_optimization/selected_candidate/autoencoder")
    from app.ai.training.phase4_candidates import CandidateConfig
    official_cfg = CandidateConfig(
        candidate_id="selected_candidate", stage="final_selected",
        training_config=winning_config.training_config,
        notes="Official Phase 4 selected candidate, trained fresh (not reusing any stage-exploration artifact).",
    )

    run1 = run_candidate(official_cfg, split.training, split.validation, official_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        run2_path = Path(tmpdir) / "repro" / "autoencoder.pt"
        run2 = run_candidate(official_cfg, split.training, split.validation, run2_path)

        split_identical = True  # same split object reused for both runs by construction
        config_identical = run1.config.training_config == run2.config.training_config
        model_hash_identical = run1.model_metadata.md5 == run2.model_metadata.md5
        validation_errors_identical = run1.validation_errors == run2.validation_errors
        threshold_identical = (
            run1.selection.selected is not None and run2.selection.selected is not None
            and run1.selection.selected.threshold == run2.selection.selected.threshold
        )
        print(f"  Run 1 MD5: {run1.model_metadata.md5}   Run 2 MD5: {run2.model_metadata.md5}")
        print(f"  Config identical: {config_identical}   Model hash identical: {model_hash_identical}")
        print(f"  Validation errors identical: {validation_errors_identical}")
        print(f"  Selected threshold identical: {threshold_identical}")

        # --- Final-test evaluation for BOTH runs, purely as a reproducibility check on the
        # --- already-locked configuration (same precedent as Phase 3) - never to compare or
        # --- choose between candidates; the winning config was already locked above.
        test_samples = discover_test_samples(CATEGORY)
        test_labels = [s.label for s in test_samples]

        def _final_test_eval(run):
            t0 = time.perf_counter()
            errors = []
            per_image_preprocess_ms = []
            per_image_inference_ms = []
            from app.ai.evaluation.evaluate import compute_reconstruction_error
            from app.ai.training.dataset import load_sample
            import torch
            for sample in test_samples:
                p0 = time.perf_counter()
                result = load_sample(sample, target_size=run.config.training_config.image_size)
                tensor = torch.from_numpy(result.preprocessing.normalized_image).permute(2, 0, 1).contiguous()
                p1 = time.perf_counter()
                error = compute_reconstruction_error(run.model, tensor)
                p2 = time.perf_counter()
                errors.append(error)
                per_image_preprocess_ms.append((p1 - p0) * 1000)
                per_image_inference_ms.append((p2 - p1) * 1000)
            total_ms = (time.perf_counter() - t0) * 1000
            candidate_result = evaluate_candidate_on_final_test(run.selection.selected, test_labels, errors)
            return candidate_result, total_ms, per_image_preprocess_ms, per_image_inference_ms

        test_result_1, test_total_ms_1, preprocess_ms_1, inference_ms_1 = _final_test_eval(run1)
        test_result_2, test_total_ms_2, _, _ = _final_test_eval(run2)

        predictions_identical = (
            (test_result_1.true_negatives, test_result_1.false_positives,
             test_result_1.false_negatives, test_result_1.true_positives)
            == (test_result_2.true_negatives, test_result_2.false_positives,
                test_result_2.false_negatives, test_result_2.true_positives)
        )
        metrics_identical = (
            (test_result_1.accuracy, test_result_1.precision, test_result_1.recall, test_result_1.f1_score)
            == (test_result_2.accuracy, test_result_2.precision, test_result_2.recall, test_result_2.f1_score)
        )
        print(f"  Final-test predictions identical across runs: {predictions_identical}")
        print(f"  Final-test metrics identical across runs: {metrics_identical}")
        print()

        overall_reproducible = (
            config_identical and model_hash_identical and validation_errors_identical
            and threshold_identical and predictions_identical and metrics_identical
        )
        if not overall_reproducible:
            print("!!! STOP: Phase 4 selected candidate is NOT reproducible across two independent "
                  "training runs. !!!")
            sys.exit(1)
        print("  Reproducibility verified (config/model hash/validation errors/threshold/predictions/"
              "metrics all identical). Using run 1 as the official Phase 4 result.")
        print()

    print("=== Phase 4 selected candidate: final-test result ===")
    print(f"  threshold={run1.selection.selected.threshold:.6f} accuracy={test_result_1.accuracy:.4f} "
          f"precision={test_result_1.precision:.4f} recall={test_result_1.recall:.4f} "
          f"f1={test_result_1.f1_score:.4f}")
    print(f"  TN={test_result_1.true_negatives} FP={test_result_1.false_positives} "
          f"FN={test_result_1.false_negatives} TP={test_result_1.true_positives}")
    print()

    # ============================= Phase 3 baseline (loaded, not recomputed) =============================
    phase3_data = json.loads(phase3_report_file.read_text(encoding="utf-8")) if phase3_report_file.is_file() else None
    phase3_final = None
    if phase3_data is not None:
        selected_summary = phase3_data["selection"]["selected_final_test_result"]
        phase3_final = selected_summary["final_test"]
        print(f"Phase 3 baseline (loaded from report, not recomputed): threshold={selected_summary['threshold']:.6f} "
              f"accuracy={phase3_final['accuracy']:.4f} precision={phase3_final['precision']:.4f} "
              f"recall={phase3_final['recall']:.4f} f1={phase3_final['f1_score']:.4f}")
    print()

    # ============================= Performance benchmarking =============================
    model_load_times_ms = []
    from app.ai.training.artifacts import load_model as _load_model
    from app.ai.training.model import build_model as _build_model
    for _ in range(5):
        t0 = time.perf_counter()
        _load_model(run1.model_path, _build_model(latent_channels=winning_config.training_config.model.latent_channels))
        model_load_times_ms.append((time.perf_counter() - t0) * 1000)

    from dataclasses import asdict as _asdict
    from app.ai.evaluation.benchmark_stats import summarize_timings

    def _stats(values):
        return _asdict(summarize_timings(values))

    performance = {
        "model_loading_ms": _stats(model_load_times_ms),
        "preprocessing_ms_per_image": _stats(preprocess_ms_1),
        "inference_ms_per_image": _stats(inference_ms_1),
        "total_final_test_ms": test_total_ms_1,
        "final_test_sample_count": len(test_samples),
        "training_duration_s": run1.training_duration_s,
        "model_file_size_bytes": run1.model_metadata.artifact_size_bytes,
        "environment": {
            "os": sys.platform,
            "python_version": sys.version.split()[0],
        },
    }
    import torch as _torch
    import cv2 as _cv2
    performance["environment"]["torch_version"] = _torch.__version__
    performance["environment"]["opencv_version"] = _cv2.__version__
    performance["environment"]["cuda_used"] = _torch.cuda.is_available()

    print("=== Performance ===")
    print(f"  Model loading (n={len(model_load_times_ms)}): mean={performance['model_loading_ms']['mean']:.2f}ms "
          f"median={performance['model_loading_ms']['median']:.2f}ms min={performance['model_loading_ms']['min']:.2f}ms "
          f"max={performance['model_loading_ms']['max']:.2f}ms")
    print(f"  Preprocessing per image (n={len(preprocess_ms_1)}): mean={performance['preprocessing_ms_per_image']['mean']:.2f}ms "
          f"median={performance['preprocessing_ms_per_image']['median']:.2f}ms")
    print(f"  Inference per image (n={len(inference_ms_1)}): mean={performance['inference_ms_per_image']['mean']:.2f}ms "
          f"median={performance['inference_ms_per_image']['median']:.2f}ms")
    print()

    # ============================= Manufacturing metrics =============================
    manufacturing_metrics = {
        "defect_identification_accuracy": {
            "definition": "recall on the final 83-image test set (TP / (TP+FN)) - fraction of actually "
                           "defective bottles correctly flagged",
            "value": test_result_1.recall,
        },
        "false_defect_detection_rate": {
            "definition": "false-positive rate on the final test set (FP / (FP+TN)) - fraction of "
                           "actually good bottles incorrectly flagged as defective",
            "value": test_result_1.false_positive_rate,
        },
        "inspection_automation_rate": "NOT AVAILABLE / NOT YET IMPLEMENTED - no existing definition of "
                                       "'automated' vs 'manual' inspection exists in the codebase or "
                                       "database; would require the application to explicitly record how "
                                       "each inspection's final status was decided (purely AI, "
                                       "AI-plus-human-confirmation, or fully manual) before this could be "
                                       "computed without inventing a definition.",
        "ai_analyzed_inspection_rate": None,  # filled below from live DB, if reachable
        "average_inspection_processing_time": "NOT AVAILABLE - app.models.inspection.Inspection has no "
                                                "persisted timing field; app.ai.inference.predict.PredictionResult."
                                                "processing_time_ms is computed at prediction time but is not "
                                                "written to the database, so no historical average exists. "
                                                "Would require adding a column and persisting it on each inspection.",
    }
    try:
        from app.database.session import SessionLocal
        from app.inspections.analytics import get_inspection_analytics_summary
        db = SessionLocal()
        try:
            summary = get_inspection_analytics_summary(db)
            total = summary.total_inspections if hasattr(summary, "total_inspections") else None
            ai_analyzed = summary.ai_analyzed_count
            manufacturing_metrics["ai_analyzed_inspection_rate"] = {
                "definition": "count(Inspection.ai_prediction IS NOT NULL) / count(*) - from the live "
                               "application database via the existing "
                               "app.inspections.analytics.get_inspection_analytics_summary (read-only, "
                               "not a Phase 4 computation)",
                "ai_analyzed_count": ai_analyzed,
                "total_inspections": total,
                "rate": (ai_analyzed / total) if total else None,
            }
        finally:
            db.close()
    except Exception as exc:
        manufacturing_metrics["ai_analyzed_inspection_rate"] = f"NOT AVAILABLE - could not query the live database: {exc}"

    print("=== Manufacturing metrics ===")
    print(f"  Defect identification accuracy (recall): {manufacturing_metrics['defect_identification_accuracy']['value']:.4f}")
    print(f"  False defect detection rate (FP rate): {manufacturing_metrics['false_defect_detection_rate']['value']:.4f}")
    print(f"  AI-analyzed inspection rate: {manufacturing_metrics['ai_analyzed_inspection_rate']}")
    print(f"  Inspection automation rate: {manufacturing_metrics['inspection_automation_rate']}")
    print(f"  Average inspection processing time: {manufacturing_metrics['average_inspection_processing_time']}")
    print()

    map_note = ("mAP is not applicable to the current reconstruction-based anomaly detection model "
                "because it does not produce object-level bounding boxes or segmentation detections.")
    production_status = "Phase 4 optimization candidate evaluated; production model not replaced."
    print(map_note)
    print(production_status)
    print()

    # ============================= Verify protected artifacts unchanged =============================
    original_md5_after = _file_md5(original_path)
    phase3_md5_after = _file_md5(phase3_path)
    print(f"Original model MD5 after Phase 4: {original_md5_after} (unchanged={original_md5_after == original_md5_before})")
    print(f"Phase 3 model MD5 after Phase 4:  {phase3_md5_after} (unchanged={phase3_md5_after == phase3_md5_before})")
    if original_md5_after != original_md5_before or phase3_md5_after != phase3_md5_before:
        print("!!! STOP: a protected model artifact was modified during Phase 4. !!!")
        sys.exit(1)
    print()

    # ============================= Build & save report =============================
    dataset_counts = {
        "train_good_count": stats.train_good_count, "test_good_count": stats.test_good_count,
        "test_defective_count": stats.test_defective_count, "test_total_count": stats.test_total_count,
        "test_defect_type_counts": stats.test_defect_type_counts,
    }
    split_summary = {"training_count": len(split.training), "validation_count": len(split.validation),
                      "training_fraction": split.training_fraction, "seed": split.seed}

    selected_model_metadata = compute_model_metadata(
        run1.model_path, category=CATEGORY, model_name="phase4_optimization/selected_candidate",
        latent_channels=winning_config.training_config.model.latent_channels,
    )

    final_test_metrics = {
        "threshold": run1.selection.selected.threshold,
        "method": run1.selection.selected.method,
        "parameter": run1.selection.selected.parameter,
        "accuracy": test_result_1.accuracy, "precision": test_result_1.precision,
        "recall": test_result_1.recall, "f1_score": test_result_1.f1_score,
        "true_negatives": test_result_1.true_negatives, "false_positives": test_result_1.false_positives,
        "false_negatives": test_result_1.false_negatives, "true_positives": test_result_1.true_positives,
        "false_positive_rate": test_result_1.false_positive_rate,
        "false_negative_rate": test_result_1.false_negative_rate,
    }

    phase3_baseline_summary = phase3_final if phase3_final is not None else "Phase 3 report not found"

    reproducibility = {
        "run_count": 2, "config_identical": config_identical, "model_hash_identical": model_hash_identical,
        "validation_errors_identical": validation_errors_identical, "threshold_identical": threshold_identical,
        "predictions_identical": predictions_identical, "metrics_identical": metrics_identical,
        "overall_reproducible": overall_reproducible,
    }

    limitations = [
        "Only 209 total normal images exist (167 training / 42 validation); every candidate shares this "
        "small pool, so differences between candidates are measured on a small validation set.",
        "The 42-image validation set contains only good/normal images - it cannot measure defect "
        "recall/precision/F1 or true class separation; stage selection relies on reconstruction-error "
        "consistency (coefficient of variation) and the 10% false-positive ceiling instead.",
        "The final test set is only 83 images (20 good, 63 defective); small differences in the final "
        "comparison table should be read as suggestive, not statistically significant.",
        "CPU-only benchmarking on a single machine; timing numbers are not representative of GPU or "
        "production-server performance and should not be used for capacity planning.",
        "The 10% validation false-positive ceiling is an inherited engineering constraint (from Phase 2/3), "
        "not a value derived from a project specification.",
    ]

    recommendation = None  # filled in the final written report below, based on measured results

    report = build_phase4_report(
        git_branch=branch, git_head=head, dataset_counts=dataset_counts, split=split_summary,
        stage1_results=stage1_results, stage1_selection_reasoning=stage1_selection.reasoning,
        stage2_results=stage2_new_results, stage2_selection_reasoning=stage2_selection.reasoning,
        stage3_result=stage3_result, stage3_decision_reasoning=stage3_selection.reasoning,
        selected_config_summary={
            "candidate_id": winning_config.candidate_id,
            "image_size": list(winning_config.training_config.image_size),
            "epochs": winning_config.training_config.epochs,
            "learning_rate": winning_config.training_config.learning_rate,
            "seed": winning_config.training_config.seed,
        },
        selected_model_metadata={
            "artifact_path": str(run1.model_path), "md5": selected_model_metadata.md5,
            "sha256": selected_model_metadata.sha256,
        },
        final_test_metrics=final_test_metrics,
        phase3_baseline=phase3_baseline_summary,
        reproducibility=reproducibility,
        performance=performance,
        manufacturing_metrics=manufacturing_metrics,
        map_note=map_note,
        production_status=production_status,
        limitations=limitations,
        recommendation="See the written Phase 4 report accompanying this JSON artifact.",
    )
    report_path = save_phase4_report(report, default_phase4_report_path(CATEGORY))
    print(f"Phase 4 report written to: {report_path}")
    print(f"Selected candidate model saved to: {run1.model_path}")


if __name__ == "__main__":
    main()
