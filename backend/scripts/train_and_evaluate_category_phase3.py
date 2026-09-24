"""Train + validate + evaluate ONE MVTec category with the validated Bottle Phase 3 methodology.

Category-parameterized counterpart of scripts/train_and_evaluate_bottle_phase3.py (which is left
untouched). All logic lives in app.ai.evaluation.category_phase3; this script only orchestrates:

  1. Refuse to run if the category already has a Phase 3 model or report (never overwrites -
     in particular it can never overwrite Bottle's validated artifacts).
  2. Run 1: split -> train -> validation errors -> LOCK threshold selection -> final test (once),
     saving the model to backend/ai_models/<category>/phase3_validation/autoencoder.pt.
  3. Run 2: the complete process again, into a scratch directory, for reproducibility.
  4. Compare the runs, audit leakage for both, and write
     backend/ai_models/<category>/evaluation_reports/phase3_validated_model_report.json.

Trains ONLY the category given. Does not register anything for production serving.

Usage (from backend/, venv active):
    python scripts/train_and_evaluate_category_phase3.py --category hazelnut
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.category_phase3 import (  # noqa: E402
    PHASE3_MODEL_NAME,
    audit_leakage,
    compare_runs,
    dataset_counts,
    ensure_no_overwrite,
    per_defect_type_recall,
    phase3_training_config,
    resolve_outputs,
    run_category_phase3,
    selected_final_metrics,
)
from app.ai.evaluation.category_phase3_report import build_category_phase3_report, protocol_status  # noqa: E402
from app.ai.evaluation.model_metadata import compute_model_metadata  # noqa: E402
from app.ai.evaluation.phase3_report import save_phase3_report  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"<unavailable: {exc}>"


def _print_table(rows: list[tuple], header: tuple) -> None:
    widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(len(header))]
    fmt = lambda row: " | ".join(str(v).rjust(widths[i]) for i, v in enumerate(row))  # noqa: E731
    print(fmt(header))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--category", required=True, help="MVTec category directory name, e.g. hazelnut")
    category = parser.parse_args().category

    branch, head = _git("rev-parse", "--abbrev-ref", "HEAD"), _git("rev-parse", "HEAD")
    print(f"Git branch: {branch}  HEAD: {head}\nCategory: {category}")

    outputs = resolve_outputs(category)
    ensure_no_overwrite(outputs)  # raises FileExistsError - never overwrite an existing model/report
    config = phase3_training_config(category)
    counts = dataset_counts(category)
    print(f"Dataset: {counts}")
    print(f"Config: image_size={config.image_size} batch={config.batch_size} epochs={config.epochs} "
          f"lr={config.learning_rate} seed={config.seed} device={config.device}\n")

    print("=== Run 1 (saved to the canonical artifact path) ===")
    run1 = run_category_phase3(category, outputs.model_path, config)
    print(f"  trained on {len(run1.split.training)}, validated on {len(run1.split.validation)}, "
          f"final test {len(run1.scored.test_labels)}")
    print(f"  training {run1.training_result.duration_seconds:.2f}s final loss {run1.training_result.final_loss:.6f}"
          f"  MD5 {run1.model_md5}")

    print("=== Run 2 (independent full re-run into a scratch directory) ===")
    with tempfile.TemporaryDirectory() as tmp:
        run2 = run_category_phase3(category, Path(tmp) / "phase3_repro_check" / "autoencoder.pt", config)
    print(f"  training {run2.training_result.duration_seconds:.2f}s final loss {run2.training_result.final_loss:.6f}"
          f"  MD5 {run2.model_md5}\n")

    reproducibility = compare_runs(run1, run2)
    leakage = [audit_leakage(run1), audit_leakage(run2)]
    status, problems = protocol_status(run1, leakage, reproducibility)

    print("=== Reproducibility ===")
    for key, value in reproducibility.items():
        print(f"  {key}: {value}")
    print("=== Leakage audit (run 1) ===")
    for key, value in leakage[0].items():
        print(f"  {key}: {value}")

    scored = run1.scored
    print(f"\n=== Validation ({len(scored.validation_errors)} images) ===")
    print(f"  errors: mean={scored.validation_mean:.6f} std={scored.validation_std:.6f} "
          f"min={min(scored.validation_errors):.6f} max={max(scored.validation_errors):.6f}")
    _print_table(
        [(s.candidate.method, s.candidate.parameter, f"{s.candidate.threshold:.6f}",
          s.validation_false_positive_count, f"{s.validation_false_positive_rate:.1%}", s.passes_stability_check)
         for s in scored.selection.stability_by_candidate],
        ("Method", "Param", "Threshold", "ValFP", "ValFPRate", "Passes"),
    )
    print(f"  selection: {scored.selection.status}")
    if scored.selection.selected is not None:
        sel = scored.selection.selected
        print(f"  selected: {sel.method} {sel.parameter} threshold={sel.threshold!r}")
    print(f"  {scored.selection.reasoning}")

    metrics = selected_final_metrics(scored)
    if metrics is not None:
        print(f"\n=== Final test at the locked threshold ({metrics['total_test_samples']} images) ===")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print(f"  per defect type: {json.dumps(per_defect_type_recall(scored))}")

    model_metadata = compute_model_metadata(
        run1.model_path, category=category, model_name=PHASE3_MODEL_NAME,
        latent_channels=config.model.latent_channels, device=config.device,
    )
    report = build_category_phase3_report(
        run=run1, model_metadata=model_metadata, git_branch=branch, git_head=head,
        leakage_runs=leakage, reproducibility=reproducibility, run2_total_ms=run2.total_ms,
    )
    path = save_phase3_report(report, outputs.report_path)
    print(f"\nProtocol status: {status} {problems}")
    print(f"MD5 {model_metadata.md5}\nSHA-256 {model_metadata.sha256}\nsize {model_metadata.artifact_size_bytes} bytes")
    print(f"Model: {run1.model_path}\nReport: {path}")


if __name__ == "__main__":
    main()
