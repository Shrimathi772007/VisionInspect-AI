"""Phase 1 baseline for one MVTec category (existing ConvAutoencoder, K=3 train threshold, one test scoring),
run twice for a reproducibility comparison.

Run 1 writes the real artifact backend/ai_models/<category>/phase1_baseline/autoencoder.pt (refusing to overwrite),
a threshold lock record, and reports/phase1_report.json. Run 2 repeats the identical experiment into a temporary
directory (its model is discarded after hashing) and is only compared with run 1.

Nothing is registered for serving; no threshold other than K=3 is computed; no other phase is run.

Usage (from backend/, venv active):
    python scripts/run_category_phase1.py --category leather --train-good 245 --test-good 32 --test-defective 92
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.category_phase1 import (  # noqa: E402
    compare_runs,
    default_model_path,
    file_hashes,
    phase1_dir,
    run_phase1,
    verify_dataset_counts,
)
from app.ai.training import artifacts  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _existing_artifact_hashes(category: str) -> dict:
    """SHA-256 of every model artifact outside this category's phase1_baseline directory."""
    own = phase1_dir(category).resolve()
    out = {}
    for path in sorted(artifacts.ARTIFACTS_ROOT.rglob("*")):
        if path.is_file() and path.suffix in {".pt", ".pth", ".json"} and own not in path.resolve().parents:
            out[str(path.relative_to(artifacts.ARTIFACTS_ROOT)).replace("\\", "/")] = file_hashes(path)["sha256"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--train-good", type=int, required=True)
    parser.add_argument("--test-good", type=int, required=True)
    parser.add_argument("--test-defective", type=int, required=True)
    parser.add_argument("--defect-types", default=None,
                        help="optional 'name=count,...' (good included); STOP if the enumerated test folders differ")
    args = parser.parse_args()
    category = args.category
    expected = {"train_good": args.train_good, "test_good": args.test_good, "test_defective": args.test_defective,
                "test_total": args.test_good + args.test_defective}

    counts = verify_dataset_counts(category, expected)  # raises (before any training) on a mismatch
    print("Dataset counts verified:", json.dumps(counts))
    if args.defect_types:
        declared = {k: int(v) for k, v in (item.split("=") for item in args.defect_types.split(","))}
        if declared != counts["test_by_type"]:
            sys.exit(f"STOP: test folders {counts['test_by_type']} differ from the declared {declared}")
    model_path = default_model_path(category)
    if model_path.exists() or (phase1_dir(category) / "reports" / "phase1_report.json").exists():
        sys.exit(f"STOP: Phase 1 output already exists under {phase1_dir(category)}; refusing to overwrite.")

    before = _existing_artifact_hashes(category)
    print(f"Snapshot of {len(before)} pre-existing artifact files taken.")

    print("=== RUN 1 (real artifact) ===")
    run1 = run_phase1(category, model_path, expected, lock_path=phase1_dir(category) / "threshold_lock.json")
    print(f"  trained in {run1['training']['duration_seconds']:.1f}s, final loss {run1['training']['final_loss']:.6f}, "
          f"threshold {run1['threshold']['value']!r}, sha256 {run1['artifact']['sha256']}")

    print("=== RUN 2 (identical experiment, temporary artifact) ===")
    with tempfile.TemporaryDirectory() as tmp:
        run2 = run_phase1(category, Path(tmp) / "autoencoder.pt", expected)
    print(f"  threshold {run2['threshold']['value']!r}, sha256 {run2['artifact']['sha256']}")
    reproducibility = compare_runs(run1, run2)
    print(json.dumps(reproducibility, indent=1))

    after = _existing_artifact_hashes(category)
    run1["leakage_partial"]["9_existing_artifacts_not_overwritten"] = before == after
    run1["leakage_partial"]["9b_serving_registry_unchanged"] = category.lower() not in (
        Path(__file__).resolve().parent.parent / "app" / "ai" / "inference" / "serving.py").read_text(encoding="utf-8").lower()

    report = {"git_head": _git("rev-parse", "HEAD"), "run1": run1, "run2_summary": {
        "artifact": run2["artifact"] | {"note": "temporary file, deleted after hashing"},
        "threshold": run2["threshold"]["value"], "metrics": run2["metrics"], "loss_history": run2["training"]["loss_history"],
        "duration_seconds": run2["training"]["duration_seconds"]}, "reproducibility": reproducibility,
        "preexisting_artifact_count": len(before)}
    out = phase1_dir(category) / "reports" / "phase1_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    hide = ("predictions", "train_errors", "training_files")
    print("\n=== RUN 1 RESULT ===")
    print(json.dumps({k: v for k, v in run1.items() if k not in hide}, indent=1))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
