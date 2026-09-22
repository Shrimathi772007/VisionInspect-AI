"""Phase 2 threshold-robustness / calibration study for the already-selected, FROZEN model-family candidate of one
category. Never touches the protected final test; never changes the candidate, the selection lock, the selected
artifact, or the final-test result. See app.ai.evaluation.phase2_threshold_calibration for the pre-declared rule.

  study   verify the frozen state, run the threshold study (normal-only splits + synthetic diagnostic) for the one
          locked candidate, verify the frozen state again, and write reports/phase2_threshold_report.json.
  repro   repeat the COMPLETE study into a scratch directory and require it to reproduce the first run; writes
          reports/phase2_reproducibility.json.

Usage (from backend/, venv active):
    python scripts/run_phase2_threshold_calibration.py study|repro --category pill
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.model_family_pipeline import backbone_info, load_inputs  # noqa: E402
from app.ai.evaluation.phase2_threshold_calibration import (  # noqa: E402
    compare_threshold_studies,
    frozen_candidate_spec,
    phase2_candidates_dir,
    phase2_report_path,
    phase2_reports_dir,
    read_frozen_state,
    run_threshold_study,
    verify_frozen_state_unchanged,
)

EXPECTED_BY_CATEGORY = {
    "pill": {"train_good": 267, "test_good": 26, "test_defective": 141, "test_total": 167,
             "test_by_type": {"color": 25, "combined": 17, "contamination": 21, "crack": 26, "faulty_imprint": 19,
                              "good": 26, "pill_type": 9, "scratch": 24}},
}
BACKBONE_SIZE_BYTES = 46_830_571
EXPECTED_ARTIFACT_SHA256 = {"pill": "6222a2fc968f73b9cd1081f0377b9850d9400daa2f4857bb56266ec3f0f5c469"}
EXPECTED_LOCK_SHA256 = {"pill": "fb36b05648ffab99fab00a5a9c020f4a096208185ac39c9d2cb6028ceb669a4a"}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _check_backbone() -> None:
    info = backbone_info()  # raises on a SHA-256 mismatch
    if info["size_bytes"] != BACKBONE_SIZE_BYTES:
        sys.exit(f"STOP: backbone size {info['size_bytes']} != {BACKBONE_SIZE_BYTES}")


def _check_frozen_before(category: str) -> dict:
    before = read_frozen_state(category)
    expected_artifact, expected_lock = EXPECTED_ARTIFACT_SHA256.get(category), EXPECTED_LOCK_SHA256.get(category)
    if not before["selection_lock_digest_valid"]:
        sys.exit("STOP: the selection lock's digest does not match its content.")
    if expected_artifact and before["selected_artifact_sha256"] != expected_artifact:
        sys.exit(f"STOP: selected artifact sha256 {before['selected_artifact_sha256']} != expected {expected_artifact}")
    if expected_lock and before["selection_lock_sha256"] != expected_lock:
        sys.exit(f"STOP: selection lock sha256 {before['selection_lock_sha256']} != expected {expected_lock}")
    return before


def cmd_study(category: str) -> None:
    if phase2_report_path(category).exists() or phase2_candidates_dir(category).exists():
        sys.exit("STOP: Phase 2 output already exists; refusing to overwrite.")
    _check_backbone()
    before = _check_frozen_before(category)
    spec, baseline_policy, lock = frozen_candidate_spec(category)
    print(f"frozen candidate {spec.candidate_id!r} | baseline policy {baseline_policy!r} | locked threshold {lock['selected_threshold']!r}")
    inputs = load_inputs(category, EXPECTED_BY_CATEGORY[category])
    print(f"inputs ready in {inputs.build_seconds:.0f}s; counts {json.dumps(inputs.counts)}", flush=True)
    report = run_threshold_study(inputs, spec, baseline_policy, phase2_candidates_dir(category), log=lambda m: print(m, flush=True))
    after = read_frozen_state(category)
    frozen_check = verify_frozen_state_unchanged(before, after)
    if not frozen_check["all_passed"]:
        sys.exit(f"STOP: frozen state changed during the Phase 2 study: {frozen_check}")
    report["git_head"] = _git("rev-parse", "HEAD")
    report["frozen_state_before"], report["frozen_state_after"], report["frozen_state_check"] = before, after, frozen_check
    report["recomputed_artifact_matches_locked_selected_artifact"] = report["recomputed_artifact_sha256"] == before["selected_artifact_sha256"]
    phase2_reports_dir(category).mkdir(parents=True, exist_ok=True)
    phase2_report_path(category).write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\nSELECTION: {report['selection']['classification']} | selected policy {report['selection']['selected_policy']} "
          f"threshold {report['selection']['selected_threshold']!r}")
    print(report["selection"]["reasoning"])
    print(f"recomputed artifact matches locked selected artifact: {report['recomputed_artifact_matches_locked_selected_artifact']}")
    print(f"\nWrote {phase2_report_path(category)}")


def cmd_repro(category: str) -> None:
    out = phase2_reports_dir(category) / "phase2_reproducibility.json"
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    first = json.loads(phase2_report_path(category).read_text(encoding="utf-8"))
    before = _check_frozen_before(category)
    spec, baseline_policy, _ = frozen_candidate_spec(category)
    with tempfile.TemporaryDirectory() as tmp:
        _check_backbone()
        inputs = load_inputs(category, EXPECTED_BY_CATEGORY[category])
        second = json.loads(json.dumps(run_threshold_study(inputs, spec, baseline_policy, Path(tmp) / "candidates",
                                                            log=lambda m: print(m, flush=True)), sort_keys=True))
    after = read_frozen_state(category)
    frozen_check = verify_frozen_state_unchanged(before, after)
    repro = compare_threshold_studies(first, second)
    repro["frozen_state_unchanged"] = frozen_check["all_passed"]
    repro["second_run_seconds"] = second["study_seconds"]
    print(json.dumps(repro, indent=1))
    out.write_text(json.dumps(repro, indent=1), encoding="utf-8")
    if not all(v for v in repro.values() if isinstance(v, bool)):
        sys.exit("!!! Run 2 did not reproduce run 1, or the frozen state changed; STOP and investigate. !!!")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["study", "repro"])
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    {"study": cmd_study, "repro": cmd_repro}[args.command](args.category)


if __name__ == "__main__":
    main()
