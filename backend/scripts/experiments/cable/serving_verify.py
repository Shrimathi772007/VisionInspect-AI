"""Read-only verification of the locked Cable files + full ai_models snapshot. Usage: serving_verify.py before|after"""
import hashlib
import json
import sys
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
SCRATCH = Path(__file__).resolve().parent
ROOT = BACKEND / "ai_models"
STUDY = ROOT / "cable" / "model_family_study"
EXPECTED = {
    "model_sha256": "432859c5e39aaa85f8adbbd952f82ff0be1e6a905e360cfcca978a5638de7f5a",
    "model_md5": "c878207964fc22865743f77c43ed06ea",
    "config_sha256": "f6696e7fbb3af5f275d8678c66339210ec7230195c212cad70dd22777bcf3230",
    "manifest_sha256": "6bdd6543d1fd3a247673ab701fa85725a46e62685e52a54724dd375ab59093a3",
    "lock_sha256": "72c2bf182b71e8fefb9e5907cec9380989f0322caaf2bcb711be64230bef534e",
    "lock_digest": "31d1aed63843aa0e316c3343b5a82309b2b349fde803d93219e22d012bcbc675",
    "threshold": 2.244786389430004,
    "candidate": "knn_l23_128",
    "experiment_id": "cable-model-family-cable-ce2a8f015143",
}
# Cable Phase 1 hashes as recorded in the Phase 1 report and in the selection lock's prior_phase_artifacts
PHASE1_RECORDED = {
    "phase1_baseline/autoencoder.pt": "fc3456255a195fa12a28abf8e089a60847a107ce950058987eec075ab282432e",
    "phase1_baseline/threshold_lock.json": "f3b2ad40be4137559c85a06f2ff780e7142eb768515bbfe568d6731e4c69932d",
    "phase1_baseline/reports/phase1_report.json": "70a9562058663e99bfc03fa087d0aa512488e70e17adfc6156be95b0385a537d",
}


def h(path: Path, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def main() -> None:
    step = sys.argv[1]
    lock = json.loads((STUDY / "selection_lock.json").read_text(encoding="utf-8"))
    sel = STUDY / "selected_candidate"
    actual = {
        "model_sha256": h(sel / "model_state.pt"), "model_md5": h(sel / "model_state.pt", "md5"),
        "config_sha256": h(sel / "model_config.json"), "manifest_sha256": h(sel / "candidate_manifest.json"),
        "lock_sha256": h(STUDY / "selection_lock.json"), "lock_digest": lock["lock_digest"],
        "threshold": lock["selected_threshold"], "candidate": lock["selected_candidate"], "experiment_id": lock["experiment_id"],
    }
    checks = {k: actual[k] == v for k, v in EXPECTED.items()}
    checks["threshold_repr_exact"] = repr(lock["selected_threshold"]) == "2.244786389430004"
    checks["lock_artifact_sha256_matches_file"] = lock["selected_artifact"]["sha256"] == actual["model_sha256"]
    phase1 = {k: h(ROOT / "cable" / k) for k in PHASE1_RECORDED}
    checks["phase1_matches_recorded"] = phase1 == PHASE1_RECORDED
    checks["phase1_matches_lock_prior_phase_artifacts"] = phase1 == lock["prior_phase_artifacts"]
    snap = {str(p.relative_to(ROOT)).replace("\\", "/"): [p.stat().st_size, p.stat().st_mtime_ns, h(p)]
            for p in sorted(ROOT.rglob("*")) if p.is_file()}
    (SCRATCH / f"serving_verify_{step}.json").write_text(
        json.dumps({"actual": actual, "checks": checks, "phase1": phase1, "snapshot": snap}, indent=1), encoding="utf-8")
    print(json.dumps({"step": step, "actual": actual, "checks": checks, "phase1": phase1, "snapshot_files": len(snap)}, indent=1))
    if step == "after":
        before = json.loads((SCRATCH / "serving_verify_before.json").read_text(encoding="utf-8"))
        b = before["snapshot"]
        print(json.dumps({"snapshot_identical_size_mtime_sha256": b == snap,
                          "changed": sorted(k for k in b if k in snap and b[k] != snap[k]),
                          "added": sorted(set(snap) - set(b)), "removed": sorted(set(b) - set(snap)),
                          "locked_values_identical_to_before": before["actual"] == actual,
                          "phase1_identical_to_before": before["phase1"] == phase1}, indent=1))
    if not all(checks.values()):
        sys.exit("ABORT: locked value mismatch: " + str([k for k, v in checks.items() if not v]))


if __name__ == "__main__":
    main()
