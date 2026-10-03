"""Tile wrapper around the UNCHANGED backend/scripts/run_model_family_study.py (study | lock | repro | audit).

Only additions, all runtime-only (no project file is edited):
  * Tile's expected dataset counts are injected into EXPECTED_BY_CATEGORY (the same dict Leather/Metal Nut/Pill use).
  * Final-test guards: discover_test_samples is replaced by a function that raises, and every image decode through
    the project loader is recorded and REFUSED if it lies under dataset/tile/test. Directory-entry counting
    (count_dataset_entries, os.scandir, no decode) is the only permitted touch of the test tree, as in the framework.
  * A hash snapshot of every artifact outside tile/model_family_study, compared at `audit`.
The `final` command is deliberately not exposed.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

import run_model_family_study as runner  # noqa: E402
from app.ai.evaluation import model_family_final, model_family_pipeline  # noqa: E402
from app.ai.evaluation.category_phase1 import file_hashes  # noqa: E402
from app.ai.preprocessing import pipeline as prep  # noqa: E402
from app.ai.training import artifacts, dataset  # noqa: E402

CATEGORY = "tile"
runner.EXPECTED_BY_CATEGORY[CATEGORY] = {
    "train_good": 230, "test_good": 33, "test_defective": 84, "test_total": 117,
    "test_by_type": {"crack": 17, "glue_strip": 18, "good": 33, "gray_stroke": 16, "oil": 18, "rough": 15}}

DECODED: list[str] = []
TEST_ROOT = (BACKEND.parent / "dataset" / CATEGORY / "test").resolve()


def _forbidden(*_a, **_k):
    raise RuntimeError("FINAL-TEST GUARD: discover_test_samples called during the normal-only study.")


_real_load = prep._load_image


def _guarded_load(path):
    p = Path(path).resolve()
    if TEST_ROOT in p.parents:
        raise RuntimeError(f"FINAL-TEST GUARD: attempted to decode {p}")
    DECODED.append(str(p))
    return _real_load(path)


dataset.discover_test_samples = _forbidden
model_family_final.discover_test_samples = _forbidden
prep._load_image = _guarded_load
model_family_pipeline._load_image = _guarded_load


def snapshot() -> dict:
    own = model_family_pipeline.study_root(CATEGORY).resolve()
    return {str(p.relative_to(artifacts.ARTIFACTS_ROOT)).replace("\\", "/"): file_hashes(p)["sha256"]
            for p in sorted(artifacts.ARTIFACTS_ROOT.rglob("*")) if p.is_file() and own not in p.resolve().parents}


def record_decodes(step: str) -> None:
    train_dir = (BACKEND.parent / "dataset" / CATEGORY / "train" / "good").resolve()
    out = SCRATCH / f"decodes_{step}.json"
    out.write_text(json.dumps({"step": step, "decode_calls": len(DECODED), "unique_files": len(set(DECODED)),
                               "all_under_train_good": all(Path(p).parent == train_dir for p in DECODED),
                               "any_under_test": any(TEST_ROOT in Path(p).parents for p in DECODED)}, indent=1))
    print(out.read_text())


def main() -> None:
    step = sys.argv[1]
    snap = SCRATCH / "artifact_snapshot_before.json"
    if step == "study":
        if not snap.exists():
            snap.write_text(json.dumps(snapshot(), indent=1))
        runner.cmd_study(CATEGORY)
        record_decodes(step)
    elif step == "lock":
        runner.cmd_lock(CATEGORY)
    elif step == "repro":
        runner.cmd_repro(CATEGORY)
        record_decodes(step)
    elif step == "audit":
        before, after = json.loads(snap.read_text()), snapshot()
        print(json.dumps({"artifacts_outside_study_dir": len(after), "unchanged": before == after,
                          "changed": sorted(k for k in before if before.get(k) != after.get(k)),
                          "added": sorted(set(after) - set(before)),
                          "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "runner_script_sha256": hashlib.sha256((BACKEND / "scripts" / "run_model_family_study.py").read_bytes()).hexdigest(),
                          "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    else:
        sys.exit("usage: study | lock | repro | audit")


if __name__ == "__main__":
    main()
