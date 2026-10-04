"""Pure logic of the WRN-50-2 PatchCore final test (FINAL_TEST_DECLARATION.md).

Everything here is deterministic and testable without images or weights: which lock each category is scored
with, lock verification (digest, model state, config, selection table re-derivation, protected-file hashes),
the test-only dataset path guard, test listing and labels, the prediction rule, the metrics, and the one-shot
sentinel/consumed guard. run_wrn50_final.py only wires these together with the backbone and the filesystem.
"""

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import grid_full320_logic as g
import wrn50_study_logic as logic
from app.ai.evaluation.category_phase1 import gate_classification
from app.ai.models.patchcore import CONFIG_FILENAME, STATE_FILENAME
from app.ai.models.wide_resnet50 import WRN50_2_SHA256
from app.inspections.storage import DATASET_ROOT

HERE = Path(__file__).resolve().parent
RUN_ORDER = ("bottle", "capsule", "zipper", "wood", "grid", "pill", "carpet", "screw", "hazelnut")
STANDARD_DIRNAME = "patchcore_wrn50"
GRID_FULL320_DIRNAME = "patchcore_wrn50_full320"
FINAL_DIRNAME = "final_test"
TEST_SPLIT = "test"
GOOD = "good"
IMAGE_SUFFIX = ".png"
FORBIDDEN_PART = "ground_truth"  # the only place this name is spelled; every path builder checks against it

# Lock digests recorded BEFORE any test scoring and committed with the declaration. A lock and digest edited
# together are therefore still rejected.
EXPECTED_LOCK_DIGESTS = {
    "bottle": "830aa484045ed1b900b4435937091ee0c4d100112e201b6a08ba841e440d1550",
    "capsule": "7821104664f380b69342ca579d18007badf4a4eabad9a49e19fbe95a5e9cf22f",
    "zipper": "2a72a8fdf8163b0772165dd17fd60e040d2652ad4ee4518bcfe3ca2d7c53a9ab",
    "wood": "49b048415e53e2b324fd503e9062ffdfa3ca38c0953c16dd071c21e28f63c842",
    "grid": "4cdf78b2b2b6563bcb8b96693e8bf32b5fa99040444c4599a0c284efc5919efc",  # Addendum 1 full320 lock
    "pill": "072271786f13f887f8815d6f38b488d778d727b839b38d26fcea0efd9f463f18",
    "carpet": "2527ed6ac34680c51d2f3298f9c5c6697f9a16e160fccd7d7efc530d1e085127",
    "screw": "0857280ca50e702acef1768040f04e192daab07175f1dacca3efbdfb3d4fb07b",
    "hazelnut": "894cc81a91859f45e5b6f0c8b73e9100adfc75c099ac717dabfc19f47814cfc1",
}
GRID_SUPERSEDED_LOCK_DIGEST = "de0bd22acf60ca5abee0a32cffe1980f54f2dcda7ddb1113acccd7d288c8e263"  # verified, never scored
GRID_FULL320_THRESHOLD = 1.88393018105021


class LockVerificationError(Exception):
    """A lock, its model or a protected file differs from what was locked."""


class DatasetAccessError(Exception):
    """A dataset path outside dataset/<category>/test/<subfolder>/ was requested, or one naming the mask folder."""


class OneShotError(Exception):
    """The final test of this category was already started or consumed."""


# ---------------------------------------------------------------------------
# Which model each category is scored with
# ---------------------------------------------------------------------------

def lock_dir(category: str, models_root: Path) -> Path:
    """Folder holding the lock of the model that is scored: grid -> the Addendum 1 full320 lock."""
    if category not in RUN_ORDER:
        raise ValueError(f"{category!r} is not one of the nine WRN-50 categories.")
    return Path(models_root) / category / (GRID_FULL320_DIRNAME if category == "grid" else STANDARD_DIRNAME)


def model_label(category: str) -> str:
    return "WRN-50 full320 (addendum 1)" if category == "grid" else "WRN-50"


# ---------------------------------------------------------------------------
# Lock verification
# ---------------------------------------------------------------------------

def _check(condition: bool, message: str) -> None:
    if not condition:
        raise LockVerificationError(message)


def _verify_model_files(lock: dict, directory: Path) -> dict:
    model_dir = directory / "final_model"
    state_sha = logic.sha256_file(model_dir / STATE_FILENAME)
    config_sha = logic.sha256_file(model_dir / CONFIG_FILENAME)
    _check(state_sha == lock["model_state_sha256"], "model_state.pt does not match the lock's model_state_sha256")
    _check(config_sha == lock["model_config_sha256"], "model_config.json does not match the lock's model_config_sha256")
    record = json.loads((model_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
    _check(record.get("state_sha256") == state_sha, "model_config.json's state_sha256 differs from the state file")
    _check({k: record.get(k) for k in lock["config"]} == lock["config"], "model_config.json differs from the lock's config")
    _check(record.get("bank_shape") == lock["final_bank_shape"], "bank shape differs from the lock")
    _check(lock["backbone_weights_sha256"] == WRN50_2_SHA256 == record.get("weights_sha256"), "backbone weights hash differs")
    _check(lock["config"]["preprocessing"] == lock["mode"] and lock["config"]["aggregation"] == lock["aggregation"],
           "lock mode/aggregation differ from its config")
    _check(lock.get("test_directory_listed") is False, "lock does not certify that the test directory was unlisted")
    return {"model_state_sha256": state_sha, "model_config_sha256": config_sha}


def _decision_matches(decision: dict, lock: dict) -> bool:
    return all(decision[k] == lock[k] for k in ("mode", "aggregation", "policy", "threshold"))


def verify_standard_lock(directory: Path, category: str, study_dir: Path = HERE, expected_digest: str | None = None) -> tuple[dict, dict]:
    """Digest (content and expected value), model state + config, selection table hash AND re-derived selection,
    and the SHA-256 of every protected file the lock records. Returns a verification record."""
    directory = Path(directory)
    lock = json.loads((directory / "lock.json").read_text(encoding="utf-8"))
    _check(lock.get("category") == category, f"lock is for {lock.get('category')!r}, not {category!r}")
    _check(logic.lock_digest(lock) == lock.get("lock_digest"), "lock content does not match its lock_digest")
    if expected_digest is not None:
        _check(lock["lock_digest"] == expected_digest, "lock_digest differs from the digest declared before scoring")
    _check(lock["mode"] in logic.MODES, f"unexpected mode {lock['mode']!r} for a standard lock")
    files = _verify_model_files(lock, directory)
    table_path = directory / "selection_table.json"
    _check(logic.sha256_file(table_path) == lock["selection_table_sha256"], "selection_table.json differs from the lock")
    rows = json.loads(table_path.read_text(encoding="utf-8"))["rows"]
    _check(_decision_matches(logic.select(rows), lock), "the selection rule applied to the saved table does not give the lock")
    protected = {"runner_sha256": "run_wrn50_study.py", "logic_sha256": "wrn50_study_logic.py",
                 "protocol_sha256": "PROTOCOL.md", "declaration_sha256": "STUDY_DECLARATION.md"}
    for key, name in protected.items():
        _check(logic.sha256_file(Path(study_dir) / name) == lock[key], f"protected file {name} differs from the lock")
    return {"lock_dir": str(directory), "lock_digest": lock["lock_digest"], "lock_file_sha256": logic.sha256_file(directory / "lock.json"),
            **files, "selection_table_sha256": lock["selection_table_sha256"], "selection_rederived": True,
            "protected_files_verified": sorted(protected.values()), "verified": True}, lock


def verify_grid_full320_lock(directory: Path, original_dir: Path, study_dir: Path = HERE,
                             expected_digest: str | None = EXPECTED_LOCK_DIGESTS["grid"],
                             expected_superseded: str | None = GRID_SUPERSEDED_LOCK_DIGEST) -> tuple[dict, dict, dict]:
    """The Addendum 1 lock: digest, model, combined table (hash, provenance, re-derived selection), every protected
    file it records, decision.json; and the ORIGINAL Grid lock must still verify (it is never scored)."""
    directory = Path(directory)
    original_record, original = verify_standard_lock(original_dir, "grid", study_dir, expected_superseded)
    lock = json.loads((directory / "lock.json").read_text(encoding="utf-8"))
    _check(lock.get("category") == "grid" and lock.get("study") == "patchcore_wrn50_addendum1_grid_full320", "not the Grid full320 lock")
    _check(logic.lock_digest(lock) == lock.get("lock_digest"), "full320 lock content does not match its lock_digest")
    if expected_digest is not None:
        _check(lock["lock_digest"] == expected_digest, "full320 lock_digest differs from the digest declared before scoring")
    _check(lock["mode"] == g.MODE_FULL320 and lock["config"]["input_size"] == g.FULL320_SIZE, "full320 lock is not full320")
    _check(lock["superseded_lock_digest"] == original["lock_digest"], "full320 lock does not supersede the original Grid lock")
    files = _verify_model_files(lock, directory)
    table_path = directory / "selection_table_combined.json"
    _check(logic.sha256_file(table_path) == lock["selection_table_combined_sha256"], "selection_table_combined.json differs from the lock")
    table = json.loads(table_path.read_text(encoding="utf-8"))
    _check(table["existing_rows_sha256"] == original["selection_table_sha256"] and table["existing_lock_digest"] == original["lock_digest"],
           "combined table does not come from the original Grid lock's table")
    _check(_decision_matches(g.select_combined(table["rows"]), lock), "the combined rule applied to the saved table does not give the lock")
    protected = {"addendum_runner_sha256": "run_wrn50_grid_full320.py", "addendum_logic_sha256": "grid_full320_logic.py",
                 "addendum_sha256": "ADDENDUM_1_GRID_FULL320.md", "protocol_sha256": "PROTOCOL.md",
                 "declaration_sha256": "STUDY_DECLARATION.md", "study_runner_sha256": "run_wrn50_study.py",
                 "study_logic_sha256": "wrn50_study_logic.py"}
    for key, name in protected.items():
        _check(logic.sha256_file(Path(study_dir) / name) == lock[key], f"protected file {name} differs from the full320 lock")
    decision = json.loads((directory / "decision.json").read_text(encoding="utf-8"))
    _check(decision.get("full320_won") is True and _decision_matches(decision, lock), "decision.json does not name this full320 model")
    record = {"lock_dir": str(directory), "lock_digest": lock["lock_digest"], "lock_file_sha256": logic.sha256_file(directory / "lock.json"),
              **files, "selection_table_combined_sha256": lock["selection_table_combined_sha256"], "selection_rederived": True,
              "protected_files_verified": sorted(protected.values()), "decision_json_full320_won": True, "verified": True,
              "original_grid_lock": {**original_record, "used_for_scoring": False}}
    return record, lock, original


def verify_category_lock(category: str, models_root: Path, study_dir: Path = HERE) -> tuple[dict, dict]:
    """(verification record, lock) of the model that category is scored with. Raises LockVerificationError."""
    directory = lock_dir(category, models_root)
    if category == "grid":
        record, lock, _ = verify_grid_full320_lock(directory, Path(models_root) / "grid" / STANDARD_DIRNAME, study_dir)
        _check(lock["threshold"] == GRID_FULL320_THRESHOLD and lock["aggregation"] == "max", "Grid full320 threshold/aggregation differ")
        return record, lock
    record, lock = verify_standard_lock(directory, category, study_dir, EXPECTED_LOCK_DIGESTS[category])
    return record, lock


# ---------------------------------------------------------------------------
# Test-only dataset paths, listing and labels
# ---------------------------------------------------------------------------

def _forbidden(part: str) -> bool:
    return FORBIDDEN_PART in part.lower()


def guarded_path(category: str, subfolder: str | None = None, filename: str | None = None,
              dataset_root: Path = DATASET_ROOT, split: str = TEST_SPLIT) -> Path:
    """THE only way the final test builds dataset paths: dataset/<category>/<split>[/<subfolder>[/<filename>]].
    `split` is "test" for scoring ("train/good" only for the dry run). Refuses any part naming the mask folder, any
    part with path structure, and any resolved path outside dataset/<category>/<split>."""
    parts = [category, *split.split("/")]
    for extra in (subfolder, filename):
        if extra is None:
            continue
        if not extra or any(sep in extra for sep in ("/", "\\")) or extra in (".", ".."):
            raise DatasetAccessError(f"{extra!r} must be a plain name.")
        parts.append(extra)
    if filename is not None and subfolder is None and split == TEST_SPLIT:
        raise DatasetAccessError("A test file needs its subfolder.")
    if any(_forbidden(p) for p in parts):
        raise DatasetAccessError(f"Refusing a dataset path naming {FORBIDDEN_PART}: {'/'.join(parts)}")
    if split not in (TEST_SPLIT, "train/good"):
        raise DatasetAccessError(f"Unknown split {split!r}.")
    base = (Path(dataset_root) / category / split).resolve()
    path = Path(dataset_root).joinpath(*parts).resolve()
    if path != base and not path.is_relative_to(base):
        raise DatasetAccessError(f"Path escapes {category}/{split}: {'/'.join(parts)}")
    return path


def label_from_folder(subfolder: str) -> tuple[int, str]:
    """(label, defect type): "good" -> (0, "good"); any other test subfolder -> (1, its name)."""
    if _forbidden(subfolder):
        raise DatasetAccessError(f"Refusing a test subfolder named {subfolder!r}.")
    return (0, GOOD) if subfolder == GOOD else (1, subfolder)


@dataclass(frozen=True)
class EvalImage:
    relative: str  # relative to the dataset root, e.g. "bottle/test/broken_large/000.png"
    label: int  # 0 good, 1 defective
    defect_type: str
    path: Path


def list_test_images(category: str, dataset_root: Path = DATASET_ROOT, record=None) -> tuple[list[EvalImage], dict]:
    """Sorted subfolders of dataset/<category>/test, then sorted *.png files in each. `record(action, path)` is
    called for every listing. Returns (images, listing summary incl. ignored non-png entries)."""
    root = guarded_path(category, dataset_root=dataset_root)
    if record:
        record("list", root)
    with os.scandir(root) as entries:
        subfolders = sorted(e.name for e in entries if e.is_dir())
    images, ignored = [], []
    for sub in subfolders:
        label, defect_type = label_from_folder(sub)
        directory = guarded_path(category, sub, dataset_root=dataset_root)
        if record:
            record("list", directory)
        with os.scandir(directory) as entries:
            names = sorted(e.name for e in entries if e.is_file())
        for name in names:
            if not name.lower().endswith(IMAGE_SUFFIX):
                ignored.append(f"{sub}/{name}")
                continue
            path = guarded_path(category, sub, name, dataset_root=dataset_root)
            images.append(EvalImage(f"{category}/{TEST_SPLIT}/{sub}/{name}", label, defect_type, path))
    return images, {"subfolders": subfolders, "ignored_non_png": ignored}


def predict(score: float, threshold: float) -> int:
    """Serving rule: score <= threshold -> good (0); score > threshold -> defective (1)."""
    return 1 if score > threshold else 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion(labels, predictions) -> dict:
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if labels.shape != predictions.shape or not set(np.unique(labels)) <= {0, 1} or not set(np.unique(predictions)) <= {0, 1}:
        raise ValueError("labels and predictions must be equal-length 0/1 sequences")
    return {"true_positives": int(((labels == 1) & (predictions == 1)).sum()),
            "true_negatives": int(((labels == 0) & (predictions == 0)).sum()),
            "false_positives": int(((labels == 0) & (predictions == 1)).sum()),
            "false_negatives": int(((labels == 1) & (predictions == 0)).sum())}


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Defective is the positive class. Undefined ratios (zero denominator) are reported as 0.0 (sklearn's
    zero_division=0 convention; MCC likewise 0)."""
    total = tp + tn + fp + fn
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    fpr, specificity = _ratio(fp, fp + tn), _ratio(tn, fp + tn)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": _ratio(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1_score": _ratio(2 * precision * recall, precision + recall),
        "f2_score": _ratio(5 * precision * recall, 4 * precision + recall),
        "false_positive_rate": fpr,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": _ratio(tp * tn - fp * fn, mcc_den),
        "true_positives": tp, "true_negatives": tn, "false_positives": fp, "false_negatives": fn,
        "total": total,
    }


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    """95% Wilson score interval for k successes in n trials (same formula as the Tile/Cable runners); None if n == 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def ranking_metrics(labels, scores) -> dict:
    """AUROC (ties count one half) and average precision (step-wise, sklearn) of the image scores; None when the
    labels hold only one class."""
    labels = np.asarray(labels, dtype=int)
    if len(set(labels.tolist())) < 2:
        return {"auroc": None, "average_precision": None}
    scores = np.asarray(scores, dtype=np.float64)
    return {"auroc": float(roc_auc_score(labels, scores)), "average_precision": float(average_precision_score(labels, scores))}


def per_defect_recall(labels, defect_types, predictions) -> dict:
    out: dict[str, dict] = {}
    for label, defect_type, prediction in zip(labels, defect_types, predictions):
        if label != 1:
            continue
        entry = out.setdefault(defect_type, {"total": 0, "detected": 0})
        entry["total"] += 1
        entry["detected"] += int(prediction)
    for entry in out.values():
        entry["missed"] = entry["total"] - entry["detected"]
        entry["recall"] = entry["detected"] / entry["total"]
        entry["wilson95"] = wilson(entry["detected"], entry["total"])
    return dict(sorted(out.items()))


def evaluate(labels, defect_types, scores, threshold: float) -> dict:
    """Everything final_test_result.json reports about the predictions, from (labels, types, scores, threshold)."""
    predictions = [predict(float(s), threshold) for s in scores]
    cells = confusion(labels, predictions)
    metrics = binary_metrics(cells["true_positives"], cells["true_negatives"], cells["false_positives"], cells["false_negatives"])
    return {
        "predictions": predictions,
        "metrics": metrics,
        **ranking_metrics(labels, scores),
        "gate_classification": gate_classification(metrics["recall"], metrics["f1_score"], metrics["false_positive_rate"]),
        "wilson95": {"recall": wilson(metrics["true_positives"], metrics["true_positives"] + metrics["false_negatives"]),
                     "false_positive_rate": wilson(metrics["false_positives"], metrics["false_positives"] + metrics["true_negatives"])},
        "per_defect": per_defect_recall(labels, defect_types, predictions),
    }


# ---------------------------------------------------------------------------
# One-shot guard
# ---------------------------------------------------------------------------

SENTINEL = "final_test_sentinel.json"
CONSUMED = "final_test_consumed.json"
RESULT = "final_test_result.json"
PER_IMAGE = "per_image.csv"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def begin_one_shot(out_dir: Path, record: dict, technical_rerun: bool = False) -> dict:
    """Writes the sentinel BEFORE the test directory is listed. Refuses when the category was consumed, when any
    result file exists, or when an earlier attempt left a sentinel (unless this is the single permitted technical
    re-run of an attempt that died before writing results)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / CONSUMED).exists():
        raise OneShotError(f"{CONSUMED} exists: this category's test set was already scored.")
    if (out_dir / RESULT).exists() or (out_dir / PER_IMAGE).exists():
        raise OneShotError("Results already exist: never re-run.")
    sentinel = out_dir / SENTINEL
    if sentinel.exists():
        previous = json.loads(sentinel.read_text(encoding="utf-8"))
        if not technical_rerun:
            raise OneShotError(f"{SENTINEL} exists without results: an earlier attempt died. Only one technical re-run "
                               "(--technical-rerun) is permitted.")
        if previous.get("attempts", 1) >= 2:
            raise OneShotError("The single permitted technical re-run was already used.")
        if previous.get("lock_digest") != record.get("lock_digest"):
            raise OneShotError("A technical re-run must use the identical locked model.")
        body = {**record, "attempts": 2, "technical_rerun": True, "first_attempt": previous}
        sentinel.write_text(json.dumps(body, indent=1), encoding="utf-8")
        return body
    if technical_rerun:
        raise OneShotError("--technical-rerun given but no earlier attempt exists.")
    body = {**record, "attempts": 1, "technical_rerun": False}
    with open(sentinel, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(body, indent=1))
    return body


def finish_one_shot(out_dir: Path, record: dict) -> dict:
    """Writes the consumed marker after every result file is saved (exclusive create)."""
    out_dir = Path(out_dir)
    if not (out_dir / RESULT).exists():
        raise OneShotError("Refusing to mark consumed before final_test_result.json exists.")
    body = {**record, "consumed_utc": utc_now()}
    with open(out_dir / CONSUMED, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(body, indent=1))
    return body
