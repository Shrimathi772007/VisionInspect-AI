"""WRN-50-2 PatchCore final test: the single scoring of ONE locked category (FINAL_TEST_DECLARATION.md).

Usage (from backend/):
    python scripts/experiments/patchcore_wrn50/run_wrn50_final.py --category bottle
    python scripts/experiments/patchcore_wrn50/run_wrn50_final.py --category grid --technical-rerun   # only after a crash
    python scripts/experiments/patchcore_wrn50/run_wrn50_final.py --category bottle \
        --dry-run-train-good 10 --output-dir <temp dir>                       # pipeline check, never touches test/

Order (enforced): verify lock (grid: the Addendum 1 full320 lock AND the original Grid lock, which is never
scored) -> write final_test_sentinel.json -> list dataset/<category>/test -> load the locked detector (standard
loader; grid: grid_full320_logic.load_full320_detector) -> score every test image once, one at a time, with the
locked mode and aggregation -> metrics -> save outputs -> re-verify lock and model -> final_test_consumed.json.

Every file open and directory listing of the process is recorded by an audit hook; leakage_audit.json is built
from that record. Only dataset/<category>/test/<subfolder>/*.png is read from the dataset. Per-image results go
to files only. Exit codes: 3 peak RAM > 3.5 GB, 4 lock verification failed, 5 one-shot guard refused,
6 dataset access refused, 7 not committed (declaration/runner/logic must be committed before real scoring).
"""

import argparse
import csv
import gc
import json
import os
import platform
import site
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import sklearn
import torch

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent.parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(HERE))

import grid_full320_logic as g  # noqa: E402
import run_wrn50_study as study  # noqa: E402  (read-only reuse: RAM measurement, backbone loader)
import wrn50_final_logic as fl  # noqa: E402
import wrn50_study_logic as logic  # noqa: E402
from app.ai.evaluation.category_phase3 import metrics_from_predictions  # noqa: E402
from app.ai.models.patchcore import PatchCoreDetector  # noqa: E402
from app.ai.models.wide_resnet50 import wide_resnet50_weights_path  # noqa: E402
from app.ai.preprocessing.patchcore_preprocess import decode_image, normalize, resize_and_crop  # noqa: E402
from app.ai.training.artifacts import ARTIFACTS_ROOT  # noqa: E402
from app.inspections.storage import DATASET_ROOT  # noqa: E402

DECLARATION = HERE / "FINAL_TEST_DECLARATION.md"
COMMITTED_FILES = (Path(__file__).resolve(), HERE / "wrn50_final_logic.py", DECLARATION)
RUNTIME_SUFFIXES = {".py", ".pyc", ".pyd", ".dll", ".so", ".pth", ".zip", ".typed"}


class Exit(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Process-wide file-access record
# ---------------------------------------------------------------------------

class AccessRecord:
    """sys.addaudithook callback: every `open`, `os.listdir` and `os.scandir` of this process with the phase it
    happened in. The hook only appends; classification happens afterwards."""

    def __init__(self):
        self.phase = "startup"
        self.events: list[tuple[float, str, str, str, str]] = []

    def __call__(self, event, args):
        if event not in ("open", "os.listdir", "os.scandir"):
            return
        target = args[0] if args else None
        if target is None or isinstance(target, int):
            return
        try:
            target = os.fsdecode(os.fspath(target))
        except TypeError:
            return
        mode = str(args[1]) if event == "open" and len(args) > 1 else event
        self.events.append((time.time(), self.phase, event, target, mode))


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def classify_events(events, category: str, allowed_split: str, model_dir: Path, out_dir: Path) -> dict:
    """Dataset accesses (must all lie under dataset/<category>/<allowed_split>), any path naming the mask folder
    (must be none), and, for the listing/scoring phases, every non-dataset file split into model files, backbone
    weights, Python runtime, outputs and OTHER (must be none)."""
    dataset, model, weights, out = _norm(DATASET_ROOT), _norm(model_dir), _norm(wide_resnet50_weights_path()), _norm(out_dir)
    allowed = _norm(Path(DATASET_ROOT) / category / allowed_split)
    runtime_roots = tuple({_norm(r) for r in (sys.prefix, sys.base_prefix, BACKEND / "app", HERE, site.getusersitepackages(),
                                              *site.getsitepackages())})
    forbidden, outside, dataset_opens, dataset_lists = [], [], 0, 0
    scoring = {"model_files": set(), "backbone_weights": set(), "python_runtime": 0, "outputs": set(), "other": set()}
    for _, phase, event, target, _mode in events:
        path = _norm(target)
        if fl.FORBIDDEN_PART in path.lower():
            forbidden.append(target)
        if path == dataset or path.startswith(dataset + os.sep):
            if path == allowed or path.startswith(allowed + os.sep):
                dataset_opens += event == "open"
                dataset_lists += event != "open"
            else:
                outside.append(f"{event} {target}")
            continue
        if phase not in ("listing", "scoring"):
            continue
        if path.startswith(model + os.sep):
            scoring["model_files"].add(os.path.basename(path))
        elif path == weights:
            scoring["backbone_weights"].add(os.path.basename(path))
        elif path.startswith(out + os.sep) or path == out:
            scoring["outputs"].add(os.path.basename(path))
        elif os.path.splitext(path)[1] in RUNTIME_SUFFIXES or path.startswith(runtime_roots):
            scoring["python_runtime"] += 1
        else:
            scoring["other"].add(target)
    return {
        "events_recorded": len(events),
        "ground_truth_paths_seen": forbidden,
        "no_ground_truth_path_opened_or_listed": not forbidden,
        "dataset_files_opened": dataset_opens,
        "dataset_directories_listed": dataset_lists,
        "dataset_accesses_outside_allowed_split": outside,
        "all_dataset_accesses_under": f"{category}/{allowed_split}",
        "all_dataset_accesses_inside_allowed_split": not outside,
        "files_opened_during_listing_and_scoring": {k: (sorted(v) if isinstance(v, set) else v) for k, v in scoring.items()},
        "no_other_file_opened_during_listing_and_scoring": not scoring["other"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=BACKEND, capture_output=True, text=True, check=True).stdout.strip()


def committed_state() -> dict:
    """HEAD and whether the runner, logic module and declaration are committed and unmodified."""
    rel = [os.path.relpath(p, BACKEND).replace("\\", "/") for p in COMMITTED_FILES]
    dirty = git("status", "--porcelain", "--", *rel)
    tracked = [r for r in rel if git("ls-files", "--", r)]
    return {"head": git("rev-parse", "HEAD"), "declaration_commit": git("log", "-1", "--format=%H", "--", rel[-1]) or None,
            "files": rel, "all_tracked": len(tracked) == len(rel), "working_tree_clean": dirty == "", "porcelain": dirty}


def environment() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
            "sklearn": sklearn.__version__, "torch_threads": torch.get_num_threads(), "cpu_count": os.cpu_count(),
            "platform": platform.platform()}


def summary_stats(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {"count": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)), "min": float(arr.min()),
            "max": float(arr.max()), "std": float(arr.std())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args, access: AccessRecord) -> dict:
    started_wall = time.time()
    category = args.category
    dry = args.dry_run_train_good is not None
    access.phase = "verify"

    # 1. Lock verification (before anything under the dataset is touched).
    try:
        verification, lock = fl.verify_category_lock(category, ARTIFACTS_ROOT)
    except (fl.LockVerificationError, KeyError, FileNotFoundError) as exc:
        raise Exit(4, f"STOP {category}: lock verification failed: {exc}")
    verified_at = time.time()
    directory = fl.lock_dir(category, ARTIFACTS_ROOT)
    model_dir = directory / "final_model"

    if dry:
        if args.output_dir is None:
            raise Exit(6, "--dry-run-train-good needs --output-dir")
        out_dir = Path(args.output_dir).resolve()
        if out_dir == Path(ARTIFACTS_ROOT).resolve() or out_dir.is_relative_to(Path(ARTIFACTS_ROOT).resolve()):
            raise Exit(6, "a dry run must not write under ai_models/")
        out_dir = out_dir / category
        allowed_split = "train/good"
    else:
        if args.output_dir is not None:
            raise Exit(6, "real scoring writes only to <lock folder>/final_test/")
        out_dir = directory / fl.FINAL_DIRNAME
        allowed_split = fl.TEST_SPLIT

    commit = committed_state()
    if not dry and not (commit["all_tracked"] and commit["working_tree_clean"]):
        raise Exit(7, f"STOP {category}: runner/logic/declaration must be committed and unmodified before scoring ({commit['porcelain']!r})")

    runner_sha = logic.sha256_file(Path(__file__))
    sentinel_record = {
        "category": category, "model": fl.model_label(category), "sentinel_utc": fl.utc_now(), "sentinel_unix": time.time(),
        "lock_dir": str(directory), "lock_digest": lock["lock_digest"], "threshold": lock["threshold"],
        "threshold_repr": repr(lock["threshold"]), "mode": lock["mode"], "aggregation": lock["aggregation"],
        "runner_sha256": runner_sha, "final_logic_sha256": logic.sha256_file(HERE / "wrn50_final_logic.py"),
        "declaration_sha256": logic.sha256_file(DECLARATION), "git_head": commit["head"],
        "declaration_commit": commit["declaration_commit"], "dry_run": dry,
    }

    # 2. One-shot sentinel BEFORE the evaluation directory is listed.
    access.phase = "sentinel"
    try:
        sentinel = fl.begin_one_shot(out_dir, sentinel_record, technical_rerun=args.technical_rerun)
    except fl.OneShotError as exc:
        raise Exit(5, f"STOP {category}: {exc}")
    sentinel_at = time.time()

    # 3. List the evaluation images (test/, or train/good for the dry run).
    access.phase = "listing"
    listed_at = time.time()
    try:
        if dry:
            root = fl.guarded_path(category, dataset_root=DATASET_ROOT, split="train/good")
            with os.scandir(root) as entries:
                names = sorted(e.name for e in entries if e.is_file() and e.name.lower().endswith(fl.IMAGE_SUFFIX))
            names = names[: args.dry_run_train_good]
            images = [fl.EvalImage(f"{category}/train/good/{n}", i % 2, "fake_defect" if i % 2 else fl.GOOD,
                                   fl.guarded_path(category, filename=n, dataset_root=DATASET_ROOT, split="train/good"))
                      for i, n in enumerate(names)]
            listing = {"subfolders": ["(dry run) train/good"], "ignored_non_png": [], "fake_labels": "index % 2"}
        else:
            images, listing = fl.list_test_images(category, DATASET_ROOT)
    except fl.DatasetAccessError as exc:
        raise Exit(6, f"STOP {category}: {exc}")

    # 4. Load the locked detector.
    access.phase = "scoring"
    t_load = time.perf_counter()
    backbone = study.load_backbone()
    if category == "grid":
        detector = g.load_full320_detector(model_dir, extractor=backbone)
        preprocess = lambda image: normalize(g.resize_full320(image))  # noqa: E731
    else:
        detector = PatchCoreDetector.load(model_dir, extractor=backbone)
        preprocess = lambda image: normalize(resize_and_crop(image, lock["mode"]))  # noqa: E731
    load_ms = (time.perf_counter() - t_load) * 1000
    if detector.config.to_dict() != lock["config"] or list(detector.bank.shape) != lock["final_bank_shape"]:
        raise Exit(4, f"STOP {category}: loaded detector differs from the lock")
    peaks = [study.peak_ram_mb()]

    # 5. Score every image once, one at a time.
    side, size, n = detector.config.grid_side, detector.config.input_size, len(images)
    patch_grids = np.empty((n, side, side), dtype=np.float32)
    maps = np.empty((n, size, size), dtype=np.float16)
    scores, prep_ms, inf_ms, map_ms, decode_matches = [], [], [], [], []
    t_scoring = time.perf_counter()
    with torch.no_grad():
        for i, item in enumerate(images):
            t0 = time.perf_counter()
            with open(item.path, "rb") as handle:
                data = handle.read()
            decoded = decode_image(data)
            if dry:  # dry run only (train/good): the bytes decode equals the study's path decode
                decode_matches.append(bool(np.array_equal(decoded, decode_image(item.path))))
            tensor = preprocess(decoded)[None]
            t1 = time.perf_counter()
            patch = detector.patch_scores_from_features(detector.extract_features(tensor))
            score = float(detector.image_scores(patch, lock["aggregation"])[0])
            t2 = time.perf_counter()
            amap = detector.anomaly_map(patch)
            t3 = time.perf_counter()
            patch_grids[i] = patch.reshape(side, side).numpy()
            maps[i] = amap[0].numpy().astype(np.float16)
            scores.append(score)
            prep_ms.append((t1 - t0) * 1000)
            inf_ms.append((t2 - t1) * 1000)
            map_ms.append((t3 - t2) * 1000)
            peaks.append(study.peak_ram_mb())
            if peaks[-1] > study.PEAK_RAM_LIMIT_MB:
                raise study.RamBudgetExceeded(f"peak RAM {peaks[-1]:.0f} MB exceeded {study.PEAK_RAM_LIMIT_MB:.0f} MB at image {i}")
    scoring_s = time.perf_counter() - t_scoring
    del detector, backbone
    gc.collect()

    # 6. Metrics (computed once, from the locked threshold).
    access.phase = "metrics"
    threshold = lock["threshold"]
    labels = [im.label for im in images]
    types = [im.defect_type for im in images]
    evaluation = fl.evaluate(labels, types, scores, threshold)
    predictions = evaluation.pop("predictions")
    project = metrics_from_predictions([{"label": l, "predicted_label": p} for l, p in zip(labels, predictions)])
    m = evaluation["metrics"]
    cross_check = {k: project[k] == m[k] for k in ("accuracy", "precision", "recall", "f1_score", "true_positives",
                                                   "true_negatives", "false_positives", "false_negatives")}
    cross_check["fpr"] = project["false_defect_detection_rate"] == m["false_positive_rate"]
    by_type: dict[str, int] = {}
    for t in types:
        by_type[t] = by_type.get(t, 0) + 1

    # 7. Re-verify lock and model after scoring.
    access.phase = "post_verify"
    try:
        after, _ = fl.verify_category_lock(category, ARTIFACTS_ROOT)
        unchanged = after == verification
    except fl.LockVerificationError as exc:
        after, unchanged = {"error": str(exc)}, False

    # 8. Outputs.
    access.phase = "writing"
    with open(out_dir / fl.PER_IMAGE, "x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "label", "defect_type", "score", "prediction"])
        for im, s, p in zip(images, scores, predictions):
            writer.writerow([im.relative, "defective" if im.label else "good", im.defect_type, repr(s), "defective" if p else "good"])
    files = np.array([im.relative for im in images])
    np.savez_compressed(out_dir / "patch_scores.npz", patch_scores=patch_grids, files=files)
    np.savez_compressed(out_dir / "anomaly_maps.npz", anomaly_maps=maps, files=files)

    access_summary = classify_events(access.events, category, allowed_split, model_dir, out_dir)
    sentinel_mtime = (out_dir / fl.SENTINEL).stat().st_mtime
    leakage = {
        "lock_verified_before_listing": verified_at < sentinel_at <= listed_at,
        "sentinel_written_before_test_listing": sentinel_at <= listed_at and sentinel_mtime <= listed_at,
        "sentinel_mtime_unix": sentinel_mtime, "lock_verified_at_unix": verified_at, "listed_at_unix": listed_at,
        "no_ground_truth_path_built": all(fl.FORBIDDEN_PART not in im.relative.lower() for im in images),
        **access_summary,
        "lock_and_model_unchanged_after_scoring": unchanged,
        "metrics_match_project_metrics_from_predictions": all(cross_check.values()),
        "original_grid_lock_verified_and_not_scored": (verification["original_grid_lock"]["verified"]
                                                       and not verification["original_grid_lock"]["used_for_scoring"])
        if category == "grid" else None,
    }
    leakage["all_passed"] = all(v for k, v in leakage.items() if isinstance(v, bool))
    (out_dir / "leakage_audit.json").write_text(json.dumps(leakage, indent=1), encoding="utf-8")

    result = {
        "category": category, "model": fl.model_label(category), "dry_run": dry, "technical_rerun": sentinel["technical_rerun"],
        "lock_dir": str(directory), "lock_digest": lock["lock_digest"], "lock_verification": verification,
        "mode": lock["mode"], "aggregation": lock["aggregation"], "policy": lock["policy"],
        "locked_threshold": threshold, "threshold_repr": repr(threshold), "prediction_rule": "score > threshold -> defective",
        "counts": {"test_total": n, "test_good": labels.count(0), "test_defective": labels.count(1), "test_by_type": dict(sorted(by_type.items())),
                   "ignored_non_png": listing["ignored_non_png"]},
        "metrics": m, "auroc": evaluation["auroc"], "average_precision": evaluation["average_precision"],
        "gate_classification": evaluation["gate_classification"], "wilson95": evaluation["wilson95"],
        "per_defect": evaluation["per_defect"], "project_metrics_cross_check": cross_check,
        "score_distributions": {"good": summary_stats([s for s, l in zip(scores, labels) if l == 0]),
                                "defective": summary_stats([s for s, l in zip(scores, labels) if l == 1])},
        "timing": {"model_load_ms": load_ms, "read_decode_preprocess_ms_per_image": statistics.fmean(prep_ms),
                   "inference_ms_per_image": statistics.fmean(inf_ms), "anomaly_map_ms_per_image": statistics.fmean(map_ms),
                   "total_ms_per_image": statistics.fmean(prep_ms) + statistics.fmean(inf_ms) + statistics.fmean(map_ms),
                   "median_inference_ms": statistics.median(inf_ms), "scoring_seconds": scoring_s,
                   "wall_seconds": time.time() - started_wall},
        "peak_ram_mb": max(peaks), "grid_side": side, "map_size": size,
        "outputs": [fl.PER_IMAGE, "patch_scores.npz", "anomaly_maps.npz", "leakage_audit.json", fl.SENTINEL, fl.CONSUMED],
        "sentinel": sentinel, "test_listed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(listed_at)),
        "leakage_audit_all_passed": leakage["all_passed"], "environment": environment(), "git": commit,
    }
    if dry:
        result["dry_run_bytes_decode_equals_path_decode"] = all(decode_matches) and len(decode_matches) == n
    (out_dir / fl.RESULT).write_text(json.dumps(result, indent=1), encoding="utf-8")
    fl.finish_one_shot(out_dir, {"category": category, "lock_digest": lock["lock_digest"], "runner_sha256": runner_sha,
                                 "result_sha256": logic.sha256_file(out_dir / fl.RESULT), "dry_run": dry,
                                 "technical_rerun": sentinel["technical_rerun"]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=fl.RUN_ORDER)
    parser.add_argument("--technical-rerun", action="store_true",
                        help="ONLY when an earlier attempt died after the sentinel and before results were written")
    parser.add_argument("--dry-run-train-good", type=int, default=None,
                        help="PIPELINE CHECK ONLY: score the first N train/good images with fake labels into --output-dir")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    access = AccessRecord()
    sys.addaudithook(access)
    try:
        result = run(args, access)
    except Exit as exc:
        print(str(exc), flush=True)
        return exc.code
    except study.RamBudgetExceeded as exc:
        print(f"STOPPED {args.category}: {exc}", flush=True)
        return 3
    m = result["metrics"]
    print(f"[{args.category}] {result['model']}{' DRY RUN' if result['dry_run'] else ''}: {result['counts']['test_total']} images "
          f"({result['counts']['test_good']} good / {result['counts']['test_defective']} defective); "
          f"TP {m['true_positives']} TN {m['true_negatives']} FP {m['false_positives']} FN {m['false_negatives']}; "
          f"gate {result['gate_classification']}; leakage audit passed: {result['leakage_audit_all_passed']}; "
          f"wall {result['timing']['wall_seconds']:.0f}s; peak RAM {result['peak_ram_mb']:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
