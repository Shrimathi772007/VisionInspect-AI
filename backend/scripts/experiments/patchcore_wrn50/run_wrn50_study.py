"""WRN-50-2 PatchCore normal-data study + lock for ONE category (PROTOCOL.md, STUDY_DECLARATION.md).

Usage (from backend/):
    python scripts/experiments/patchcore_wrn50/run_wrn50_study.py --category bottle [--output-root DIR] [--log-dir DIR]

Reads ONLY dataset/<category>/train/good (every path built by wrn50_study_logic.train_good_path; every
listing and file open is appended to dataset_access_audit.txt). The category's evaluation split is never
listed or read. Outputs go to <output-root>/<category>/patchcore_wrn50/ (default output root: ai_models/).

For each preprocessing mode, one at a time: features of all train/good images into one pre-allocated tensor;
for each of the 5 folds a 10% greedy-coreset bank (seed 0) from the other four folds scores the fold's
images (out-of-fold) and 4 synthetic defects per fold image, keeping both aggregations; checkpoint; free.
Then the 12-row selection table, the selection, the final bank on ALL train/good images, and lock.json.
Exit code 3 if peak RAM exceeds the 3.5 GB budget (the run stops; nothing is switched to float16).
"""

import argparse
import ctypes
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent.parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(HERE))

import wrn50_study_logic as logic  # noqa: E402
from app.ai.evaluation.synthetic_defects import make_synthetic_defect  # noqa: E402
from app.ai.models.patchcore import PatchCoreConfig, PatchCoreDetector  # noqa: E402
from app.ai.models.wide_resnet50 import WRN50_2_SHA256, load_pretrained_wide_resnet50_2  # noqa: E402
from app.ai.preprocessing.patchcore_preprocess import decode_image, input_size, normalize, resize_and_crop  # noqa: E402
from app.ai.training.artifacts import ARTIFACTS_ROOT  # noqa: E402

PEAK_RAM_LIMIT_MB = 3500.0
PROTOCOL = HERE / "PROTOCOL.md"
DECLARATION = HERE / "STUDY_DECLARATION.md"


def peak_ram_mb() -> float:
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                           "PagefileUsage", "PeakPagefileUsage")]

    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    c = Counters()
    c.cb = ctypes.sizeof(Counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise RuntimeError("could not read process memory counters")
    return round(c.PeakWorkingSetSize / 2**20, 1)


class RamBudgetExceeded(Exception):
    pass


def check_ram(stage: str, log) -> float:
    peak = peak_ram_mb()
    log(f"  peak RAM after {stage}: {peak:.0f} MB")
    if peak > PEAK_RAM_LIMIT_MB:
        raise RamBudgetExceeded(f"peak RAM {peak:.0f} MB exceeded {PEAK_RAM_LIMIT_MB:.0f} MB after {stage}")
    return peak


def load_backbone() -> nn.Module:
    """Verified, strictly loaded WRN-50-2; layer4/fc (never used for layer2+3 features) are released from memory
    AFTER the strict load. This changes no feature value: forward() stops after layer3."""
    model = load_pretrained_wide_resnet50_2()
    model.layer4 = nn.Identity()
    model.fc = nn.Identity()
    gc.collect()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=logic.SCOPE)
    parser.add_argument("--output-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None,
                        help="SMOKE TEST ONLY: use the first N train/good images; refused for the default output root")
    args = parser.parse_args()
    if args.max_images is not None and Path(args.output_root).resolve() == Path(ARTIFACTS_ROOT).resolve():
        parser.error("--max-images is only allowed with an --output-root outside ai_models/")

    category = args.category
    out_dir = Path(args.output_root) / category / "patchcore_wrn50"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = (Path(args.log_dir) if args.log_dir else out_dir) / f"{category}_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{category}] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    started = time.time()
    audit = logic.AccessAudit(out_dir / "dataset_access_audit.txt")
    names = logic.list_train_good(category, audit)
    if args.max_images is not None:
        names = names[: args.max_images]
    n = len(names)
    folds = [logic.fold_of(i) for i in range(n)]
    fingerprint = logic.file_list_fingerprint(category, names)
    code_hashes = {
        "runner_sha256": logic.sha256_file(Path(__file__)),
        "logic_sha256": logic.sha256_file(HERE / "wrn50_study_logic.py"),
        "protocol_sha256": logic.sha256_file(PROTOCOL),
        "declaration_sha256": logic.sha256_file(DECLARATION),
    }
    log(f"{n} train/good images; folds {[folds.count(k) for k in range(logic.N_FOLDS)]}; output {out_dir}")

    def open_image(index: int):
        path = logic.train_good_path(category, names[index])
        audit.record("open", path)
        return decode_image(path)

    backbone = load_backbone()
    check_ram("backbone load", log)
    peaks = []
    mode_results = {}
    synthetic_digests = {}

    for mode in logic.MODES:
        checkpoint = out_dir / f"checkpoint_{mode}.json"
        resume_key = {"file_list_sha256": fingerprint["files_sha256"], **code_hashes, "mode": mode}
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("resume_key") == resume_key:
                log(f"mode {mode}: resuming from checkpoint")
                mode_results[mode] = saved["results"]
                synthetic_digests[mode] = saved["synthetic_digest_sha256"]
                peaks.append(saved["peak_ram_mb"])
                continue
            log(f"mode {mode}: checkpoint does not match this run; recomputing")

        t_mode = time.time()
        side = input_size(mode)
        detector = PatchCoreDetector(PatchCoreConfig(preprocessing=mode), extractor=backbone)
        patches = detector.config.patches_per_image
        features = detector.extract_features_into(
            (normalize(resize_and_crop(open_image(i), mode)) for i in range(n)), n)
        flat = features.view(n * patches, features.shape[2])
        log(f"mode {mode}: features {tuple(features.shape)} ({features.numel() * 4 / 2**20:.0f} MB) in {time.time() - t_mode:.0f}s")
        peaks.append(check_ram(f"{mode} extraction", log))

        oof = {agg: [None] * n for agg in logic.AGGREGATIONS}
        synthetic = {agg: [] for agg in logic.AGGREGATIONS}
        synthetic_meta = []
        digest = hashlib.sha256()
        batch = torch.empty((logic.DEFECTS_PER_IMAGE, 3, side, side), dtype=torch.float32)
        for fold in range(logic.N_FOLDS):
            t_fold = time.time()
            train_images = [i for i in range(n) if folds[i] != fold]
            rows = logic.greedy_coreset_over_images(flat, train_images, patches)
            fold_detector = PatchCoreDetector(detector.config, extractor=backbone, bank=flat[rows].clone())
            t_bank = time.time() - t_fold
            for position, index in enumerate(logic.fold_members(n, fold)):
                normal = fold_detector.patch_scores_from_features(features[index : index + 1])
                for agg in logic.AGGREGATIONS:
                    oof[agg][index] = float(fold_detector.image_scores(normal, agg)[0])
                image = open_image(index)
                plan = logic.synthetic_plan(fold, position)
                ok = []
                for j, item in enumerate(plan):
                    try:
                        painted, _ = make_synthetic_defect(image, item["defect_type"], item["severity"], seed=item["seed"])
                    except RuntimeError as exc:
                        log(f"  synthetic failed (counted as not detected): {names[index]} {item} - {exc}")
                        ok.append(False)
                        continue
                    digest.update(painted.tobytes())
                    batch[j] = normalize(resize_and_crop(painted, mode))
                    ok.append(True)
                good = [j for j, flag in enumerate(ok) if flag]
                scores = {}
                if good:
                    patch_scores = fold_detector.patch_scores_from_features(detector.extract_features(batch[good]))
                    for agg in logic.AGGREGATIONS:
                        scores[agg] = fold_detector.image_scores(patch_scores, agg).tolist()
                for j, item in enumerate(plan):
                    synthetic_meta.append({"source": names[index], "fold": fold, "position": position, **item, "generated": ok[j]})
                    for agg in logic.AGGREGATIONS:
                        synthetic[agg].append(scores[agg][good.index(j)] if ok[j] else None)
            log(f"  fold {fold}: bank {tuple(fold_detector.bank.shape)} in {t_bank:.0f}s; fold done in {time.time() - t_fold:.0f}s")
            del fold_detector, rows
            gc.collect()

        peaks.append(check_ram(f"{mode} folds", log))
        types = [m["defect_type"] for m in synthetic_meta]
        mode_results[mode] = {agg: {"oof": oof[agg], "folds": folds, "synthetic": synthetic[agg], "synthetic_types": types}
                              for agg in logic.AGGREGATIONS}
        mode_results[mode]["synthetic_meta"] = synthetic_meta
        synthetic_digests[mode] = digest.hexdigest()
        checkpoint.write_text(json.dumps({"resume_key": resume_key, "results": mode_results[mode],
                                          "synthetic_digest_sha256": synthetic_digests[mode], "peak_ram_mb": peaks[-1],
                                          "seconds": round(time.time() - t_mode, 1)}), encoding="utf-8")
        log(f"mode {mode}: done in {time.time() - t_mode:.0f}s; checkpoint written")
        del features, flat, detector, batch
        gc.collect()

    table = logic.build_selection_table(mode_results)
    table_path = out_dir / "selection_table.json"
    table_path.write_text(json.dumps({"category": category, "rows": table}, indent=1), encoding="utf-8")
    decision = logic.select(table)
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")
    log(f"selected {decision['mode']} / {decision['aggregation']} / {decision['policy']}: threshold {decision['threshold']!r}, "
        f"worst-fold FPR {decision['worst_fold_fpr']:.3f}, synthetic recall {decision['synthetic_recall']:.3f} "
        f"(any met limit: {decision['any_combination_met_fpr_limit']})")

    # Final model: bank on ALL train/good images with the chosen configuration.
    t_final = time.time()
    final_config = PatchCoreConfig(preprocessing=decision["mode"], aggregation=decision["aggregation"])
    final = PatchCoreDetector(final_config, extractor=backbone)
    features = final.extract_features_into(
        (normalize(resize_and_crop(open_image(i), decision["mode"])) for i in range(n)), n)
    final.fit_from_features(features)
    del features
    gc.collect()
    model_dir = out_dir / "final_model"
    state_path = final.save(model_dir)
    peaks.append(check_ram("final fit", log))
    log(f"final bank {tuple(final.bank.shape)} saved in {time.time() - t_final:.0f}s")

    audit_result = audit.verify()
    timestamp = time.time()
    lock = {
        "category": category,
        "study": "patchcore_wrn50_normal_data",
        "mode": decision["mode"],
        "aggregation": decision["aggregation"],
        "policy": decision["policy"],
        "threshold": decision["threshold"],
        "worst_fold_fpr": decision["worst_fold_fpr"],
        "mean_fold_fpr": decision["mean_fold_fpr"],
        "synthetic_recall": decision["synthetic_recall"],
        "any_combination_met_fpr_limit": decision["any_combination_met_fpr_limit"],
        "selection_reason": decision["reason"],
        "train_good_count": n,
        "fold_sizes": [folds.count(k) for k in range(logic.N_FOLDS)],
        "config": final_config.to_dict(),
        "backbone_weights_sha256": WRN50_2_SHA256,
        "model_state_sha256": logic.sha256_file(state_path),
        "model_config_sha256": logic.sha256_file(model_dir / "model_config.json"),
        "final_bank_shape": list(final.bank.shape),
        "selection_table_sha256": logic.sha256_file(table_path),
        **code_hashes,
        "train_good_file_list_sha256": fingerprint["file_list_sha256"],
        "train_good_files_sha256": fingerprint["files_sha256"],
        "train_good_per_file_sha256": fingerprint["per_file_sha256"],
        "synthetic_digest_sha256": synthetic_digests,
        "dataset_access_audit": audit_result,
        "peak_ram_mb": max(peaks),
        "run_seconds": round(time.time() - started, 1),
        "timestamp_unix": timestamp,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "test_directory_listed": False,
        "smoke_test_max_images": args.max_images,
    }
    lock["lock_digest"] = logic.lock_digest(lock)
    (out_dir / "lock.json").write_text(json.dumps(lock, indent=1), encoding="utf-8")
    log(f"lock written ({lock['lock_digest'][:16]}); run {lock['run_seconds']:.0f}s; peak RAM {lock['peak_ram_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RamBudgetExceeded as exc:
        print(f"STOPPED: {exc}", flush=True)
        sys.exit(3)
