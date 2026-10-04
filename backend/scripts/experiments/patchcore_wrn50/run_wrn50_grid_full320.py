"""Addendum 1: Grid full320 candidate on normal data only (ADDENDUM_1_GRID_FULL320.md).

Usage (from backend/):
    python scripts/experiments/patchcore_wrn50/run_wrn50_grid_full320.py [--log-dir DIR]
    python scripts/experiments/patchcore_wrn50/run_wrn50_grid_full320.py --equivalence-check [--log-dir DIR]

Reads ONLY dataset/grid/train/good (every path built by wrn50_study_logic.train_good_path; every listing and
file open is appended to dataset_access_audit.txt). Outputs go to ai_models/grid/patchcore_wrn50_full320/;
the locked study folder ai_models/grid/patchcore_wrn50/ is only read (its lock and selection_table.json).

Study: for each of the 5 folds, the other four folds' full320 features go into one pre-allocated buffer, the
10% coreset bank is built and the buffer released; then the held-out fold is scored with 4 synthetic defects
per image (checkpoint per fold). The 6 full320 rows are pooled with the 12 existing rows, the combined rule
is applied, and a final model + lock are written ONLY if a full320 row wins.
Exit code 3 if peak RAM exceeds 3.5 GB (nothing is switched to float16); 4 if a consistency check fails.
"""

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent.parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(HERE))

import grid_full320_logic as g  # noqa: E402
import run_wrn50_study as study  # noqa: E402  (read-only reuse: RAM check, backbone loader)
import wrn50_study_logic as logic  # noqa: E402
from app.ai.evaluation.synthetic_defects import make_synthetic_defect  # noqa: E402
from app.ai.models.patchcore import PatchCoreDetector  # noqa: E402
from app.ai.models.wide_resnet50 import WRN50_2_SHA256  # noqa: E402
from app.ai.preprocessing.patchcore_preprocess import decode_image, normalize  # noqa: E402
from app.ai.training.artifacts import ARTIFACTS_ROOT  # noqa: E402

CATEGORY = g.CATEGORY
LOCKED_DIR = Path(ARTIFACTS_ROOT) / CATEGORY / "patchcore_wrn50"
OUT_DIR = Path(ARTIFACTS_ROOT) / CATEGORY / "patchcore_wrn50_full320"
ADDENDUM = HERE / "ADDENDUM_1_GRID_FULL320.md"


class ConsistencyError(Exception):
    pass


def code_hashes() -> dict:
    return {
        "addendum_runner_sha256": logic.sha256_file(Path(__file__)),
        "addendum_logic_sha256": logic.sha256_file(HERE / "grid_full320_logic.py"),
        "addendum_sha256": logic.sha256_file(ADDENDUM),
        "protocol_sha256": logic.sha256_file(study.PROTOCOL),
        "declaration_sha256": logic.sha256_file(study.DECLARATION),
        "study_runner_sha256": logic.sha256_file(HERE / "run_wrn50_study.py"),
        "study_logic_sha256": logic.sha256_file(HERE / "wrn50_study_logic.py"),
    }


def read_locked_grid() -> tuple[dict, list[dict], str]:
    """Grid's existing lock (digest verified) and its 12 selection rows (file hash verified against the lock)."""
    lock = json.loads((LOCKED_DIR / "lock.json").read_text(encoding="utf-8"))
    if logic.lock_digest(lock) != lock["lock_digest"]:
        raise ConsistencyError("Grid's existing lock.json does not match its lock_digest.")
    table_path = LOCKED_DIR / "selection_table.json"
    table_sha = logic.sha256_file(table_path)
    if table_sha != lock["selection_table_sha256"]:
        raise ConsistencyError("Grid's selection_table.json does not match the hash recorded in its lock.")
    rows = json.loads(table_path.read_text(encoding="utf-8"))["rows"]
    if len(rows) != 12:
        raise ConsistencyError(f"Expected 12 existing Grid rows, found {len(rows)}.")
    return lock, rows, table_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--equivalence-check", action="store_true",
                        help="compare the memory-light fold method and streaming final fit with an all-in-memory "
                             "reference on the first N train/good images (fold 0); writes equivalence_check.json")
    parser.add_argument("--equivalence-images", type=int, default=20)
    args = parser.parse_args()

    if OUT_DIR.resolve() == LOCKED_DIR.resolve():
        raise ConsistencyError("Refusing to write into the locked study folder.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = (Path(args.log_dir) if args.log_dir else OUT_DIR) / ("grid_full320_equivalence.log" if args.equivalence_check
                                                                     else "grid_full320_run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [grid/full320] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    started = time.time()
    locked, existing_rows, existing_table_sha = read_locked_grid()
    audit = logic.AccessAudit(OUT_DIR / "dataset_access_audit.txt")
    names = logic.list_train_good(CATEGORY, audit)
    hashes = code_hashes()

    def open_image(index: int):
        path = logic.train_good_path(CATEGORY, names[index])
        audit.record("open", path)
        return decode_image(path)

    def preprocessed(index: int) -> torch.Tensor:
        return normalize(g.resize_full320(open_image(index)))

    backbone = study.load_backbone()
    peaks = [study.check_ram("backbone load", log)]
    config = g.Full320Config()
    detector = PatchCoreDetector(config, extractor=backbone)
    patches, dim = config.patches_per_image, 1536

    if args.equivalence_check:
        return equivalence_check(args.equivalence_images, names, preprocessed, detector, backbone, config, audit, log)

    n = len(names)
    folds = [logic.fold_of(i) for i in range(n)]
    fingerprint = logic.file_list_fingerprint(CATEGORY, names)
    if fingerprint["files_sha256"] != locked["train_good_files_sha256"]:
        raise ConsistencyError("Grid's train/good files differ from the files recorded in its lock.")
    log(f"{n} train/good images; folds {[folds.count(k) for k in range(logic.N_FOLDS)]}; {patches} patches/image; output {OUT_DIR}")

    oof = {agg: [None] * n for agg in logic.AGGREGATIONS}
    synthetic = {agg: [] for agg in logic.AGGREGATIONS}
    synthetic_meta = []
    digest = hashlib.sha256()
    resumed_any = False
    batch = torch.empty((logic.DEFECTS_PER_IMAGE, 3, g.FULL320_SIZE, g.FULL320_SIZE), dtype=torch.float32)

    for fold in range(logic.N_FOLDS):
        checkpoint = OUT_DIR / f"checkpoint_fold{fold}.json"
        resume_key = {"file_list_sha256": fingerprint["files_sha256"], **hashes, "fold": fold}
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("resume_key") == resume_key:
                log(f"fold {fold}: resuming from checkpoint")
                for agg in logic.AGGREGATIONS:
                    for index, score in saved["oof"][agg].items():
                        oof[agg][int(index)] = score
                    synthetic[agg].extend(saved["synthetic"][agg])
                synthetic_meta.extend(saved["synthetic_meta"])
                peaks.append(saved["peak_ram_mb"])
                resumed_any = True
                continue
            log(f"fold {fold}: checkpoint does not match this run; recomputing")

        t_fold = time.time()
        train_images = [i for i in range(n) if folds[i] != fold]
        buffer = detector.extract_features_into((preprocessed(i) for i in train_images), len(train_images))
        log(f"fold {fold}: train buffer {tuple(buffer.shape)} ({buffer.numel() * 4 / 2**20:.0f} MB) in {time.time() - t_fold:.0f}s")
        peaks.append(study.check_ram(f"fold {fold} buffer", log))
        t_bank = time.time()
        bank = g.fold_bank_from_buffer(buffer)
        peaks.append(study.check_ram(f"fold {fold} bank", log))
        del buffer
        gc.collect()
        log(f"  bank {tuple(bank.shape)} in {time.time() - t_bank:.0f}s; buffer released")
        fold_detector = PatchCoreDetector(config, extractor=backbone, bank=bank)

        members = logic.fold_members(n, fold)
        held = detector.extract_features_into((preprocessed(i) for i in members), len(members))
        fold_oof = {agg: {} for agg in logic.AGGREGATIONS}
        fold_synthetic = {agg: [] for agg in logic.AGGREGATIONS}
        fold_meta = []
        for position, index in enumerate(members):
            normal = fold_detector.patch_scores_from_features(held[position : position + 1])
            for agg in logic.AGGREGATIONS:
                fold_oof[agg][index] = float(fold_detector.image_scores(normal, agg)[0])
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
                batch[j] = normalize(g.resize_full320(painted))
                ok.append(True)
            good = [j for j, flag in enumerate(ok) if flag]
            scores = {}
            if good:
                patch_scores = fold_detector.patch_scores_from_features(detector.extract_features(batch[good]))
                for agg in logic.AGGREGATIONS:
                    scores[agg] = fold_detector.image_scores(patch_scores, agg).tolist()
            for j, item in enumerate(plan):
                fold_meta.append({"source": names[index], "fold": fold, "position": position, **item, "generated": ok[j]})
                for agg in logic.AGGREGATIONS:
                    fold_synthetic[agg].append(scores[agg][good.index(j)] if ok[j] else None)
        peaks.append(study.check_ram(f"fold {fold} scoring", log))
        for agg in logic.AGGREGATIONS:
            for index, score in fold_oof[agg].items():
                oof[agg][index] = score
            synthetic[agg].extend(fold_synthetic[agg])
        synthetic_meta.extend(fold_meta)
        checkpoint.write_text(json.dumps({"resume_key": resume_key, "oof": fold_oof, "synthetic": fold_synthetic,
                                          "synthetic_meta": fold_meta, "peak_ram_mb": max(peaks),
                                          "seconds": round(time.time() - t_fold, 1)}), encoding="utf-8")
        log(f"  fold {fold} done in {time.time() - t_fold:.0f}s; checkpoint written")
        del held, fold_detector, bank
        gc.collect()

    if resumed_any:  # the running digest is incomplete: repaint (no extraction) to recompute it in order
        digest = hashlib.sha256()
        for fold in range(logic.N_FOLDS):
            for position, index in enumerate(logic.fold_members(n, fold)):
                image = open_image(index)
                for item in logic.synthetic_plan(fold, position):
                    try:
                        painted, _ = make_synthetic_defect(image, item["defect_type"], item["severity"], seed=item["seed"])
                    except RuntimeError:
                        continue
                    digest.update(painted.tobytes())
    synthetic_digest = digest.hexdigest()
    expected_digest = locked["synthetic_digest_sha256"]["crop224"]
    if synthetic_digest != expected_digest:
        raise ConsistencyError(f"Synthetic digest {synthetic_digest} differs from Grid's locked {expected_digest}: "
                               "the synthetic defects are not the same as the study's.")
    log(f"synthetic digest matches Grid's lock ({synthetic_digest[:16]})")

    types = [m["defect_type"] for m in synthetic_meta]
    results = {agg: {"oof": oof[agg], "folds": folds, "synthetic": synthetic[agg], "synthetic_types": types}
               for agg in logic.AGGREGATIONS}
    new_rows = g.build_full320_rows(results)
    (OUT_DIR / "selection_table_full320.json").write_text(json.dumps({"category": CATEGORY, "rows": new_rows,
                                                                      "synthetic_meta": synthetic_meta}, indent=1), encoding="utf-8")
    combined_path = OUT_DIR / "selection_table_combined.json"
    combined_path.write_text(json.dumps({
        "category": CATEGORY,
        "existing_rows_source": str(LOCKED_DIR / "selection_table.json"),
        "existing_rows_sha256": existing_table_sha,
        "existing_lock_digest": locked["lock_digest"],
        "rows": existing_rows + new_rows,
    }, indent=1), encoding="utf-8")
    combined_sha = logic.sha256_file(combined_path)
    decision = g.select_combined(existing_rows + new_rows)
    full320_won = decision["mode"] == g.MODE_FULL320
    log(f"combined winner {decision['mode']} / {decision['aggregation']} / {decision['policy']}: threshold "
        f"{decision['threshold']!r}, worst-fold FPR {decision['worst_fold_fpr']:.3f}, synthetic recall "
        f"{decision['synthetic_recall']:.3f}; full320 won: {full320_won}")

    lock_written = None
    if full320_won:
        t_final = time.time()
        final_config = g.Full320Config(aggregation=decision["aggregation"])
        final = PatchCoreDetector(final_config, extractor=backbone)
        selected = g.streaming_coreset_select(
            g.iter_image_features(final, (preprocessed(i) for i in range(n))), n, patches, dim)
        bank = g.gather_selected_rows(
            g.iter_image_features(final, (preprocessed(i) for i in range(n))), selected, patches, dim)
        final.bank = bank
        model_dir = OUT_DIR / "final_model"
        state_path = final.save(model_dir)
        reloaded = g.load_full320_detector(model_dir, extractor=backbone)
        if not torch.equal(reloaded.bank, bank):
            raise ConsistencyError("Saved full320 model does not reload to the same bank.")
        peaks.append(study.check_ram("final fit", log))
        log(f"final bank {tuple(bank.shape)} saved in {time.time() - t_final:.0f}s")
        timestamp = time.time()
        lock = {
            "category": CATEGORY,
            "study": "patchcore_wrn50_addendum1_grid_full320",
            "mode": decision["mode"],
            "aggregation": decision["aggregation"],
            "policy": decision["policy"],
            "threshold": decision["threshold"],
            "worst_fold_fpr": decision["worst_fold_fpr"],
            "mean_fold_fpr": decision["mean_fold_fpr"],
            "synthetic_recall": decision["synthetic_recall"],
            "selection_reason": decision["reason"],
            "train_good_count": n,
            "fold_sizes": [folds.count(k) for k in range(logic.N_FOLDS)],
            "config": final_config.to_dict(),
            "backbone_weights_sha256": WRN50_2_SHA256,
            "model_state_sha256": logic.sha256_file(state_path),
            "model_config_sha256": logic.sha256_file(model_dir / "model_config.json"),
            "final_bank_shape": list(bank.shape),
            "selection_table_combined_sha256": combined_sha,
            **hashes,
            "train_good_file_list_sha256": fingerprint["file_list_sha256"],
            "train_good_files_sha256": fingerprint["files_sha256"],
            "synthetic_digest_sha256": synthetic_digest,
            "superseded_lock_digest": locked["lock_digest"],
            "dataset_access_audit": audit.verify(),
            "peak_ram_mb": max(peaks),
            "run_seconds": round(time.time() - started, 1),
            "timestamp_unix": timestamp,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "test_directory_listed": False,
        }
        lock["lock_digest"] = logic.lock_digest(lock)
        (OUT_DIR / "lock.json").write_text(json.dumps(lock, indent=1), encoding="utf-8")
        lock_written = lock["lock_digest"]
        log(f"full320 lock written ({lock_written[:16]})")

    audit_result = audit.verify()
    record = {
        **decision,
        "full320_won": full320_won,
        "grid_final_model": (f"full320 model in {OUT_DIR / 'final_model'} (lock {lock_written})" if full320_won else
                             f"unchanged: the existing locked {locked['mode']} / {locked['aggregation']} / "
                             f"{locked['policy']} model in {LOCKED_DIR} (lock {locked['lock_digest']}); no new model adopted"),
        "selection_table_combined_sha256": combined_sha,
        "existing_lock_digest": locked["lock_digest"],
        "synthetic_digest_sha256": synthetic_digest,
        "synthetic_digest_matches_locked_study": True,
        **hashes,
        "dataset_access_audit": audit_result,
        "peak_ram_mb": max(peaks),
        "run_seconds": round(time.time() - started, 1),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_directory_listed": False,
    }
    (OUT_DIR / "decision.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    log(f"decision written; run {record['run_seconds']:.0f}s; peak RAM {record['peak_ram_mb']:.0f} MB; "
        f"audit {audit_result['entries']} entries, all under train/good")
    return 0


@torch.no_grad()
def equivalence_check(count, names, preprocessed, detector, backbone, config, audit, log) -> int:
    """Fold 0 of the first `count` train/good images: memory-light fold method and streaming final fit vs an
    all-in-memory reference (the study's method: all features in one tensor)."""
    m = min(count, len(names))
    patches, dim = config.patches_per_image, 1536
    train = [i for i in range(m) if logic.fold_of(i) != 0]
    members = logic.fold_members(m, 0)

    def max_diff(a, b) -> float:
        return float((a - b).abs().max()) if a.shape == b.shape else float("inf")

    reference = detector.extract_features_into((preprocessed(i) for i in range(m)), m)
    flat = reference.view(m * patches, dim)
    rows_ref = logic.greedy_coreset_over_images(flat, train, patches)
    bank_ref = flat[rows_ref].clone()
    scores_ref = PatchCoreDetector(config, extractor=backbone, bank=bank_ref).patch_scores_from_features(reference[members])

    buffer = detector.extract_features_into((preprocessed(i) for i in train), len(train))
    rows_light_local = logic.greedy_coreset_over_images(buffer.view(-1, dim), list(range(len(train))), patches)
    rows_light = torch.tensor(train)[rows_light_local // patches] * patches + rows_light_local % patches
    bank_light = g.fold_bank_from_buffer(buffer)
    held = detector.extract_features_into((preprocessed(i) for i in members), len(members))
    light = PatchCoreDetector(config, extractor=backbone, bank=bank_light)
    scores_light = light.patch_scores_from_features(held)

    final_ref = PatchCoreDetector(config, extractor=backbone)
    final_ref.fit_from_features(reference)
    rows_final_over = logic.greedy_coreset_over_images(flat, list(range(m)), patches)
    selected = g.streaming_coreset_select(g.iter_image_features(detector, (preprocessed(i) for i in range(m))), m, patches, dim)
    bank_stream = g.gather_selected_rows(g.iter_image_features(detector, (preprocessed(i) for i in range(m))), selected, patches, dim)

    report = {
        "images": m, "fold": 0, "train_images": train, "held_out_images": members, "patches_per_image": patches,
        "train_feature_max_abs_diff": max_diff(buffer, reference[train]),
        "held_out_feature_max_abs_diff": max_diff(held, reference[members]),
        "fold_bank_rows_identical": bool(torch.equal(rows_light, rows_ref)),
        "fold_bank_max_abs_diff": max_diff(bank_light, bank_ref),
        "patch_score_max_abs_diff": max_diff(scores_light, scores_ref),
        "image_score_max_abs_diff": {agg: max_diff(light.image_scores(scores_light, agg), light.image_scores(scores_ref, agg))
                                     for agg in logic.AGGREGATIONS},
        "final_streaming_rows_identical_to_study_fold_method": bool(torch.equal(selected, rows_final_over)),
        "final_streaming_bank_max_abs_diff_vs_fit_from_features": max_diff(bank_stream, final_ref.bank),
        "final_streaming_bank_identical_to_fit_from_features": bool(torch.equal(bank_stream, final_ref.bank)),
        "peak_ram_mb": study.peak_ram_mb(),
        "audit": audit.verify(),
    }
    (OUT_DIR / "equivalence_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for key, value in report.items():
        if key not in ("train_images", "held_out_images"):
            log(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except study.RamBudgetExceeded as exc:
        print(f"STOPPED: {exc}", flush=True)
        sys.exit(3)
    except ConsistencyError as exc:
        print(f"STOPPED: {exc}", flush=True)
        sys.exit(4)
