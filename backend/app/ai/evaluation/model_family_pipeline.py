"""Model-family study pipeline: load train/good inputs -> evaluate every registered candidate on the normal-only
splits + the synthetic diagnostic -> fit and save all 8 candidate artifacts -> apply the pre-declared selection rule
-> write the hash-locked selection lock. See app.ai.evaluation.model_family_study for the registry and the rule.

It cannot reach the final test set: it never imports test discovery (asserted from source by tests). The only
touch of the test tree before the lock is `count_dataset_entries`, which counts directory entries per folder (no
image is opened, no file name is kept) so the mandatory dataset-count verification can run.

Artifacts go under backend/ai_models/<category>/model_family_study/ (Git-ignored); nothing is registered for serving.
"""

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from app.ai.evaluation import model_family_study as mfs
from app.ai.evaluation.anomaly_model_selection import SYNTHETIC_SEED, to_tensor
from app.ai.evaluation.category_phase1 import describe, file_hashes
from app.ai.evaluation.synthetic_defects import DEFECT_TYPES, SEVERITIES, make_synthetic_defect
from app.ai.models.patch_anomaly import (
    GaussianPatchScorer,
    KNNPatchScorer,
    PatchAnomalyDetector,
    extract_patch_features,
)
from app.ai.models.resnet18 import RESNET18_SHA256, ResNet18, load_pretrained_resnet18, pretrained_weights_path
from app.ai.preprocessing.pipeline import _build_preprocessing_result, _load_image
from app.ai.training import artifacts
from app.ai.training.dataset import discover_train_samples, get_category_dir
from app.inspections.storage import ALLOWED_EXTENSIONS

STUDY_DIRNAME = "model_family_study"
SIDES = (128, 256)
SYNTHETIC_BATCH = 16
PRIOR_PHASE_DIRS = ("phase1_baseline", "phase2_threshold", "phase3_validation")


def study_root(category: str) -> Path:
    return artifacts.ARTIFACTS_ROOT / category / STUDY_DIRNAME


def candidates_dir(category: str) -> Path:
    return study_root(category) / "candidates"


def selected_dir(category: str) -> Path:
    return study_root(category) / "selected_candidate"


def reports_dir(category: str) -> Path:
    return study_root(category) / "reports"


def lock_path(category: str) -> Path:
    return study_root(category) / "selection_lock.json"


def study_report_path(category: str) -> Path:
    return reports_dir(category) / "model_family_study.json"


# ---------------------------------------------------------------------------
# Dataset counts (metadata only), fingerprints, prior-phase references
# ---------------------------------------------------------------------------

def count_dataset_entries(category: str, expected: dict) -> dict:
    """Directory-entry COUNTS per folder; raises on a mismatch. Opens no image and retains no file name."""
    root = get_category_dir(category)

    def count(directory: Path) -> int:
        if not directory.is_dir():
            return 0
        return sum(1 for e in os.scandir(directory) if e.is_file() and Path(e.name).suffix.lower() in ALLOWED_EXTENSIONS)

    by_type = {d.name: count(d) for d in sorted(os.scandir(root / "test"), key=lambda e: e.name) if d.is_dir()}
    actual = {"train_good": count(root / "train" / "good"), "test_good": by_type.get("good", 0),
              "test_defective": sum(v for k, v in by_type.items() if k != "good")}
    actual["test_total"] = actual["test_good"] + actual["test_defective"]
    actual["test_by_type"] = dict(sorted(by_type.items()))
    mismatches = {k: {"expected": v, "actual": actual[k]} for k, v in expected.items() if actual[k] != v}
    if mismatches:
        raise ValueError(f"Dataset counts differ from the expected counts: {mismatches}")
    return actual


def dataset_fingerprint(samples, counts: dict) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(f"{s.path.name}:{hashlib.sha256(s.path.read_bytes()).hexdigest()}\n".encode())
    h.update(json.dumps({k: counts[k] for k in ("train_good", "test_good", "test_defective", "test_total")}, sort_keys=True).encode())
    return h.hexdigest()


def prior_phase_reference(category: str) -> dict:
    root = artifacts.ARTIFACTS_ROOT / category
    ref = {}
    for sub in PRIOR_PHASE_DIRS:
        base = root / sub
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    ref[str(path.relative_to(root)).replace("\\", "/")] = file_hashes(path)["sha256"]
    return ref


def backbone_info() -> dict:
    path = pretrained_weights_path()
    h = file_hashes(path)
    if h["sha256"] != RESNET18_SHA256:
        raise RuntimeError("Verified ResNet-18 backbone hash mismatch.")
    return {"path": str(path), "sha256": h["sha256"], "md5": h["md5"], "size_bytes": h["size_bytes"], "fine_tuned": False}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class StudyInputs:
    category: str
    names: list[str]
    arrays: dict[int, np.ndarray]
    synthetic: dict[int, np.ndarray]
    synthetic_meta: list[dict]
    synthetic_digest: str
    canonical_held_out: list[int]
    schemes: dict
    counts: dict
    dataset_fingerprint: str
    extractor: ResNet18
    build_seconds: float = 0.0
    feature_ms: dict = field(default_factory=dict)


def load_inputs(category: str, expected_counts: dict, extractor: ResNet18 | None = None) -> StudyInputs:
    """Decode every train/good image once (at 128 and 256), and paint the synthetic diagnostic defects on the
    canonical split's held-out images at full resolution. No test image is listed, decoded or scored."""
    t0 = time.perf_counter()
    counts = count_dataset_entries(category, expected_counts)
    samples = discover_train_samples(category)
    if any(s.split != "train" or s.defect_type != "good" for s in samples):
        raise RuntimeError("Calibration pool contains a non-train/good sample.")
    names = [s.path.name for s in samples]
    if names != sorted(names):
        raise RuntimeError("Train/good samples are not in sorted file-name order.")
    n = len(samples)
    schemes = mfs.scheme_splits(n)
    canonical = next(s for s in schemes[mfs.CANONICAL_SPLIT[0]] if s["name"] == mfs.CANONICAL_SPLIT[1])["held_out"].tolist()
    canonical_set = set(canonical)

    lists = {s: [] for s in SIDES}
    syn = {s: [] for s in SIDES}
    meta, digest = [], hashlib.sha256()
    for index, sample in enumerate(samples):
        image = _load_image(sample.path)
        for side in SIDES:
            lists[side].append(_build_preprocessing_result(sample.path, image, (side, side)).resized_image)
        if index in canonical_set:
            for defect_type in DEFECT_TYPES:
                for severity in SEVERITIES:
                    painted, _ = make_synthetic_defect(image, defect_type, severity, seed=SYNTHETIC_SEED + index)
                    for side in SIDES:
                        syn[side].append(_build_preprocessing_result(sample.path, painted, (side, side)).resized_image)
                    digest.update(painted.tobytes())
                    meta.append({"source_file": sample.path.name, "source_index": index, "defect_type": defect_type, "severity": severity})
    return StudyInputs(
        category=category, names=names, arrays={s: np.stack(v) for s, v in lists.items()},
        synthetic={s: np.stack(v) for s, v in syn.items()}, synthetic_meta=meta, synthetic_digest=digest.hexdigest(),
        canonical_held_out=canonical, schemes=schemes, counts=counts, dataset_fingerprint=dataset_fingerprint(samples, counts),
        extractor=extractor if extractor is not None else load_pretrained_resnet18(), build_seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Candidate evaluation + fitting
# ---------------------------------------------------------------------------

def _synthetic_scores(inputs: StudyInputs, spec: mfs.CandidateSpec, scorer) -> np.ndarray:
    out = []
    arr = inputs.synthetic[spec.image_size]
    for start in range(0, len(arr), SYNTHETIC_BATCH):
        patches = extract_patch_features(inputs.extractor, to_tensor(arr[start : start + SYNTHETIC_BATCH]), spec.layers)
        out.append(mfs.image_scores(scorer, patches, spec.aggregation))
    return np.concatenate(out)


def evaluate_group(inputs: StudyInputs, specs: list[mfs.CandidateSpec], out_dir: Path, log=print) -> dict:
    """All candidates sharing (image_size, layers): one feature pass, both scorer families evaluated on every split,
    the synthetic diagnostic scored by the canonical split's model, and the final model fitted on all normals."""
    side, layers = specs[0].image_size, specs[0].layers
    t0 = time.perf_counter()
    patches = extract_patch_features(inputs.extractor, to_tensor(inputs.arrays[side]), layers)
    feature_seconds = time.perf_counter() - t0
    n, h, w, d = patches.shape
    all_idx = np.arange(n)
    canon_cal = next(s for s in inputs.schemes[mfs.CANONICAL_SPLIT[0]] if s["name"] == mfs.CANONICAL_SPLIT[1])["calibration"]
    results = {}

    for spec in specs:
        t_fit = time.perf_counter()
        if spec.family == mfs.FAMILY_GAUSSIAN:
            sums, outers = mfs.image_statistics(patches)
            score_split = lambda cal, held: mfs.gaussian_split_scores(patches, sums, outers, cal, held, spec.shrinkage, spec.aggregation)  # noqa: E731
        else:
            banks = mfs.per_image_coresets(patches, spec.coreset_fraction, mfs.CORESET_SEED)
            minima = mfs.pairwise_minima(patches, banks)
            score_split = lambda cal, held: mfs.knn_split_scores(minima, h, w, cal, held, spec.aggregation)  # noqa: E731
        prep_seconds = time.perf_counter() - t_fit

        t_eval = time.perf_counter()
        split_scores = {}
        for scheme, splits in inputs.schemes.items():
            split_scores[scheme] = []
            for s in splits:
                cal_scores, held_scores = score_split(s["calibration"], s["held_out"])
                split_scores[scheme].append({"name": s["name"], "calibration": cal_scores, "held_out": held_scores})
        all_scores, _ = score_split(all_idx, np.empty(0, dtype=int))  # leave-one-image-out over all normals

        # synthetic diagnostic: scored by the canonical split's model (fitted on its calibration images only)
        if spec.family == mfs.FAMILY_GAUSSIAN:
            cal = torch.as_tensor(canon_cal)
            scorer0 = GaussianPatchScorer.from_statistics(h * w * len(canon_cal), sums[cal].sum(0), outers[cal].sum(0), spec.shrinkage)
        else:
            scorer0 = mfs.knn_bank(banks, canon_cal)
        syn_scores = _synthetic_scores(inputs, spec, scorer0)
        eval_seconds = time.perf_counter() - t_eval

        summary = mfs.summarize_candidate(split_scores, all_scores, {"scores": syn_scores, "meta": inputs.synthetic_meta})

        # final model fitted on ALL normal images -> artifact
        if spec.family == mfs.FAMILY_GAUSSIAN:
            scorer = GaussianPatchScorer.from_statistics(h * w * n, sums.sum(0), outers.sum(0), spec.shrinkage)
        else:
            scorer = mfs.knn_bank(banks, all_idx)
        detector = PatchAnomalyDetector(inputs.extractor, spec.layers, side, scorer, spec.aggregation)
        cdir = Path(out_dir) / spec.candidate_id
        if cdir.exists():
            raise FileExistsError(f"Refusing to overwrite an existing candidate artifact: {cdir}")
        state_path = detector.save(cdir)
        hashes = file_hashes(state_path)
        probe = to_tensor(inputs.arrays[side][:16])
        detector.score_images(probe[:2])  # warm-up
        t_score = time.perf_counter()
        probe_scores = detector.score_images(probe)
        score_ms = (time.perf_counter() - t_score) * 1000 / len(probe)

        manifest = {
            "candidate_id": spec.candidate_id, "config": spec.config(), "artifact": {**hashes, "path": str(state_path)},
            "config_file_sha256": file_hashes(cdir / "model_config.json")["sha256"],
            "training_data_fingerprint": inputs.dataset_fingerprint, "training_image_count": n, "fitted_on": f"all {n} train/good images",
            "feature_config": {"layers": list(spec.layers), "image_size": side, "pooling": spec.pooling, "feature_dim": d,
                               "patches_per_image": h * w, "normalization": "imagenet_mean_std_on_top_of_project_preprocessing"},
            "thresholds_full_pool": {p: e["full_pool_threshold"] for p, e in summary["policies"].items()},
            "backbone_sha256": RESNET18_SHA256,
        }
        (cdir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        results[spec.candidate_id] = {
            "spec": spec.config(), "summary": summary, "artifact": manifest["artifact"], "manifest": manifest,
            "cost": {"feature_extraction_seconds_shared_group": feature_seconds, "prep_seconds": prep_seconds,
                     "cv_and_synthetic_seconds": eval_seconds, "score_ms_per_image_end_to_end": score_ms,
                     "state_size_bytes": hashes["size_bytes"], "probe_score_mean": float(np.mean(probe_scores))},
        }
        log(f"  {spec.candidate_id}: prep {prep_seconds:.0f}s, cv+synthetic {eval_seconds:.0f}s, state {hashes['size_bytes']/1e6:.2f} MB, "
            f"synthetic AUROC {summary['synthetic_auroc']:.3f}")
        if spec.family == mfs.FAMILY_GAUSSIAN:
            del sums, outers
        else:
            del minima, banks
    return results


def run_study(inputs: StudyInputs, out_dir: Path, log=print) -> dict:
    specs = mfs.registry()
    if len(specs) != mfs.EXPECTED_CANDIDATES or len(mfs.policy_ids()) != mfs.EXPECTED_POLICIES:
        raise RuntimeError("Candidate/policy registry differs from the declared 8 x 8 grid.")
    groups: dict[tuple, list] = {}
    for s in specs:
        groups.setdefault((s.image_size, s.layers), []).append(s)
    t0 = time.perf_counter()
    results = {}
    for key in sorted(groups):
        log(f"group side={key[0]} layers={key[1]}")
        results.update(evaluate_group(inputs, groups[key], out_dir, log))
    results = {s.candidate_id: results[s.candidate_id] for s in specs}  # report in registry order
    selection = mfs.select_configuration({cid: r["summary"] for cid, r in results.items()}, specs)
    return {
        "category": inputs.category, "registry": [s.config() for s in specs], "registry_fingerprint": mfs.registry_fingerprint(specs),
        "policies": mfs.policy_ids(), "counts": inputs.counts, "dataset_fingerprint_sha256": inputs.dataset_fingerprint,
        "normal_file_list_sha256": hashlib.sha256("\n".join(inputs.names).encode()).hexdigest(),
        "splits_sha256": mfs.splits_fingerprint(inputs.schemes), "split_audit": mfs.split_audit(len(inputs.names), inputs.schemes),
        "scheme_definitions": {k: {"splits": len(v), "held_out": len(v[0]["held_out"]), "calibration": len(v[0]["calibration"])} for k, v in inputs.schemes.items()},
        "synthetic": {"seed_base": SYNTHETIC_SEED, "images": len(inputs.synthetic_meta), "digest_sha256": inputs.synthetic_digest,
                      "held_out_images": len(inputs.canonical_held_out), "variants_per_image": len(DEFECT_TYPES) * len(SEVERITIES),
                      "source_split": "holdout_80_20/seed42 held-out images",
                      "label": f"DIAGNOSTIC ONLY - not real {inputs.category.replace('_', ' ').title()} defect recall"},
        "backbone": backbone_info(), "candidates": {cid: {k: v for k, v in r.items() if k != "manifest"} for cid, r in results.items()},
        "selection": selection, "declared_rule": mfs.__doc__,
        "input_build_seconds": inputs.build_seconds, "study_seconds": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Selection lock
# ---------------------------------------------------------------------------

def lock_digest(lock: dict) -> str:
    """Canonical content hash, excluding wall-clock fields and the artifact's filesystem location."""
    volatile = ("timestamp_unix", "timestamp_utc", "lock_digest")
    body = {k: v for k, v in lock.items() if k not in volatile}
    body["selected_artifact"] = {k: v for k, v in body.get("selected_artifact", {}).items() if k != "path"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def build_lock(category: str, study: dict, selected_state: Path, prior: dict, timestamp: float) -> dict:
    sel = study["selection"]
    if sel["winner"] is None:
        raise RuntimeError(mfs.NO_CANDIDATE_MESSAGE + " Nothing to lock.")
    cid = sel["winner_candidate"]
    art = file_hashes(selected_state)
    if art["sha256"] != study["candidates"][cid]["artifact"]["sha256"]:
        raise RuntimeError("Selected artifact does not match the candidate's fitted artifact.")
    lock = {
        "category": category, "experiment_id": f"{category.replace('_', '-')}-model-family-{category}-{study['splits_sha256'][:12]}",
        "timestamp_unix": timestamp, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "candidate_list": [c["candidate_id"] for c in study["registry"]], "candidate_configurations": study["registry"],
        "registry_fingerprint": study["registry_fingerprint"], "backbone": study["backbone"],
        "normal_dataset_fingerprint_sha256": study["dataset_fingerprint_sha256"], "dataset_counts": study["counts"],
        "normal_file_list_sha256": study["normal_file_list_sha256"], "split_fingerprint_sha256": study["splits_sha256"],
        "scheme_definitions": study["scheme_definitions"],
        "threshold_candidates": {c: {p: e["full_pool_threshold"] for p, e in r["summary"]["policies"].items()} for c, r in study["candidates"].items()},
        "normal_robustness_results": [{k: r[k] for k in ("candidate", "policy", "worst_fold_fpr", "mean_fpr", "threshold_cv", "passes_gate")} for r in sel["rows"]],
        "synthetic_diagnostic_results": {"label": "DIAGNOSTIC ONLY", "per_configuration": [{k: r[k] for k in ("candidate", "policy", "synthetic_recall")} for r in sel["rows"]],
                                         "per_candidate_auroc": {c: r["summary"]["synthetic_auroc"] for c, r in study["candidates"].items()},
                                         "synthetic_digest_sha256": study["synthetic"]["digest_sha256"]},
        "selection_rule": study["declared_rule"], "selection_rule_sha256": hashlib.sha256(study["declared_rule"].encode("utf-8")).hexdigest(),
        "selection_reasoning": sel["reasoning"], "preferred_synthetic_floor_met": sel["preferred_floor_met"],
        "selected_candidate": cid, "selected_policy": sel["winner_policy"], "selected_threshold": sel["threshold"],
        "selected_configuration": next(c for c in study["registry"] if c["candidate_id"] == cid),
        "selected_artifact": {**art, "path": str(selected_state)}, "prior_phase_artifacts": prior,
        "final_test_data_used_for_selection": False,
    }
    lock["lock_digest"] = lock_digest(lock)
    return lock


def create_selection_lock(category: str, study: dict, timestamp: float | None = None) -> dict:
    """Copy the winning candidate to selected_candidate/, verify the copy's hash, and write selection_lock.json."""
    if lock_path(category).exists() or selected_dir(category).exists():
        raise FileExistsError("Refusing to overwrite an existing selection lock / selected candidate.")
    sel = study["selection"]
    if sel["winner"] is None:
        return {"locked": False, "message": mfs.NO_CANDIDATE_MESSAGE}
    src = candidates_dir(category) / sel["winner_candidate"]
    shutil.copytree(src, selected_dir(category))
    lock = build_lock(category, study, selected_dir(category) / "model_state.pt", prior_phase_reference(category),
                      time.time() if timestamp is None else timestamp)
    lock_path(category).write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
    return {"locked": True, "lock": lock, "lock_file": file_hashes(lock_path(category))}


def compare_studies(a: dict, b: dict) -> dict:
    """Everything the second complete run must reproduce (no final-test data involved)."""
    canon = lambda x: json.loads(json.dumps(x, sort_keys=True))  # noqa: E731
    hashes = lambda s: {c: r["artifact"]["sha256"] for c, r in s["candidates"].items()}  # noqa: E731
    return {
        "candidate_configurations_identical": canon(a["registry"]) == canon(b["registry"]) and a["registry_fingerprint"] == b["registry_fingerprint"],
        "dataset_fingerprint_identical": a["dataset_fingerprint_sha256"] == b["dataset_fingerprint_sha256"],
        "split_fingerprints_identical": a["splits_sha256"] == b["splits_sha256"],
        "synthetic_digest_identical": a["synthetic"]["digest_sha256"] == b["synthetic"]["digest_sha256"],
        "candidate_artifact_hashes_identical": hashes(a) == hashes(b),
        "threshold_values_identical": canon({c: {p: e["full_pool_threshold"] for p, e in r["summary"]["policies"].items()} for c, r in a["candidates"].items()})
        == canon({c: {p: e["full_pool_threshold"] for p, e in r["summary"]["policies"].items()} for c, r in b["candidates"].items()}),
        "robustness_metrics_identical": canon({c: {p: (e["worst_fold_fpr"], e["mean_fpr"], e["pooled_threshold"]) for p, e in r["summary"]["policies"].items()} for c, r in a["candidates"].items()})
        == canon({c: {p: (e["worst_fold_fpr"], e["mean_fpr"], e["pooled_threshold"]) for p, e in r["summary"]["policies"].items()} for c, r in b["candidates"].items()}),
        "synthetic_diagnostics_identical": canon({c: (r["summary"]["synthetic_auroc"], {p: e["synthetic_recall"] for p, e in r["summary"]["policies"].items()}) for c, r in a["candidates"].items()})
        == canon({c: (r["summary"]["synthetic_auroc"], {p: e["synthetic_recall"] for p, e in r["summary"]["policies"].items()}) for c, r in b["candidates"].items()}),
        "full_summaries_identical": canon({c: r["summary"] for c, r in a["candidates"].items()}) == canon({c: r["summary"] for c, r in b["candidates"].items()}),
        "selection_identical": canon(a["selection"]) == canon(b["selection"]),
        "selected_candidate_identical": a["selection"]["winner"] == b["selection"]["winner"],
        "selected_threshold_identical": a["selection"]["threshold"] == b["selection"]["threshold"],
    }
