"""Carpet-style calibration optimization pipeline: study -> pre-declared selection -> freeze.

Runs entirely on the category's train/good images (plus labelled-synthetic defects painted on the existing
validation-good images as a diagnostic). It cannot reach the final test set: it does not import test discovery
(asserted by a test), and the final test is scored separately by app.ai.evaluation.calibration_final_test.

What is held fixed (Phase D): the ResNet-18 backbone and its verified weights, layers 2+3, 3x3 local pooling,
256x256 input, and one global Gaussian with shrinkage 1e-2. Only the image-level aggregation, the threshold policy
and the calibration procedure vary. No network is trained and no augmentation is used.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from app.ai.evaluation import calibration_study as cs
from app.ai.evaluation.anomaly_model_selection import DevArrays, build_dev_arrays, to_tensor
from app.ai.evaluation.category_phase3 import PHASE3_SPLIT_SEED, PHASE3_TRAINING_FRACTION
from app.ai.models.patch_anomaly import GaussianPatchScorer, PatchAnomalyDetector, extract_patch_features
from app.ai.models.resnet18 import RESNET18_SHA256, ResNet18, load_pretrained_resnet18
from app.ai.training import artifacts, discover_train_samples
from app.ai.training.validation_split import TrainValidationSplit, split_train_validation

SIDE = 256
LAYERS = (2, 3)
BASELINE_AGGREGATION = "top1pct_mean"
BASELINE_POLICY = ("mean_std", 1.5)


def optimization_root(category: str) -> Path:
    return artifacts.ARTIFACTS_ROOT / category / "optimization"


def reports_dir(category: str) -> Path:
    return optimization_root(category) / "reports"


def selected_dir(category: str) -> Path:
    return optimization_root(category) / "selected_candidate"


@dataclass
class StudyInputs:
    category: str
    split: TrainValidationSplit
    data: DevArrays
    pool: cs.GoodPool
    extractor: ResNet18
    build_seconds: float = 0.0


def load_inputs(category: str, extractor: ResNet18 | None = None, log=print) -> StudyInputs:
    """Decode the category's train/good images once (224 training + 56 validation, the existing split), paint the
    synthetic diagnostic defects on the VALIDATION images, and cache patch features for all 280 good images."""
    t0 = time.perf_counter()
    split = split_train_validation(discover_train_samples(category), PHASE3_TRAINING_FRACTION, PHASE3_SPLIT_SEED)
    data = build_dev_arrays(split, (SIDE,))
    names = data.train_files + data.validation_files
    order = np.argsort(names, kind="stable")
    files = [names[i] for i in order]
    images = np.concatenate([data.train[SIDE], data.validation[SIDE]])[order]
    extractor = extractor or load_pretrained_resnet18()
    pool = cs.build_pool(extractor, to_tensor(images), files, LAYERS)
    seconds = time.perf_counter() - t0
    log(f"  inputs ready in {seconds:.0f}s: {len(files)} good images, {len(data.synthetic_meta)} synthetic "
        f"(digest {data.synthetic_digest[:12]}), patches/image {pool.patches_per_image}")
    return StudyInputs(category, split, data, pool, extractor, seconds)


@dataclass
class StudyResult:
    summary: dict
    rows: list[cs.ConfigRow]
    selection: cs.ConfigSelection
    thresholds: dict = field(default_factory=dict)  # config_id -> {"random": t, "block": t, "final": t}


def _scheme_stats(fold_scores, agg, policy, key) -> dict:
    return cs.summarize_transfer(cs.pooled_transfer(fold_scores, agg, policy, group_key=key))


def run_study(inp: StudyInputs, log=print) -> StudyResult:
    pool, n = inp.pool, len(inp.pool.files)
    aggs = cs.STUDY_AGGREGATIONS
    schemes = cs.all_schemes(n)
    log("  scoring every good-only fold out-of-fold ...")
    scored = {name: cs.score_folds(pool, folds, aggs) for name, folds in schemes.items()}
    existing = cs.existing_split(pool.files, inp.data.validation_files)
    existing_scored = cs.score_folds(pool, [existing], aggs)[0]

    # --- Consistency with the original frozen model's validation scores (same split, same aggregation) ---
    stage_a = artifacts.ARTIFACTS_ROOT / inp.category / "model_selection" / "reports" / "stage_a_results.json"
    consistency = None
    if stage_a.is_file():
        prev = next(r for r in json.loads(stage_a.read_text())["results"]
                    if r["spec"]["candidate_id"] == "pgauss_l23_256_top1pct_mean")
        # stage A validation scores are in split.validation order; existing fold heldout is in pool order.
        by_file = dict(zip([pool.files[i] for i in existing.heldout_idx], existing_scored.scores[BASELINE_AGGREGATION]))
        ours = np.array([by_file[f] for f in inp.data.validation_files])
        consistency = {"max_relative_difference": float(np.max(np.abs(ours - np.array(prev["validation_scores"])) /
                                                              np.array(prev["validation_scores"])))}

    # ---------------- Phase A: calibration robustness (baseline aggregation, baseline rule) ----------------
    base_agg, base_policy = BASELINE_AGGREGATION, BASELINE_POLICY
    phase_a = {"distributions": {}, "folds": {}, "drift": None, "in_sample_vs_out_of_sample": None}
    phase_a["existing_split"] = {
        "validation_size": len(existing.heldout_idx),
        "threshold_baseline_rule": cs.policy_thresholds(existing_scored.scores[base_agg])[base_policy],
        "distribution": cs.distribution_summary(existing_scored.scores[base_agg]),
    }
    tau_existing = phase_a["existing_split"]["threshold_baseline_rule"]
    phase_a["existing_split"]["validation_false_positives"] = int(np.sum(existing_scored.scores[base_agg] > tau_existing))
    for name, fs in scored.items():
        pooled = cs.pooled_oof(fs, base_agg)
        phase_a["distributions"][name] = cs.distribution_summary(pooled)
        rows = []
        for f in fs:
            s = f.scores[base_agg]
            tau = cs.policy_thresholds(s)[base_policy]
            rows.append({"fold": f.fold.fold_id, "validation_size": len(s), "threshold": tau,
                         "false_positives": int(np.sum(s > tau)), "fpr": cs.fpr(tau, s), "mean": float(s.mean()),
                         "std": float(s.std()), "median": float(np.median(s)), "cov": float(s.std() / s.mean()),
                         "p95": float(np.percentile(s, 95)), "p99": float(np.percentile(s, 99))})
        phase_a["folds"][name] = rows
        thr = np.array([r["threshold"] for r in rows])
        phase_a.setdefault("threshold_stability", {})[name] = {
            "threshold_mean": float(thr.mean()), "threshold_std": float(thr.std()),
            "threshold_cv": float(thr.std() / thr.mean()), "threshold_min": float(thr.min()), "threshold_max": float(thr.max()),
            "single_fold_transfer": cs.summarize_transfer(cs.single_fold_transfer(fs, base_agg, base_policy)),
        }
    phase_a["drift"] = cs.drift_diagnostics(scored[cs.SCHEME_BLOCK], scored[cs.SCHEME_RANDOM], base_agg, n)
    a1_scorer = cs.fit_scorer(pool, existing.train_idx)
    in_sample = cs.aggregate_np(a1_scorer.patch_scores(pool.patches[list(existing.train_idx)]), base_agg)
    phase_a["in_sample_vs_out_of_sample"] = {
        "training_images_in_sample": cs.distribution_summary(in_sample),
        "validation_out_of_sample": cs.distribution_summary(existing_scored.scores[base_agg]),
    }

    # ---------------- Synthetic diagnostic scores (model fitted on the 224 training images) ----------------
    log("  scoring the synthetic-defect diagnostic set ...")
    synthetic = inp.data.synthetic[SIDE]
    maps = []
    for start in range(0, len(synthetic), 8):
        feats = extract_patch_features(inp.extractor, to_tensor(synthetic[start:start + 8]), LAYERS)
        maps.append(a1_scorer.patch_scores(feats))
    syn_maps = torch.cat(maps)
    syn_scores = {a: cs.aggregate_np(syn_maps, a) for a in aggs}
    syn_auroc = {a: cs.synthetic_auroc(existing_scored.scores[a], syn_scores[a]) for a in aggs}

    # ---------------- Phases B + C: every (aggregation, policy) configuration ----------------
    block_key = lambda f: "blocks"  # noqa: E731
    seed_key = lambda f: f.fold.fold_id.rsplit("_fold", 1)[0]  # noqa: E731
    rows, thresholds, table = [], {}, []
    for agg in aggs:
        pooled_r = cs.pooled_oof(scored[cs.SCHEME_RANDOM], agg)
        pooled_b = cs.pooled_oof(scored[cs.SCHEME_BLOCK], agg)
        tau_r_all, tau_b_all = cs.policy_thresholds(pooled_r), cs.policy_thresholds(pooled_b)
        oof_cov = float(pooled_r.std() / pooled_r.mean())
        for policy in cs.POLICIES:
            sr = _scheme_stats(scored[cs.SCHEME_RANDOM], agg, policy, seed_key)
            sb = _scheme_stats(scored[cs.SCHEME_BLOCK], agg, policy, block_key)
            final_tau = max(tau_r_all[policy], tau_b_all[policy])
            row = cs.ConfigRow(agg, policy, max(sr["fpr_max"], sb["fpr_max"]), max(sr["fpr_mean"], sb["fpr_mean"]),
                               final_tau, cs.synthetic_recall(final_tau, syn_scores[agg]), oof_cov)
            rows.append(row)
            thresholds[row.config_id] = {"random": tau_r_all[policy], "block": tau_b_all[policy], "final": final_tau}
            table.append({"config": row.config_id, "aggregation": agg, "policy": list(policy),
                          "random_5fold_pooled_transfer": sr, "block_pooled_transfer": sb,
                          "threshold_random_pooled": tau_r_all[policy], "threshold_block_pooled": tau_b_all[policy],
                          "final_threshold": final_tau, "synthetic_recall_diagnostic": row.synthetic_recall,
                          "synthetic_auroc_diagnostic": syn_auroc[agg], "oof_cov_random": oof_cov})

    selection = cs.select_configuration(rows)
    single = {}
    for agg in aggs:  # small-calibration-set transfer for every (aggregation, policy) on the existing 56-size folds
        for policy in cs.POLICIES:
            single[f"{agg}|{policy[0]}_{policy[1]:g}"] = {
                sch: cs.summarize_transfer(cs.single_fold_transfer(scored[sch], agg, policy)) for sch in scored}
    summary = {
        "category": inp.category, "side": SIDE, "layers": list(LAYERS), "n_good": n,
        "consistency_with_original_stage_a": consistency,
        "phase_a": phase_a, "configurations": table, "single_fold_transfer": single,
        "synthetic_diagnostic": {"auroc_by_aggregation": syn_auroc, "note": "SYNTHETIC defects on validation good images - not real Carpet recall."},
        "selection": {"winner": selection.winner.config_id, "passes_primary_gate": selection.passes_primary_gate,
                      "meets_secondary_target": selection.meets_secondary_target, "reasoning": selection.reasoning,
                      "eligible": selection.eligible},
        "declared_gates": {"primary_fpr": cs.PRIMARY_FPR, "secondary_fpr": cs.SECONDARY_FPR,
                           "synthetic_recall_floor": cs.RECALL_FLOOR, "recall_tie": cs.RECALL_TIE},
        "split_design": {"random_kfold_seeds": list(cs.RANDOM_KFOLD_SEEDS), "k": cs.RANDOM_KFOLD_K,
                         "holdout_fractions": list(cs.HOLDOUT_FRACTIONS), "holdout_repeats": cs.HOLDOUT_REPEATS,
                         "holdout_seed_base": cs.HOLDOUT_SEED, "contiguous_blocks": cs.BLOCK_K,
                         "folds_per_scheme": {k: len(v) for k, v in schemes.items()}},
    }
    return StudyResult(summary, rows, selection, thresholds)


def compare_studies(a: StudyResult, b: StudyResult) -> dict:
    """Exact comparison of two complete, independent study runs."""
    out = {
        "configuration_rows_identical": a.rows == b.rows,
        "selected_configuration_identical": a.selection.winner == b.selection.winner,
        "final_thresholds_identical": a.thresholds == b.thresholds,
        "phase_a_identical": a.summary["phase_a"] == b.summary["phase_a"],
        "configurations_table_identical": a.summary["configurations"] == b.summary["configurations"],
        "synthetic_diagnostic_identical": a.summary["synthetic_diagnostic"] == b.summary["synthetic_diagnostic"],
    }
    out["all_identical"] = all(out.values())
    return out


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

def _md5_sha256(path: Path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest(), path.stat().st_size


def fit_final_detector(inp: StudyInputs, aggregation: str) -> PatchAnomalyDetector:
    """The frozen Gaussian, fitted on ALL good images (train + validation pool) - final-test images are not involved."""
    scorer = GaussianPatchScorer.fit(inp.pool.patches)
    return PatchAnomalyDetector(inp.extractor, LAYERS, SIDE, scorer, aggregation)


def freeze_optimized(inp: StudyInputs, study: StudyResult, directory: Path, baseline_reference: dict) -> dict:
    directory = Path(directory)
    if (directory / "frozen_model.json").exists():
        raise FileExistsError(f"A frozen optimized model already exists in {directory}; refusing to overwrite.")
    winner = study.selection.winner
    detector = fit_final_detector(inp, winner.aggregation)
    state_path = detector.save(directory)
    md5, sha, size = _md5_sha256(state_path)
    t = study.thresholds[winner.config_id]
    lock = {
        "candidate_id": f"optimized_{winner.config_id}",
        "family": "patch_gaussian",
        "config": detector.config(),
        "backbone": {"name": "resnet18_imagenet_frozen", "sha256": RESNET18_SHA256},
        "fitted_on": {"training_images": len(inp.pool.files),
                      "file_list_sha256": hashlib.sha256("\n".join(inp.pool.files).encode()).hexdigest(),
                      "pool": "all train/good images (224 training + 56 validation)"},
        "threshold": {"method": winner.policy[0], "parameter": winner.policy[1], "value": winner.final_threshold,
                      "rule": "policy applied to pooled out-of-fold scores of good images; the larger of the random "
                              "5-fold and contiguous-block thresholds",
                      "random_5fold_pooled_threshold": t["random"], "contiguous_block_pooled_threshold": t["block"],
                      "note": "Fold models were fitted on ~224 images; the frozen model on 280, so scores on new "
                              "images are expected to be equal or marginally lower (threshold marginally conservative)."},
        "selection": {"config_id": winner.config_id, "reasoning": study.selection.reasoning,
                      "passes_primary_gate": study.selection.passes_primary_gate,
                      "meets_secondary_target": study.selection.meets_secondary_target,
                      "worst_fold_transfer_fpr": winner.fpr_max, "mean_transfer_fpr": winner.fpr_mean,
                      "synthetic_recall_diagnostic": winner.synthetic_recall},
        "state_file": {"name": state_path.name, "md5": md5, "sha256": sha, "size_bytes": size},
        "baseline_reference": baseline_reference,
        "digest": None,
    }
    lock["digest"] = hashlib.sha256(json.dumps({k: v for k, v in lock.items() if k != "digest"}, sort_keys=True,
                                               default=str).encode()).hexdigest()
    (directory / "frozen_model.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


__all__ = ["StudyInputs", "StudyResult", "compare_studies", "fit_final_detector", "freeze_optimized", "load_inputs",
           "optimization_root", "reports_dir", "run_study", "selected_dir", "BASELINE_AGGREGATION", "BASELINE_POLICY",
           "SIDE", "LAYERS", "RESNET18_SHA256"]
