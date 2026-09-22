"""Model-family / resolution / scoring-rule selection for one category - TRAIN + VALIDATION ONLY.

Compares anomaly-detection candidates without ever touching the category's final test set:

  * FITTING data: the 224 training/good images (ConvAE weights, PatchCore memory bank, Gaussian statistics).
  * THRESHOLD data: the 56 validation/good images, with the existing Phase 3 rule (lowest of mean+K*std
    K in 3,2.5,2,1.5,1 and percentiles 95/97/98/99 whose validation false-positive rate <= 10%).
  * MODEL-SELECTION evidence: because validation holds only good images (real defects exist only in the
    forbidden test split), candidates are ranked on SYNTHETIC defects painted onto the validation good
    images (app.ai.evaluation.synthetic_defects). This is development evidence, labelled synthetic
    everywhere; it is not, and must never be reported as, real defect recall.

Pre-declared rules (fixed BEFORE any candidate was scored, never changed afterwards)
------------------------------------------------------------------------------------
  Eligibility   the candidate has a validation-selected threshold (some threshold option <= 10% FPR).
  Ranking       pooled synthetic recall at that locked threshold, descending
                (5 defect types x 3 severities x 56 images = 840 synthetic images).
  Ties          recalls within 0.02 are tied -> highest synthetic AUROC; AUROCs within 0.005 are tied ->
                cheaper candidate (smaller input side, smaller model state, candidate id).
  Escalation    if the best Stage-A pooled synthetic recall is < 0.90, a stronger backbone (Stage B) is
                added to the pool before anything is frozen. (Stage B only exists if triggered.)
  Freeze        the winner is fitted on the 224 training images, its threshold re-derived from ITS
                validation scores, and everything is written to a frozen artifact + digest. Only then is
                the final test evaluated (app.ai.evaluation.anomaly_final_test), exactly once.

This module does not import `discover_test_samples`; a test asserts that from the source.
"""

import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from app.ai.evaluation.phase3_threshold_selection import (
    MAX_VALIDATION_FALSE_POSITIVE_RATE,
    STATUS_SELECTED,
    select_threshold_from_validation,
)
from app.ai.evaluation.synthetic_defects import DEFECT_TYPES, SEVERITIES, SEVERITY_NAMES, make_synthetic_defect
from app.ai.evaluation.threshold_experiments import generate_candidates
from app.ai.models.patch_anomaly import (
    AGG_MAX,
    AGG_TOP1PCT,
    SCORER_GAUSSIAN,
    SCORER_KNN,
    GaussianPatchScorer,
    KNNPatchScorer,
    PatchAnomalyDetector,
    aggregate_scores,
    extract_patch_features,
)
from app.ai.models.resnet18 import ResNet18
from app.ai.preprocessing.pipeline import _build_preprocessing_result, _load_image
from app.ai.training.artifacts import load_model
from app.ai.training.model import build_model
from app.ai.training.schemas import DatasetSample
from app.ai.training.validation_split import TrainValidationSplit

FAMILY_CONVAE = "convae"
FAMILY_KNN = "patch_knn"
FAMILY_GAUSSIAN = "patch_gaussian"
CONVAE_GLOBAL = "global_mean"
CONVAE_PATCHMAX = "patch_max"

SYNTHETIC_SEED = 20240607
CORESET_FRACTION = 0.10
CORESET_SEED = 42
ESCALATION_RECALL = 0.90
RECALL_TIE = 0.02
AUROC_TIE = 0.005
STAGE_A = "stage_a"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    image_size: int
    aggregation: str
    layers: tuple[int, ...] | None = None
    coreset_fraction: float | None = None
    convae_artifact: str | None = None  # model_name under ai_models/<category>/ for ConvAE candidates
    stage: str = STAGE_A
    notes: str = ""


def stage_a_candidates() -> list[CandidateSpec]:
    """The fixed, documented Stage-A pool (16 candidates)."""
    specs = [
        CandidateSpec("convae128_global", FAMILY_CONVAE, 128, CONVAE_GLOBAL,
                      convae_artifact="phase3_validation/autoencoder",
                      notes="The failed Phase 3 baseline, unchanged (whole-image mean squared reconstruction error)."),
        CandidateSpec("convae128_patchmax", FAMILY_CONVAE, 128, CONVAE_PATCHMAX,
                      convae_artifact="phase3_validation/autoencoder",
                      notes="Same model; localized scoring: max of the local-window mean error map."),
        CandidateSpec("convae256_global", FAMILY_CONVAE, 256, CONVAE_GLOBAL,
                      convae_artifact="model_selection/convae_input256/autoencoder",
                      notes="ConvAE retrained at 256x256 (train images only) to test resolution alone."),
        CandidateSpec("convae256_patchmax", FAMILY_CONVAE, 256, CONVAE_PATCHMAX,
                      convae_artifact="model_selection/convae_input256/autoencoder"),
    ]
    for side, layers in ((224, (2, 3)), (256, (2, 3)), (320, (2, 3)), (256, (2,))):
        tag = "l" + "".join(str(x) for x in layers)
        for agg in (AGG_MAX, AGG_TOP1PCT):
            specs.append(CandidateSpec(f"pknn_{tag}_{side}_{agg}", FAMILY_KNN, side, agg, layers, CORESET_FRACTION,
                                       notes="ResNet-18 (ImageNet, frozen) patch features + nearest-neighbour memory bank."))
    for side in (256, 320):
        for agg in (AGG_MAX, AGG_TOP1PCT):
            specs.append(CandidateSpec(f"pgauss_l23_{side}_{agg}", FAMILY_GAUSSIAN, side, agg, (2, 3),
                                       notes="ResNet-18 patch features + one global Gaussian (Mahalanobis)."))
    return specs


# ---------------------------------------------------------------------------
# Development data: training / validation images + synthetic defects, at every needed resolution
# ---------------------------------------------------------------------------

@dataclass
class DevArrays:
    sides: tuple[int, ...]
    train_files: list[str]
    validation_files: list[str]
    train: dict[int, np.ndarray] = field(default_factory=dict)  # side -> uint8 [N,S,S,3] RGB
    validation: dict[int, np.ndarray] = field(default_factory=dict)
    synthetic: dict[int, np.ndarray] = field(default_factory=dict)
    synthetic_meta: list[dict] = field(default_factory=list)
    synthetic_digest: str = ""
    build_seconds: float = 0.0


def _resize_all(path: Path, image_bgr: np.ndarray, sides: tuple[int, ...]) -> dict[int, np.ndarray]:
    """The project's existing preprocessing (BGR->RGB, INTER_AREA resize) at each side; uint8 RGB result."""
    return {s: _build_preprocessing_result(path, image_bgr, (s, s)).resized_image for s in sides}


def build_dev_arrays(split: TrainValidationSplit, sides: tuple[int, ...], seed: int = SYNTHETIC_SEED) -> DevArrays:
    """Decode each training/validation image ONCE and resize it to every side; paint the synthetic defects on
    the full-resolution VALIDATION images before resizing. Uses no test image, label or statistic."""
    start = time.perf_counter()
    data = DevArrays(
        sides=tuple(sides),
        train_files=[s.path.name for s in split.training],
        validation_files=[s.path.name for s in split.validation],
    )
    train_lists: dict[int, list] = {s: [] for s in sides}
    for sample in split.training:
        for side, arr in _resize_all(sample.path, _load_image(sample.path), sides).items():
            train_lists[side].append(arr)

    val_lists: dict[int, list] = {s: [] for s in sides}
    syn_lists: dict[int, list] = {s: [] for s in sides}
    digest = hashlib.sha256()
    for index, sample in enumerate(split.validation):
        image = _load_image(sample.path)
        for side, arr in _resize_all(sample.path, image, sides).items():
            val_lists[side].append(arr)
        for defect_type in DEFECT_TYPES:
            for severity in SEVERITIES:
                painted, _ = make_synthetic_defect(image, defect_type, severity, seed=seed + index)
                small = _resize_all(sample.path, painted, sides)
                for side, arr in small.items():
                    syn_lists[side].append(arr)
                digest.update(painted.tobytes())  # resolution-independent fingerprint of the painted image
                data.synthetic_meta.append(
                    {"source_file": sample.path.name, "defect_type": defect_type, "severity": severity}
                )
    data.train = {s: np.stack(v) for s, v in train_lists.items()}
    data.validation = {s: np.stack(v) for s, v in val_lists.items()}
    data.synthetic = {s: np.stack(v) for s, v in syn_lists.items()}
    data.synthetic_digest = digest.hexdigest()
    data.build_seconds = time.perf_counter() - start
    return data


def to_tensor(images_uint8: np.ndarray) -> torch.Tensor:
    """uint8 NHWC RGB -> float32 NCHW in [0,1] (identical values to the pipeline's normalized_image)."""
    return torch.from_numpy(images_uint8.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def convae_scores(model, images_uint8: np.ndarray, aggregation: str, batch_size: int = 16) -> list[float]:
    """Image scores from the (unchanged) ConvAutoencoder's per-pixel squared-error map."""
    model.eval()
    out: list[float] = []
    for start in range(0, len(images_uint8), batch_size):
        x = to_tensor(images_uint8[start : start + batch_size])
        error_map = ((model(x) - x) ** 2).mean(dim=1)  # N,H,W
        if aggregation == CONVAE_GLOBAL:
            scores = error_map.flatten(1).mean(dim=1)
        elif aggregation == CONVAE_PATCHMAX:
            k = max(3, x.shape[-1] // 16)
            scores = F.avg_pool2d(error_map[:, None], kernel_size=k, stride=1).flatten(1).max(dim=1).values
        else:
            raise ValueError(aggregation)
        out.extend(float(s) for s in scores)
    return out


@dataclass
class Timing:
    fit_seconds: float = 0.0
    feature_ms_per_image: float = 0.0
    score_ms_per_image: float = 0.0


@dataclass
class DevResult:
    spec: CandidateSpec
    validation_scores: list[float]
    synthetic_scores: list[float]
    validation_stats: dict
    threshold_candidates: list[dict]
    selection_status: str
    selected_threshold: dict | None
    validation_fpr_at_selected: float | None
    synthetic_recall_pooled: float | None
    synthetic_recall_by_type: dict | None
    synthetic_recall_by_severity: dict | None
    synthetic_recall_by_type_severity: dict | None
    synthetic_auroc: float
    state_bytes: int
    timing: Timing


def dev_metrics(spec: CandidateSpec, val_scores: list[float], syn_scores: list[float], syn_meta: list[dict],
                state_bytes: int, timing: Timing) -> DevResult:
    """Threshold from VALIDATION GOOD scores only (existing Phase 3 rule); synthetic metrics at that threshold."""
    candidates = generate_candidates(val_scores)
    selection = select_threshold_from_validation(candidates, val_scores)
    mean = statistics.fmean(val_scores)
    std = (sum((s - mean) ** 2 for s in val_scores) / len(val_scores)) ** 0.5

    y = np.array([0] * len(val_scores) + [1] * len(syn_scores))
    auroc = float(roc_auc_score(y, np.array(val_scores + syn_scores)))

    recall = by_type = by_sev = by_cell = fpr = threshold = None
    if selection.status == STATUS_SELECTED:
        chosen = selection.selected
        threshold = {"method": chosen.method, "parameter": chosen.parameter, "threshold": chosen.threshold}
        flags = np.array([s > chosen.threshold for s in syn_scores])
        recall = float(flags.mean())
        by_type = {t: float(np.mean([f for f, m in zip(flags, syn_meta) if m["defect_type"] == t])) for t in DEFECT_TYPES}
        by_sev = {SEVERITY_NAMES[v]: float(np.mean([f for f, m in zip(flags, syn_meta) if m["severity"] == v]))
                  for v in SEVERITIES}
        by_cell = {f"{t}/{SEVERITY_NAMES[v]}": float(np.mean([f for f, m in zip(flags, syn_meta)
                                                              if m["defect_type"] == t and m["severity"] == v]))
                   for t in DEFECT_TYPES for v in SEVERITIES}
        fpr = float(np.mean([s > chosen.threshold for s in val_scores]))

    stability = {(s.candidate.method, s.candidate.parameter): s for s in selection.stability_by_candidate}
    return DevResult(
        spec=spec,
        validation_scores=list(val_scores),
        synthetic_scores=list(syn_scores),
        validation_stats={"count": len(val_scores), "mean": mean, "std": std, "min": min(val_scores),
                          "max": max(val_scores), "cov": std / mean if mean else float("inf")},
        threshold_candidates=[
            {"method": c.method, "parameter": c.parameter, "threshold": c.threshold,
             "validation_fp_count": stability[(c.method, c.parameter)].validation_false_positive_count,
             "validation_fpr": stability[(c.method, c.parameter)].validation_false_positive_rate,
             "passes_ceiling": stability[(c.method, c.parameter)].passes_stability_check}
            for c in candidates
        ],
        selection_status=selection.status,
        selected_threshold=threshold,
        validation_fpr_at_selected=fpr,
        synthetic_recall_pooled=recall,
        synthetic_recall_by_type=by_type,
        synthetic_recall_by_severity=by_sev,
        synthetic_recall_by_type_severity=by_cell,
        synthetic_auroc=auroc,
        state_bytes=state_bytes,
        timing=timing,
    )


def evaluate_convae(spec: CandidateSpec, category: str, data: DevArrays, model_root: Path) -> DevResult:
    from app.ai.training.artifacts import get_model_path

    path = get_model_path(category, spec.convae_artifact)
    model = load_model(path, build_model())
    t0 = time.perf_counter()
    val = convae_scores(model, data.validation[spec.image_size], spec.aggregation)
    syn = convae_scores(model, data.synthetic[spec.image_size], spec.aggregation)
    per_image_ms = (time.perf_counter() - t0) * 1000 / (len(val) + len(syn))
    return dev_metrics(spec, val, syn, data.synthetic_meta, path.stat().st_size,
                       Timing(fit_seconds=0.0, feature_ms_per_image=0.0, score_ms_per_image=per_image_ms))


def evaluate_patch_group(
    specs: list[CandidateSpec], extractor: ResNet18, data: DevArrays, log=print
) -> list[DevResult]:
    """All patch candidates sharing (image_size, layers): one feature pass, every scorer fitted on the TRAINING
    images only and applied to validation + synthetic images."""
    side, layers = specs[0].image_size, specs[0].layers
    wanted = {(s.family, s.aggregation): s for s in specs}
    kinds = {s.family for s in specs}

    t0 = time.perf_counter()
    train_patches = extract_patch_features(extractor, to_tensor(data.train[side]), layers)
    extract_train_s = time.perf_counter() - t0

    scorers, fit_seconds = {}, {}
    if FAMILY_KNN in kinds:
        t = time.perf_counter()
        fraction = next(s.coreset_fraction for s in specs if s.family == FAMILY_KNN) or CORESET_FRACTION
        scorers[FAMILY_KNN] = KNNPatchScorer.fit(train_patches, fraction, CORESET_SEED)  # same as fit_detector
        fit_seconds[FAMILY_KNN] = extract_train_s + time.perf_counter() - t
    if FAMILY_GAUSSIAN in kinds:
        t = time.perf_counter()
        scorers[FAMILY_GAUSSIAN] = GaussianPatchScorer.fit(train_patches)
        fit_seconds[FAMILY_GAUSSIAN] = extract_train_s + time.perf_counter() - t
    del train_patches
    log(f"    fitted {sorted(scorers)} at side {side}, layers {layers}: "
        + ", ".join(f"{k} {v:.0f}s" for k, v in fit_seconds.items()))

    images = {"val": data.validation[side], "syn": data.synthetic[side]}
    scores = {(fam, agg): {"val": [], "syn": []} for (fam, agg) in wanted}
    feature_s = 0.0
    scoring_s = {fam: 0.0 for fam in scorers}
    for split_name, arr in images.items():
        for start in range(0, len(arr), 8):
            t = time.perf_counter()
            patches = extract_patch_features(extractor, to_tensor(arr[start : start + 8]), layers)
            feature_s += time.perf_counter() - t
            for fam, scorer in scorers.items():
                t = time.perf_counter()
                maps = scorer.patch_scores(patches)
                scoring_s[fam] += time.perf_counter() - t
                for (f, agg) in wanted:
                    if f == fam:
                        scores[(f, agg)][split_name].extend(float(x) for x in aggregate_scores(maps, agg))
    n_images = len(images["val"]) + len(images["syn"])

    results = []
    for (fam, agg), spec in wanted.items():
        timing = Timing(fit_seconds=fit_seconds[fam], feature_ms_per_image=feature_s * 1000 / n_images,
                        score_ms_per_image=scoring_s[fam] * 1000 / n_images)
        results.append(dev_metrics(spec, scores[(fam, agg)]["val"], scores[(fam, agg)]["syn"], data.synthetic_meta,
                                   scorers[fam].size_bytes, timing))
    return results


def run_stage(specs: list[CandidateSpec], category: str, extractor: ResNet18, data: DevArrays,
              model_root: Path, log=print) -> list[DevResult]:
    """Evaluate every candidate in `specs` (grouped so feature extraction is shared)."""
    results: dict[str, DevResult] = {}
    for spec in [s for s in specs if s.family == FAMILY_CONVAE]:
        log(f"  {spec.candidate_id}")
        results[spec.candidate_id] = evaluate_convae(spec, category, data, model_root)
    groups: dict[tuple, list[CandidateSpec]] = {}
    for spec in [s for s in specs if s.family != FAMILY_CONVAE]:
        groups.setdefault((spec.image_size, spec.layers), []).append(spec)
    for (side, layers), members in groups.items():
        log(f"  patch group side={side} layers={layers}: {[m.candidate_id for m in members]}")
        for r in evaluate_patch_group(members, extractor, data, log):
            results[r.spec.candidate_id] = r
    return [results[s.candidate_id] for s in specs]


# ---------------------------------------------------------------------------
# Pre-declared selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Selection:
    winner_id: str
    reasoning: str
    ranked: list[tuple[str, float]]
    disqualified: list[str]
    escalation_needed: bool


def select_best(results: list[DevResult]) -> Selection:
    eligible = [r for r in results if r.selection_status == STATUS_SELECTED and r.synthetic_recall_pooled is not None]
    disqualified = [r.spec.candidate_id for r in results if r not in eligible]
    if not eligible:
        raise ValueError("No candidate has a validation-selected threshold under the 10% false-positive ceiling.")
    ranked = sorted(eligible, key=lambda r: (-r.synthetic_recall_pooled, r.spec.candidate_id))
    best_recall = ranked[0].synthetic_recall_pooled
    tied = [r for r in ranked if best_recall - r.synthetic_recall_pooled <= RECALL_TIE]
    best_auroc = max(r.synthetic_auroc for r in tied)
    top = [r for r in tied if best_auroc - r.synthetic_auroc <= AUROC_TIE]
    winner = min(top, key=lambda r: (r.spec.image_size, r.state_bytes, r.spec.candidate_id))
    reasoning = (
        f"Best pooled synthetic recall {best_recall:.3f}; {len(tied)} candidate(s) within {RECALL_TIE} of it; "
        f"{len(top)} within {AUROC_TIE} of the best synthetic AUROC ({best_auroc:.4f}) among those; cheapest "
        f"(smallest input side, then smallest model state) -> '{winner.spec.candidate_id}'. Synthetic evidence "
        "ranks candidates; it does not measure real defect recall."
    )
    return Selection(
        winner_id=winner.spec.candidate_id,
        reasoning=reasoning,
        ranked=[(r.spec.candidate_id, r.synthetic_recall_pooled) for r in ranked],
        disqualified=disqualified,
        escalation_needed=best_recall < ESCALATION_RECALL,
    )


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

def _md5_sha256(path: Path) -> tuple[str, str, int]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest(), path.stat().st_size


def fit_detector(spec: CandidateSpec, extractor: ResNet18, data: DevArrays) -> tuple[PatchAnomalyDetector, float]:
    """Fit ONE patch candidate on the TRAINING images only. Returns (detector, fit seconds)."""
    if spec.family not in (FAMILY_KNN, FAMILY_GAUSSIAN):
        raise ValueError("Only patch-based candidates can be frozen by fit_detector.")
    start = time.perf_counter()
    patches = extract_patch_features(extractor, to_tensor(data.train[spec.image_size]), spec.layers)
    if spec.family == FAMILY_KNN:
        scorer = KNNPatchScorer.fit(patches, spec.coreset_fraction or CORESET_FRACTION, CORESET_SEED)
    else:
        scorer = GaussianPatchScorer.fit(patches)
    detector = PatchAnomalyDetector(extractor, tuple(spec.layers), spec.image_size, scorer, spec.aggregation)
    return detector, time.perf_counter() - start


def score_uint8(detector: PatchAnomalyDetector, images_uint8: np.ndarray, batch_size: int = 8) -> list[float]:
    out: list[float] = []
    for start in range(0, len(images_uint8), batch_size):
        out.extend(detector.score_images(to_tensor(images_uint8[start : start + batch_size])))
    return out


@dataclass
class FrozenModel:
    directory: Path
    spec: CandidateSpec
    threshold_method: str
    threshold_parameter: float
    threshold: float
    validation_scores: list[float]
    state_md5: str
    state_sha256: str
    state_bytes: int
    fit_seconds: float
    digest: str


def freeze_detector(spec: CandidateSpec, extractor: ResNet18, data: DevArrays, directory: Path,
                    backbone_sha256: str, dev_result: DevResult) -> FrozenModel:
    """Fit on the training images, derive the threshold from this detector's OWN validation scores, save.

    Refuses to overwrite. Writes model_state.pt, model_config.json and frozen_model.json (the lock record).
    """
    directory = Path(directory)
    if (directory / "frozen_model.json").exists():
        raise FileExistsError(f"A frozen model already exists in {directory}; refusing to overwrite.")
    detector, fit_seconds = fit_detector(spec, extractor, data)
    val_scores = score_uint8(detector, data.validation[spec.image_size])
    selection = select_threshold_from_validation(generate_candidates(val_scores), val_scores)
    if selection.selected is None:
        raise ValueError("The frozen candidate has no threshold under the validation false-positive ceiling.")
    chosen = selection.selected

    state_path = detector.save(directory)
    md5, sha, size = _md5_sha256(state_path)
    digest = hashlib.sha256(json.dumps(
        {"config": detector.config(), "state_sha256": sha, "threshold": chosen.threshold,
         "backbone_sha256": backbone_sha256, "synthetic_digest": data.synthetic_digest}, sort_keys=True
    ).encode()).hexdigest()
    lock = {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "config": detector.config(),
        "backbone": {"name": "resnet18_imagenet_frozen", "sha256": backbone_sha256},
        "coreset_fraction": spec.coreset_fraction,
        "coreset_seed": CORESET_SEED,
        "fitted_on": {"training_images": len(data.train_files), "file_list_sha256":
                      hashlib.sha256("\n".join(data.train_files).encode()).hexdigest()},
        "threshold": {"method": chosen.method, "parameter": chosen.parameter, "value": chosen.threshold,
                      "rule": "lowest threshold candidate with validation FPR <= 10% (existing Phase 3 rule)",
                      "validation_fp_ceiling": MAX_VALIDATION_FALSE_POSITIVE_RATE,
                      "validation_images": len(val_scores)},
        "state_file": {"name": state_path.name, "md5": md5, "sha256": sha, "size_bytes": size},
        "development_evidence_synthetic": {
            "pooled_recall": dev_result.synthetic_recall_pooled, "auroc": dev_result.synthetic_auroc,
            "note": "SYNTHETIC defects on validation good images - not real MVTec performance."},
        "digest": digest,
    }
    (directory / "frozen_model.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return FrozenModel(directory, spec, chosen.method, chosen.parameter, chosen.threshold, val_scores, md5, sha,
                       size, fit_seconds, digest)


__all__ = [
    "AGG_MAX", "AGG_TOP1PCT", "SCORER_GAUSSIAN", "SCORER_KNN",
    "CandidateSpec", "DevArrays", "DevResult", "FrozenModel", "Selection", "Timing",
    "build_dev_arrays", "convae_scores", "dev_metrics", "evaluate_convae", "evaluate_patch_group",
    "fit_detector", "freeze_detector", "run_stage", "score_uint8", "select_best", "stage_a_candidates", "to_tensor",
]
