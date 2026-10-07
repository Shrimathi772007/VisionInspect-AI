"""Category-specific AI serving configuration and cached model loading.

    category -> CategoryServingConfig (model artifact + locked threshold + reliability metadata)
             -> cached, hash-verified model (shared frozen backbone + per-category memory bank)

Answers, in O(1) and without any dataset access: which locked model artifact serves a category, which
locked threshold its predictions are compared against, and how reliable that model measured on its single
final test. Until this module existed, inference always loaded ai_models/<category>/autoencoder.pt and
recomputed a K=3 threshold from every train/good image on every call.

Policy: every one of the 15 MVTec categories has exactly one registered final model - the model each
category's locked study selected and scored once on its final test. Registration is NOT a quality claim:
the `gate` field (EXCELLENT / GOOD / ACCEPTABLE / NOT_PRODUCTION_READY, the project's evidence gates applied
to that final test) says how far a category's predictions can be relied on, and the final-test recall, FPR,
AUROC and average precision are carried alongside it. A category with no entry has no AI support -
predict_image raises ModelArtifactNotFoundError for it (which app.inspections.service already turns into
"ai_* stay NULL"). There is deliberately no default entry and no fallback to another category's model, or
from a configured artifact to a different one.

Three model families are served:
  - convae          ConvAutoencoder, per-image reconstruction MSE (no category uses it any more; Bottle's
                    Phase 3 ConvAE was replaced by its WRN-50 PatchCore model - the file stays on disk, and
                    old inspection rows keep their stored ai_model_name "autoencoder")
  - patch_anomaly   frozen ImageNet ResNet-18 PatchAnomalyDetector (model-family study winners)
  - patchcore_wrn50 frozen ImageNet WideResNet-50-2 PatchCoreDetector (WRN-50 study; Grid uses the
                    Addendum 1 full320 model and its own loader)
The family, input size/mode, aggregation and artifact hashes are part of each category's configuration, so
a category can only be served by its own locked model.

Memory: one frozen ResNet-18 and one frozen WRN-50-2 are loaded at most once per process and shared by every
category of that family; only the per-category memory banks are cached, at most `model_cache_limit()` of
them (default 3, see app.ai.config.model_cache_size), least recently used evicted first.

This module never trains, evaluates, tunes or selects anything: thresholds are copied in from
already-locked reports, not derived here.
"""

import hashlib
import importlib.util
import json
import math
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from torch import nn

from app.ai import config as ai_config
from app.ai.evaluation.evaluate import DEFAULT_IMAGE_SIZE
from app.ai.inference.errors import ModelArtifactNotFoundError, ModelIntegrityError
from app.ai.models.patch_anomaly import PatchAnomalyDetector
from app.ai.models.patchcore import CONFIG_FILENAME, PatchCoreDetector, PatchCoreModelError
from app.ai.models.resnet18 import RESNET18_SHA256, load_pretrained_resnet18
from app.ai.models.wide_resnet50 import WRN50_2_SHA256, load_pretrained_wide_resnet50_2
from app.ai.training.artifacts import get_model_path, load_model_for_category

# How a category's artifact is loaded and scored.
MODEL_FAMILY_CONVAE = "convae"  # ConvAutoencoder state_dict -> per-image reconstruction MSE
MODEL_FAMILY_PATCH_ANOMALY = "patch_anomaly"  # frozen ResNet-18 patch features -> PatchAnomalyDetector score
MODEL_FAMILY_PATCHCORE_WRN50 = "patchcore_wrn50"  # frozen WRN-50-2 patch features -> PatchCoreDetector score
MODEL_FAMILIES = (MODEL_FAMILY_CONVAE, MODEL_FAMILY_PATCH_ANOMALY, MODEL_FAMILY_PATCHCORE_WRN50)

# WRN-50 input modes. crop224/full256 are app.ai.preprocessing.patchcore_preprocess modes; full320 exists only
# in the Grid addendum (PatchCoreDetector.load rejects it, so Grid is loaded with the addendum's own loader).
INPUT_MODE_CROP224 = "crop224"
INPUT_MODE_FULL256 = "full256"
INPUT_MODE_FULL320 = "full320"
PATCHCORE_INPUT_SIZES = {INPUT_MODE_CROP224: 224, INPUT_MODE_FULL256: 256, INPUT_MODE_FULL320: 320}

# Evidence gates (app.ai.evaluation.category_phase1.gate_classification), spelled as API values.
GATE_EXCELLENT = "EXCELLENT"
GATE_GOOD = "GOOD"
GATE_ACCEPTABLE = "ACCEPTABLE"
GATE_NOT_PRODUCTION_READY = "NOT_PRODUCTION_READY"
GATES = (GATE_EXCELLENT, GATE_GOOD, GATE_ACCEPTABLE, GATE_NOT_PRODUCTION_READY)

# The Grid Addendum 1 code that defines full320 (Full320Config, load_full320_detector, resize_full320). It is a
# protected study file, so it is imported read-only from its locked location and only when its SHA-256 (and that
# of the study logic it imports) equals the value recorded in ai_models/grid/patchcore_wrn50_full320/lock.json
# (addendum_logic_sha256 / study_logic_sha256).
PATCHCORE_STUDY_DIR = Path(__file__).resolve().parents[3] / "scripts" / "experiments" / "patchcore_wrn50"
WRN50_STUDY_LOGIC_SHA256 = "47fa5ae6543bd216c561b2c81291aa8fefb18e7aee953484342c5c1ea1bc277d"
GRID_FULL320_LOGIC_SHA256 = "f4645bfaebcfe43ac8c77b302e87f160e3e74310ca027188a3b0bc02bdceaaf2"

# Image-level average precision of the six ResNet-18 categories, whose model-family-study final-test reports store
# none. Computed post-hoc from the per-image scores those reports saved (no re-scoring, no selection use; see the
# file's note, method and per-category source SHA-256). Display metadata only: no model, threshold or decision
# reads it. Loaded once at import; a missing or unreadable file, or a missing category, leaves the AP None.
AP_ADDENDUM_PATH = Path(__file__).resolve().parents[1] / "ap_addendum.json"


def load_ap_addendum(path: Path = AP_ADDENDUM_PATH) -> dict[str, float]:
    """category -> AP from the addendum file; {} when the file is missing or unreadable. An entry without a finite
    average_precision in [0, 1] is skipped."""
    try:
        categories = json.loads(Path(path).read_text(encoding="utf-8"))["categories"]
        items = categories.items()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}
    addendum = {}
    for category, entry in items:
        value = entry.get("average_precision") if isinstance(entry, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
            addendum[category] = float(value)
    return addendum


_AP_ADDENDUM = load_ap_addendum()


@dataclass(frozen=True)
class CategoryServingConfig:
    """Everything inference needs to serve one category, and where each value came from."""

    category: str
    # Artifact name relative to ai_models/<category>/, without ".pt" - the same convention
    # as app.ai.training.artifacts.get_model_path's `model_name` (e.g. "phase3_validation/autoencoder").
    artifact_name: str
    # Value reported as PredictionResult.model_name and stored in Inspection.ai_model_name.
    # "autoencoder" for ConvAE entries (the existing database contract); a patch-anomaly entry
    # uses its locked candidate id (e.g. "knn_l23_256"); a WRN-50 entry "wrn50_patchcore_<mode>".
    # Which artifact actually served a prediction is identified by `artifact_name`, not by this label.
    model_name: str
    # Locked decision threshold: anomaly score (reconstruction error for ConvAE) <= threshold -> "good".
    threshold: float
    threshold_method: str
    threshold_parameter: float
    # MD5 of the expected artifact. When set, the artifact is verified on first load so a
    # replaced or retrained file is refused instead of silently served. None skips the check.
    expected_md5: str | None
    # Human-readable pointer to the authoritative record of the threshold and model hash.
    provenance: str
    # --- Fields below default to the original ConvAE behaviour, so existing entries are unchanged. ---
    model_family: str = MODEL_FAMILY_CONVAE
    # (width, height) the image is resized to - part of the validated model, not a caller choice
    # for patch-anomaly / WRN-50 models (see app.ai.inference.predict.predict_image).
    input_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    # Patch-anomaly / WRN-50: SHA-256 of the model state and of its model_config.json (layers, input size,
    # scorer, aggregation). Both are verified on first load, like expected_md5.
    expected_sha256: str | None = None
    expected_config_sha256: str | None = None
    # Identifier of the locked experiment the model and threshold come from, when there is one.
    experiment_id: str | None = None
    # WRN-50 only: preprocessing mode (crop224 / full256 / full320). None for the other families, which use the
    # project's existing resize-to-input_size pipeline.
    input_mode: str | None = None
    # Patch models: image score aggregation of the locked model ("max", "top1pct_mean"), checked against the
    # loaded model's own configuration. None skips the check (ConvAE).
    aggregation: str | None = None
    # Pinned SHA-256 of the frozen ImageNet backbone weights the model was built on (patch / WRN-50 models).
    expected_backbone_sha256: str | None = None
    # lock_digest of the lock the model, threshold and hashes are copied from.
    lock_digest: str | None = None
    # Reliability metadata (static, copied from the category's single final-test report): evidence gate and
    # final-test recall, false-positive rate, AUROC and average precision (None where the report has none).
    gate: str | None = None
    final_test_recall: float | None = None
    final_test_fpr: float | None = None
    final_test_auroc: float | None = None
    final_test_average_precision: float | None = None


def _resnet18_entry(category: str, model_name: str, threshold: float, threshold_method: str, threshold_parameter: float,
                    md5: str, sha256: str, config_sha256: str, input_side: int, experiment_id: str, lock_digest: str,
                    gate: str, recall: float, fpr: float, auroc: float) -> CategoryServingConfig:
    """A ResNet-18 model-family-study winner (selected_candidate/model_state.pt + model_config.json). The study
    reports no average precision; it comes from the post-hoc AP addendum (None when absent)."""
    return CategoryServingConfig(
        category=category,
        artifact_name="model_family_study/selected_candidate/model_state",
        model_name=model_name,
        threshold=threshold,
        threshold_method=threshold_method,
        threshold_parameter=threshold_parameter,
        expected_md5=md5,
        provenance=(
            f"backend/ai_models/{category}/model_family_study/selection_lock.json (selected_threshold, selected_artifact; "
            f"lock_digest {lock_digest}) and reports/final_test_result.json"
        ),
        model_family=MODEL_FAMILY_PATCH_ANOMALY,
        input_size=(input_side, input_side),
        expected_sha256=sha256,
        expected_config_sha256=config_sha256,
        experiment_id=experiment_id,
        aggregation="top1pct_mean",
        expected_backbone_sha256=RESNET18_SHA256,
        lock_digest=lock_digest,
        gate=gate,
        final_test_recall=recall,
        final_test_fpr=fpr,
        final_test_auroc=auroc,
        final_test_average_precision=_AP_ADDENDUM.get(category),
    )


def _wrn50_entry(category: str, mode: str, threshold: float, threshold_method: str, threshold_parameter: float,
                 sha256: str, config_sha256: str, lock_digest: str, gate: str, recall: float, fpr: float, auroc: float,
                 average_precision: float) -> CategoryServingConfig:
    """A WRN-50 PatchCore final model (<study dir>/final_model/model_state.pt + model_config.json). Every WRN-50
    lock uses aggregation "max" and the pinned WRN-50-2 ImageNet weights."""
    directory = "patchcore_wrn50_full320" if mode == INPUT_MODE_FULL320 else "patchcore_wrn50"
    side = PATCHCORE_INPUT_SIZES[mode]
    return CategoryServingConfig(
        category=category,
        artifact_name=f"{directory}/final_model/model_state",
        model_name=f"wrn50_patchcore_{mode}",
        threshold=threshold,
        threshold_method=threshold_method,
        threshold_parameter=threshold_parameter,
        expected_md5=None,  # the WRN-50 locks pin SHA-256 only
        provenance=(
            f"backend/ai_models/{category}/{directory}/lock.json (threshold, mode, aggregation, model_state_sha256, "
            f"model_config_sha256, backbone_weights_sha256; lock_digest {lock_digest}) and final_test/final_test_result.json"
        ),
        model_family=MODEL_FAMILY_PATCHCORE_WRN50,
        input_size=(side, side),
        expected_sha256=sha256,
        expected_config_sha256=config_sha256,
        input_mode=mode,
        aggregation="max",
        expected_backbone_sha256=WRN50_2_SHA256,
        lock_digest=lock_digest,
        gate=gate,
        final_test_recall=recall,
        final_test_fpr=fpr,
        final_test_auroc=auroc,
        final_test_average_precision=average_precision,
    )


SERVING_CONFIGS: dict[str, CategoryServingConfig] = {
    # ------------------------------------------------------------------ WRN-50-2 PatchCore (9 categories)
    # Every value is copied verbatim from backend/ai_models/<category>/patchcore_wrn50/lock.json (Grid:
    # patchcore_wrn50_full320/lock.json, the Addendum 1 lock): threshold, mode, aggregation ("max" for all nine),
    # model_state_sha256, model_config_sha256 and backbone_weights_sha256; policy -> threshold_method/parameter.
    # Thresholds were selected from normal (train/good) data only, before each category's final test was scored
    # once (scripts/experiments/patchcore_wrn50/FINAL_TEST_DECLARATION.md). Gate and metrics: final_test/
    # final_test_result.json (gate_classification, metrics.recall, metrics.false_positive_rate, auroc,
    # average_precision).
    #
    # Bottle: replaces the Milestone 4 Phase 3 ConvAE (phase3_validation/autoencoder.pt, MD5 76478dd6..., threshold
    # 0.0028031117030001018; final test recall 73.02%) - by protocol a replacement regardless of the result. The
    # ConvAE file stays untouched on disk and is no longer served.
    "bottle": _wrn50_entry(
        "bottle", INPUT_MODE_CROP224, 1.6858729828595898, "mean_std", 2.5,
        sha256="0c7bd7f45c4769980b7c4e6cb5d6d8762cef35a9088eed045d03bebda8ecb7cb",
        config_sha256="9d7261a3116427c5a63fcc890cc3f92185865dedc6e15dac996407c8b4a0030e",
        lock_digest="830aa484045ed1b900b4435937091ee0c4d100112e201b6a08ba841e440d1550",
        gate=GATE_EXCELLENT, recall=1.0, fpr=0.0, auroc=1.0, average_precision=0.9999999999999998,
    ),
    "capsule": _wrn50_entry(
        "capsule", INPUT_MODE_CROP224, 1.55368220243819, "mean_std", 2.5,
        sha256="902e411a684d1cf9e07eba73e7daf9e7ef9917e6762e1ceb0202b03b87e3f5ea",
        config_sha256="d94aca2dcfa9e9c8fe53b441f0f10fe667166eae5a0f9686dec99962e27fff7d",
        lock_digest="7821104664f380b69342ca579d18007badf4a4eabad9a49e19fbe95a5e9cf22f",
        gate=GATE_ACCEPTABLE, recall=0.7889908256880734, fpr=0.043478260869565216, auroc=0.9780614280015956,
        average_precision=0.9951537644854546,
    ),
    "zipper": _wrn50_entry(
        "zipper", INPUT_MODE_FULL256, 1.416393740819701, "mean_std", 2.5,
        sha256="f3bde4e4b6c9469560ba0af43e5987fd12e828de4c4edcf2bdc87b7fcf58fdbd",
        config_sha256="c11c20223d019f0a086f06e734974982917c844634ff178cec1a52403ca36e78",
        lock_digest="2a72a8fdf8163b0772165dd17fd60e040d2652ad4ee4518bcfe3ca2d7c53a9ab",
        gate=GATE_ACCEPTABLE, recall=0.9915966386554622, fpr=0.125, auroc=0.989233193277311,
        average_precision=0.9969916300662345,
    ),
    "wood": _wrn50_entry(
        "wood", INPUT_MODE_FULL256, 1.8853798671039956, "mean_std", 2.5,
        sha256="14c6af91d6c1dc091b113ff3167080b51a2de66856cb0514e82d8c39810d159a",
        config_sha256="6ec79266db2dc601568c8ab5e7d48ce192e65eedc065923bdf20fc9d74fe842c",
        lock_digest="49b048415e53e2b324fd503e9062ffdfa3ca38c0953c16dd071c21e28f63c842",
        gate=GATE_NOT_PRODUCTION_READY, recall=0.9833333333333333, fpr=0.3684210526315789, auroc=0.986842105263158,
        average_precision=0.9961573954852546,
    ),
    # Grid: the Addendum 1 full320 model (whole image resized to 320x320). The original crop224/full256 Grid lock
    # (patchcore_wrn50/lock.json) was superseded before any test scoring and is never served.
    "grid": _wrn50_entry(
        "grid", INPUT_MODE_FULL320, 1.88393018105021, "mean_std", 2.5,
        sha256="473e486a889c4a01ca18cc1ef0761eddc4555559d6fd4e7bd0c25ed09a84460c",
        config_sha256="10ce21f45edce594038e30ca919088a76e14197247e859ab6dbf1bbd543c66b1",
        lock_digest="4cdf78b2b2b6563bcb8b96693e8bf32b5fa99040444c4599a0c284efc5919efc",
        gate=GATE_EXCELLENT, recall=0.9824561403508771, fpr=0.047619047619047616, auroc=0.9908103592314119,
        average_precision=0.9970737264738176,
    ),
    "pill": _wrn50_entry(
        "pill", INPUT_MODE_CROP224, 2.071415785373315, "mean_std", 2.5,
        sha256="b3ccd3b3879af14344a059e85b931b4b7694a06c0ac29dd41b21b231bb43d111",
        config_sha256="0cec430e1f45616e3fb685d928e85835a13d9b4804e8debe9ef05b4e3906033c",
        lock_digest="072271786f13f887f8815d6f38b488d778d727b839b38d26fcea0efd9f463f18",
        gate=GATE_NOT_PRODUCTION_READY, recall=0.6595744680851063, fpr=0.038461538461538464, auroc=0.947899618112384,
        average_precision=0.9899105320054837,
    ),
    "carpet": _wrn50_entry(
        "carpet", INPUT_MODE_CROP224, 1.559494003088166, "mean_std", 2.5,
        sha256="90293b8de7051746abb44ef6cef9682fc238bda7097076a37d75d60036702b0b",
        config_sha256="82b98830ccfc50d58d345e300c448b61d570cda125cb6391c71bcf600225963d",
        lock_digest="2527ed6ac34680c51d2f3298f9c5c6697f9a16e160fccd7d7efc530d1e085127",
        gate=GATE_NOT_PRODUCTION_READY, recall=0.9887640449438202, fpr=0.32142857142857145, auroc=0.9891653290529696,
        average_precision=0.9967858790094236,
    ),
    "screw": _wrn50_entry(
        "screw", INPUT_MODE_FULL256, 1.692118835724279, "mean_std", 2.5,
        sha256="8a63f0d55332c2f159e1149d098ecb02dd46924e87628e0b5b2db945235aa631",
        config_sha256="7791215b06a1e1ca1ae1416b1072abe8d1182bbee00759ff99e2d7dabec75c2d",
        lock_digest="0857280ca50e702acef1768040f04e192daab07175f1dacca3efbdfb3d4fb07b",
        gate=GATE_NOT_PRODUCTION_READY, recall=0.7058823529411765, fpr=0.0, auroc=0.9508095921295348,
        average_precision=0.9834617248701818,
    ),
    "hazelnut": _wrn50_entry(
        "hazelnut", INPUT_MODE_CROP224, 2.5406732082366945, "percentile", 99.0,
        sha256="659efe018a1a8d3d0d5c77a533bc520dea0a079c79e94b29cafdecb96e4f94b3",
        config_sha256="d9e91e02515344498e0e3b128bd34131b2421df1bf7ebaedfb39c77b63fb2281",
        lock_digest="894cc81a91859f45e5b6f0c8b73e9100adfc75c099ac717dabfc19f47814cfc1",
        gate=GATE_EXCELLENT, recall=1.0, fpr=0.0, auroc=1.0, average_precision=0.9999999999999998,
    ),
    # ------------------------------------------------------------------ ResNet-18 patch models (6 categories)
    # Tile: the locked winner of the normal-only alternative model-family study (the Phase 1
    # ConvAE, ai_models/tile/phase1_baseline/, failed with AUROC 0.481 and is NOT served).
    # Every value is copied verbatim from backend/ai_models/tile/model_family_study/selection_lock.json:
    #   threshold        <- selected_threshold (policy mean_std_3: mean + 3*std of the leave-one-image-out
    #                       scores of all 230 train/good images; selected from normal data only)
    #   expected_md5/sha <- selected_artifact.md5 / .sha256
    # Model: frozen ImageNet ResNet-18, layers 2+3 (384-d, 3x3 avg pooling), 256x256 input, k=1
    # Euclidean nearest neighbour to a 10% per-image coreset bank, image score = mean of the top 1%
    # of patch scores. Final test (scored once): TN 33 FP 0 FN 7 TP 77, recall 91.67%, F1 95.65%,
    # FPR 0%, AUROC 0.9971 (EXCELLENT); gray_stroke recall is 62.5% (6 of the 7 misses).
    "tile": CategoryServingConfig(
        category="tile",
        artifact_name="model_family_study/selected_candidate/model_state",
        model_name="knn_l23_256",
        threshold=1.7393077017650718,
        threshold_method="mean_std",
        threshold_parameter=3.0,
        expected_md5="cb9530d76709319ec0eed58f55717356",
        provenance=(
            "backend/ai_models/tile/model_family_study/selection_lock.json (selected_threshold, "
            "selected_artifact; lock_digest d3302166d3b7088302286fdc46243f8316d39e848677ce045cd22597e51d472b) "
            "and reports/final_test_result.json"
        ),
        model_family=MODEL_FAMILY_PATCH_ANOMALY,
        input_size=(256, 256),
        expected_sha256="de00e2774daf02d97e1414fe75b10305e2ad392696449b1627a250b1ec0dd4d7",
        expected_config_sha256="6ab1c5e603cf2b63cb9c9f0e17320d83c2a5242c6df7c3cf386dedb63ed3812d",
        experiment_id="tile-model-family-tile-0ed823168064",
        aggregation="top1pct_mean",
        expected_backbone_sha256=RESNET18_SHA256,
        lock_digest="d3302166d3b7088302286fdc46243f8316d39e848677ce045cd22597e51d472b",
        gate=GATE_EXCELLENT,
        final_test_recall=0.9166666666666666,
        final_test_fpr=0.0,
        final_test_auroc=0.9971139971139972,
        final_test_average_precision=_AP_ADDENDUM.get("tile"),  # post-hoc, see AP_ADDENDUM_PATH
    ),
    # Cable: the locked winner of the normal-only alternative model-family study (the Phase 1
    # ConvAE, ai_models/cable/phase1_baseline/, failed with recall 0% / AUROC 0.574 and is NOT served).
    # Every value is copied verbatim from backend/ai_models/cable/model_family_study/selection_lock.json:
    #   threshold        <- selected_threshold (policy mean_std_3: mean + 3*std of the leave-one-image-out
    #                       scores of all 224 train/good images; selected from normal data only)
    #   expected_md5/sha <- selected_artifact.md5 / .sha256
    # Model: frozen ImageNet ResNet-18, layers 2+3 (384-d, 3x3 avg pooling), 128x128 input, k=1
    # Euclidean nearest neighbour to a 10% per-image coreset bank, image score = mean of the top 1%
    # of patch scores. Final test (scored once): TN 57 FP 1 FN 13 TP 79, recall 85.87%, F1 91.86%,
    # FPR 1.72%, AUROC 0.9880 (GOOD on point estimates; recall Wilson 95% lower bound 77.3%);
    # poke_insulation recall is 40% (6 of the 13 misses).
    "cable": CategoryServingConfig(
        category="cable",
        artifact_name="model_family_study/selected_candidate/model_state",
        model_name="knn_l23_128",
        threshold=2.244786389430004,
        threshold_method="mean_std",
        threshold_parameter=3.0,
        expected_md5="c878207964fc22865743f77c43ed06ea",
        provenance=(
            "backend/ai_models/cable/model_family_study/selection_lock.json (selected_threshold, "
            "selected_artifact; lock_digest 31d1aed63843aa0e316c3343b5a82309b2b349fde803d93219e22d012bcbc675) "
            "and reports/final_test_result.json"
        ),
        model_family=MODEL_FAMILY_PATCH_ANOMALY,
        input_size=(128, 128),
        expected_sha256="432859c5e39aaa85f8adbbd952f82ff0be1e6a905e360cfcca978a5638de7f5a",
        expected_config_sha256="f6696e7fbb3af5f275d8678c66339210ec7230195c212cad70dd22777bcf3230",
        experiment_id="cable-model-family-cable-ce2a8f015143",
        aggregation="top1pct_mean",
        expected_backbone_sha256=RESNET18_SHA256,
        lock_digest="31d1aed63843aa0e316c3343b5a82309b2b349fde803d93219e22d012bcbc675",
        gate=GATE_GOOD,
        final_test_recall=0.8586956521739131,
        final_test_fpr=0.017241379310344827,
        final_test_auroc=0.9880059970014994,
        final_test_average_precision=_AP_ADDENDUM.get("cable"),  # post-hoc, see AP_ADDENDUM_PATH
    ),
    # Leather, metal_nut, toothbrush, transistor: the locked winners of the same normal-only model-family study.
    # Every value is copied verbatim from backend/ai_models/<category>/model_family_study/selection_lock.json
    # (selected_candidate, selected_policy -> threshold_method/parameter, selected_threshold, selected_artifact.md5
    # / .sha256, experiment_id, lock_digest) and the SHA-256 of selected_candidate/model_config.json; gate, recall,
    # false_defect_detection_rate (FPR) and auroc from reports/final_test_result.json.
    "leather": _resnet18_entry(
        "leather", "gauss_l23_128", 29.944206466674807, "percentile", 99.0,
        md5="2386a7846e7383b5daaa9465a62e1843",
        sha256="a1cda7e4d8430b5f980c2445100ce5fdfec220b2c5c6141373e41ab9658720bf",
        config_sha256="77301eaf024e9716174a1e2cd4896900432746f05fb4ced4d226adb6d9c52705",
        input_side=128, experiment_id="leather-model-family-leather-d1b3df14779e",
        lock_digest="b3f7b30c99961c71138590fe6e6d6f45959869b6ebccba75cf6fc34f3b082260",
        gate=GATE_EXCELLENT, recall=1.0, fpr=0.0625, auroc=1.0,
    ),
    "metal_nut": _resnet18_entry(
        "metal_nut", "knn_l23_256", 2.017129956435393, "mean_std", 3.0,
        md5="ad62adb1022f36bd4f8c6d0b27d0e2a6",
        sha256="8e8fdd7756d75c3e2c15d2ee41738adb040be702014a5c1a8116d528c5e75141",
        config_sha256="6ab1c5e603cf2b63cb9c9f0e17320d83c2a5242c6df7c3cf386dedb63ed3812d",
        input_side=256, experiment_id="metal-nut-model-family-metal_nut-9a781605647b",
        lock_digest="fd94efa7446f88406e509d65240d9c524708b022c3a26c70dc5f02cd76f45360",
        gate=GATE_EXCELLENT, recall=0.967741935483871, fpr=0.045454545454545456, auroc=0.9912023460410557,
    ),
    "toothbrush": _resnet18_entry(
        "toothbrush", "gauss_l23_256", 37.871115668567995, "mean_std", 2.5,
        md5="4adc6d90032a1a147e576061229bc49d",
        sha256="4dcd4de93b0a5f387d04b2f96b7e8aa2876e0d906295f1924fb0cef1631556cc",
        config_sha256="b4815c40406dac199f515ad3a872d7770f50e4b6913bb23c3313c20c0667ae13",
        input_side=256, experiment_id="toothbrush-model-family-toothbrush-daebb965a2ab",
        lock_digest="19f60ebab4ec0dd4deab6321da68fe2532bfb51cee6f52f25804297ba7081b7c",
        gate=GATE_EXCELLENT, recall=0.9, fpr=0.08333333333333333, auroc=0.9722222222222223,
    ),
    "transistor": _resnet18_entry(
        "transistor", "knn_l23_256", 2.00266815662384, "percentile", 99.0,
        md5="e7e851d25a9095be32d6033fa0efe78c",
        sha256="37eae159275f5541e444868a6753b270b7ee4392d667a074507328c9aeadc11b",
        config_sha256="6ab1c5e603cf2b63cb9c9f0e17320d83c2a5242c6df7c3cf386dedb63ed3812d",
        input_side=256, experiment_id="transistor-model-family-transistor-8c5e27723221",
        lock_digest="c7d0ea4327fb6502c20f1b1ab88e5848c7a5a016f7762eb218474aeff52a94c4",
        gate=GATE_EXCELLENT, recall=0.925, fpr=0.0, auroc=0.98625,
    ),
}


def get_serving_config(category: str) -> CategoryServingConfig:
    """The serving configuration for `category`.

    Raises ModelArtifactNotFoundError if the category has no registered model.
    """
    config = SERVING_CONFIGS.get(category)
    if config is None:
        raise ModelArtifactNotFoundError(f"No AI model is configured for category '{category}'.")
    return config


def get_supported_categories() -> tuple[str, ...]:
    """Categories that currently have a registered AI model."""
    return tuple(SERVING_CONFIGS)


def _file_digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_md5(path: Path) -> str:
    return _file_digest(path, "md5")


def _missing(config: CategoryServingConfig) -> ModelArtifactNotFoundError:
    return ModelArtifactNotFoundError(
        f"No trained model artifact available for category '{config.category}' "
        f"(artifact '{config.artifact_name}')."
    )


def _verify(config: CategoryServingConfig, path: Path, algorithm: str, expected: str | None, what: str) -> None:
    if expected is None:
        return
    actual = _file_digest(path, algorithm)
    if actual != expected:
        raise ModelIntegrityError(
            f"{what} for category '{config.category}' (artifact '{config.artifact_name}') does not "
            f"match the validated model: expected {algorithm.upper()} {expected}, found {actual}."
        )


# ---------------------------------------------------------------------------
# Grid full320 loader (Addendum 1 code, imported read-only and hash-verified)
# ---------------------------------------------------------------------------

def _import_study_module(name: str, expected_sha256: str) -> ModuleType:
    path = PATCHCORE_STUDY_DIR / f"{name}.py"
    if not path.is_file():
        raise ModelArtifactNotFoundError(f"WRN-50 study module '{name}' is not available.")
    actual = _file_digest(path, "sha256")
    if actual != expected_sha256:
        raise ModelIntegrityError(
            f"WRN-50 study module '{name}' does not match its lock: expected SHA256 {expected_sha256}, found {actual}."
        )
    module = sys.modules.get(name)
    if module is not None:
        if Path(getattr(module, "__file__", "") or "").resolve() != path.resolve():
            raise ModelIntegrityError(f"A different module named '{name}' is already imported.")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # grid_full320_logic imports wrn50_study_logic by this bare name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def full320_module() -> ModuleType:
    """The Grid Addendum 1 logic module (load_full320_detector, resize_full320), after verifying its SHA-256 and
    that of the study logic it depends on against the values pinned from the Grid full320 lock."""
    _import_study_module("wrn50_study_logic", WRN50_STUDY_LOGIC_SHA256)
    return _import_study_module("grid_full320_logic", GRID_FULL320_LOGIC_SHA256)


# ---------------------------------------------------------------------------
# Shared backbones + LRU cache of per-category models
# ---------------------------------------------------------------------------

# One frozen backbone per family for the whole process, shared by every category of that family (never one
# backbone per category). Keyed by family; loaded on first use, kept until clear_model_cache().
_BACKBONES: dict[str, nn.Module] = {}

# Per-category models (memory bank / ConvAE weights + a reference to the shared backbone), keyed by resolved
# artifact path, least recently used first. value: (file signature, model); an entry is invalidated if the file
# on disk changes. Guarded by one lock because request handlers run in a thread pool; loading is serialized, as
# before. The models themselves are only ever used in eval()/no_grad mode.
_MODEL_CACHE: "OrderedDict[str, tuple[tuple, nn.Module | PatchAnomalyDetector | PatchCoreDetector]]" = OrderedDict()
_MODEL_CACHE_LOCK = threading.Lock()


def model_cache_limit() -> int:
    """How many per-category models (banks) may be resident at once (app.ai.config.model_cache_size)."""
    return ai_config.model_cache_size()


def cached_model_count() -> int:
    with _MODEL_CACHE_LOCK:
        return len(_MODEL_CACHE)


def cached_model_keys() -> tuple[str, ...]:
    """Cache keys (artifact paths), least recently used first - for tests and diagnostics; never sent to clients."""
    with _MODEL_CACHE_LOCK:
        return tuple(_MODEL_CACHE)


def loaded_backbone_count() -> int:
    with _MODEL_CACHE_LOCK:
        return len(_BACKBONES)


def clear_model_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
        _BACKBONES.clear()


def _shared_backbone(config: CategoryServingConfig) -> nn.Module:
    """The process-wide frozen backbone for `config`'s family (caller holds _MODEL_CACHE_LOCK). The pinned
    weights hash must be the one this code was validated with; the loaders verify the file against it."""
    family = config.model_family
    if family == MODEL_FAMILY_PATCHCORE_WRN50:
        pinned, loader = WRN50_2_SHA256, load_pretrained_wide_resnet50_2
    else:
        pinned, loader = RESNET18_SHA256, load_pretrained_resnet18
    if config.expected_backbone_sha256 is not None and config.expected_backbone_sha256 != pinned:
        raise ModelIntegrityError(
            f"Backbone weights pinned for category '{config.category}' ({config.expected_backbone_sha256}) are not "
            f"the validated weights ({pinned})."
        )
    backbone = _BACKBONES.get(family)
    if backbone is None:
        backbone = loader()
        _BACKBONES[family] = backbone
    return backbone


def _evict_for_one_more() -> None:
    """Drop least recently used models until one more fits (caller holds _MODEL_CACHE_LOCK). Runs BEFORE a new
    bank is read, so peak memory is the limit's worth of banks, never limit + 1."""
    limit = max(1, model_cache_limit())
    while len(_MODEL_CACHE) >= limit:
        _MODEL_CACHE.popitem(last=False)


def _load_patchcore(config: CategoryServingConfig, path: Path) -> PatchCoreDetector:
    mode = config.input_mode
    if mode not in PATCHCORE_INPUT_SIZES:
        raise ModelIntegrityError(f"Unknown WRN-50 input mode '{mode}' configured for category '{config.category}'.")
    backbone = _shared_backbone(config)
    try:
        if mode == INPUT_MODE_FULL320:
            detector = full320_module().load_full320_detector(path.parent, extractor=backbone)
        else:
            detector = PatchCoreDetector.load(path.parent, extractor=backbone)
    except PatchCoreModelError as exc:
        raise ModelIntegrityError(f"WRN-50 model for category '{config.category}' failed verification: {exc}") from exc
    except FileNotFoundError as exc:
        raise _missing(config) from exc
    loaded = detector.config
    side = PATCHCORE_INPUT_SIZES[mode]
    if (loaded.preprocessing, loaded.input_size, loaded.aggregation, loaded.weights_sha256) != (
        mode, side, config.aggregation, config.expected_backbone_sha256
    ) or tuple(config.input_size) != (side, side):
        raise ModelIntegrityError(
            f"Model configuration for category '{config.category}' ({loaded.preprocessing}, {loaded.input_size}, "
            f"{loaded.aggregation}) differs from the serving configuration ({mode}, {config.input_size}, "
            f"{config.aggregation})."
        )
    return detector


def _load_patch_anomaly(config: CategoryServingConfig, path: Path) -> PatchAnomalyDetector:
    backbone = _shared_backbone(config)
    try:
        model = PatchAnomalyDetector.load(path.parent, backbone)
    except FileNotFoundError as exc:
        raise _missing(config) from exc
    if (model.image_size, model.image_size) != tuple(config.input_size):
        raise ModelIntegrityError(
            f"Model configuration for category '{config.category}' expects {model.image_size}x"
            f"{model.image_size} input, but the serving configuration says {config.input_size}."
        )
    if config.aggregation is not None and model.aggregation != config.aggregation:
        raise ModelIntegrityError(
            f"Model configuration for category '{config.category}' aggregates with '{model.aggregation}', but the "
            f"serving configuration says '{config.aggregation}'."
        )
    return model


def load_serving_model(config: CategoryServingConfig) -> nn.Module | PatchAnomalyDetector | PatchCoreDetector:
    """The model for `config`, loaded from its configured artifact (cached, hash-verified).

    ConvAE configs return the ConvAutoencoder; patch-anomaly configs a PatchAnomalyDetector built from the
    artifact's directory (model_state.pt + model_config.json) and the shared frozen ResNet-18; WRN-50 configs a
    PatchCoreDetector (Grid: the Addendum 1 full320 loader) on the shared frozen WRN-50-2. Loading is read-only:
    nothing is fitted and no file is written.

    Raises ModelArtifactNotFoundError if that exact artifact (or its model_config.json) is missing - never falls
    back to any other file - and ModelIntegrityError if a configured MD5/SHA-256, the backbone hash, the input
    mode/size or the aggregation differs from the locked model.
    """
    path = get_model_path(config.category, config.artifact_name)
    family = config.model_family
    if family not in MODEL_FAMILIES:
        raise ModelArtifactNotFoundError(
            f"Unknown model family '{family}' configured for category '{config.category}'."
        )
    has_config_file = family != MODEL_FAMILY_CONVAE
    config_path = path.parent / CONFIG_FILENAME
    try:
        stat = path.stat()
        signature: tuple = (stat.st_mtime_ns, stat.st_size)
        if has_config_file:
            config_stat = config_path.stat()
            signature += (config_stat.st_mtime_ns, config_stat.st_size)
    except OSError as exc:
        raise _missing(config) from exc
    key = str(path)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            _MODEL_CACHE.move_to_end(key)
            return cached[1]
        _MODEL_CACHE.pop(key, None)

        _verify(config, path, "md5", config.expected_md5, "Model artifact")
        _verify(config, path, "sha256", config.expected_sha256, "Model artifact")
        if has_config_file:
            _verify(config, config_path, "sha256", config.expected_config_sha256, "Model configuration")

        _evict_for_one_more()
        if family == MODEL_FAMILY_PATCHCORE_WRN50:
            model = _load_patchcore(config, path)
        elif family == MODEL_FAMILY_PATCH_ANOMALY:
            model = _load_patch_anomaly(config, path)
        else:
            try:
                model = load_model_for_category(config.category, model_name=config.artifact_name)
            except FileNotFoundError as exc:
                raise _missing(config) from exc

        _MODEL_CACHE[key] = (signature, model)
        return model
