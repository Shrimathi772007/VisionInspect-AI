"""All 15 MVTec categories served: registry, shared backbones, LRU bank cache, GET /ai/models.

Covers the registration of every category's locked final model in app.ai.inference.serving - six ResNet-18 patch
models (model-family study winners) and nine WRN-50-2 PatchCore models (Grid: the Addendum 1 full320 model) - and
checks every pinned constant against the lock / final-test report it was copied from.

Real-model tests use COPIES of dataset/<category>/train/good/000.png only. Nothing here lists, reads or scores
dataset/<category>/test/ or ground_truth/: an autouse guard refuses to decode any such file. Tests that need the
real (Git-ignored) artifacts, backbones or dataset skip when they are absent, like the other AI serving tests.
"""

import json
import math
import shutil
import threading
from pathlib import Path

import pytest

import app.ai.inference.predict as predict_module
import app.ai.inference.serving as serving
from app.ai import config as ai_config
from app.ai.inference import ModelArtifactNotFoundError, ModelIntegrityError, predict_image
from app.ai.inference.serving import (
    GATES,
    MODEL_FAMILY_PATCH_ANOMALY,
    MODEL_FAMILY_PATCHCORE_WRN50,
    SERVING_CONFIGS,
    CategoryServingConfig,
    clear_model_cache,
    get_serving_config,
    load_serving_model,
)
from app.ai.models.patch_anomaly import PatchAnomalyDetector
from app.ai.models.patchcore import PatchCoreDetector
from app.ai.models.resnet18 import RESNET18_SHA256, pretrained_weights_path
from app.ai.models.wide_resnet50 import WRN50_2_SHA256, wide_resnet50_weights_path
from app.ai.preprocessing import patchcore_preprocess, pipeline
from app.ai.training.artifacts import ARTIFACTS_ROOT, get_model_path
from app.database import SessionLocal
from app.dataset.categories import MVTEC_CATEGORIES
from app.inspections.storage import DATASET_ROOT
from app.models.inspection import Inspection
from tests.conftest import make_image_bytes

RESNET18_CATEGORIES = ("tile", "cable", "leather", "metal_nut", "toothbrush", "transistor")
WRN50_CATEGORIES = ("bottle", "capsule", "zipper", "wood", "grid", "pill", "carpet", "screw", "hazelnut")
WRN50_DIRS = {c: ("patchcore_wrn50_full320" if c == "grid" else "patchcore_wrn50") for c in WRN50_CATEGORIES}
EXPECTED_GATES = {
    "bottle": "EXCELLENT", "hazelnut": "EXCELLENT", "grid": "EXCELLENT", "leather": "EXCELLENT",
    "metal_nut": "EXCELLENT", "transistor": "EXCELLENT", "tile": "EXCELLENT", "toothbrush": "EXCELLENT",
    "cable": "GOOD",
    "capsule": "ACCEPTABLE", "zipper": "ACCEPTABLE",
    "wood": "NOT_PRODUCTION_READY", "carpet": "NOT_PRODUCTION_READY", "screw": "NOT_PRODUCTION_READY",
    "pill": "NOT_PRODUCTION_READY",
}
EXPECTED_MODES = {
    "bottle": "crop224", "capsule": "crop224", "pill": "crop224", "carpet": "crop224", "hazelnut": "crop224",
    "zipper": "full256", "wood": "full256", "screw": "full256", "grid": "full320",
}
API_FIELDS = {
    "category", "model_name", "family", "input_mode", "input_size", "threshold", "gate", "final_test_recall",
    "final_test_fpr", "final_test_auroc", "final_test_average_precision",
}
AI_FIELDS = ("ai_prediction", "ai_reconstruction_error", "ai_threshold", "ai_model_name", "ai_inference_time_ms")
DATASET_DIR = Path(DATASET_ROOT).resolve()


def _artifact_present(category: str) -> bool:
    config = SERVING_CONFIGS[category]
    path = get_model_path(category, config.artifact_name)
    return path.is_file() and (path.parent / "model_config.json").is_file()


ALL_PRESENT = (
    all(_artifact_present(c) for c in MVTEC_CATEGORIES)
    and pretrained_weights_path().is_file()
    and wide_resnet50_weights_path().is_file()
    and all((DATASET_DIR / c / "train" / "good" / "000.png").is_file() for c in MVTEC_CATEGORIES)
)
requires_all_models = pytest.mark.skipif(
    not ALL_PRESENT, reason="the 15 final models, both backbones or MVTec train/good images are not present"
)
requires_wrn_backbone = pytest.mark.skipif(not wide_resnet50_weights_path().is_file(), reason="WRN-50-2 weights not present")


def _requires(*categories):
    present = all(_artifact_present(c) for c in categories) and pretrained_weights_path().is_file() \
        and wide_resnet50_weights_path().is_file()
    return pytest.mark.skipif(not present, reason=f"model artifacts for {categories} not present")


def _is_forbidden_dataset_path(path) -> bool:
    try:
        resolved = Path(path).resolve()
    except (TypeError, ValueError):
        return False
    if not resolved.is_relative_to(DATASET_DIR):
        return False
    parts = resolved.relative_to(DATASET_DIR).parts
    return (len(parts) >= 2 and parts[1] == "test") or any("ground_truth" in p.lower() for p in parts)


@pytest.fixture(autouse=True)
def _fresh_cache_and_test_split_guard(monkeypatch):
    """Refuse to decode any dataset test/ or ground_truth/ file (both preprocessing loaders), for every test."""
    real_load = pipeline._load_image

    def guarded(path):
        if _is_forbidden_dataset_path(path):
            raise AssertionError(f"dataset test/ground_truth file must not be read: {path}")
        return real_load(path)

    monkeypatch.setattr(pipeline, "_load_image", guarded)
    monkeypatch.setattr(patchcore_preprocess, "_load_image", guarded)
    clear_model_cache()
    yield
    clear_model_cache()


def _train_good_copy(category: str, tmp_path: Path) -> Path:
    target = tmp_path / f"{category}_train_good_000.png"
    shutil.copyfile(DATASET_DIR / category / "train" / "good" / "000.png", target)
    return target


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _policy(policy: str) -> tuple[str, float]:
    method, _, parameter = policy.rpartition("_")
    return method, float(parameter)


# ---------------------------------------------------------------------------
# Registry: 15 categories, constants equal to the locks and reports
# ---------------------------------------------------------------------------

def test_all_15_categories_and_nothing_else_are_registered():
    assert len(MVTEC_CATEGORIES) == 15
    assert set(SERVING_CONFIGS) == set(MVTEC_CATEGORIES)
    assert {c for c, cfg in SERVING_CONFIGS.items() if cfg.model_family == MODEL_FAMILY_PATCH_ANOMALY} == set(RESNET18_CATEGORIES)
    assert {c for c, cfg in SERVING_CONFIGS.items() if cfg.model_family == MODEL_FAMILY_PATCHCORE_WRN50} == set(WRN50_CATEGORIES)
    for category, config in SERVING_CONFIGS.items():
        assert config.category == category


def test_gates_modes_and_model_names_are_as_declared():
    for category, config in SERVING_CONFIGS.items():
        assert config.gate in GATES
        assert config.gate == EXPECTED_GATES[category], category
    for category in WRN50_CATEGORIES:
        config = SERVING_CONFIGS[category]
        mode = EXPECTED_MODES[category]
        side = {"crop224": 224, "full256": 256, "full320": 320}[mode]
        assert config.input_mode == mode and config.input_size == (side, side)
        assert config.model_name == f"wrn50_patchcore_{mode}"
        assert config.aggregation == "max"
        assert config.expected_backbone_sha256 == WRN50_2_SHA256
        assert config.expected_md5 is None and len(config.expected_sha256) == 64 and len(config.expected_config_sha256) == 64
        assert config.artifact_name == f"{WRN50_DIRS[category]}/final_model/model_state"
    for category in RESNET18_CATEGORIES:
        config = SERVING_CONFIGS[category]
        assert config.input_mode is None and config.aggregation == "top1pct_mean"
        assert config.expected_backbone_sha256 == RESNET18_SHA256
        # Not reported by the model-family study: the post-hoc AP addendum (tests/test_ap_addendum.py).
        assert config.final_test_average_precision == serving.load_ap_addendum()[category]
        assert config.artifact_name == "model_family_study/selected_candidate/model_state"


@pytest.mark.parametrize("category", WRN50_CATEGORIES)
def test_wrn50_constants_equal_the_lock_and_final_test_report(category):
    directory = ARTIFACTS_ROOT / category / WRN50_DIRS[category]
    if not (directory / "lock.json").is_file():
        pytest.skip(f"{category} WRN-50 lock not present")
    lock = json.loads((directory / "lock.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "final_test" / "final_test_result.json").read_text(encoding="utf-8"))
    config = get_serving_config(category)

    assert config.threshold == lock["threshold"] == report["locked_threshold"]
    assert (config.threshold_method, config.threshold_parameter) == _policy(lock["policy"])
    assert config.input_mode == lock["mode"] == lock["config"]["preprocessing"]
    assert config.input_size == (lock["config"]["input_size"],) * 2
    assert config.aggregation == lock["aggregation"]
    assert config.expected_sha256 == lock["model_state_sha256"]
    assert config.expected_config_sha256 == lock["model_config_sha256"]
    assert config.expected_backbone_sha256 == lock["backbone_weights_sha256"]
    assert config.lock_digest == lock["lock_digest"] == report["lock_digest"]
    assert config.gate == report["gate_classification"].replace(" ", "_")
    assert config.final_test_recall == report["metrics"]["recall"]
    assert config.final_test_fpr == report["metrics"]["false_positive_rate"]
    assert config.final_test_auroc == report["auroc"]
    assert config.final_test_average_precision == report["average_precision"]
    assert report["dry_run"] is False


@pytest.mark.parametrize("category", RESNET18_CATEGORIES)
def test_resnet18_constants_equal_the_selection_lock_and_final_test_report(category):
    study = ARTIFACTS_ROOT / category / "model_family_study"
    if not (study / "selection_lock.json").is_file():
        pytest.skip(f"{category} selection lock not present")
    lock = json.loads((study / "selection_lock.json").read_text(encoding="utf-8"))
    report = json.loads((study / "reports" / "final_test_result.json").read_text(encoding="utf-8"))
    config = get_serving_config(category)

    assert config.model_name == lock["selected_candidate"] == report["selected_candidate"]
    assert config.threshold == lock["selected_threshold"] == report["locked_threshold"]
    assert (config.threshold_method, config.threshold_parameter) == _policy(lock["selected_policy"])
    assert config.expected_md5 == lock["selected_artifact"]["md5"]
    assert config.expected_sha256 == lock["selected_artifact"]["sha256"]
    assert config.expected_config_sha256 == _sha256(study / "selected_candidate" / "model_config.json")
    assert config.experiment_id == lock["experiment_id"]
    assert config.lock_digest == lock["lock_digest"] == report["selection_lock_digest"]
    model_config = json.loads((study / "selected_candidate" / "model_config.json").read_text(encoding="utf-8"))
    assert config.input_size == (model_config["image_size"],) * 2
    assert config.aggregation == model_config["aggregation"]
    assert config.gate == report["gate_classification"].replace(" ", "_")
    assert config.final_test_recall == report["metrics"]["recall"]
    assert config.final_test_fpr == report["metrics"]["false_defect_detection_rate"]
    assert config.final_test_auroc == report["auroc"]


def test_wrn50_constants_equal_the_summary():
    path = ARTIFACTS_ROOT / "patchcore_wrn50_summary.json"
    if not path.is_file():
        pytest.skip("WRN-50 summary not present")
    rows = {row["category"]: row for row in json.loads(path.read_text(encoding="utf-8"))["rows"]}
    assert set(rows) == set(WRN50_CATEGORIES)
    for category, row in rows.items():
        config = SERVING_CONFIGS[category]
        assert config.gate == row["gate"].replace(" ", "_")
        assert (config.threshold, config.final_test_recall, config.final_test_fpr) == (row["threshold"], row["recall"], row["fpr"])
        assert (config.final_test_auroc, config.final_test_average_precision) == (row["auroc"], row["average_precision"])
        assert config.lock_digest == row["lock_digest"]


def test_pinned_study_module_hashes_equal_the_grid_full320_lock():
    lock_path = ARTIFACTS_ROOT / "grid" / "patchcore_wrn50_full320" / "lock.json"
    if not lock_path.is_file():
        pytest.skip("Grid full320 lock not present")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert serving.GRID_FULL320_LOGIC_SHA256 == lock["addendum_logic_sha256"]
    assert serving.WRN50_STUDY_LOGIC_SHA256 == lock["study_logic_sha256"]
    assert _sha256(serving.PATCHCORE_STUDY_DIR / "grid_full320_logic.py") == lock["addendum_logic_sha256"]
    assert _sha256(serving.PATCHCORE_STUDY_DIR / "wrn50_study_logic.py") == lock["study_logic_sha256"]


@pytest.mark.parametrize("category", ["", "Bottle", "bottles", "unknown", "mvtec_ad", "grid_full320"])
def test_unknown_category_has_no_model_and_loads_nothing(category, tmp_path, monkeypatch):
    def _must_not_load(*args, **kwargs):
        raise AssertionError("no model may be loaded for an unknown category")

    monkeypatch.setattr(predict_module, "load_serving_model", _must_not_load)
    image = tmp_path / "x.png"
    image.write_bytes(make_image_bytes("PNG", size=(32, 32)))
    with pytest.raises(ModelArtifactNotFoundError):
        predict_image(image, category)


def test_registry_is_immutable():
    with pytest.raises(Exception):
        get_serving_config("grid").threshold = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Cache size configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 3), ("", 3), ("3", 3), ("1", 1), ("5", 5), ("15", 15), ("0", 3), ("-2", 3), ("16", 3), ("abc", 3), (" 4 ", 4)],
)
def test_model_cache_size_env_var_has_a_safe_default(raw, expected, monkeypatch):
    if raw is None:
        monkeypatch.delenv(ai_config.MODEL_CACHE_SIZE_ENV, raising=False)
    else:
        monkeypatch.setenv(ai_config.MODEL_CACHE_SIZE_ENV, raw)
    assert ai_config.model_cache_size() == expected
    assert serving.model_cache_limit() == expected


def _convae_widget(tmp_path, monkeypatch, names):
    from app.ai.training import build_model, save_model

    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path / "ai_models")
    configs = {}
    for name in names:
        save_model(build_model(), get_model_path(name))
        configs[name] = CategoryServingConfig(
            category=name, artifact_name="autoencoder", model_name="autoencoder", threshold=0.01,
            threshold_method="mean_std", threshold_parameter=1.5, expected_md5=None, provenance="test fixture")
    return configs


def test_lru_cache_evicts_the_least_recently_used_model(tmp_path, monkeypatch):
    monkeypatch.setenv(ai_config.MODEL_CACHE_SIZE_ENV, "2")
    configs = _convae_widget(tmp_path, monkeypatch, ["a", "b", "c"])
    first_a = load_serving_model(configs["a"])
    load_serving_model(configs["b"])
    assert load_serving_model(configs["a"]) is first_a  # hit: a becomes most recently used
    load_serving_model(configs["c"])  # evicts b (least recently used), not a
    keys = serving.cached_model_keys()
    assert serving.cached_model_count() == 2
    assert [Path(k).parent.name for k in keys] == ["a", "c"]
    assert load_serving_model(configs["a"]) is first_a
    load_serving_model(configs["b"])  # reloaded; evicts c
    assert [Path(k).parent.name for k in serving.cached_model_keys()] == ["a", "b"]


def test_concurrent_loads_of_one_category_load_it_once(tmp_path, monkeypatch):
    configs = _convae_widget(tmp_path, monkeypatch, ["w"])
    calls = []
    real = serving.load_model_for_category

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(serving, "load_model_for_category", counting)
    results, errors = [], []

    def worker():
        try:
            results.append(load_serving_model(configs["w"]))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(results) == 8
    assert all(r is results[0] for r in results) and len(calls) == 1


# ---------------------------------------------------------------------------
# Real models: every category scores a train/good image, at most 3 banks resident
# ---------------------------------------------------------------------------

@requires_all_models
def test_every_category_scores_a_train_good_image_with_at_most_three_banks_resident(tmp_path, monkeypatch):
    monkeypatch.setenv(ai_config.MODEL_CACHE_SIZE_ENV, "3")
    backbone_loads = {"resnet18": 0, "wrn50": 0}
    real_r18, real_wrn = serving.load_pretrained_resnet18, serving.load_pretrained_wide_resnet50_2

    def r18():
        backbone_loads["resnet18"] += 1
        return real_r18()

    def wrn():
        backbone_loads["wrn50"] += 1
        return real_wrn()

    monkeypatch.setattr(serving, "load_pretrained_resnet18", r18)
    monkeypatch.setattr(serving, "load_pretrained_wide_resnet50_2", wrn)

    max_resident = 0
    extractors = {MODEL_FAMILY_PATCH_ANOMALY: set(), MODEL_FAMILY_PATCHCORE_WRN50: set()}
    for category in MVTEC_CATEGORIES:
        config = get_serving_config(category)
        result = predict_image(_train_good_copy(category, tmp_path), category)
        max_resident = max(max_resident, serving.cached_model_count())

        assert math.isfinite(result.reconstruction_error) and result.reconstruction_error >= 0.0, category
        assert result.prediction in ("good", "defective")
        assert result.prediction == ("good" if result.reconstruction_error <= config.threshold else "defective")
        assert result.threshold == config.threshold
        assert result.model_name == config.model_name
        assert result.input_size == config.input_size
        assert result.category == category

        model = load_serving_model(config)
        assert isinstance(model, PatchCoreDetector if config.model_family == MODEL_FAMILY_PATCHCORE_WRN50 else PatchAnomalyDetector)
        extractors[config.model_family].add(id(model.extractor))
        assert serving.cached_model_count() <= 3

    assert max_resident <= 3
    assert serving.cached_model_count() == 3
    assert serving.loaded_backbone_count() == 2
    assert backbone_loads == {"resnet18": 1, "wrn50": 1}  # one shared backbone per family, never one per category
    assert all(len(ids) == 1 for ids in extractors.values())


# ---------------------------------------------------------------------------
# Grid: the Addendum 1 full320 loader
# ---------------------------------------------------------------------------

@_requires("grid")
def test_grid_is_loaded_with_the_full320_loader_and_scored_at_320(tmp_path, monkeypatch):
    module = serving.full320_module()
    calls = []
    real_loader = module.load_full320_detector

    def spy(directory, extractor=None):
        calls.append(Path(directory).name)
        return real_loader(directory, extractor=extractor)

    def refuse(*args, **kwargs):
        raise AssertionError("PatchCoreDetector.load must not be used for Grid (it rejects full320)")

    monkeypatch.setattr(module, "load_full320_detector", spy)
    monkeypatch.setattr(PatchCoreDetector, "load", classmethod(refuse))

    detector = load_serving_model(get_serving_config("grid"))
    assert calls == ["final_model"]
    assert isinstance(detector.config, module.Full320Config)
    assert (detector.config.preprocessing, detector.config.input_size, detector.config.aggregation) == ("full320", 320, "max")
    assert list(detector.bank.shape) == [42240, 1536]

    if (DATASET_DIR / "grid" / "train" / "good" / "000.png").is_file():
        shapes = []
        real_extract = detector.extract_features

        def spy_extract(images):
            shapes.append(tuple(images.shape))
            return real_extract(images)

        monkeypatch.setattr(detector, "extract_features", spy_extract)
        result = predict_image(_train_good_copy("grid", tmp_path), "grid")
        assert result.model_name == "wrn50_patchcore_full320" and result.input_size == (320, 320)
        assert shapes == [(1, 3, 320, 320)]
        assert result.threshold == 1.88393018105021


def test_grid_study_module_with_a_wrong_hash_is_refused(monkeypatch):
    monkeypatch.setattr(serving, "GRID_FULL320_LOGIC_SHA256", "0" * 64)
    with pytest.raises(ModelIntegrityError, match="grid_full320_logic"):
        serving.full320_module()


# ---------------------------------------------------------------------------
# Tampering: a wrong pinned value refuses to load - and never breaks a request
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("category", "field", "match"),
    [
        ("bottle", "expected_sha256", "expected SHA256 " + "0" * 64),
        ("zipper", "expected_config_sha256", "Model configuration"),
        ("metal_nut", "expected_md5", "expected MD5 " + "0" * 32),
        ("toothbrush", "expected_sha256", "expected SHA256 " + "0" * 64),
    ],
)
def test_wrong_pinned_hash_refuses_to_load(category, field, match, tmp_path, monkeypatch):
    if not _artifact_present(category):
        pytest.skip(f"{category} artifact not present")
    value = "0" * (32 if field == "expected_md5" else 64)
    tampered = CategoryServingConfig(**{**get_serving_config(category).__dict__, field: value})
    monkeypatch.setitem(SERVING_CONFIGS, category, tampered)

    def _no_backbone():
        raise AssertionError("a refused model must be rejected before any backbone is loaded")

    monkeypatch.setattr(serving, "load_pretrained_resnet18", _no_backbone)
    monkeypatch.setattr(serving, "load_pretrained_wide_resnet50_2", _no_backbone)
    image = tmp_path / "x.png"
    image.write_bytes(make_image_bytes("PNG", size=(32, 32)))
    with pytest.raises(ModelIntegrityError, match=match):
        predict_image(image, category)
    assert serving.cached_model_count() == 0


@_requires("capsule")
def test_wrong_pinned_backbone_hash_refuses_to_load(monkeypatch):
    tampered = CategoryServingConfig(**{**get_serving_config("capsule").__dict__, "expected_backbone_sha256": "1" * 64})
    with pytest.raises(ModelIntegrityError, match="Backbone weights pinned"):
        load_serving_model(tampered)


@_requires("pill")
def test_wrong_pinned_input_mode_or_aggregation_refuses_to_load():
    base = get_serving_config("pill")
    for change in ({"input_mode": "full256", "input_size": (256, 256)}, {"aggregation": "top1pct_mean"}):
        with pytest.raises(ModelIntegrityError, match="differs from the serving configuration"):
            load_serving_model(CategoryServingConfig(**{**base.__dict__, **change}))
        clear_model_cache()


def test_tampered_wrn50_bank_in_place_is_refused(tmp_path, monkeypatch):
    import torch

    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    directory = root / "grid" / "patchcore_wrn50_full320" / "final_model"
    directory.mkdir(parents=True)
    torch.save({"bank": torch.zeros(4, 1536)}, directory / "model_state.pt")
    (directory / "model_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="expected SHA256 " + get_serving_config("grid").expected_sha256):
        load_serving_model(get_serving_config("grid"))


def test_missing_wrn50_model_raises_not_found_and_never_falls_back(tmp_path, monkeypatch):
    from app.ai.training import build_model, save_model

    root = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", root)
    save_model(build_model(), root / "capsule" / "autoencoder.pt")  # an unrelated ConvAE exists
    image = tmp_path / "x.png"
    image.write_bytes(make_image_bytes("PNG", size=(32, 32)))
    with pytest.raises(ModelArtifactNotFoundError, match="capsule"):
        predict_image(image, "capsule")


def test_tampered_model_never_fails_an_upload_and_leaks_nothing(category_products, monkeypatch):
    tampered = CategoryServingConfig(**{**get_serving_config("bottle").__dict__, "expected_sha256": "0" * 64})
    monkeypatch.setitem(SERVING_CONFIGS, "bottle", tampered)
    product = category_products.create(category="bottle")

    response = category_products.upload(product, make_image_bytes("PNG"))

    assert response.status_code == 201
    body = response.json()
    for field in AI_FIELDS:
        assert body[field] is None, field
    assert "0" * 64 not in response.text and "ai_models" not in response.text and "model_state" not in response.text


# ---------------------------------------------------------------------------
# End to end through the API: upload and import, one WRN-50 and one ResNet-18 category
# ---------------------------------------------------------------------------

def _stored(inspection_id: int) -> Inspection:
    with SessionLocal() as session:
        return session.get(Inspection, inspection_id)


def _assert_persisted(body: dict, category: str) -> None:
    config = get_serving_config(category)
    assert body["ai_prediction"] in ("good", "defective")
    assert body["ai_model_name"] == config.model_name
    assert body["ai_threshold"] == config.threshold
    assert isinstance(body["ai_reconstruction_error"], float) and math.isfinite(body["ai_reconstruction_error"])
    assert body["ai_inference_time_ms"] > 0
    assert body["ai_prediction"] == ("good" if body["ai_reconstruction_error"] <= config.threshold else "defective")
    stored = _stored(body["id"])
    assert stored.ai_prediction == body["ai_prediction"]
    assert stored.ai_model_name == config.model_name
    assert stored.ai_threshold == config.threshold
    assert stored.ai_reconstruction_error == body["ai_reconstruction_error"]
    assert stored.ai_inference_time_ms == body["ai_inference_time_ms"]


@_requires("zipper", "metal_nut")
@pytest.mark.parametrize("category", ["zipper", "metal_nut"])  # WRN-50 full256, ResNet-18 256
def test_upload_persists_ai_fields(category, category_products, tmp_path):
    image = _train_good_copy(category, tmp_path)
    product = category_products.create(category=category)

    response = category_products.upload(product, image.read_bytes(), filename=f"{category}.png")

    assert response.status_code == 201, response.text
    body = response.json()
    _assert_persisted(body, category)
    assert body["product_category"] == category


@_requires("capsule", "transistor")
@pytest.mark.parametrize("category", ["capsule", "transistor"])  # WRN-50 crop224, ResNet-18 256
def test_import_persists_ai_fields(category, client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={"product_id": test_product["id"], "category": category, "split": "train", "defect_type": "good",
              "filename": "000.png"},
    )
    assert response.status_code == 201, response.text
    _assert_persisted(response.json(), category)


def test_upload_for_category_with_no_registered_model_stays_null(category_products, monkeypatch):
    monkeypatch.delitem(SERVING_CONFIGS, "wood")
    product = category_products.create(category="wood")
    response = category_products.upload(product, make_image_bytes("PNG"))
    assert response.status_code == 201
    for field in AI_FIELDS:
        assert response.json()[field] is None, field


def test_inference_failure_during_upload_stays_null(category_products, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError(r"bank load failed at C:\secret\ai_models\grid\final_model\model_state.pt")

    monkeypatch.setattr(predict_module, "load_serving_model", _boom)
    product = category_products.create(category="grid")
    response = category_products.upload(product, make_image_bytes("PNG"))
    assert response.status_code == 201
    for field in AI_FIELDS:
        assert response.json()[field] is None, field
    assert "secret" not in response.text


# ---------------------------------------------------------------------------
# GET /ai/models
# ---------------------------------------------------------------------------

def test_ai_models_requires_authentication(client):
    assert client.get("/ai/models").status_code == 401
    assert client.get("/ai/models", headers={"Authorization": "Bearer not-a-token"}).status_code == 401


@pytest.mark.parametrize("headers_fixture", ["supervisor_headers", "qe_headers"])
def test_ai_models_lists_the_15_registered_models(headers_fixture, client, request, monkeypatch):
    def _must_not_load(*args, **kwargs):
        raise AssertionError("GET /ai/models must not load any model")

    monkeypatch.setattr(serving, "load_serving_model", _must_not_load)
    response = client.get("/ai/models", headers=request.getfixturevalue(headers_fixture))
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 15
    assert [row["category"] for row in rows] == sorted(MVTEC_CATEGORIES)
    for row in rows:
        assert set(row) == API_FIELDS
        config = SERVING_CONFIGS[row["category"]]
        assert row["model_name"] == config.model_name
        assert row["family"] == config.model_family
        assert row["input_mode"] == (config.input_mode or "resize")
        assert tuple(row["input_size"]) == config.input_size
        assert row["threshold"] == config.threshold
        assert row["gate"] == EXPECTED_GATES[row["category"]]
        assert row["final_test_recall"] == config.final_test_recall
        assert row["final_test_fpr"] == config.final_test_fpr
        assert row["final_test_auroc"] == config.final_test_auroc
        assert row["final_test_average_precision"] == config.final_test_average_precision
    grid = next(row for row in rows if row["category"] == "grid")
    assert (grid["model_name"], grid["input_mode"], grid["input_size"]) == ("wrn50_patchcore_full320", "full320", [320, 320])
    tile = next(row for row in rows if row["category"] == "tile")
    assert (tile["family"], tile["input_mode"], tile["final_test_average_precision"]) == ("patch_anomaly", "resize", 0.99886)


def test_ai_models_exposes_no_paths_or_hashes(client, qe_headers):
    text = client.get("/ai/models", headers=qe_headers).text
    for config in SERVING_CONFIGS.values():
        for secret in (config.expected_md5, config.expected_sha256, config.expected_config_sha256,
                       config.expected_backbone_sha256, config.lock_digest, config.artifact_name, config.provenance):
            if secret:
                assert secret not in text
    for marker in ("ai_models", "model_state", ".pt", ".json", "\\\\", "sha", "md5", "digest", "artifact", "provenance"):
        assert marker not in text.lower()
