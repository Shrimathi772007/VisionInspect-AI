

# =====================================================================================================================
# Genuinely held-out ConvAutoencoder baseline (app.ai.evaluation.category_phase3_heldout / ..._final)
# 196/49 split, pre-declared selection rule, hash-locked threshold, exactly-once protected final test.
# =====================================================================================================================

import hashlib  # noqa: E402

import numpy as np  # noqa: E402

from app.ai.evaluation import category_phase3_heldout as heldout  # noqa: E402
from app.ai.evaluation import category_phase3_heldout_final as heldout_final  # noqa: E402
from app.ai.evaluation.threshold_experiments import METHOD_MEAN_STD, METHOD_PERCENTILE, ThresholdCandidate, generate_candidates  # noqa: E402
from app.ai.training.schemas import DatasetSample  # noqa: E402

HELDOUT_EXPECTED = {"train_good": 30, "test_good": 6, "test_defective": 8, "test_total": 14}


def _fake_samples(n: int) -> list[DatasetSample]:
    return [DatasetSample(path=Path(f"/x/train/good/{i:03d}.png"), category="c", split="train", defect_type="good", label=0) for i in range(n)]


def test_heldout_split_is_exactly_196_49_and_deterministic_seed_42():
    samples = _fake_samples(245)
    a, b = heldout.make_split(samples), heldout.make_split(samples)
    assert (len(a.training), len(a.validation)) == (196, 49) and a.seed == 42 and a.training_fraction == 0.8
    assert [s.path for s in a.training] == [s.path for s in b.training] and [s.path for s in a.validation] == [s.path for s in b.validation]
    assert not {s.path for s in a.training} & {s.path for s in a.validation}
    assert {s.path for s in a.training} | {s.path for s in a.validation} == {s.path for s in samples}
    shared = split_train_validation(samples, training_fraction=0.8, seed=42)  # the existing project convention
    assert [s.path for s in a.validation] == [s.path for s in shared.validation]
    expected_val = sorted(np.random.default_rng(42).permutation(245)[196:].tolist())
    assert [int(s.path.stem) for s in a.validation] == expected_val
    assert [s.path for s in a.validation] != [s.path for s in split_train_validation(samples, 0.8, 43).validation]
    fp = heldout.split_fingerprint(a)
    assert fp["training_count"] == 196 and fp["validation_count"] == 49 and fp["seed"] == 42


def test_candidate_generation_covers_the_declared_families_with_population_std():
    errors = list(np.random.default_rng(0).lognormal(-7.5, 0.25, 49))
    candidates = generate_candidates(errors)
    assert sorted(heldout.candidate_id(c) for c in candidates) == sorted([
        "mean_std_3", "mean_std_2.5", "mean_std_2", "mean_std_1.5", "mean_std_1",
        "percentile_95", "percentile_97", "percentile_98", "percentile_99"])
    by_id = {heldout.candidate_id(c): c for c in candidates}
    arr = np.asarray(errors)
    assert by_id["mean_std_3"].threshold == pytest.approx(arr.mean() + 3 * arr.std(ddof=0))
    assert by_id["percentile_95"].threshold == pytest.approx(np.percentile(arr, 95))
    assert all(c.calibration_sample_count == 49 for c in candidates)


def _cand(method, parameter, threshold):
    return ThresholdCandidate(method=method, parameter=parameter, threshold=threshold, calibration_sample_count=10,
                              calibration_mean=0.0, calibration_std=0.0, calibration_percentile=None)


ERRS = [float(i) for i in range(1, 11)]  # 1..10 -> 10 images, each false positive = 10%


def test_selection_only_considers_candidates_with_validation_fpr_at_most_ten_percent():
    sel = heldout.select_candidate([_cand(METHOD_MEAN_STD, 1.0, 5.5), _cand(METHOD_MEAN_STD, 2.0, 9.5)], ERRS)
    assert sel["winner"] == "mean_std_2"  # 5.5 -> 5 FP (50%) is ineligible; 9.5 -> 1 FP (10%) is eligible (boundary)
    assert [r["eligible"] for r in sel["candidates"]] == [False, True]


def test_selection_prefers_the_lower_validation_fpr():
    sel = heldout.select_candidate([_cand(METHOD_MEAN_STD, 3.0, 9.5), _cand(METHOD_PERCENTILE, 99.0, 10.5)], ERRS)
    assert sel["winner"] == "percentile_99" and sel["winner_row"]["validation_false_positives"] == 0


def test_selection_ties_go_to_mean_std_then_lowest_threshold_then_order():
    sel = heldout.select_candidate([_cand(METHOD_PERCENTILE, 98.0, 10.2), _cand(METHOD_MEAN_STD, 3.0, 10.9)], ERRS)
    assert sel["winner"] == "mean_std_3"  # identical FPR (0/10): the simpler family wins even with a higher threshold
    sel = heldout.select_candidate([_cand(METHOD_MEAN_STD, 3.0, 10.9), _cand(METHOD_MEAN_STD, 2.5, 10.2)], ERRS)
    assert sel["winner"] == "mean_std_2.5"  # still tied: the existing policy prefers the lowest threshold
    sel = heldout.select_candidate([_cand(METHOD_MEAN_STD, 2.0, 10.5), _cand(METHOD_MEAN_STD, 2.0, 10.5)], ERRS)
    assert sel["winner_row"]["order"] == 0


def test_no_eligible_candidate_reports_the_declared_message():
    sel = heldout.select_candidate([_cand(METHOD_MEAN_STD, 1.0, 3.0), _cand(METHOD_PERCENTILE, 95.0, 5.0)], ERRS)
    assert sel["winner"] is None and sel["message"] == "No threshold candidate satisfied the predeclared validation FPR requirement."


def test_selection_signature_has_no_parameter_for_test_data():
    import inspect

    assert list(inspect.signature(heldout.select_candidate).parameters) == ["candidates", "validation_errors"]


@pytest.fixture
def hworld(tmp_path, monkeypatch):
    """30 train/good (-> 24 training / 6 validation), 6 test/good, 8 defective; phase 1/2 artifacts faked in place."""
    root = tmp_path / "dataset"
    for i in range(30):
        _write(root / CATEGORY / "train" / "good" / f"{i:03d}.png", (10 + i * 5, 60 + i % 7, 90))
    for i in range(6):
        _write(root / CATEGORY / "test" / "good" / f"{i:03d}.png", (12 + i * 11, 61, 91))
    for defect, base in (("scratch", 200), ("dent", 150)):
        for i in range(4):
            _write(root / CATEGORY / "test" / defect / f"{i:03d}.png", (base, 20 + i * 9, 30))
    monkeypatch.setattr("app.ai.training.dataset.DATASET_ROOT", root)
    art = tmp_path / "ai_models"
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", art)

    metrics = {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1_score": 0.5, "false_defect_detection_rate": 0.9,
               "true_negatives": 1, "false_positives": 5, "false_negatives": 4, "true_positives": 4}
    dist = {"mean": 1.0, "std": 0.1, "median": 1.0, "min": 0.5, "max": 2.0}
    per_defect = {"scratch": {"recall": 0.5, "detected": 2, "total": 4}, "dent": {"recall": 0.25, "detected": 1, "total": 4}}
    p1 = art / CATEGORY / "phase1_baseline"
    p2 = art / CATEGORY / "phase2_threshold"
    (p1 / "reports").mkdir(parents=True)
    (p2 / "reports").mkdir(parents=True)
    (p1 / "autoencoder.pt").write_bytes(b"phase1-weights")
    (p1 / "reports" / "phase1_report.json").write_text(json.dumps({"run1": {
        "threshold": {"value": 0.001}, "metrics": metrics, "separation": {"auroc_descriptive_only": 0.51}, "per_defect": per_defect,
        "distributions": {"train_good": dist, "test_good": dist, "test_defective": dist}}}))
    (p2 / "threshold_lock.json").write_text("{}")
    (p2 / "reports" / "final_test_result.json").write_text(json.dumps({"locked_threshold": 0.001, "metrics": metrics, "per_defect": per_defect,
        "predictions": [{"label": 0, "reconstruction_error": 0.1}, {"label": 1, "reconstruction_error": 0.2}]}))
    return {"root": root, "art": art, "p1": p1, "p2": p2}


def _tiny(category=CATEGORY):
    return TrainingConfig(category=category, image_size=IMAGE_SIZE, batch_size=4, epochs=1, device="cpu")


def _stage(hworld, write_lock=True, target=None):
    target = target or heldout.model_path(CATEGORY)
    return heldout.train_validate_lock(CATEGORY, target, HELDOUT_EXPECTED, heldout.lock_path(CATEGORY) if write_lock else None, _tiny())


def _save_stage_report(stage):
    heldout.reports_dir(CATEGORY).mkdir(parents=True, exist_ok=True)
    heldout.validation_report_path(CATEGORY).write_text(json.dumps(stage, sort_keys=True), encoding="utf-8")


def test_stage_one_trains_only_on_training_images_and_never_decodes_a_test_image(hworld, monkeypatch):
    import app.ai.preprocessing.pipeline as pipeline
    import app.ai.training.phase3_dataset as p3ds

    decoded, trained, training_decodes = [], {}, []
    real_load = pipeline._load_image
    monkeypatch.setattr(pipeline, "_load_image", lambda path: (decoded.append(Path(path)), real_load(path))[1])
    real_ds = p3ds.load_sample
    monkeypatch.setattr(p3ds, "load_sample", lambda s, target_size: (training_decodes.append(s.path.name), real_ds(s, target_size=target_size))[1])
    real_train = heldout.train_anomaly_model_on_samples
    monkeypatch.setattr(heldout, "train_anomaly_model_on_samples",
                        lambda samples, cfg, path: (trained.update(names=[s.path.name for s in samples]), real_train(samples, cfg, path))[1])

    stage = _stage(hworld)
    assert (stage["split"]["training_count"], stage["split"]["validation_count"]) == (24, 6)
    assert trained["names"] == stage["training_files"] and not set(trained["names"]) & set(stage["validation_files"])
    assert set(training_decodes) == set(stage["training_files"])  # the DataLoader only ever decoded training images
    assert not [p for p in decoded if "test" in p.parts], "a final-test image was decoded before the lock"
    assert stage["events"] == ["split", "trained_on_training_subset_only", "validation_and_training_errors_computed", "selection_made", "threshold_locked"]


def test_heldout_stage_module_cannot_list_or_score_test_images():
    source = Path(heldout.__file__).read_text(encoding="utf-8")
    assert "discover_test_samples" not in source
    imported = {n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not [m for m in imported if "heldout_final" in m or "calibration_final_test" in m]


def test_count_mismatch_stops_before_training(hworld, monkeypatch):
    monkeypatch.setattr(heldout, "train_anomaly_model_on_samples", lambda *a, **k: (_ for _ in ()).throw(AssertionError("trained")))
    with pytest.raises(ValueError, match="train_good"):
        heldout.train_validate_lock(CATEGORY, heldout.model_path(CATEGORY), {**HELDOUT_EXPECTED, "train_good": 245}, None, _tiny())
    assert not heldout.model_path(CATEGORY).exists()


def test_lock_contents_hashes_and_artifact_isolation(hworld):
    before = heldout.prior_phase_reference(CATEGORY)
    stage = _stage(hworld)
    lock = json.loads(heldout.lock_path(CATEGORY).read_text())
    for key in ("category", "phase", "model_path", "model_sha256", "model_md5", "training_image_count", "validation_image_count", "split_seed",
                "threshold", "threshold_method", "all_candidates", "selected_candidate", "dataset_fingerprint_sha256", "experiment_id",
                "timestamp_utc", "lock_digest"):
        assert key in lock
    assert lock["training_image_count"] == 24 and lock["validation_image_count"] == 6 and lock["split_seed"] == 42
    assert len(lock["all_candidates"]) == 9 and lock["final_test_data_used_for_selection"] is False
    assert lock["model_sha256"] == hashlib.sha256(heldout.model_path(CATEGORY).read_bytes()).hexdigest()
    assert lock["model_md5"] == hashlib.md5(heldout.model_path(CATEGORY).read_bytes()).hexdigest()
    assert lock["threshold"] == stage["selection"]["winner_row"]["threshold"] and heldout.lock_digest(lock) == lock["lock_digest"]
    assert stage["model_hash_verified_after_lock"] and stage["reload_matches_trained_weights"]
    # artifact isolation: Phase 3 lives in its own directory; Phase 1/2 files are untouched
    assert heldout.model_path(CATEGORY).parent.name == "phase3_validation" and heldout.lock_path(CATEGORY).parent.name == "phase3_validation"
    assert heldout.prior_phase_reference(CATEGORY) == before == stage["prior_phase_artifacts_at_end"]
    assert (hworld["p1"] / "autoencoder.pt").read_bytes() == b"phase1-weights"


def test_stage_one_refuses_to_overwrite_existing_model_or_lock(hworld):
    _stage(hworld)
    with pytest.raises(FileExistsError):
        _stage(hworld)
    heldout.model_path(CATEGORY).unlink()
    with pytest.raises(FileExistsError):  # the lock is still there
        _stage(hworld)


def test_heldout_modules_do_not_register_serving():
    serving = Path(heldout.__file__).parent.parent / "inference" / "serving.py"
    assert "leather" not in serving.read_text(encoding="utf-8").lower()


def test_no_eligible_candidate_writes_no_lock(hworld, monkeypatch):
    monkeypatch.setattr(heldout, "select_candidate", lambda c, e: {"winner": None, "candidates": [], "message": heldout.NO_CANDIDATE_MESSAGE,
                                                                   "reasoning": heldout.NO_CANDIDATE_MESSAGE})
    stage = _stage(hworld)
    assert stage["lock"] is None and not heldout.lock_path(CATEGORY).exists() and "threshold_locked" not in stage["events"]


def test_two_independent_stage_runs_reproduce_everything(hworld, tmp_path):
    first = json.loads(json.dumps(_stage(hworld), sort_keys=True))
    second = json.loads(json.dumps(_stage(hworld, write_lock=False, target=tmp_path / "scratch" / "autoencoder.pt"), sort_keys=True))
    repro = heldout.compare_stages(first, second)
    assert all(v for k, v in repro.items() if isinstance(v, bool)), repro
    assert repro["loss_history_max_abs_difference"] == 0.0 and repro["validation_scores_max_abs_difference"] == 0.0
    tampered = json.loads(json.dumps(second))
    tampered["validation_errors"][0] += 1e-9
    assert not heldout.compare_stages(first, tampered)["validation_scores_identical"]


def _lock_and_report(hworld):
    stage = json.loads(json.dumps(_stage(hworld), sort_keys=True))
    _save_stage_report(stage)
    return stage


def test_final_test_runs_once_after_lock_with_a_clean_leakage_audit(hworld, monkeypatch):
    _lock_and_report(hworld)
    seen = {}
    real = heldout_final.discover_test_samples

    def spy(category):
        seen["lock_on_disk"] = heldout.lock_path(category).is_file()
        seen["result_existed"] = heldout_final.result_path(category).exists()
        return real(category)

    monkeypatch.setattr(heldout_final, "discover_test_samples", spy)
    result = heldout_final.evaluate_locked_on_final_test(CATEGORY, IMAGE_SIZE)
    assert seen == {"lock_on_disk": True, "result_existed": False}
    lock = json.loads(heldout.lock_path(CATEGORY).read_text())
    assert result["locked_threshold"] == lock["threshold"] and result["counts"] == {"test_total": 14, "test_good": 6, "test_defective": 8}
    m = result["metrics"]
    assert m["true_positives"] + m["false_negatives"] == 8 and m["true_negatives"] + m["false_positives"] == 6
    assert m["defect_identification_accuracy"] == m["recall"] and 0.0 <= result["auroc_descriptive_only"] <= 1.0
    for p in result["predictions"]:
        assert p["predicted_label"] == int(p["reconstruction_error"] > lock["threshold"])
    assert result["leakage_audit"]["all_passed"], result["leakage_audit"]
    assert set(result["comparison"]) >= {"phase1", "phase2", "phase3", "per_defect", "distributions"}
    assert result["gate_classification"] in {"EXCELLENT", "GOOD", "ACCEPTABLE", "NOT PRODUCTION READY"}
    with pytest.raises(FileExistsError):  # exactly once
        heldout_final.evaluate_locked_on_final_test(CATEGORY, IMAGE_SIZE)


def _forbid_test_access(monkeypatch):
    monkeypatch.setattr(heldout_final, "discover_test_samples", lambda c: (_ for _ in ()).throw(AssertionError("test set touched")))


def test_final_test_rejects_a_tampered_model_before_touching_the_test_set(hworld, monkeypatch):
    _lock_and_report(hworld)
    with open(heldout.model_path(CATEGORY), "ab") as handle:
        handle.write(b"tamper")
    _forbid_test_access(monkeypatch)
    with pytest.raises(RuntimeError, match="model file no longer matches"):
        heldout_final.evaluate_locked_on_final_test(CATEGORY, IMAGE_SIZE)
    assert not heldout_final.result_path(CATEGORY).exists()


def test_final_test_rejects_a_tampered_threshold_lock(hworld, monkeypatch):
    _lock_and_report(hworld)
    _forbid_test_access(monkeypatch)
    path = heldout.lock_path(CATEGORY)
    lock = json.loads(path.read_text())

    lock["threshold"] *= 2  # 1) edited threshold, stale digest
    path.write_text(json.dumps(lock))
    with pytest.raises(RuntimeError, match="does not match its digest"):
        heldout_final.evaluate_locked_on_final_test(CATEGORY, IMAGE_SIZE)

    lock["lock_digest"] = heldout.lock_digest(lock)  # 2) digest recomputed too: the stage report still holds the original
    path.write_text(json.dumps(lock))
    with pytest.raises(RuntimeError, match="differs from the lock recorded"):
        heldout_final.evaluate_locked_on_final_test(CATEGORY, IMAGE_SIZE)

    report = json.loads(heldout.validation_report_path(CATEGORY).read_text())  # 3) lock AND report edited consistently
    report["lock"] = lock
    heldout.validation_report_path(CATEGORY).write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="not what the pre-declared rule selects"):
        heldout_final.evaluate_locked_on_final_test(CATEGORY, IMAGE_SIZE)
    assert not heldout_final.result_path(CATEGORY).exists()


def test_final_test_rejects_a_modified_prior_phase_artifact(hworld, monkeypatch):
    _lock_and_report(hworld)
    (hworld["p1"] / "autoencoder.pt").write_bytes(b"overwritten")
    _forbid_test_access(monkeypatch)
    with pytest.raises(RuntimeError, match="Phase 1 or Phase 2 artifact"):
        heldout_final.evaluate_locked_on_final_test(CATEGORY, IMAGE_SIZE)


def test_final_test_rejects_a_changed_training_dataset(hworld, monkeypatch):
    _lock_and_report(hworld)
    _write(hworld["root"] / CATEGORY / "train" / "good" / "000.png", (250, 250, 250))
    _forbid_test_access(monkeypatch)
    with pytest.raises(RuntimeError, match="dataset no longer matches"):
        heldout_final.evaluate_locked_on_final_test(CATEGORY, IMAGE_SIZE)


def test_final_module_only_scores_with_the_locked_threshold():
    source = Path(heldout_final.__file__).read_text(encoding="utf-8")
    assert "candidate_results" not in source and "evaluate_candidate_on_final_test" not in source
