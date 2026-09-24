"""Category Phase 4 optimization experiment: candidates, validation-only selection, freeze, single
final-test scoring, adoption rule, reproducibility, leakage audit, artifact metadata.

The experiment tests run the REAL engine on a tiny synthetic category (32x32-ish images) with the stage
grids shrunk via the stage constants, so they finish in seconds; the baseline configuration is kept in
the grid so the protected-artifact reuse path is exercised too. Selection unit tests use light fakes.
One read-only test uses the real Hazelnut artifact/dataset (skipped when absent).
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from app.ai.evaluation import category_phase4 as engine
from app.ai.evaluation.category_phase3 import PHASE3_SPLIT_SEED, PHASE3_TRAINING_FRACTION, resolve_outputs
from app.ai.evaluation.category_phase4 import (
    EVENT_SELECTION_FROZEN,
    ExperimentResult,
    LockedSelection,
    Phase3Reference,
    compare_experiments,
    run_candidate,
    run_optimization_experiment,
    select_stage_winner,
)
from app.ai.evaluation.category_phase4_final import (
    EVENT_FINAL_TEST_SCORED,
    audit_phase4_leakage,
    compare_to_baseline,
    evaluate_locked_on_final_test,
    module_references,
    sign_test_p_value,
)
from app.ai.evaluation.category_phase4_report import (
    build_category_phase4_report,
    ensure_no_phase4_overwrite,
    phase4_report_path,
    phase4_root,
)
from app.ai.evaluation.phase3_threshold_selection import (
    STATUS_NO_DEFENSIBLE_WINNER,
    STATUS_SELECTED,
    select_threshold_from_validation,
)
from app.ai.evaluation.threshold_experiments import generate_candidates
from app.ai.training import discover_train_samples
from app.ai.training import category_phase4_candidates as cands
from app.ai.training.artifacts import ARTIFACTS_ROOT, get_model_path
from app.ai.training.phase3_train import train_anomaly_model_on_samples
from app.ai.training.validation_split import split_train_validation
from tests.conftest import make_image_bytes

CATEGORY = "widget"


# ---------------------------------------------------------------------------
# Candidate configuration generation (real constants)
# ---------------------------------------------------------------------------

def test_stage1_candidates_vary_only_input_size():
    configs = cands.stage1_candidates("hazelnut")
    assert [c.training_config.image_size for c in configs] == [(128, 128), (160, 160), (192, 192)]
    for c in configs:
        tc = c.training_config
        assert (tc.category, tc.epochs, tc.learning_rate, tc.batch_size, tc.seed, tc.device) == (
            "hazelnut", 15, 1e-3, 8, 42, "cpu",
        )
        assert tc.model.architecture == "conv_autoencoder" and tc.model.latent_channels == 128


def test_stage2_and_stage3_candidates_vary_one_parameter():
    s2 = cands.stage2_candidates("hazelnut", (160, 160))
    assert [(c.training_config.epochs, c.training_config.learning_rate) for c in s2] == [(15, 1e-3), (25, 1e-3), (35, 1e-3)]
    assert {c.training_config.image_size for c in s2} == {(160, 160)}
    s3 = cands.stage3_candidates("hazelnut", (160, 160), 25)
    assert [c.training_config.learning_rate for c in s3] == [1e-3, 5e-4]
    assert {(c.training_config.image_size, c.training_config.epochs) for c in s3} == {((160, 160), 25)}


def test_candidate_ids_are_unique_per_configuration_and_stable():
    assert cands.candidate_id((160, 160), 25, 5e-4) == "input160_epochs25_lr0.0005"
    ids = [c.candidate_id for c in cands.stage1_candidates("hazelnut") + cands.stage2_candidates("hazelnut", (128, 128))]
    # the same configuration reached from two stages is the same candidate id (trained once)
    assert ids.count("input128_epochs15_lr0.001") == 2 and len(set(ids)) == 5


def test_only_the_exact_baseline_configuration_may_reuse_the_phase3_artifact():
    everything = (
        cands.stage1_candidates("hazelnut")
        + cands.stage2_candidates("hazelnut", (128, 128))
        + cands.stage2_candidates("hazelnut", (192, 192))
        + cands.stage3_candidates("hazelnut", (128, 128), 35)
    )
    for c in everything:
        tc = c.training_config
        exact = (tc.image_size, tc.epochs, tc.learning_rate) == ((128, 128), 15, 1e-3)
        assert c.reuse_phase3_artifact == exact
    assert not cands.is_baseline((128, 128), 25, 1e-3) and not cands.is_baseline((128, 128), 15, 5e-4)
    assert not cands.is_baseline((160, 160), 15, 1e-3)


def test_candidates_are_category_parameterized():
    assert {c.training_config.category for c in cands.stage1_candidates("cable")} == {"cable"}


# ---------------------------------------------------------------------------
# Paths / overwrite protection
# ---------------------------------------------------------------------------

def test_phase4_paths_are_derived_from_category_and_separate_from_phase3():
    root = phase4_root("hazelnut")
    assert root == ARTIFACTS_ROOT / "hazelnut" / "phase4_optimization"
    assert phase4_report_path("hazelnut") == root / "reports" / "phase4_optimization_report.json"
    assert resolve_outputs("hazelnut").model_path.parent.name == "phase3_validation"
    assert not str(resolve_outputs("hazelnut").model_path).startswith(str(root))


def test_phase4_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp_path)
    ensure_no_phase4_overwrite("hazelnut")  # nothing there yet
    (tmp_path / "hazelnut" / "phase4_optimization").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ensure_no_phase4_overwrite("hazelnut")


# ---------------------------------------------------------------------------
# Validation-only stage selection (light fakes)
# ---------------------------------------------------------------------------

def _fake(cid, cov, *, epochs=15, size=128, lr=1e-3, nbytes=1000, status=STATUS_SELECTED):
    return SimpleNamespace(
        candidate_id=cid,
        coefficient_of_variation=cov,
        selection=SimpleNamespace(status=status),
        config=SimpleNamespace(training_config=SimpleNamespace(epochs=epochs, image_size=(size, size), learning_rate=lr)),
        model_metadata=SimpleNamespace(artifact_size_bytes=nbytes),
    )


def test_lowest_validation_cov_wins():
    winner, sel = select_stage_winner("s", [_fake("a", 0.40), _fake("b", 0.30), _fake("c", 0.35)])
    assert winner.candidate_id == "b" and [r[0] for r in sel.ranked] == ["b", "c", "a"] and not sel.tied


def test_disqualified_candidate_cannot_win_even_with_best_cov():
    winner, sel = select_stage_winner(
        "s", [_fake("great", 0.01, status=STATUS_NO_DEFENSIBLE_WINNER), _fake("ok", 0.30)]
    )
    assert winner.candidate_id == "ok" and sel.disqualified == ["great"]


def test_all_disqualified_or_empty_raises():
    with pytest.raises(ValueError):
        select_stage_winner("s", [_fake("x", 0.1, status=STATUS_NO_DEFENSIBLE_WINNER)])
    with pytest.raises(ValueError):
        select_stage_winner("s", [])


@pytest.mark.parametrize(
    "a_kwargs,b_kwargs,expected",
    [
        (dict(epochs=25), dict(epochs=15), "b"),  # fewer epochs
        (dict(size=160), dict(size=128), "b"),  # smaller input
        (dict(nbytes=2000), dict(nbytes=1000), "b"),  # smaller model file
        (dict(lr=5e-4), dict(lr=1e-3), "b"),  # learning rate nearest baseline
    ],
)
def test_ties_within_two_percent_prefer_the_simpler_candidate(a_kwargs, b_kwargs, expected):
    winner, sel = select_stage_winner("s", [_fake("a", 0.300, **a_kwargs), _fake("b", 0.305, **b_kwargs)])
    assert winner.candidate_id == expected and set(sel.tied) == {"a", "b"}


def test_outside_the_tolerance_the_lower_cov_wins_regardless_of_simplicity():
    winner, _ = select_stage_winner("s", [_fake("complex_better", 0.20, epochs=35, size=192), _fake("simple", 0.30)])
    assert winner.candidate_id == "complex_better"


def test_wall_clock_time_can_never_change_a_selection():
    """Fakes carry no timing attributes at all - selection must not need (or be swayed by) them."""
    first, _ = select_stage_winner("s", [_fake("a", 0.30), _fake("b", 0.30)])
    second, _ = select_stage_winner("s", [_fake("b", 0.30), _fake("a", 0.30)])
    assert first.candidate_id == second.candidate_id == "a"  # deterministic id tie-break, order-independent


# ---------------------------------------------------------------------------
# The experiment itself (synthetic data, real engine)
# ---------------------------------------------------------------------------

def _write(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_image_bytes("PNG", size=(40, 40), color=color))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phase4")
    root = tmp / "dataset"
    for i in range(20):  # 20 train/good -> 16 training, 4 validation
        _write(root / CATEGORY / "train" / "good" / f"{i:03d}.png", (10 + i * 6, 60 + i, 90))
    for i in range(6):
        _write(root / CATEGORY / "test" / "good" / f"{i:03d}.png", (14 + i * 9, 61 + i, 91))
    for defect, base in (("scratch", 210), ("dent", 150)):
        for i in range(5):
            _write(root / CATEGORY / "test" / defect / f"{i:03d}.png", (base, 20 + i * 9, 30 + i))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.ai.training.dataset.DATASET_ROOT", root)
        mp.setattr("app.ai.training.artifacts.ARTIFACTS_ROOT", tmp / "ai_models")
        # Shrink the grids; the exact baseline (128, 15 epochs, 1e-3) stays in stage 1 to exercise reuse.
        mp.setattr(cands, "STAGE1_IMAGE_SIZES", ((128, 128), (32, 32), (48, 48)))
        mp.setattr(cands, "STAGE2_EPOCHS", (15, 1, 2))

        split = split_train_validation(discover_train_samples(CATEGORY), PHASE3_TRAINING_FRACTION, PHASE3_SPLIT_SEED)
        baseline_cfg = cands.make_candidate(CATEGORY, "baseline", (128, 128), 15, 1e-3).training_config
        baseline_path = get_model_path(CATEGORY, "phase3_validation/autoencoder")
        result, _ = train_anomaly_model_on_samples(split.training, baseline_cfg, baseline_path)
        reference = Phase3Reference(
            model_path=baseline_path,
            training_duration_s=result.duration_seconds,
            final_loss=result.final_loss,
            num_training_images=result.num_training_images,
            loss_history=result.loss_history,
        )
        import hashlib

        baseline_md5 = hashlib.md5(baseline_path.read_bytes()).hexdigest()

        logs: list[str] = []
        exp1 = run_optimization_experiment(CATEGORY, split, tmp / "run1", reference, log=logs.append)
        exp2 = run_optimization_experiment(CATEGORY, split, tmp / "run2", reference, log=logs.append)
        final = evaluate_locked_on_final_test(CATEGORY, exp1.locked, exp1.events)
        yield SimpleNamespace(
            split=split, reference=reference, exp1=exp1, exp2=exp2, final=final, tmp=tmp,
            baseline_md5=baseline_md5, baseline_path=baseline_path,
        )


def test_experiment_evaluates_the_predefined_stages_and_dedupes_candidates(world):
    exp = world.exp1
    assert [s.stage for s in exp.stage_selections] == ["stage1_input_size", "stage2_epochs", "stage3_learning_rate"]
    ids = list(exp.outcomes)
    assert len(ids) == len(set(ids))  # every configuration became exactly one candidate
    assert {"input128_epochs15_lr0.001", "input32_epochs15_lr0.001", "input48_epochs15_lr0.001"} <= set(ids)


def test_every_candidate_trained_on_exactly_the_training_subset(world):
    n = len(world.split.training)
    assert n == 16 and len(world.split.validation) == 4
    assert all(o.num_training_images == n for o in world.exp1.outcomes.values())


def test_baseline_artifact_is_reused_only_for_the_baseline_configuration_and_never_modified(world):
    import hashlib

    for o in world.exp1.outcomes.values():
        tc = o.config.training_config
        exact = (tc.image_size, tc.epochs, tc.learning_rate) == ((128, 128), 15, 1e-3)
        assert o.reused_phase3_artifact == exact
        if exact:
            assert o.model_path == world.baseline_path
        else:
            assert o.model_path != world.baseline_path and o.model_path.is_file()
    assert hashlib.md5(world.baseline_path.read_bytes()).hexdigest() == world.baseline_md5


def test_run_candidate_refuses_reuse_without_a_reference(world):
    cfg = cands.make_candidate(CATEGORY, "s", (128, 128), 15, 1e-3)
    with pytest.raises(ValueError, match="reuses the Phase 3 artifact"):
        run_candidate(cfg, world.split, world.tmp / "x" / "a.pt", None)


def test_candidate_threshold_uses_only_its_own_validation_errors(world):
    for o in world.exp1.outcomes.values():
        expected = select_threshold_from_validation(generate_candidates(o.validation_errors), o.validation_errors)
        assert o.selection.status == expected.status
        if expected.selected is not None:
            assert o.selection.selected.threshold == expected.selected.threshold
        assert len(o.validation_errors) == 4
        assert o.coefficient_of_variation == pytest.approx(o.validation_std / o.validation_mean)


def test_frozen_selection_matches_the_stage3_winner_and_is_immutable(world):
    exp, locked = world.exp1, world.exp1.locked
    assert locked.candidate_id == exp.winner.candidate_id == exp.stage_selections[-1].winner_id
    assert locked.threshold == exp.winner.selection.selected.threshold
    assert locked.model_md5 == exp.winner.model_metadata.md5
    assert exp.events[-1] == EVENT_SELECTION_FROZEN
    with pytest.raises(dataclasses.FrozenInstanceError):
        locked.threshold = 1.0  # type: ignore[misc]


def test_experiment_module_cannot_list_test_images_but_the_final_module_can():
    import app.ai.evaluation.category_phase4_final as final_module

    assert module_references(engine, {"discover_test_samples"}) == set()
    assert module_references(final_module, {"discover_test_samples"}) == {"discover_test_samples"}


def test_experiment_does_not_touch_test_images_even_if_discovery_is_booby_trapped(world, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("experiment must not list test images")

    monkeypatch.setattr("app.ai.training.dataset.discover_test_samples", _boom)
    cfgs = [cands.make_candidate(CATEGORY, "s", (32, 32), 1, 1e-3)]
    outcome = run_candidate(cfgs[0], world.split, world.tmp / "trap" / "a.pt", world.reference)
    assert outcome.num_training_images == 16


# ---------------------------------------------------------------------------
# Final test: ordering, single scoring, protections
# ---------------------------------------------------------------------------

def test_final_test_is_scored_once_after_the_freeze_with_the_locked_threshold(world):
    final, locked = world.final, world.exp1.locked
    assert final.events.index(EVENT_SELECTION_FROZEN) < final.events.index(EVENT_FINAL_TEST_SCORED)
    assert final.events.count(EVENT_FINAL_TEST_SCORED) == 1
    assert final.locked_digest == locked.digest
    assert len(final.predictions) == 16 and final.metrics["total_test_samples"] == 16
    assert all(p["predicted_label"] == (1 if p["reconstruction_error"] > locked.threshold else 0) for p in final.predictions)
    m = final.metrics
    assert m["defect_identification_accuracy"] == m["recall"]
    assert (m["true_negatives"] + m["false_positives"], m["false_negatives"] + m["true_positives"]) == (6, 10)


def test_final_test_refuses_before_freeze_and_after_model_tampering(world):
    with pytest.raises(RuntimeError, match="not been frozen"):
        evaluate_locked_on_final_test(CATEGORY, world.exp1.locked, [])
    tampered = dataclasses.replace(world.exp1.locked, model_md5="0" * 32)
    with pytest.raises(RuntimeError, match="Locked model changed"):
        evaluate_locked_on_final_test(CATEGORY, tampered, world.exp1.events)


def test_leakage_audit_passes_for_a_clean_experiment(world):
    checks = audit_phase4_leakage(CATEGORY, world.split, [world.exp1, world.exp2], world.final)
    assert checks["all_passed"], {k: v for k, v in checks.items() if not v}
    for key in (
        "training_and_validation_disjoint", "training_plus_validation_equals_all_train_good",
        "test_disjoint_from_training_by_content", "every_candidate_trained_on_exactly_the_training_subset",
        "experiment_module_cannot_list_test_images", "threshold_selection_receives_validation_data_only",
        "candidate_run_and_stage_selection_receive_no_test_data", "final_test_scored_after_selection_frozen",
        "final_test_scored_only_the_frozen_selection",
    ):
        assert checks[key] is True


def test_leakage_audit_detects_violations(world):
    exp1, exp2, final = world.exp1, world.exp2, world.final

    early = dataclasses.replace(final, events=[EVENT_FINAL_TEST_SCORED] + final.events[:-1])
    assert not audit_phase4_leakage(CATEGORY, world.split, [exp1, exp2], early)["final_test_scored_after_selection_frozen"]

    twice = dataclasses.replace(final, events=final.events + [EVENT_FINAL_TEST_SCORED])
    assert not audit_phase4_leakage(CATEGORY, world.split, [exp1, exp2], twice)["final_test_scored_exactly_once"]

    other = dataclasses.replace(final, locked_digest="deadbeef")
    assert not audit_phase4_leakage(CATEGORY, world.split, [exp1, exp2], other)["final_test_scored_only_the_frozen_selection"]

    cid, first = next(iter(exp1.outcomes.items()))
    bad = dict(exp1.outcomes, **{cid: dataclasses.replace(first, num_training_images=first.num_training_images + 1)})
    bad_exp = dataclasses.replace(exp1, outcomes=bad)
    assert not audit_phase4_leakage(CATEGORY, world.split, [bad_exp, exp2], final)[
        "every_candidate_trained_on_exactly_the_training_subset"
    ]

    leaked_split = dataclasses.replace(world.split, training=world.split.training + world.split.validation[:1])
    assert not audit_phase4_leakage(CATEGORY, leaked_split, [exp1, exp2], final)["training_and_validation_disjoint"]

    reused_wrongly = dataclasses.replace(
        first, reused_phase3_artifact=True,
        config=dataclasses.replace(first.config, training_config=dataclasses.replace(
            first.config.training_config, epochs=99)),
    )
    bad2 = dataclasses.replace(exp1, outcomes=dict(exp1.outcomes, **{cid: reused_wrongly}))
    assert not audit_phase4_leakage(CATEGORY, world.split, [bad2, exp2], final)[
        "baseline_reuse_only_for_exact_baseline_configuration"
    ]


def test_audit_checks_split_parity_with_the_baseline_validation_statistics(world):
    base = next(o for o in world.exp1.outcomes.values() if o.reused_phase3_artifact)
    good = {"mean_error": base.validation_mean, "std_error": base.validation_std}
    bad = {"mean_error": base.validation_mean * 1.01, "std_error": base.validation_std}
    assert audit_phase4_leakage(CATEGORY, world.split, [world.exp1], world.final, good)[
        "split_matches_phase3_baseline_validation_statistics"
    ]
    assert not audit_phase4_leakage(CATEGORY, world.split, [world.exp1], world.final, bad)[
        "split_matches_phase3_baseline_validation_statistics"
    ]


# ---------------------------------------------------------------------------
# Reproducibility comparison
# ---------------------------------------------------------------------------

def test_an_experiment_compared_with_itself_is_identical(world):
    assert compare_experiments(world.exp1, world.exp1)["all_identical"]


def test_two_independent_experiments_share_configs_and_report_every_field(world):
    result = compare_experiments(world.exp1, world.exp2)
    assert result["candidate_configurations_identical"]
    assert set(result) >= {
        "model_hashes_identical", "training_losses_identical", "validation_errors_identical",
        "threshold_candidates_identical", "stage_winners_identical", "selected_candidate_identical",
        "locked_threshold_identical", "locked_digest_identical", "all_identical",
    }


def test_compare_experiments_detects_a_difference(world):
    cid, o = next(iter(world.exp2.outcomes.items()))
    changed = dataclasses.replace(o, validation_errors=[e + 1e-9 for e in o.validation_errors])
    diff = compare_experiments(world.exp1, dataclasses.replace(world.exp2, outcomes=dict(world.exp2.outcomes, **{cid: changed})))
    assert not diff["validation_errors_identical"] and not diff["all_identical"]


# ---------------------------------------------------------------------------
# Baseline comparison + adoption rule
# ---------------------------------------------------------------------------

def test_sign_test_p_values():
    assert sign_test_p_value(0, 0) == 1.0
    assert sign_test_p_value(10, 0) == pytest.approx(1 / 1024)
    assert sign_test_p_value(5, 5) == pytest.approx(638 / 1024)
    assert sign_test_p_value(0, 4) == 1.0


def _preds(spec):
    """spec: list of (label, predicted) for 10 defective then 10 good, keyed by index."""
    return [
        {"file": f"f{i}", "defect_type": "d" if lbl else "good", "label": lbl, "reconstruction_error": 0.0, "predicted_label": p}
        for i, (lbl, p) in enumerate(spec)
    ]


def _cmp(base_spec, cand_spec):
    from app.ai.evaluation.category_phase3 import metrics_from_predictions

    base, cand = _preds(base_spec), _preds(cand_spec)
    fake = SimpleNamespace(predictions=cand, metrics=metrics_from_predictions(cand))
    return compare_to_baseline(base, metrics_from_predictions(base), fake)


DEFECTS, GOODS = 30, 10


def _spec(detected_defects, false_positives):
    return [(1, 1 if i < detected_defects else 0) for i in range(DEFECTS)] + [
        (0, 1 if i < false_positives else 0) for i in range(GOODS)
    ]


def test_clear_significant_improvement_is_retained():
    result = _cmp(_spec(10, 0), _spec(22, 0))
    assert result["decision"] == "OPTIMIZED CANDIDATE RETAINED (pending review)"
    assert result["paired_defective_images"]["newly_detected_by_candidate"] == 12
    assert all(result["adoption_criteria"].values())


def test_improvement_that_is_not_significant_keeps_the_baseline():
    result = _cmp(_spec(10, 0), _spec(13, 0))  # 3 wins, 0 losses -> p = 0.125
    assert result["adoption_criteria"]["A1_f1_strictly_higher"]
    assert not result["adoption_criteria"]["A3_recall_gain_significant_sign_test_p_lt_0_05"]
    assert result["decision"] == "PHASE 3 BASELINE RETAINED"


def test_excess_false_positives_keep_the_baseline_even_with_higher_f1():
    result = _cmp(_spec(10, 0), _spec(28, 2))  # FPR 20% > 10%
    assert result["adoption_criteria"]["A1_f1_strictly_higher"]
    assert not result["adoption_criteria"]["A2_false_positive_rate_at_most_10_percent"]
    assert result["decision"] == "PHASE 3 BASELINE RETAINED"


def test_worse_candidate_keeps_the_baseline_and_reports_negative_deltas():
    result = _cmp(_spec(20, 0), _spec(10, 0))
    assert result["decision"] == "PHASE 3 BASELINE RETAINED"
    assert result["metrics"]["recall"]["absolute_change"] < 0
    assert result["metrics"]["false_negatives"]["absolute_change"] == 10


def test_comparison_rejects_mismatched_test_images():
    base = _preds(_spec(10, 0))
    cand = _preds(_spec(10, 0))[:-1]
    with pytest.raises(ValueError):
        compare_to_baseline(base, {}, SimpleNamespace(predictions=cand, metrics={}))


# ---------------------------------------------------------------------------
# Artifact metadata / report
# ---------------------------------------------------------------------------

def test_candidate_artifact_metadata_matches_the_files(world):
    import hashlib

    for o in world.exp1.outcomes.values():
        data = o.model_path.read_bytes()
        assert o.model_metadata.md5 == hashlib.md5(data).hexdigest()
        assert o.model_metadata.sha256 == hashlib.sha256(data).hexdigest()
        assert o.model_metadata.artifact_size_bytes == len(data)


def test_report_records_every_candidate_and_never_claims_serving(world):
    report = build_category_phase4_report(
        category=CATEGORY, git_branch="t", git_head="0" * 40, baseline={"md5": world.baseline_md5},
        experiment=world.exp1, reproducibility=compare_experiments(world.exp1, world.exp2), final=world.final,
        comparison=None, decision="PHASE 3 BASELINE RETAINED",
        leakage=audit_phase4_leakage(CATEGORY, world.split, [world.exp1, world.exp2], world.final),
        experiment2_total_ms=1.0, protected_artifacts={"unchanged": True},
    )
    json.dumps(report)
    assert len(report["candidates"]) == len(world.exp1.outcomes)
    first = report["candidates"][0]
    assert {"md5", "sha256", "size_bytes"} <= set(first["model"])
    assert len(first["validation"]["threshold_candidates"]) == 9
    assert {"coefficient_of_variation", "mean_error", "std_error", "min_error", "max_error"} <= set(first["validation"])
    assert {"training_duration_s", "model_load_ms", "validation_ms_per_image_total"} <= set(first["timing"])
    assert "not registered" in report["serving_status"].lower()
    assert report["frozen_selection"]["digest"] == world.exp1.locked.digest


def test_phase4_never_registers_serving():
    from app.ai.inference.serving import SERVING_CONFIGS

    assert set(SERVING_CONFIGS) == {"bottle"}
    for module in (engine,):
        assert module_references(module, {"SERVING_CONFIGS", "get_serving_config"}) == set()
