import re

src = open("tests/test_ai_metal_nut_model_family.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old)
    src = src.replace(old, new)


rep('"""Metal Nut alternative model-family study (app.ai.evaluation.model_family_study / _pipeline / _final applied to `metal_nut`).',
    '"""Pill alternative model-family study (app.ai.evaluation.model_family_study / _pipeline / _final applied to `pill`).')
rep("220-image split geometry", "267-image split geometry")
rep("tiny synthetic `metal_nut`\nworld", "tiny synthetic `pill`\nworld")
rep('CATEGORY = "metal_nut"\nREAL_EXPECTED = {"train_good": 220, "test_good": 22, "test_defective": 93, "test_total": 115}\nREAL_TYPES = {"bent": 25, "color": 22, "flip": 23, "good": 22, "scratch": 23}\nTINY_EXPECTED = {"train_good": 20, "test_good": 4, "test_defective": 8, "test_total": 12}',
    'CATEGORY = "pill"\nREAL_EXPECTED = {"train_good": 267, "test_good": 26, "test_defective": 141, "test_total": 167}\n'
    'REAL_TYPES = {"color": 25, "combined": 17, "contamination": 21, "crack": 26, "faulty_imprint": 19, "good": 26, "pill_type": 9, "scratch": 24}\n'
    'DEFECT_NAMES = ("color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch")\n'
    'TINY_EXPECTED = {"train_good": 20, "test_good": 4, "test_defective": 14, "test_total": 18}\nTINY_TYPES = {"good": 4, **{name: 2 for name in DEFECT_NAMES}}')
rep('reason="MVTec metal_nut not present"', 'reason="MVTec pill not present"')

# --- replace the split-geometry and synthetic-count tests with Pill's ---
start = src.index("def test_metal_nut_splits_are_30")
end = src.index("# ---------------------------------------------------------------------------\n# Real dataset (no fitting)")
pill_split_tests = '''def test_pill_splits_are_30_deterministic_disjoint_splits_with_derived_sizes():
    a, b = mfs.scheme_splits(267), mfs.scheme_splits(267)
    assert sum(len(v) for v in a.values()) == 30 == 15 + 5 + 10 and mfs.splits_fingerprint(a) == mfs.splits_fingerprint(b)
    assert [s["name"] for s in a["holdout_80_20"]] == [f"seed{i}" for i in range(42, 52)]
    assert sorted(a) == ["contiguous_kfold5", "holdout_80_20", "random_kfold5_seed42", "random_kfold5_seed43", "random_kfold5_seed44"]
    audit = mfs.split_audit(267, a)
    assert audit["calibration_and_held_out_disjoint_in_every_split"] and audit["every_split_covers_the_pool"] and audit["kfold_held_out_folds_partition_the_pool"]
    # the established rule: repeated 80/20 holds out round(0.2 * 267) = 53; 5 does not divide 267, so folds hold out 54 or 53
    assert {(len(s["calibration"]), len(s["held_out"])) for s in a["holdout_80_20"]} == {(214, 53)}
    for name in ("random_kfold5_seed42", "random_kfold5_seed43", "random_kfold5_seed44", "contiguous_kfold5"):
        assert sorted(len(s["held_out"]) for s in a[name]) == [53, 53, 53, 54, 54]
        assert all(len(s["calibration"]) + len(s["held_out"]) == 267 for s in a[name])
    assert audit["held_out_sizes"] == [53, 54] and audit["calibration_sizes"] == [213, 214]
    assert [len(s["held_out"]) for s in a["contiguous_kfold5"]] == [54, 54, 53, 53, 53]
    assert a["holdout_80_20"][0]["held_out"].tolist() == sorted(np.random.default_rng(42).permutation(267)[:53].tolist())
    assert mfs.splits_fingerprint(a) not in (mfs.splits_fingerprint(mfs.scheme_splits(245)), mfs.splits_fingerprint(mfs.scheme_splits(220)))


def test_pill_synthetic_diagnostic_size_is_derived_from_the_canonical_held_out_split_times_15():
    canonical = next(s for s in mfs.scheme_splits(267)[mfs.CANONICAL_SPLIT[0]] if s["name"] == mfs.CANONICAL_SPLIT[1])
    held_out, variants = len(canonical["held_out"]), len(mfs.DEFECT_TYPES) * len(mfs.SEVERITIES)
    assert (len(canonical["calibration"]), held_out, variants) == (214, 53, 15)
    # 795, not the 810 (= 54 x 15) the study brief expected: 54 would need ceil(0.2 * n), which is NOT the established rule
    assert held_out * variants == 795 != 54 * variants
    # no other category's size is inherited: Leather 49 x 15 = 735, Metal Nut 44 x 15 = 660
    sizes = {n: len(mfs.scheme_splits(n)[mfs.CANONICAL_SPLIT[0]][0]["held_out"]) * variants for n in (245, 220, 267)}
    assert sizes == {245: 735, 220: 660, 267: 795}
    declared = mfs.__doc__
    assert "held_out_images x 15" in declared and "Pill: n = 267" in declared and "53 x 15 = 795" in declared
    assert "Metal Nut: 44 x 15 = 660" in declared and "Leather: 49 x 15 = 735" in declared and "= 735 synthetic defects" not in declared


def test_pill_expected_counts_and_defect_types_are_pinned_in_the_cli():
    script = Path(mfs.__file__).parents[3] / "scripts" / "run_model_family_study.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    table = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "EXPECTED_BY_CATEGORY" for t in n.targets))
    assert table["pill"] == {**REAL_EXPECTED, "test_by_type": REAL_TYPES}
    assert table["leather"]["train_good"] == 245 and table["metal_nut"]["train_good"] == 220  # earlier categories untouched
    assert sum(v for k, v in REAL_TYPES.items() if k != "good") == 141 and sum(REAL_TYPES.values()) == 167


'''
src = src[:start] + pill_split_tests + src[end:]

# --- real-dataset tests ---
rep('''    assert (counts["train_good"], counts["test_good"], counts["test_defective"], counts["test_total"]) == (220, 22, 93, 115)
    assert counts["test_by_type"] == REAL_TYPES
    with pytest.raises(ValueError, match="test_defective"):
        pipe.count_dataset_entries(CATEGORY, {**REAL_EXPECTED, "test_defective": 92})''',
    '''    assert (counts["train_good"], counts["test_good"], counts["test_defective"], counts["test_total"]) == (267, 26, 141, 167)
    assert counts["test_by_type"] == REAL_TYPES and set(counts["test_by_type"]) - {"good"} == set(DEFECT_NAMES)
    with pytest.raises(ValueError, match="test_defective"):
        pipe.count_dataset_entries(CATEGORY, {**REAL_EXPECTED, "test_defective": 140})
    with pytest.raises(ValueError, match="test_by_type"):  # an unexpected or missing defect folder stops the study
        pipe.count_dataset_entries(CATEGORY, {**REAL_EXPECTED, "test_by_type": {**REAL_TYPES, "crack": 27}})
    assert pipe.count_dataset_entries(CATEGORY, {**REAL_EXPECTED, "test_by_type": REAL_TYPES}) == counts''')
rep("assert len(samples) == 220 and", "assert len(samples) == 267 and")
rep("assert len(discover_test_samples(CATEGORY)) == 115", "assert len(discover_test_samples(CATEGORY)) == 167")

# --- serving must not register pill ---
rep("def test_final_module_only_scores_the_locked_candidate_and_serving_has_no_metal_nut():",
    "def test_final_module_only_scores_the_locked_candidate_and_serving_has_no_pill():")
rep('    assert "metal_nut" not in serving and "metal nut" not in serving', '    assert "pill" not in serving.replace("pillow", "")')

# --- tiny world ---
rep("End to end on one tiny synthetic metal_nut world", "End to end on one tiny synthetic pill world")
rep('    per_defect = {t: {"recall": 0.0, "detected": 0, "total": 2} for t in ("bent", "color", "flip", "scratch")}',
    '    per_defect = {t: {"recall": 0.0, "detected": 0, "total": 2} for t in DEFECT_NAMES}')
rep('"true_negatives": 4, "false_positives": 0, "false_negatives": 8, "true_positives": 0}',
    '"true_negatives": 4, "false_positives": 0, "false_negatives": 14, "true_positives": 0}')
rep('    for k, defect in enumerate(("bent", "color", "flip", "scratch")):', '    for k, defect in enumerate(DEFECT_NAMES):')
rep('tmp_path_factory.mktemp("mn_world")', 'tmp_path_factory.mktemp("pill_world")')
rep('assert "Metal Nut" in synthetic["label"] and "Leather" not in synthetic["label"]',
    'assert "Pill" in synthetic["label"] and "Leather" not in synthetic["label"] and "Metal Nut" not in synthetic["label"]')
rep('assert world["study"]["counts"]["test_by_type"] == {"bent": 2, "color": 2, "flip": 2, "good": 4, "scratch": 2}',
    'assert world["study"]["counts"]["test_by_type"] == TINY_TYPES')
rep('manifest["training_image_count"] == 20',
    'manifest["training_image_count"] == 20 and manifest["fitted_on"] == "all 20 train/good images"  # derived, not "245"')
rep('lock["experiment_id"].startswith("metal-nut-model-family-metal_nut-")', 'lock["experiment_id"].startswith("pill-model-family-pill-")')
rep('"selected_candidate", "selected_threshold", "selected_artifact", "timestamp_utc", "lock_digest"):',
    '"selected_candidate", "selected_policy", "selected_threshold", "selected_artifact", "timestamp_utc", "lock_digest",\n                "selection_rule_sha256", "prior_phase_artifacts"):')
rep('    assert lock["synthetic_diagnostic_results"]["label"] == "DIAGNOSTIC ONLY"',
    '    assert lock["selection_rule_sha256"] == hashlib.sha256(mfs.__doc__.encode("utf-8")).hexdigest()  # rule version pinned in the lock\n'
    '    assert lock["synthetic_diagnostic_results"]["synthetic_digest_sha256"] == json.loads(pipe.study_report_path(CATEGORY).read_text())["synthetic"]["digest_sha256"]\n'
    '    assert lock["synthetic_diagnostic_results"]["label"] == "DIAGNOSTIC ONLY"')
rep('assert m["true_positives"] + m["false_negatives"] == 8 and m["true_negatives"] + m["false_positives"] == 4',
    'assert m["true_positives"] + m["false_negatives"] == 14 and m["true_negatives"] + m["false_positives"] == 4')
rep('assert set(result["per_defect"]) == {"bent", "color", "flip", "scratch"}',
    'assert set(result["per_defect"]) == set(DEFECT_NAMES) and sum(v["total"] for v in result["per_defect"].values()) == 14')
rep('assert set(result["comparison"]["per_defect"]["bent"]) ==', 'assert set(result["comparison"]["per_defect"]["crack"]) ==')
rep('''    assert final.verify_lock_unchanged_since_final_test(CATEGORY)
    with pytest.raises(FileExistsError):
        final.evaluate_locked_on_final_test(CATEGORY)''',
    '''    assert result["selection_lock_file_sha256"] == file_hashes(pipe.lock_path(CATEGORY))["sha256"]  # the result references the exact lock file
    assert result["artifact_sha256"] == lock["selected_artifact"]["sha256"] and result["backbone_sha256"] == RESNET18_SHA256
    assert final.verify_lock_unchanged_since_final_test(CATEGORY)
    with pytest.raises(FileExistsError):
        final.evaluate_locked_on_final_test(CATEGORY)''')
rep("only Phase 1 exists for Metal Nut", "only Phase 1 exists for Pill")
open("tests/test_ai_pill_model_family.py", "w", encoding="utf-8").write(src)
print("written", len(src.splitlines()), "lines")
print("leftover metal mentions:", [l.strip()[:100] for l in src.splitlines() if "metal" in l.lower() and "Metal Nut:" not in l and "Metal Nut 44" not in l])
