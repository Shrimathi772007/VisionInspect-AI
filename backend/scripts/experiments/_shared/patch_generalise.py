import re

# ---- model_family_final.py: comparison with whichever earlier ConvAE phases exist for the category ----
p = "app/ai/evaluation/model_family_final.py"
s = open(p, encoding="utf-8").read()
start = s.index("    prior = _prior_results(category, threshold)")
end = s.index("    train_paths = {s.path for s in train_samples}")
s = s[:start] + "    comparison = _build_comparison(category, metrics, threshold, auroc, per_defect)\n\n" + s[end:]
s = s.replace("from app.ai.evaluation.category_phase3_heldout_final import _prior_results\n", "")

helper = '''def _optional_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _metric_row(m: dict, threshold: float, auroc: float) -> dict:
    return {"threshold": threshold, "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1_score": m["f1_score"],
            "false_defect_detection_rate": m["false_defect_detection_rate"], "auroc": auroc, "true_negatives": m["true_negatives"],
            "false_positives": m["false_positives"], "false_negatives": m["false_negatives"], "true_positives": m["true_positives"]}


def _build_comparison(category: str, metrics: dict, threshold: float, auroc: float, per_defect: dict) -> dict:
    """Read-only comparison with the category's earlier ConvAutoencoder phases - whichever of Phase 1/2/3 exist."""
    root = artifacts.ARTIFACTS_ROOT / category
    p1 = _optional_json(root / "phase1_baseline" / "reports" / "phase1_report.json")
    p2 = _optional_json(root / "phase2_threshold" / "reports" / "final_test_result.json")
    p3 = _optional_json(root / "phase3_validation" / "reports" / "final_test_result.json")
    comparison, earlier = {}, {}
    if p1:
        r1 = p1["run1"]
        comparison["phase1_convae"] = _metric_row(r1["metrics"], r1["threshold"]["value"], r1["separation"]["auroc_descriptive_only"])
        earlier["phase1"] = r1["per_defect"]
    if p2:
        scores = p2["predictions"]
        comparison["phase2_convae"] = _metric_row(p2["metrics"], p2["locked_threshold"], float(roc_auc_score(
            [q["label"] for q in scores], [q["reconstruction_error"] for q in scores])))
        earlier["phase2"] = p2["per_defect"]
    if p3:
        comparison["phase3_convae"] = _metric_row(p3["metrics"], p3["locked_threshold"], p3["auroc_descriptive_only"])
        earlier["phase3"] = p3["per_defect"]
    comparison["alternative"] = _metric_row(metrics, threshold, auroc)
    comparison["per_defect"] = {n: {**{phase: d[n]["recall"] for phase, d in earlier.items() if n in d}, "alternative": v["recall"],
                                    "alternative_detected": v["detected"], "total": v["total"]} for n, v in per_defect.items()}
    return comparison


'''
anchor = "def evaluate_locked_on_final_test(category: str) -> dict:"
assert anchor in s
s = s.replace(anchor, helper + anchor, 1)
open(p, "w", encoding="utf-8").write(s)

# ---- model_family_pipeline.py: category-derived experiment id (unchanged text for leather) ----
p = "app/ai/evaluation/model_family_pipeline.py"
s = open(p, encoding="utf-8").read()
old = '''"experiment_id": f"{EXPERIMENT_ID_PREFIX}-{category}-{study['splits_sha256'][:12]}",'''
new = '''"experiment_id": f"{category.replace('_', '-')}-model-family-{category}-{study['splits_sha256'][:12]}",'''
assert old in s
s = s.replace(old, new)
s = s.replace('EXPERIMENT_ID_PREFIX = "leather-model-family"\n', "")
open(p, "w", encoding="utf-8").write(s)

# ---- model_family_study.py: category-agnostic clarification of the sizes in the pre-declared rule ----
p = "app/ai/evaluation/model_family_study.py"
s = open(p, encoding="utf-8").read()
old = "The LOCKED threshold is the winning policy applied to the leave-one-image-out scores of ALL 245 normal images under"
new = ("CATEGORY NOTE  The numbers above are for Leather (245 normals -> 196 calibration / 49 held out, 30 splits). The identical\n"
       "rule is applied unchanged to any other category with n normal images: each split holds out round(0.2 * n) images\n"
       "(Metal Nut: n = 220 -> 176 calibration / 44 held out, 30 splits); the registry, policies, gate and tie bands do not change.\n"
       "The LOCKED threshold is the winning policy applied to the leave-one-image-out scores of ALL n normal images under")
assert old in s
s = s.replace(old, new, 1)
s = s.replace("the final model fitted on all 245. Normal-only", "the final model fitted on all n. Normal-only")
open(p, "w", encoding="utf-8").write(s)
print("patched")
