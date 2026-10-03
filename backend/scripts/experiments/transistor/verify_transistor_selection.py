"""Read-only mechanical verification of the saved Transistor model-family selection (no image is read; nothing is written
except the verification record in this scratch folder).

1. fingerprints/hashes of the saved study vs the pre-study declaration and the live framework;
2. the 64 rows are recomputed from the saved per-candidate summaries and compared with the saved selection rows;
3. the winner is recomputed (a) with the framework's select_configuration and (b) with an independent step-by-step
   reimplementation of the documented rule, producing the full trace.
"""

import hashlib
import json
import sys
from pathlib import Path

BACKEND = Path(r"C:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
sys.path.insert(0, str(BACKEND))

from app.ai.evaluation import model_family_study as mfs  # noqa: E402
from app.ai.evaluation import model_family_pipeline as mfp  # noqa: E402

CATEGORY = "transistor"
EXPECTED = ("knn_l23_256", "percentile_99")
SCRATCH = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def independent_select(study: dict, specs) -> dict:
    by_id = {s.candidate_id: s for s in specs}
    rows = []
    for cid, res in study["candidates"].items():
        for pol, e in res["summary"]["policies"].items():
            rows.append({"id": f"{cid}|{pol}", "cid": cid, "pol": pol, "worst": e["worst_fold_fpr"], "recall": e["synthetic_recall"],
                         "cv": e["pooled_threshold"]["cv"], "thr": e["full_pool_threshold"]})
    trace = {"configurations": len(rows)}
    gate = [r for r in rows if r["worst"] <= 0.10 + 1e-12]
    trace["1_gate_passers"] = sorted(r["id"] for r in gate)
    floor = [r for r in gate if r["recall"] >= 0.75 - 1e-12]
    trace["2_floor_met"] = bool(floor)
    trace["2_floor_passers"] = sorted(r["id"] for r in floor)
    trace["2_excluded_by_floor"] = sorted(r["id"] for r in gate if r not in floor)
    pool = floor or gate
    best = min(round(r["worst"], 12) for r in pool)
    pool = [r for r in pool if round(r["worst"], 12) == best]
    trace["3_lowest_worst_fold_fpr"] = best
    trace["3_tied"] = [{"id": r["id"], "worst": r["worst"], "recall": r["recall"], "cv": r["cv"]} for r in pool]
    top = max(r["recall"] for r in pool)
    pool = [r for r in pool if r["recall"] >= top - 0.03 - 1e-12]
    trace["4_recall_band"] = {"best": top, "band_floor": top - 0.03, "kept": [r["id"] for r in pool]}
    low = min(r["cv"] for r in pool)
    pool = [r for r in pool if r["cv"] <= low + 0.01 + 1e-12]
    trace["5_cv_band"] = {"best": low, "band_ceiling": low + 0.01, "kept": [r["id"] for r in pool]}
    pool.sort(key=lambda r: ((0 if by_id[r["cid"]].family == "gaussian" else 1, by_id[r["cid"]].feature_dim, by_id[r["cid"]].image_size), r["cid"], r["pol"]))
    trace["6_7_simplicity_then_id_order"] = [{"id": r["id"], "key": [0 if by_id[r["cid"]].family == "gaussian" else 1,
                                                                   by_id[r["cid"]].feature_dim, by_id[r["cid"]].image_size]} for r in pool]
    trace["winner"] = pool[0]["id"]
    trace["winner_threshold"] = pool[0]["thr"]
    return trace


def main() -> None:
    study_path = mfp.study_report_path(CATEGORY)
    decl = json.loads((mfp.study_root(CATEGORY) / "study_declaration.json").read_text(encoding="utf-8"))
    study = json.loads(study_path.read_text(encoding="utf-8"))
    specs = mfs.registry()
    saved = study["selection"]
    fw = mfs.select_configuration({c: r["summary"] for c, r in study["candidates"].items()}, specs)
    ind = independent_select(study, specs)
    saved_rows = {f"{r['candidate']}|{r['policy']}": r for r in saved["rows"]}
    fw_rows = {f"{r['candidate']}|{r['policy']}": r for r in fw["rows"]}
    checks = {
        "study_artifact_sha256": sha(study_path.read_bytes()) == "d3781784551d162b12fa88298c8a7c0e1a458b59837b004769929e2c55c21abd",
        "registry_fingerprint_study_eq_live_eq_declared": study["registry_fingerprint"] == mfs.registry_fingerprint(specs) == decl["candidate_registry"]["registry_fingerprint"],
        "candidate_configs_study_eq_live": [c for c in study["registry"]] == [s.config() for s in specs],
        "policy_registry_sha256_eq_declared": sha(json.dumps(study["policies"] if "policies" in study else mfs.policy_ids()).encode()) == decl["policy_registry"]["policy_registry_sha256"],
        "policies_eq_live": list(study.get("policies", mfs.policy_ids())) == mfs.policy_ids(),
        "selection_rule_sha256_eq_declared": sha(study["declared_rule"].encode("utf-8")) == decl["selection_rule"]["declared_rule_sha256"] == sha(mfs.__doc__.encode("utf-8")),
        "dataset_fingerprint_eq_declared": study["dataset_fingerprint_sha256"] == decl["development_data"]["dataset_fingerprint_sha256"],
        "split_fingerprint_eq_declared": study["splits_sha256"] == decl["held_out_split_definition"]["expected_split_fingerprint_sha256"],
        "64_configurations": len(saved["rows"]) == 64 and len(fw["rows"]) == 64,
        "saved_rows_eq_recomputed_rows": saved_rows == fw_rows,
        "gate_passers_29": len(saved["gate_passers"]) == 29 == len(ind["1_gate_passers"]) and sorted(saved["gate_passers"]) == ind["1_gate_passers"],
        "floor_met_1": saved["preferred_floor_met"] is True and len(ind["2_floor_passers"]) == 1,
        "framework_reselection_eq_saved": fw["winner"] == saved["winner"] and fw["threshold"] == saved["threshold"],
        "independent_reselection_eq_saved": ind["winner"] == saved["winner"] and ind["winner_threshold"] == saved["threshold"],
        "winner_eq_expected": saved["winner"] == "|".join(EXPECTED),
    }
    out = {"checks": checks, "all_passed": all(checks.values()), "saved_winner": saved["winner"], "threshold": saved["threshold"],
           "threshold_repr": repr(saved["threshold"]), "framework_reasoning": saved["reasoning"], "independent_trace": ind,
           "synthetic_digest": study["synthetic"]["digest_sha256"]}
    (SCRATCH / "selection_verification.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    if not out["all_passed"]:
        sys.exit("STOP: selection verification failed")


if __name__ == "__main__":
    main()
