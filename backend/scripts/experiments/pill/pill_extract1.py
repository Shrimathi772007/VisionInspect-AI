import json, pathlib, numpy as np

root = pathlib.Path("ai_models/pill/model_family_study")
S = json.loads((root / "reports/model_family_study.json").read_bytes().decode())
L = json.loads((root / "selection_lock.json").read_bytes().decode())
sel = S["selection"]
print("counts", S["counts"])
print("fingerprints: dataset", S["dataset_fingerprint_sha256"], "| splits", S["splits_sha256"], "| registry", S["registry_fingerprint"], "| synthetic", S["synthetic"]["digest_sha256"])
print("normal_file_list", S["normal_file_list_sha256"])
print("split_audit", S["split_audit"]); print("scheme_definitions", S["scheme_definitions"])
print("synthetic", S["synthetic"])
print("backbone", S["backbone"])
print("study_seconds", S["study_seconds"], "input_build", S["input_build_seconds"], "git", S["git_head"][:7])

rows = sel["rows"]
print("\n== selection ==")
print("gate passers", len(sel["gate_passers"]), "of", len(rows), "| preferred floor met:", sel["preferred_floor_met"], "| tied_at_end", sel["tied_at_end"])
pref = sorted((r for r in rows if r["passes_gate"] and r["synthetic_recall"] >= 0.75), key=lambda r: (r["worst_fold_fpr"], r["candidate"], r["policy"]))
print("preferred (gate + recall>=0.75):", len(pref))
lowest = min(round(r["worst_fold_fpr"], 12) for r in pref)
top = [r for r in pref if round(r["worst_fold_fpr"], 12) == lowest]
print("lowest worst-fold FPR among preferred:", lowest, "-> configs:", [(r["candidate"], r["policy"], round(r["synthetic_recall"], 4), round(r["threshold_cv"], 5)) for r in top])
print("reasoning:", sel["reasoning"])
print("winner_row:", sel["winner_row"])
per_cand = {}
for r in rows:
    d = per_cand.setdefault(r["candidate"], {"gate": 0, "pref": 0})
    d["gate"] += r["passes_gate"]; d["pref"] += bool(r["passes_gate"] and r["synthetic_recall"] >= 0.75)
print("per-candidate gate/preferred counts:", per_cand)

print("\n== per candidate x policy stability (all 64) ==")
print("cand|policy|thr_mean|thr_std|thr_cv|meanFPR|medianFPR|worstFPR|synRec|gate")
for cid, res in S["candidates"].items():
    for pol, e in res["summary"]["policies"].items():
        fprs = [x["fpr"] for sch in e["per_scheme"].values() for x in sch["splits"]]
        assert len(fprs) == 30
        t = e["pooled_threshold"]
        print(f"{cid}|{pol}|{t['mean']:.5g}|{t['std']:.4g}|{t['cv']:.4f}|{e['mean_fpr']:.4f}|{np.median(fprs):.4f}|{e['worst_fold_fpr']:.4f}|{e['synthetic_recall']:.3f}|{e['worst_fold_fpr'] <= 0.10}")
print("\n== normal score distribution (LOIO, all normals) + score CV ==")
for cid, res in S["candidates"].items():
    d = res["summary"]["normal_score_distribution"] if "summary" in res else res["normal_score_distribution"]
    print(cid, "mean", round(d["mean"], 5), "std", round(d["std"], 5), "CV", round(d["cv"], 4), "median", round(d["median"], 5))
print("\n== synthetic AUROC per candidate ==")
for cid, res in S["candidates"].items():
    sm = res["summary"] if "summary" in res else res
    print(cid, round(sm["synthetic_auroc"], 4))
w = S["candidates"][sel["winner_candidate"]]; ws = w["summary"] if "summary" in w else w
we = ws["policies"][sel["winner_policy"]]
print("\n== selected synthetic recall breakdown ==", sel["winner"])
print("overall", we["synthetic_recall"], "\nby type", we["synthetic_recall_by_type"], "\nby severity", we["synthetic_recall_by_severity"])
print("canonical threshold", we["canonical_threshold"], "full-pool", we["full_pool_threshold"], "pooled thr", we["pooled_threshold"])
print("per-scheme worst FPR:", {k: round(v["worst_fold_fpr"], 4) for k, v in we["per_scheme"].items()}, "mean", {k: round(v["mean_fpr"], 4) for k, v in we["per_scheme"].items()})
print("selected: all 30 split FPRs")
for k, v in we["per_scheme"].items():
    print(" ", k, [(x["split"], x["held_out_size"], x["false_positives"], round(x["fpr"], 4), round(x["threshold"], 3)) for x in v["splits"]])
print("\n== lock ==")
print({k: L[k] for k in ("category", "experiment_id", "selected_candidate", "selected_policy", "selected_threshold", "selection_rule_sha256", "lock_digest", "timestamp_utc", "registry_fingerprint", "split_fingerprint_sha256", "normal_dataset_fingerprint_sha256", "preferred_synthetic_floor_met", "final_test_data_used_for_selection")})
print("lock backbone", L["backbone"]["sha256"], "| selected artifact", L["selected_artifact"]["sha256"], L["selected_artifact"]["md5"], L["selected_artifact"]["size_bytes"])
print("synthetic digest in lock", L["synthetic_diagnostic_results"]["synthetic_digest_sha256"])
print("prior", L["prior_phase_artifacts"])
print("\n== candidate artifacts (sha256, size, fit/prep seconds) ==")
for cid in S["candidates"]:
    m = json.loads((root / "candidates" / cid / "candidate_manifest.json").read_bytes().decode())
    a = m["artifact"]
    print(cid, a["sha256"][:12], a["size_bytes"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items() if k in ("fit_seconds", "prep_seconds", "score_ms_per_image", "load_ms", "fitted_on", "training_image_count")})
    print("   manifest keys:", sorted(k for k in m if k not in ("config", "artifact", "feature_config", "thresholds_full_pool")))
