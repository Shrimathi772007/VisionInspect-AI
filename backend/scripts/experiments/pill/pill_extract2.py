import json, math, pathlib, hashlib
import numpy as np
from sklearn.metrics import roc_auc_score

root = pathlib.Path("ai_models/pill/model_family_study")
R = json.loads((root / "reports/final_test_result.json").read_bytes().decode())
print("keys:", sorted(R))
skip = ("predictions", "comparison", "distributions", "per_defect", "leakage_audit")
print({k: v for k, v in R.items() if k not in skip and not isinstance(v, (list, dict))})
print("metrics:", R["metrics"])
print("auroc", R["auroc"], "gate", R.get("gate_classification"))
for n, d in R["per_defect"].items():
    print("per_defect", n, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()})
print("distributions keys:", list(R["distributions"]))
for k, v in R["distributions"].items():
    if isinstance(v, dict) and "count" in v:
        print("dist", k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
    else:
        print("dist", k, json.dumps(v)[:600])
print("leakage:", R["leakage_audit"])
print("comparison:", json.dumps(R["comparison"], indent=0)[:2500])
for k in R:
    if "time" in k or "ms" in k or "second" in k or "timing" in k:
        print("TIMING", k, R[k])

P = R["predictions"]; thr = R["locked_threshold"]
g = np.array([p["reconstruction_error"] for p in P if p["label"] == 0]); b = np.array([p["reconstruction_error"] for p in P if p["label"] == 1])
print("\nthreshold", thr, "| good above", int((g > thr).sum()), "/", len(g), "| defective above", int((b > thr).sum()), "/", len(b))
print("threshold pct among good", (g <= thr).mean() * 100, "among defective", (b <= thr).mean() * 100)
print("min defective", b.min(), "max good", g.max(), "| overlap:", b.min() <= g.max(), "| defective <= max good:", int((b <= g.max()).sum()), "| good >= min defective:", int((g >= b.min()).sum()), "| defective above max good", int((b > g.max()).sum()))
print("defective <= good median", int((b <= np.median(g)).sum()))
print("FP files:", [p["file"] for p in P if p["label"] == 0 and p["predicted_label"] == 1])
print("missed count", len([p for p in P if p["label"] == 1 and p["predicted_label"] == 0]))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


m = R["metrics"]
fp, tn, fn, tp = m["false_positives"], m["true_negatives"], m["false_negatives"], m["true_positives"]
print("Wilson95 FPR", fp, "/", fp + tn, [round(x * 100, 2) for x in wilson(fp, fp + tn)], "| recall", tp, "/", tp + fn, [round(x * 100, 2) for x in wilson(tp, tp + fn)])
print("Wilson95 per-defect recall:")
for n, d in R["per_defect"].items():
    print("  ", n, d["detected"], "/", d["total"], [round(x * 100, 1) for x in wilson(d["detected"], d["total"])])
print("per-type AUROC vs test-good (post-lock, descriptive):")
for t in sorted({p["defect_type"] for p in P if p["label"] == 1}):
    x = np.array([p["reconstruction_error"] for p in P if p["defect_type"] == t])
    print("  ", t, round(roc_auc_score([0] * len(g) + [1] * len(x), list(g) + list(x)), 3), "mean", round(x.mean(), 3), "min", round(x.min(), 3), "median", round(np.median(x), 3))
print("missed by type:", {t: [p["file"] for p in P if p["defect_type"] == t and p["predicted_label"] == 0] for t in sorted({p["defect_type"] for p in P if p["label"] == 1})})

lock_sha = hashlib.sha256((root / "selection_lock.json").read_bytes()).hexdigest()
print("\nresult.selection_lock_file_sha256 == lock file:", R["selection_lock_file_sha256"] == lock_sha, lock_sha)
for f in ("reports/final_test_result.json", "reports/model_family_study.json", "reports/reproducibility.json", "reports/model_family_report.json", "selection_lock.json"):
    print(f, hashlib.sha256((root / f).read_bytes()).hexdigest(), (root / f).stat().st_size)
rep = json.loads((root / "reports/reproducibility.json").read_bytes().decode()); print("repro", rep)
