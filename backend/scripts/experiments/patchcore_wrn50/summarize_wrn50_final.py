"""Summary of the nine WRN-50-2 final tests (FINAL_TEST_DECLARATION.md), from the saved final_test/ outputs only.

Usage (from backend/):
    python scripts/experiments/patchcore_wrn50/summarize_wrn50_final.py

Reads each category's final_test_result.json and leakage_audit.json (no dataset access, no scoring) and writes
ai_models/patchcore_wrn50_summary.json and ai_models/patchcore_wrn50_summary.md: new vs previous model per
category, gate counts, the mean image-level average precision (mAP) and the mean AUROC.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent.parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(HERE))

import wrn50_final_logic as fl  # noqa: E402
from app.ai.training.artifacts import ARTIFACTS_ROOT  # noqa: E402

# The previous final model's test numbers per category (recall, FPR), as recorded in the project documentation.
PREVIOUS = {
    "bottle": {"recall": 0.7302, "fpr": 0.30}, "capsule": {"recall": 0.2661, "fpr": 0.0},
    "zipper": {"recall": 0.6639, "fpr": 0.0938}, "wood": {"recall": 0.9833, "fpr": 0.2105},
    "grid": {"recall": 0.2281, "fpr": 0.0}, "pill": {"recall": 0.5532, "fpr": 0.0},
    "carpet": {"recall": 0.9775, "fpr": 0.3929, "note": "latest"}, "screw": {"recall": 0.0756, "fpr": 0.0},
    "hazelnut": {"recall": 0.3714, "fpr": 0.0},
}


def main() -> int:
    rows, missing = [], []
    for category in fl.RUN_ORDER:
        out = fl.lock_dir(category, ARTIFACTS_ROOT) / fl.FINAL_DIRNAME
        if not (out / fl.RESULT).is_file():
            missing.append(category)
            continue
        r = json.loads((out / fl.RESULT).read_text(encoding="utf-8"))
        audit = json.loads((out / "leakage_audit.json").read_text(encoding="utf-8"))
        m = r["metrics"]
        lowest = sorted(r["per_defect"].items(), key=lambda kv: (kv[1]["recall"], kv[0]))[:2]
        rows.append({
            "category": category, "model": r["model"], "gate": r["gate_classification"],
            "recall": m["recall"], "precision": m["precision"], "f1": m["f1_score"], "accuracy": m["accuracy"],
            "fpr": m["false_positive_rate"], "auroc": r["auroc"], "average_precision": r["average_precision"],
            "tp": m["true_positives"], "tn": m["true_negatives"], "fp": m["false_positives"], "fn": m["false_negatives"],
            "recall_wilson95": r["wilson95"]["recall"], "fpr_wilson95": r["wilson95"]["false_positive_rate"],
            "previous": PREVIOUS[category], "lowest_recall_defect_types": [
                {"defect_type": k, "detected": v["detected"], "total": v["total"], "recall": v["recall"]} for k, v in lowest],
            "threshold": r["locked_threshold"], "lock_digest": r["lock_digest"], "counts": r["counts"],
            "wall_seconds": r["timing"]["wall_seconds"], "ms_per_image": r["timing"]["total_ms_per_image"],
            "peak_ram_mb": r["peak_ram_mb"], "technical_rerun": r["technical_rerun"], "leakage_audit_all_passed": audit["all_passed"],
            "sentinel_present": (out / fl.SENTINEL).is_file(), "consumed_present": (out / fl.CONSUMED).is_file(),
        })
    gates = {}
    for row in rows:
        gates[row["gate"]] = gates.get(row["gate"], 0) + 1
    aps = [row["average_precision"] for row in rows]
    aurocs = [row["auroc"] for row in rows]
    summary = {
        "categories_scored": len(rows), "missing": missing, "gate_counts": gates,
        "mean_average_precision_image_level": sum(aps) / len(aps) if aps else None,
        "mean_auroc": sum(aurocs) / len(aurocs) if aurocs else None,
        "rows": rows,
    }
    (ARTIFACTS_ROOT / "patchcore_wrn50_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    lines = ["| Category | Model | Gate | Recall | Precision | F1 | Accuracy | FPR | AUROC | AP | TP/TN/FP/FN | Prev. recall | Prev. FPR |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        prev = row["previous"]
        lines.append(f"| {row['category']} | {row['model']} | {row['gate']} | {row['recall']:.4f} | {row['precision']:.4f} | {row['f1']:.4f} | "
                     f"{row['accuracy']:.4f} | {row['fpr']:.4f} | {row['auroc']:.4f} | {row['average_precision']:.4f} | "
                     f"{row['tp']}/{row['tn']}/{row['fp']}/{row['fn']} | {prev['recall']:.4f}{' (latest)' if prev.get('note') else ''} | {prev['fpr']:.4f} |")
    lines += ["", f"Mean image-level AP (mAP) over {len(rows)} categories: {summary['mean_average_precision_image_level']:.4f}; "
              f"mean AUROC: {summary['mean_auroc']:.4f}.", "",
              "Gate counts: " + ", ".join(f"{k} {v}" for k, v in sorted(gates.items()))]
    (ARTIFACTS_ROOT / "patchcore_wrn50_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"missing: {missing}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
