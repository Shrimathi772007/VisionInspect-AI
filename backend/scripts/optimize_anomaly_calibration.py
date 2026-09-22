"""Calibration / aggregation optimization for a category's frozen patch-Gaussian detector (good-only data).

  study    run the calibration-robustness study (Phase A), threshold-policy comparison (B) and aggregation
           comparison (C) on the category's 280 train/good images, apply the pre-declared selection policy, and
           write backend/ai_models/<category>/optimization/reports/calibration_study.json.
  freeze   re-run the ENTIRE study a second time and require it to be identical to the first, then fit the
           frozen model on all good images, write backend/ai_models/<category>/optimization/selected_candidate/,
           verify a second independent fit reproduces the artifact hash, and write freeze_report.json.

Neither command can reach the final test set. The final test is scored once by
scripts/evaluate_optimized_anomaly_model.py. Nothing here overwrites an existing model or report, and nothing is
registered for serving.

Usage (from backend/, venv active):
    python scripts/optimize_anomaly_calibration.py study  --category carpet
    python scripts/optimize_anomaly_calibration.py freeze --category carpet
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.calibration_pipeline import (  # noqa: E402
    SIDE,
    compare_studies,
    fit_final_detector,
    freeze_optimized,
    load_inputs,
    reports_dir,
    run_study,
    selected_dir,
    _md5_sha256,
)
from app.ai.evaluation.calibration_study import PRIMARY_FPR  # noqa: E402
from app.ai.training import artifacts  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _print_study(summary: dict) -> None:
    a = summary["phase_a"]
    print("\n=== PHASE A: good-only calibration robustness (baseline aggregation top1pct_mean, baseline rule mean+1.5std) ===")
    es = a["existing_split"]
    print(f"existing 224/56 split: threshold {es['threshold_baseline_rule']:.3f}, validation FP {es['validation_false_positives']}/{es['validation_size']}, "
          f"mean {es['distribution']['mean']:.2f} std {es['distribution']['std']:.2f} median {es['distribution']['median']:.2f}")
    ios = a["in_sample_vs_out_of_sample"]
    print(f"in-sample (training) mean {ios['training_images_in_sample']['mean']:.2f} vs out-of-sample (validation) mean {ios['validation_out_of_sample']['mean']:.2f}")
    print(f"{'scheme':20s} {'folds':>5s} {'val n':>5s} {'thr mean':>9s} {'thr std':>8s} {'thr CV':>7s} {'FPR in-fold':>11s} | 1-fold->others FPR mean/max  share<=10%")
    for name, rows in a["folds"].items():
        st = a["threshold_stability"][name]
        ft = st["single_fold_transfer"]
        infold = sum(r["false_positives"] for r in rows) / sum(r["validation_size"] for r in rows)
        print(f"{name:20s} {len(rows):5d} {rows[0]['validation_size']:5d} {st['threshold_mean']:9.2f} {st['threshold_std']:8.2f} "
              f"{st['threshold_cv']:7.3f} {infold:11.1%} | {ft['fpr_mean']:6.1%} / {ft['fpr_max']:6.1%}      {ft['share_folds_le_10pct']:.0%}")
    print("\nOut-of-fold score distribution per scheme:")
    for name, d in a["distributions"].items():
        print(f"  {name:20s} n={d['n']:4d} mean {d['mean']:.2f} std {d['std']:.2f} median {d['median']:.2f} p95 {d['p95']:.2f} p99 {d['p99']:.2f} max {d['max']:.2f} cov {d['cov']:.3f}")
    dr = a["drift"]
    print(f"\nDrift diagnostics (train/good only): block medians {[round(x, 2) for x in dr['block_medians']]}, "
          f"Kruskal-Wallis p={dr['kruskal_wallis_p_between_blocks']:.3g}, Spearman(file index, score) rho={dr['spearman_file_index_vs_score']['rho']:.3f} "
          f"(p={dr['spearman_file_index_vs_score']['p']:.3g})")
    print("\n=== PHASES B + C: pooled transfer (worst fold / mean FPR) under random 5-fold and contiguous blocks ===")
    print(f"{'config':32s} {'rand max':>8s} {'rand mean':>9s} {'block max':>9s} {'block mean':>10s} {'final thr':>9s} {'synRecall':>9s} {'synAUROC':>8s} {'oofCoV':>7s}")
    for c in summary["configurations"]:
        r, b = c["random_5fold_pooled_transfer"], c["block_pooled_transfer"]
        print(f"{c['config']:32s} {r['fpr_max']:8.1%} {r['fpr_mean']:9.1%} {b['fpr_max']:9.1%} {b['fpr_mean']:10.1%} "
              f"{c['final_threshold']:9.2f} {c['synthetic_recall_diagnostic']:9.3f} {c['synthetic_auroc_diagnostic']:8.3f} {c['oof_cov_random']:7.3f}")
    sel = summary["selection"]
    print(f"\nSELECTED (pre-declared policy): {sel['winner']} | passes gate P: {sel['passes_primary_gate']} | secondary: {sel['meets_secondary_target']}\n{sel['reasoning']}")
    print(f"(gate P = worst-fold transfer FPR <= {PRIMARY_FPR:.0%} under both schemes; {len(sel['eligible'])} eligible configuration(s))")


def study(category: str) -> None:
    out = reports_dir(category) / "calibration_study.json"
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    t0 = time.perf_counter()
    inp = load_inputs(category)
    result = run_study(inp)
    print(f"study finished in {time.perf_counter() - t0:.0f}s")
    _print_study(result.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"git_head": _git("rev-parse", "HEAD"), **result.summary}, indent=1), encoding="utf-8")
    print(f"\nWrote {out}")


def freeze(category: str) -> None:
    study_path = reports_dir(category) / "calibration_study.json"
    if not study_path.is_file():
        sys.exit("Run `study` first.")
    report_path = reports_dir(category) / "freeze_report.json"
    if report_path.exists() or (selected_dir(category) / "frozen_model.json").exists():
        sys.exit("Refusing to overwrite an existing optimized freeze.")
    first = json.loads(study_path.read_text(encoding="utf-8"))

    print("Re-running the ENTIRE study independently for reproducibility ...")
    inp = load_inputs(category)
    second = run_study(inp)
    from app.ai.evaluation.calibration_pipeline import StudyResult  # noqa: F401
    same_selection = second.summary["selection"] == first["selection"]
    same_thresholds = second.summary["configurations"] == first["configurations"]
    same_phase_a = json.loads(json.dumps(second.summary["phase_a"])) == first["phase_a"]
    repro = {"selection_identical_to_first_run": same_selection, "configurations_table_identical_to_first_run": same_thresholds,
             "phase_a_identical_to_first_run": same_phase_a,
             "synthetic_diagnostic_identical_to_first_run": json.loads(json.dumps(second.summary["synthetic_diagnostic"])) == first["synthetic_diagnostic"]}
    for k, v in repro.items():
        print(f"  {k}: {v}")
    if not all(repro.values()):
        sys.exit("!!! The study is NOT reproducible across two runs; not freezing. Investigate. !!!")

    baseline_lock = json.loads((artifacts.ARTIFACTS_ROOT / category / "selected_model" / "frozen_model.json").read_text())
    baseline_reference = {"model_state_md5": baseline_lock["state_file"]["md5"], "model_state_sha256": baseline_lock["state_file"]["sha256"],
                          "threshold": baseline_lock["threshold"]["value"], "aggregation": baseline_lock["config"]["aggregation"],
                          "note": "The original frozen Carpet detector is preserved unchanged."}
    lock = freeze_optimized(inp, second, selected_dir(category), baseline_reference)
    print(f"Frozen: {lock['candidate_id']} threshold {lock['threshold']['value']!r}\n  state sha256 {lock['state_file']['sha256']}")

    from app.ai.models.patch_anomaly import PatchAnomalyDetector

    detector2 = fit_final_detector(inp, second.selection.winner.aggregation)
    with tempfile.TemporaryDirectory() as tmp:
        sha2 = _md5_sha256(detector2.save(Path(tmp)))[1]
    reloaded = PatchAnomalyDetector.load(selected_dir(category), inp.extractor)
    import numpy as np
    from app.ai.evaluation.anomaly_model_selection import score_uint8

    probe = inp.data.validation[SIDE]
    art = {"second_independent_fit_hash_identical": sha2 == lock["state_file"]["sha256"],
           "reloaded_artifact_scores_equal_second_fit_scores": score_uint8(reloaded, probe) == score_uint8(detector2, probe),
           "artifact_does_not_overwrite_original": str(selected_dir(category)).replace("\\", "/").endswith("optimization/selected_candidate")}
    for k, v in art.items():
        print(f"  {k}: {v}")
    if not all(art.values()):
        sys.exit("!!! Artifact reproducibility failure. !!!")
    _ = np
    report_path.write_text(json.dumps({"git_head": _git("rev-parse", "HEAD"), "reproducibility": {**repro, **art},
                                       "lock": lock}, indent=1), encoding="utf-8")
    print(f"Wrote {report_path}\nFrozen candidate: {selected_dir(category)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["study", "freeze"])
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    {"study": study, "freeze": freeze}[args.command](args.category)


if __name__ == "__main__":
    main()
