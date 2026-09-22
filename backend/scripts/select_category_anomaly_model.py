"""Model selection for one category using TRAIN + VALIDATION data only (never the final test set).

Two sub-commands, so the (slow) candidate scoring is done once and the freeze is a separate, auditable step:

  stage-a   build the development arrays (train / validation / synthetic-defect validation images at every
            needed resolution), score every Stage-A candidate, apply the pre-declared selection rule, and
            write backend/ai_models/<category>/model_selection/reports/stage_a_results.json.
  freeze    fit the winner on the training images, lock its validation-derived threshold, write the frozen
            artifact to backend/ai_models/<category>/selected_model/, verify reproducibility by an
            independent second fit, and write model_selection_report.json.

Neither sub-command can reach the final test set (app.ai.evaluation.anomaly_model_selection does not import
test discovery). The final test is scored, once, by scripts/evaluate_frozen_anomaly_model.py.

Usage (from backend/, venv active):
    python scripts/select_category_anomaly_model.py stage-a --category carpet
    python scripts/select_category_anomaly_model.py freeze  --category carpet
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.evaluation.anomaly_model_selection import (  # noqa: E402
    ESCALATION_RECALL,
    SYNTHETIC_SEED,
    _md5_sha256,
    CandidateSpec,
    DevResult,
    Timing,
    build_dev_arrays,
    fit_detector,
    freeze_detector,
    run_stage,
    score_uint8,
    select_best,
    stage_a_candidates,
)
from app.ai.evaluation.category_phase3 import PHASE3_SPLIT_SEED, PHASE3_TRAINING_FRACTION, resolve_outputs  # noqa: E402
from app.ai.evaluation.phase3_threshold_selection import select_threshold_from_validation  # noqa: E402
from app.ai.evaluation.threshold_experiments import generate_candidates  # noqa: E402
from app.ai.models.patch_anomaly import PatchAnomalyDetector  # noqa: E402
from app.ai.models.resnet18 import RESNET18_SHA256, RESNET18_SOURCE_URL, load_pretrained_resnet18  # noqa: E402
from app.ai.training import artifacts, discover_train_samples  # noqa: E402
from app.ai.training.validation_split import split_train_validation  # noqa: E402


def selection_root(category: str) -> Path:
    return artifacts.ARTIFACTS_ROOT / category / "model_selection"


def reports_dir(category: str) -> Path:
    return selection_root(category) / "reports"


def selected_model_dir(category: str) -> Path:
    return artifacts.ARTIFACTS_ROOT / category / "selected_model"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _result_to_json(r: DevResult) -> dict:
    d = dataclasses.asdict(r)
    d["spec"]["layers"] = list(r.spec.layers) if r.spec.layers else None
    return d


def _result_from_json(d: dict) -> DevResult:
    spec = dict(d["spec"])
    spec["layers"] = tuple(spec["layers"]) if spec["layers"] else None
    d = dict(d, spec=CandidateSpec(**spec), timing=Timing(**d["timing"]))
    return DevResult(**d)


def _table(results: list[DevResult]) -> None:
    header = ("Candidate", "Side", "ValCoV", "FPR@sel", "SynRecall", "SynAUROC", "small", "medium", "large",
              "FitS", "Feat ms", "Score ms", "StateMB")
    rows = []
    for r in results:
        sev = r.synthetic_recall_by_severity or {}
        rows.append((
            r.spec.candidate_id, r.spec.image_size, f"{r.validation_stats['cov']:.3f}",
            "-" if r.validation_fpr_at_selected is None else f"{r.validation_fpr_at_selected:.1%}",
            "-" if r.synthetic_recall_pooled is None else f"{r.synthetic_recall_pooled:.3f}",
            f"{r.synthetic_auroc:.3f}",
            *(("-",) * 3 if not sev else (f"{sev['small']:.2f}", f"{sev['medium']:.2f}", f"{sev['large']:.2f}")),
            f"{r.timing.fit_seconds:.0f}", f"{r.timing.feature_ms_per_image:.0f}", f"{r.timing.score_ms_per_image:.0f}",
            f"{r.state_bytes / 1e6:.1f}",
        ))
    w = [max(len(str(x[i])) for x in [header] + rows) for i in range(len(header))]
    for row in [header, None] + rows:
        print("-+-".join("-" * x for x in w) if row is None else " | ".join(str(v).rjust(w[i]) for i, v in enumerate(row)))


def _split(category: str):
    return split_train_validation(discover_train_samples(category), PHASE3_TRAINING_FRACTION, PHASE3_SPLIT_SEED)


def stage_a(category: str) -> None:
    out = reports_dir(category) / "stage_a_results.json"
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    p3 = resolve_outputs(category)
    if not p3.model_path.is_file():
        sys.exit(f"Phase 3 baseline for '{category}' not found: {p3.model_path}")
    convae256 = artifacts.get_model_path(category, "model_selection/convae_input256/autoencoder")
    if not convae256.is_file():
        sys.exit(f"Expected the 256px ConvAE reference at {convae256} (train it first).")

    split = _split(category)
    print(f"Split: training={len(split.training)} validation={len(split.validation)}")
    specs = stage_a_candidates()
    sides = tuple(sorted({s.image_size for s in specs}))
    print(f"Building development arrays at sides {sides} (incl. synthetic defects on VALIDATION images) ...")
    data = build_dev_arrays(split, sides)
    print(f"  done in {data.build_seconds:.0f}s: train {len(data.train_files)}, validation {len(data.validation_files)}, "
          f"synthetic {len(data.synthetic_meta)}, synthetic digest {data.synthetic_digest[:16]}")

    extractor = load_pretrained_resnet18()
    t0 = time.perf_counter()
    results = run_stage(specs, category, extractor, data, selection_root(category))
    print(f"\nStage A scored in {time.perf_counter() - t0:.0f}s\n")
    _table(results)

    selection = select_best(results)
    print(f"\nWINNER (pre-declared rule): {selection.winner_id}\n{selection.reasoning}")
    print(f"Escalation needed (best pooled synthetic recall < {ESCALATION_RECALL}): {selection.escalation_needed}")

    payload = {
        "stage": "stage_a", "category": category, "git_head": _git("rev-parse", "HEAD"),
        "synthetic_digest": data.synthetic_digest, "synthetic_count": len(data.synthetic_meta),
        "synthetic_seed": SYNTHETIC_SEED,
        "backbone": {"name": "resnet18_imagenet_frozen", "sha256": RESNET18_SHA256, "source": RESNET18_SOURCE_URL},
        "convae256_reference": {"path": str(convae256)},
        "selection": dataclasses.asdict(selection),
        "results": [_result_to_json(r) for r in results],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nWrote {out}")


def freeze(category: str) -> None:
    stage_path = reports_dir(category) / "stage_a_results.json"
    if not stage_path.is_file():
        sys.exit("Run stage-a first.")
    report_path = reports_dir(category) / "model_selection_report.json"
    if report_path.exists() or (selected_model_dir(category) / "frozen_model.json").exists():
        sys.exit("Refusing to overwrite an existing frozen model / report.")
    payload = json.loads(stage_path.read_text(encoding="utf-8"))
    results = [_result_from_json(d) for d in payload["results"]]
    selection = select_best(results)
    winner = next(r for r in results if r.spec.candidate_id == selection.winner_id)
    print(f"Selected by the pre-declared rule: {winner.spec.candidate_id}\n{selection.reasoning}")
    if selection.escalation_needed:
        sys.exit(f"Escalation trigger fired (best pooled synthetic recall < {ESCALATION_RECALL}); Stage B must run first.")
    if winner.spec.family == "convae":
        sys.exit("A ConvAE candidate won; freezing ConvAE candidates is not implemented by this script.")

    split = _split(category)
    spec = winner.spec
    data = build_dev_arrays(split, (spec.image_size,))  # same deterministic synthetic set, one resolution
    synthetic_regenerated_identically = data.synthetic_digest == payload["synthetic_digest"]
    print(f"Synthetic development set regenerated identically to Stage A: {synthetic_regenerated_identically}")
    extractor = load_pretrained_resnet18()

    print("Freezing (fit on the training images, threshold from this detector's own validation scores) ...")
    frozen = freeze_detector(spec, extractor, data, selected_model_dir(category), RESNET18_SHA256, winner)
    print(f"  state sha256 {frozen.state_sha256}")
    print(f"  threshold {frozen.threshold!r} ({frozen.threshold_method} {frozen.threshold_parameter})")

    # --- Reproducibility: (1) the frozen fit reproduces Stage A's recorded scores (a separate process/fit),
    # --- (2) an independent second fit reproduces the frozen artifact, (3) the saved artifact reloads to the same scores.
    print("Reproducibility checks ...")
    reloaded = PatchAnomalyDetector.load(selected_model_dir(category), extractor)
    syn_frozen = score_uint8(reloaded, data.synthetic[spec.image_size])
    detector2, fit2_s = fit_detector(spec, extractor, data)
    with tempfile.TemporaryDirectory() as tmp:
        sha2 = _md5_sha256(detector2.save(Path(tmp)))[1]
    val2 = score_uint8(detector2, data.validation[spec.image_size])
    sel2 = select_threshold_from_validation(generate_candidates(val2), val2).selected
    repro = {
        "synthetic_set_regenerated_identically": synthetic_regenerated_identically,
        "frozen_validation_scores_match_stage_a": frozen.validation_scores == winner.validation_scores,
        "frozen_synthetic_scores_match_stage_a": syn_frozen == winner.synthetic_scores,
        "frozen_threshold_matches_stage_a": winner.selected_threshold is not None
        and winner.selected_threshold["threshold"] == frozen.threshold,
        "second_fit_state_hash_identical": sha2 == frozen.state_sha256,
        "second_fit_validation_scores_identical": val2 == frozen.validation_scores,
        "second_fit_threshold_identical": sel2 is not None and sel2.threshold == frozen.threshold,
        "reloaded_artifact_validation_scores_identical": score_uint8(reloaded, data.validation[spec.image_size]) == val2,
        "config_identical": detector2.config() == json.loads((selected_model_dir(category) / "model_config.json").read_text()),
    }
    repro["all_identical"] = all(repro.values())
    for k, v in repro.items():
        print(f"  {k}: {v}")
    if not repro["all_identical"]:
        sys.exit("!!! Reproducibility failure - not writing the report. Inspect before proceeding. !!!")

    report = {
        "report_type": "category_model_selection", "category": category, "git_head": _git("rev-parse", "HEAD"),
        "selected": {"candidate_id": spec.candidate_id, "reasoning": selection.reasoning},
        "frozen": json.loads((selected_model_dir(category) / "frozen_model.json").read_text()),
        "reproducibility": repro, "stage_a": payload,
        "second_fit_seconds": fit2_s, "first_fit_seconds": frozen.fit_seconds,
    }
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Wrote {report_path}\nFrozen model: {selected_model_dir(category)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stage-a", "freeze"])
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    {"stage-a": stage_a, "freeze": freeze}[args.command](args.category)


if __name__ == "__main__":
    main()
