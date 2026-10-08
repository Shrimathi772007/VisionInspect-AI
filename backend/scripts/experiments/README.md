# Archived experiment runner scripts

This folder holds the per-category experiment scripts that produced the models, threshold
locks, declarations and reports under `backend/ai_models/<category>/`. They were written and
run from temporary Claude Code session scratchpads
(`%LOCALAPPDATA%\Temp\claude\c--Users-SHRIMATHI-S-Documents-VisionInspect-AI\<session>\scratchpad\`)
and are archived here so they are not lost when those temp folders are cleaned up.

## Archived exactly as executed

Every file is a byte-for-byte copy of the original. Nothing was reformatted, fixed or
updated. `MANIFEST.json` lists, for each script, its original path, its path here, its
SHA-256, its size, and the reports that record that hash.

`match_status` in the manifest means:

| Status | Meaning |
|---|---|
| `MATCH` | A JSON report under `backend/ai_models/` records this exact SHA-256 (as `runner_sha256`, `wrapper_sha256`, `launcher_sha256`, `provenance.verifier_sha256`, a `runners` entry, etc.). |
| `NO MATCH` | A report names this file but records a different hash. (None at archive time.) |
| `NOT REFERENCED` | No report records this hash or file name: helpers that wrote declarations, lock supplements or reproducibility artifacts, model-family runners for tile/cable/capsule, and diagnostics. |

To check a file against its report:

```
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" backend/scripts/experiments/tile/run_tile_phase1.py
```

## Layout

- `<category>/`: scripts for one MVTec category (`run_<category>_phase1.py`,
  `run_<category>_model_family.py`, `run_<category>_final.py`, `*_study_launcher.py`,
  `*_repro_capture.py`, and the `write_*` / `verify_*` helpers).
- `_shared/`: scripts that are not tied to one category, such as dataset audits, one-off
  code patchers and smoke checks.

Bottle, Carpet, Hazelnut, Leather, Metal Nut and Pill have no per-category runner here
(Carpet, Hazelnut, Pill and Bottle have only diagnostic helpers). Their reports do not
record a runner hash or name. They were presumably produced through the generic scripts
in `backend/scripts/`, but this was not verified when the scripts were archived.

## Before running anything

- **Hard-coded paths.** Many scripts use absolute paths from the original Windows machine
  (the repository root there; read them as `./...` relative to the repository root) or point at the original
  scratchpad folder. Some expect to be started from `backend/` so that `app.` imports work.
- **Required data.** They expect `backend/ai_models/` (git-ignored, not in this repository)
  and `dataset/` (the MVTec AD images) to exist with the original contents.
- **Final tests are one-shot.** The `run_<category>_final.py` scripts score that category's
  held-out test set. Each category's test set has already been scored once, and the reports
  rely on that. Running a final script again would be a second scoring of the same test
  set. Any such run must be disclosed alongside the results, and it must not be used to
  select or tune a model or threshold. Several runners also write one-shot sentinel files
  (`final_test_sentinel.json`, `final_test_consumed.json`) that are meant to stop a re-run.
  Do not delete those files to get around that.
- The scripts write into `backend/ai_models/`. Running one can overwrite artifacts that the
  serving configuration pins by hash (`backend/app/ai/inference/serving.py`).

## Localization evaluation (second scoring, disclosed)

`localization_map/` holds a later, evaluation-only run: box AP@0.5 of the anomaly-map boxes and pixel AUROC of the
served models (rules declared in `localization_map/PROTOCOL.md` before scoring). It is a **second scoring of every
category's final test set, used for localization only**: no model, threshold, lock or box rule changed, and no result
was used to select or tune anything. It writes nothing into `backend/ai_models/`; the summary is
`backend/app/ai/box_eval_addendum.json`, shown on the Models page. The boxes come from the anomaly map, not from a
trained detector.
