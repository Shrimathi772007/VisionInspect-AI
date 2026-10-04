# WRN-50-2 PatchCore final test: declaration (committed BEFORE any category is scored)

This file, `run_wrn50_final.py` and `wrn50_final_logic.py` are committed together before any test set is scored.
The runner refuses real scoring unless these three files are tracked and unmodified in Git. It records the commit
hash and the SHA-256 of each file in every category's sentinel. PROTOCOL.md, STUDY_DECLARATION.md,
ADDENDUM_1_GRID_FULL320.md, the study and addendum runners, `patchcore.py`, every lock, model and threshold, and
every existing test stay unchanged.

**No result of this run will be used to modify any model, threshold or choice.**

## What is scored

Each of the nine categories is scored exactly once, by its locked model (PROTOCOL.md: "each category's test set
is scored exactly once by this model; the result is reported whatever it is"). The WRN model replaces the
previous model for all nine categories regardless of result. No choice between old and new uses test results.

| Category | Model scored | Lock folder (under `backend/ai_models/`) | Lock digest |
|---|---|---|---|
| bottle | WRN-50 | `bottle/patchcore_wrn50/` | `830aa484045e...` |
| capsule | WRN-50 | `capsule/patchcore_wrn50/` | `7821104664f3...` |
| zipper | WRN-50 | `zipper/patchcore_wrn50/` | `2a72a8fdf816...` |
| wood | WRN-50 | `wood/patchcore_wrn50/` | `49b048415e53...` |
| grid | **WRN-50 full320 (addendum 1)** | `grid/patchcore_wrn50_full320/` | `4cdf78b2b2b6...` |
| pill | WRN-50 | `pill/patchcore_wrn50/` | `072271786f13...` |
| carpet | WRN-50 | `carpet/patchcore_wrn50/` | `2527ed6ac346...` |
| screw | WRN-50 | `screw/patchcore_wrn50/` | `0857280ca50e...` |
| hazelnut | WRN-50 | `hazelnut/patchcore_wrn50/` | `894cc81a9185...` |

The full 64-character digests are in `wrn50_final_logic.EXPECTED_LOCK_DIGESTS`. A lock whose digest differs is
rejected, even if it is internally consistent.

**Grid:** Addendum 1's combined selection chose full320 / max / mean_std_2.5, threshold `1.88393018105021`. Grid is
scored with that model, loaded only through `grid_full320_logic.load_full320_detector` (`PatchCoreDetector.load`
rejects full320; `patchcore.py` is not edited). Images are resized to 320x320 with `cv2.INTER_AREA`, giving
40x40 patch grids and 320x320 maps. The original Grid lock (`grid/patchcore_wrn50/`, digest `de0bd22a...`) must
still verify, but it is never loaded for scoring and never scored.

## Procedure (one category per process, in this order: bottle, capsule, zipper, wood, grid, pill, carpet, screw, hazelnut)

1. **Verify the lock** before anything in the dataset is touched:
   - the lock digest, both recomputed and equal to the declared value;
   - the SHA-256 of the model state and config, and that `model_config.json` agrees with the lock's config, bank
     shape and backbone hash;
   - the selection table's SHA-256, plus the selection rule re-applied to the saved table giving the locked mode,
     aggregation, policy and threshold;
   - the SHA-256 of every protected file the lock records (standard locks: the study runner, study logic,
     PROTOCOL.md and STUDY_DECLARATION.md).

   For Grid, the full320 lock is checked in the same way using `selection_table_combined.json`, its provenance
   (the original table's hash and digest) and every protected file it records, including
   ADDENDUM_1_GRID_FULL320.md, `run_wrn50_grid_full320.py` and `grid_full320_logic.py`. `decision.json` must say
   that full320 won, and the original Grid lock must also verify. If anything differs, that category STOPS and is
   reported.
2. **Sentinel:** `final_test_sentinel.json` (UTC time, lock digest, threshold, runner/logic/declaration SHA-256,
   Git HEAD) is created with an exclusive create BEFORE the test directory is listed. The run refuses to start
   when `final_test_consumed.json`, `final_test_result.json` or `per_image.csv` exists. If a process dies after the
   sentinel but before results are written, the category may be repeated **once**, with `--technical-rerun`, the
   identical model and the same lock digest. That repeat is recorded in the sentinel and the result as a technical
   re-run. Once results exist, the category is never re-run.
3. **List** `dataset/<category>/test/` only: sorted subfolders, then sorted `*.png` files in each. All dataset paths
   come from one guarded builder, which refuses any path part naming the mask folder and any path outside `test/`.
   The mask folder is never listed or opened (localization is a later task that reuses the saved maps).
4. **Label:** `good` subfolder -> good; any other subfolder -> defective, with the subfolder name as the defect type.
5. **Load** the locked detector: `PatchCoreDetector.load`, or `load_full320_detector` for Grid. The backbone is the
   verified WRN-50-2 via the study's `load_backbone` (layer4/fc released; this does not change features). The
   loaded config and bank shape must equal the lock.
6. **Score** every test image once, one image at a time (batch 1, like serving), in float32. Each file is read as
   bytes and decoded with the project's `decode_image` bytes path (identical pixels to the path decode; the dry
   run checks this). Then the locked preprocessing (crop224 / full256 / full320), WRN-50-2 layer2+3 features, k=1
   distance to the locked bank, the locked aggregation and the anomaly map (bilinear upsample to the input size
   plus Gaussian blur sigma 4). **Prediction:** defective if image score > threshold; score <= threshold means
   good. No threshold, mode, aggregation or bank is changed.
7. **Metrics** are computed once from the predictions at the locked threshold. They are listed below.
8. **Re-verify** the lock and model hashes after scoring. Write the outputs, then `final_test_consumed.json`.

Peak RAM must stay under 3.5 GB (the run stops with exit code 3 if it is exceeded). Each category runs in its own
process, one at a time. Nothing is switched to float16 for computation; only the stored anomaly maps are float16.

## Metrics (defective = positive class)

Accuracy, precision, recall, F1, F2, FPR, specificity, balanced accuracy and MCC. Zero denominators give 0, as in
sklearn's `zero_division=0`. Also: AUROC (ties count one half) and average precision (sklearn
`roc_auc_score` / `average_precision_score` on the image scores); TP/TN/FP/FN; per-defect-type recall with
counts; Wilson 95% intervals for recall, FPR and each per-defect recall; per-image timing; and the threshold and
lock digest used. Accuracy, precision, recall, F1, the confusion cells and FPR are cross-checked against the
project's `category_phase3.metrics_from_predictions`. A mismatch is reported, not fixed.

**Gate** (`category_phase1.gate_classification`, on point estimates, used read-only):

- EXCELLENT: R >= 0.90, F1 >= 0.85, FPR <= 0.10
- GOOD: R >= 0.85, F1 >= 0.80, FPR <= 0.10
- ACCEPTABLE: R >= 0.75, F1 >= 0.70, FPR <= 0.15
- otherwise NOT PRODUCTION READY

## Outputs (Git-ignored), per category in `<lock folder>/final_test/`

- `final_test_result.json`: counts, all metrics above, the gate, per-defect recall, Wilson intervals, timing, peak
  RAM, threshold, lock digest and lock verification record.
- `per_image.csv`: relative path, label, defect type, score, prediction.
- `patch_scores.npz`: per-image patch score grid, float32 (N x 28 x 28, 32 x 32 or 40 x 40), with `files` in
  CSV order.
- `anomaly_maps.npz`: upsampled maps, float16, same image order.
- `leakage_audit.json`: lock verified before listing; sentinel written before listing; no mask-folder path built,
  opened or listed; every dataset access inside `<category>/test/`; files opened during listing and scoring
  limited to the model files (bank + config), the backbone weights, the Python runtime and the outputs; lock and
  model unchanged after scoring. It is built from a process-wide audit hook on every `open` and directory listing.
- `final_test_sentinel.json` and `final_test_consumed.json`.

Then `backend/ai_models/patchcore_wrn50_summary.json` (plus a Markdown table) gives the new and previous numbers
for the nine categories, the mean image-level AP and the mean AUROC.

## Dry run (before any scoring)

The full pipeline is run on the first 10 `train/good` images of one standard category and of Grid (full320 loader).
It uses fake labels and a temporary output folder; the run refuses any output folder under `ai_models/`. No `test/`
path is touched, and the temporary output is deleted afterwards.

## Known limits (declared now)

- The train/good fingerprints in the locks are not re-checked here, because that would mean reading train/good
  during the scoring run. The model state, config and bank are hash-checked instead.
- Test sets of these nine categories were scored by earlier (non-WRN) models. PROTOCOL.md already discloses this.
