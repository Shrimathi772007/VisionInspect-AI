# Addendum 1: Grid full320 candidate (normal data only)

Written and committed BEFORE the addendum is run and before any WRN-50 test scoring. PROTOCOL.md,
STUDY_DECLARATION.md, `run_wrn50_study.py` and `wrn50_study_logic.py` are unchanged (their SHA-256 values are
recorded in all nine locks). This addendum lives in new files only:

- this declaration;
- `run_wrn50_grid_full320.py` (runner);
- `grid_full320_logic.py` (full320 preprocessing, config wrapper, combined selection, memory-light coreset).

Mentor approval has not been requested. The addendum will be disclosed in the documentation.

## Why

Grid's best normal-data synthetic recall in the locked study is 0.853 (crop224 / max / mean_std_2.5), the
lowest of the nine categories. Its weakest synthetic type is object_patch (0.58). A larger input may help with
fine defects. Only normal data (train/good) and synthetic defects painted on it are used.

## New candidate

- **full320:** resize the whole image to 320x320 (`cv2.INTER_AREA`), no crop, BGR -> RGB, the same ImageNet
  normalisation as crop224/full256.
- Feature grid 40x40 = **1600 patches** per image (layer2 stride 8). top1pct_mean = mean of the top
  ceil(0.01 x 1600) = **16** patch scores (the existing `aggregate_scores`, unchanged).
- Aggregations: max, top1pct_mean. Policies: mean_std_2.5, mean_std_3, percentile_99. That gives **6 new rows**.
- Backbone, layers 2+3, 3x3 average pooling, k = 1, 10% greedy coreset, seed 0, 64-d projection: unchanged.

`PatchCoreConfig` (patchcore.py) only accepts crop224/full256, and `PatchCoreDetector.load` rebuilds that
class. patchcore.py is not edited. Instead `grid_full320_logic.Full320Config` subclasses it and accepts only
full320 at input size 320 (with the same validation otherwise), and `load_full320_detector` loads a saved full320
model with the same integrity checks as `PatchCoreDetector.load`. `PatchCoreDetector` itself is used unchanged.

## Everything else identical to the study

- Dataset access: only `dataset/grid/train/good/` is listed or read, through the study's single path function
  `wrn50_study_logic.train_good_path` (raises on `test` / `ground_truth`). Every listing and file open is
  appended to `backend/ai_models/grid/patchcore_wrn50_full320/dataset_access_audit.txt` and verified at the
  end.
- Folds: file-name-sorted train/good list, fold = index mod 5.
- Fold banks: 10% greedy coreset (seed 0) of the other four folds' patches, with the study's
  `greedy_coreset_over_images`.
- Synthetic defects: 4 per held-out image, `make_synthetic_defect` on the decoded full-resolution image BEFORE
  resizing, with the study's `synthetic_plan` (same types, severities and seeds). Failures count as not
  detected. Because painting happens before preprocessing, the synthetic-image digest must equal the digest in
  Grid's existing lock. The runner checks this.
- Metrics: the study's `combination_metrics` (pooled out-of-fold threshold, per-fold FPR at score > threshold,
  worst/mean fold FPR, pooled synthetic recall) and the same per-type breakdown.

## Selection

The 12 existing Grid rows are read from the existing
`backend/ai_models/grid/patchcore_wrn50/selection_table.json`. They are NOT recomputed. Before use, the runner
checks the file's SHA-256 against Grid's lock and checks the lock's own digest. The 12 rows are pooled with the
6 new rows (18 rows) and saved as `selection_table_combined.json`.

The PROTOCOL.md rule is applied word for word, with full320 added at the end of the mode order. Among rows with
worst-fold FPR <= 10%, choose the highest synthetic recall. Ties go to:

1. lower worst-fold FPR;
2. then crop224, then full256, then full320;
3. then max;
4. then the study's declared continuation: lower mean fold FPR, then higher threshold, then policy order
   (mean_std_2.5, mean_std_3, percentile_99).

If no row meets the limit, the study's declared fallback applies (lowest worst-fold FPR, then highest recall,
then the same mode/aggregation/continuation order). A full320 row must therefore be strictly better to win: the
existing locked row (crop224 / max) wins any exact tie.

## Outcome

- **If the winner is an existing row:** Grid stays as locked. No new model is adopted or saved. The combined
  table and `decision.json` (saying so) are written to the new folder.
- **If a full320 row wins:** a final model (bank on ALL train/good images, chosen aggregation, selected
  threshold unchanged) and `lock.json` are written to the new folder. `decision.json` states that Grid's final
  model is the full320 one.

The full320 lock records:

- the selection: threshold (full precision), mode, aggregation, policy, worst-fold FPR, mean fold FPR, synthetic
  recall;
- the backbone weights SHA-256 and the SHA-256 of the saved model state and config;
- the SHA-256 of `selection_table_combined.json`, the runner, the logic module, this addendum, PROTOCOL.md,
  STUDY_DECLARATION.md, `run_wrn50_study.py` and `wrn50_study_logic.py`;
- the train/good file-list fingerprint;
- the UTC time, `"test_directory_listed": false` and a `lock_digest`.

## Memory method

264 images x 1600 patches x 1536 floats is about 2.6 GB, so features are NOT held for all folds at once.

- **Fold k:** extract the other four folds' features into one pre-allocated buffer (sorted index order, the
  same order the study's all-in-memory method uses). Build the bank with `greedy_coreset_over_images`, copy the
  selected rows, and release the buffer. Then extract the held-out fold's features, score them, and paint and
  score its synthetic images.
- **Final model (only if full320 wins):** a two-pass streaming coreset. Pass 1 extracts features in batches of
  8 and writes only the seeded 64-d projection per image (the same per-image projection, RNG use, size
  rounding and greedy update as `greedy_coreset_over_images`). Pass 2 re-extracts with the same batches and
  copies only the selected rows.

This is a memory-only change: scores must be identical in value to the all-in-memory method. The runner's
`--equivalence-check` (20 train/good images, fold 0) compares the memory-light fold bank and scores, and the
streaming final bank, against an all-in-memory reference. It reports the maximum differences (expected: 0, or
float rounding only). Unit tests check the same on tiny random features.

The peak process RAM limit is 3.5 GB. If it is exceeded, the run stops with exit code 3. Nothing is switched to
float16. Each fold is checkpointed, and a checkpoint is reused only when the file list and all code/document
hashes match.

## Test set

No test directory of any category is listed, opened or scored by this addendum. Grid's test set will still be
scored exactly once, in Task 4D, by the final model this addendum's decision names: the locked crop224 model
or, if a full320 row wins, the full320 model.
