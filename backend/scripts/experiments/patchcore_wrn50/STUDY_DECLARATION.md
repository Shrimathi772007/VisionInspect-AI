# WRN-50-2 PatchCore normal-data study: declaration

Written and committed BEFORE any category is run. It makes PROTOCOL.md (binding, unchanged) operational for
Task 4C. The implementation is `run_wrn50_study.py` (runner) and `wrn50_study_logic.py` (logic), both committed
together with this file. Each lock file records the SHA-256 of PROTOCOL.md, this declaration, the runner and
the logic module.

## Data access

- Only `dataset/<category>/train/good/` is ever listed or read. Every dataset path is built by one function
  (`train_good_path`) that returns paths inside train/good only. It raises if any path part is `test` or
  `ground_truth` (case-insensitive), if the file name has any path structure, or if the resolved path leaves
  train/good.
- Every listing and every file open is appended to `dataset_access_audit.txt` in the category's output
  folder. The audit is verified at the end of each run: every entry must be under train/good.
- **The test directory of a category is not listed, opened or scored during Task 4C.** Each lock records
  `"test_directory_listed": false`.

## Folds

- Images are the file-name-sorted train/good list; image `i` is in fold `i mod 5`.
- For each fold k, a 10% greedy-coreset memory bank (seed 0, 64-d random projection, the same algorithm and
  size rounding as `greedy_coreset_indices`) is built from the patches of the other four folds. That bank
  scores fold k's images (out-of-fold). Every train/good image gets exactly one out-of-fold score.
- The fold banks are built without copying the other folds' features (the projection is applied image by
  image). A unit test checks that this selects exactly the same rows as `greedy_coreset_indices` on the
  copied rows.

## Synthetic defects

- Generator: `app.ai.evaluation.synthetic_defects.make_synthetic_defect` (the model-family study's
  generator), applied to the **decoded** full-resolution BGR image **before** preprocessing (resize/crop).
- 4 synthetic defects per held-out normal image, i.e. per image of fold k, painted on that image and scored
  against **the same fold-k bank** (which never contains that image). In total that is 4 x N per mode.
- For the image at 0-based position `p` within fold k's sorted members, defect `j` (j = 0..3) uses the
  counter `c = 4p + j`:
  - defect type = `DEFECT_TYPES[c mod 5]`, in the generator's fixed order
    (color_blob, dark_blob, line, wavy_line, object_patch);
  - severity = `SEVERITIES[c mod 3]` (0 small, 1 medium, 2 large);
  - seed = `1000 * k + p`. The generator additionally mixes the defect type and severity into its RNG.
- If the generator reports that a defect produced no visible change, that synthetic sample is **counted as
  not detected** (it stays in the recall denominator) and is logged.

## Metrics, per (preprocessing, aggregation, policy): 2 x 2 x 3 = 12 rows

- **Threshold:** computed from the pooled out-of-fold normal scores of all train/good images, using the
  existing `category_phase2.policy_threshold`:
  - `mean_std_2.5`: mean + 2.5 x population std (ddof 0);
  - `mean_std_3`: mean + 3 x population std;
  - `percentile_99`: numpy 99th percentile (linear interpolation).
- **Flagging:** a score is flagged (anomalous) when `score > threshold`.
- **Per-fold FPR:** the share of fold k's out-of-fold normal scores flagged at that pooled threshold.
  The worst-fold FPR is the maximum over the 5 folds; the mean FPR is the mean over the 5 folds.
- **Synthetic recall:** the share of all synthetic samples (pooled over all folds) flagged at the pooled
  threshold.
- The full 12-row table is saved as `selection_table.json`.

## Selection rule

Copied word for word from PROTOCOL.md:

> Selection: among (candidate, policy) pairs whose worst-fold FPR on normal data is <= 10%, choose the
> highest synthetic recall; ties go to lower worst-fold FPR, then to crop224, then to max. If no pair meets
> the FPR limit, choose the lowest worst-fold FPR.

Exact implementation. All comparisons are exact (no tolerance on recall or FPR ties). "<= 10%" is inclusive,
with a 1e-12 float guard only.

1. **If at least one pair has worst-fold FPR <= 10%,** among those pairs:
   1. highest synthetic recall;
   2. then lower worst-fold FPR;
   3. then crop224 before full256;
   4. then max before top1pct_mean;
   5. then lower mean fold FPR;
   6. then higher threshold;
   7. then policy order mean_std_2.5, mean_std_3, percentile_99.
2. **If no pair meets the limit,** among all 12:
   1. lowest worst-fold FPR;
   2. then highest synthetic recall;
   3. then crop224;
   4. then max;
   5. then lower mean fold FPR;
   6. then higher threshold;
   7. then the same policy order.

A category where no pair meets the limit is not a stop condition. The fact is recorded in the lock
(`any_combination_met_fpr_limit: false`).

## Final model and lock

- Final model: a 10% greedy-coreset bank (seed 0) fitted on ALL train/good images with the chosen
  preprocessing, saved with `PatchCoreDetector.save` (`final_model/model_state.pt` + `model_config.json`).
  The aggregation recorded is the chosen one.
- The threshold is the selected pooled out-of-fold threshold, **unchanged** (it is not recomputed on the
  final bank).
- `lock.json` contains:
  - the selection: category, chosen mode / aggregation / policy, the threshold (full precision), worst-fold
    FPR, mean FPR, synthetic recall, and whether any combination met the 10% limit;
  - the backbone weights SHA-256;
  - SHA-256 of the saved model state and its config;
  - SHA-256 of `selection_table.json`, the runner, the logic module, PROTOCOL.md and this declaration;
  - the SHA-256 of the sorted train/good file list, per-file SHA-256s and a combined name:sha digest;
  - the synthetic-image digest per mode, the fold sizes, the audit summary and the peak RAM;
  - the UTC timestamp, `"test_directory_listed": false`, and a `lock_digest` (SHA-256 of the lock without
    its wall-clock fields).

## Run order and resources

- Category run order: bottle, capsule, zipper, wood, grid, pill, carpet, screw, hazelnut. Each category runs
  as its own process.
- One preprocessing mode at a time: its features are freed before the next mode. The results of each mode
  are checkpointed, so an interrupted run resumes. A checkpoint is only reused when the file list and all
  code/document hashes match.
- Peak RAM budget: 3.5 GB. If it is exceeded, the run stops with exit code 3; nothing is switched to float16.
- After the strict, hash-verified load, the backbone's layer4 and fc (never used for layer2+3 features) are
  replaced by Identity to release memory. Feature values are unaffected.

## Ambiguities in PROTOCOL.md and how they are resolved (most conservative reading)

1. **Synthetic severity is not specified.** Severities are cycled (small, medium, large) together with the
   types, so every (type, severity) pair occurs equally often, as in the model-family study, which used all
   three severities. No easier or harder subset is chosen.
2. **"Held-out normal images"** is read as each fold's out-of-fold images, scored against the bank that
   excludes them.
3. **Ties beyond "then to max"** (same mode and aggregation, different policy) are not covered. The declared
   continuation is lower mean FPR, then the higher (more conservative) threshold, then a fixed policy order.
4. **No tie-break is given for the "no pair meets the limit" branch.** The declared order is highest recall,
   then crop224, then max, then the same continuation as above.
5. **Tie tolerance is not given.** Exact equality is used. No tolerance band is invented.
6. **"<= 10%"** is read as inclusive.
7. **"Threshold as selected"** is read literally: the pooled out-of-fold threshold is kept for the final
   all-images bank and is not recomputed. A bank built from more images tends to give slightly lower normal
   scores, so this does not raise the false-positive rate.
8. **Synthetic generation failures** (the generator refusing a defect that changes nothing) are counted as
   missed detections rather than skipped or replaced.
9. **Seed "0"** is applied to every fold bank and to the final bank.
