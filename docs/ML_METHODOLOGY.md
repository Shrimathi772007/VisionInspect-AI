# ML methodology and results

This document covers the dataset, the two model families that are served, how each model and threshold was chosen
before its test set was scored, the exceptions to that protocol, the full results, the localization evaluation and
the history of earlier models. The values come from `backend/app/ai/inference/serving.py`, the final-test reports
under `backend/ai_models/<category>/` (not in Git, see the README), `backend/app/ai/ap_addendum.json` and
`backend/app/ai/box_eval_addendum.json`. The protocols and declarations are in
[backend/scripts/experiments/](../backend/scripts/experiments/).

Terms used throughout:

- **Defective is the positive class.** Recall is the share of defective test images flagged. FPR is the share of
  good test images flagged.
- **Image-level AP** is the area under the image-level precision-recall curve (sklearn
  `average_precision_score`). It is **not** object-detection mAP. Box-level results are reported separately as
  box AP@0.5.
- **Gates** (`category_phase1.gate_classification`, applied to the final-test point estimates):

  | Gate | Recall | F1 | FPR |
  |---|---|---|---|
  | EXCELLENT | >= 0.90 | >= 0.85 | <= 0.10 |
  | GOOD | >= 0.85 | >= 0.80 | <= 0.10 |
  | ACCEPTABLE | >= 0.75 | >= 0.70 | <= 0.15 |
  | NOT PRODUCTION READY | otherwise | | |

## Dataset

MVTec AD (Bergmann et al., CVPR 2019; CC BY-NC-SA 4.0), used with its official train/test split. Training and
threshold selection use only `train/good`. MVTec AD has no labelled defective validation set, so defects seen
before the final test are synthetic (painted onto held-out normal images). The counts below were taken from
`dataset/` on disk. They agree with the image counts in the final-test reports and the localization evaluation.

| Category | Train good | Test good | Test defective | Test total | Defect types | GT masks |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 209 | 20 | 63 | 83 | 3 | 63 |
| cable | 224 | 58 | 92 | 150 | 8 | 92 |
| capsule | 219 | 23 | 109 | 132 | 5 | 109 |
| carpet | 280 | 28 | 89 | 117 | 5 | 89 |
| grid | 264 | 21 | 57 | 78 | 5 | 57 |
| hazelnut | 391 | 40 | 70 | 110 | 4 | 70 |
| leather | 245 | 32 | 92 | 124 | 5 | 92 |
| metal_nut | 220 | 22 | 93 | 115 | 4 | 93 |
| pill | 267 | 26 | 141 | 167 | 7 | 141 |
| screw | 320 | 41 | 119 | 160 | 5 | 119 |
| tile | 230 | 33 | 84 | 117 | 5 | 84 |
| toothbrush | 60 | 12 | 30 | 42 | 1 | 30 |
| transistor | 213 | 60 | 40 | 100 | 4 | 40 |
| wood | 247 | 19 | 60 | 79 | 5 | 60 |
| zipper | 240 | 32 | 119 | 151 | 7 | 119 |
| **Total** | **3,629** | **467** | **1,258** | **1,725** | **73** | **1,258** |

Some test sets are small. Toothbrush has 12 good images, so one false positive is 8.3 percentage points of FPR.
Wood has 19 good images, so one false positive is 5.3 points.

## Models

Both families use a frozen ImageNet backbone; no network weights are trained. "Fitting" a model means storing
patch features of the normal training images. An image's score is its distance from that normal data, and the
image is called defective when the score is above the category's locked threshold.

### WRN-50 PatchCore (9 categories)

bottle, capsule, carpet, grid, hazelnut, pill, screw, wood, zipper.

- Wide ResNet-50-2 (torchvision ImageNet weights, SHA-256 `95faca4d...50b8`), layer 2 and layer 3 features, 3x3
  average pooling.
- Memory bank: a 10% greedy coreset (seed 0) of all train/good patch features.
- Patch score: distance to the nearest patch in the bank (k = 1). Image score: the maximum patch score.
- Input modes: `crop224` (shorter side resized to 256, centre 224x224 crop), `full256` (whole image resized to
  256x256), and, for Grid only, `full320` (whole image resized to 320x320, from Addendum 1).

### ResNet-18 patch models (6 categories)

cable, leather, metal_nut, tile, toothbrush, transistor.

- ResNet-18 (ImageNet weights, SHA-256 `f37072fd...07ec`), layer 2 and layer 3 features (384-d), 3x3 average
  pooling, input 128x128 or 256x256.
- Two scorers: `knn` (Euclidean distance to the nearest patch in a bank built from 10% per-image coresets) and
  `gauss` (Mahalanobis distance to one Gaussian fitted to all normal patches, covariance shrinkage 1e-2).
- Image score: the mean of the top 1% of patch scores.

### Served configuration per category

| Category | Model name | Input | Image score | Threshold policy | Locked threshold |
|---|---|---|---|---|---:|
| bottle | wrn50_patchcore_crop224 | crop224 | max | mean + 2.5 sd | 1.68587 |
| capsule | wrn50_patchcore_crop224 | crop224 | max | mean + 2.5 sd | 1.55368 |
| carpet | wrn50_patchcore_crop224 | crop224 | max | mean + 2.5 sd | 1.55949 |
| grid | wrn50_patchcore_full320 | full320 | max | mean + 2.5 sd | 1.88393 |
| hazelnut | wrn50_patchcore_crop224 | crop224 | max | 99th percentile | 2.54067 |
| pill | wrn50_patchcore_crop224 | crop224 | max | mean + 2.5 sd | 2.07142 |
| screw | wrn50_patchcore_full256 | full256 | max | mean + 2.5 sd | 1.69212 |
| wood | wrn50_patchcore_full256 | full256 | max | mean + 2.5 sd | 1.88538 |
| zipper | wrn50_patchcore_full256 | full256 | max | mean + 2.5 sd | 1.41639 |
| cable | knn_l23_128 | 128x128 | top 1% mean | mean + 3 sd | 2.24479 |
| leather | gauss_l23_128 | 128x128 | top 1% mean | 99th percentile | 29.94421 |
| metal_nut | knn_l23_256 | 256x256 | top 1% mean | mean + 3 sd | 2.01713 |
| tile | knn_l23_256 | 256x256 | top 1% mean | mean + 3 sd | 1.73931 |
| toothbrush | gauss_l23_256 | 256x256 | top 1% mean | mean + 2.5 sd | 37.87112 |
| transistor | knn_l23_256 | 256x256 | top 1% mean | 99th percentile | 2.00267 |

The thresholds are shown rounded; `serving.py` holds them at full precision. On first load, serving verifies the
model file, its config and the backbone weights against SHA-256 values pinned in `serving.py`. A category is only
ever served by its own locked model, with no fallback to another file.

## Evaluation protocol

### Common rules

1. **Normal data only before the test.** Candidates and thresholds are chosen using `train/good` and synthetic
   defects painted on held-out normal images. Test images, test labels and masks are not used.
2. **Lock before scoring.** The chosen configuration, threshold and file hashes are written to a lock file before
   the category's test directory is listed. Each runner checks the lock again before and after scoring.
3. **Test set scored once.** Each final model scores its category's test set once, and the result is reported
   whatever it is. Later runners write one-shot sentinel files (`final_test_sentinel.json`,
   `final_test_consumed.json`) to block a second run.
4. **Gates are fixed in advance** (table above).

### WRN-50 PatchCore study

Declared in `patchcore_wrn50/PROTOCOL.md`, `STUDY_DECLARATION.md` and `FINAL_TEST_DECLARATION.md`, each
committed before the step it governs.

- **Candidates:** 4 per category, from 2 preprocessing modes ({crop224, full256}) times 2 image scores ({max,
  top1pct_mean}).
- **Threshold policies:** 3. These are mean + 2.5 sd and mean + 3 sd (population sd), and the 99th percentile,
  each computed on pooled out-of-fold normal scores.
- **Out-of-fold scoring:** train/good is split into 5 folds deterministically (sorted file names, fold = index
  mod 5). Each fold is scored by a bank built from the other four, so every normal image gets exactly one
  out-of-fold score. 4 synthetic defects are painted on each held-out image and scored against the same fold
  bank.
- **Selection:** among the 12 (candidate, policy) pairs with worst-fold FPR <= 10%, take the highest synthetic
  recall. Ties go to lower worst-fold FPR, then crop224, then max. When no pair meets the limit, take the lowest
  worst-fold FPR.
- **Final model:** a bank fitted on all train/good images. The selected threshold is kept unchanged.
- **Replacement regardless of result:** the WRN-50 model became the final model for all nine categories
  whatever its test result. No choice between the old and new models used test results.
- **Dataset-access record:** every file open and directory listing during the study and the final test was
  recorded and checked. The study read only `train/good`; the final test read only `<category>/test/` and never
  the mask folder.

### ResNet-18 model-family study

Declared in the docstring of `backend/app/ai/evaluation/model_family_study.py` before any candidate was scored,
and applied unchanged to each category.

- **Candidates:** 8, from {gauss, knn} times {layer 2 at 128, layer 3 at 128, layers 2+3 at 128, layers 2+3 at
  256}.
- **Threshold policies:** 8: mean + K sd for K in {1.5, 2.0, 2.5, 3.0}, and the 95th, 97th, 98th and 99th
  percentiles. That gives 64 configurations.
- **Normal-only evaluation:** 30 deterministic splits of train/good (random 5-fold with seeds 42, 43 and 44,
  contiguous 5-fold, and 10 repeated 80/20 splits). Calibration images get leave-one-image-out scores. A synthetic
  diagnostic of 5 defect types times 3 severities is applied to each held-out image of the canonical split.
- **Selection:** worst-fold FPR over the 30 splits <= 10% is required. Synthetic recall >= 0.75 is preferred.
  Ties go to lower worst-fold FPR, then higher synthetic recall, then lower threshold variation, then the simpler
  model.
- **Locked threshold:** the winning policy applied to leave-one-image-out scores of all train/good images.

### Disclosed exceptions

- **The nine WRN-50 test sets had already been scored by earlier models.** `PROTOCOL.md` states: "every test set
  of the nine categories is scored a further time (Carpet a third time)". So the WRN-50 result was not a first
  look at those test sets. The stored reports record these earlier image-level scorings:
  - **Carpet:** the Phase 3 ConvAE, the frozen ResNet-18 model-selection run and its calibration optimization,
    before WRN-50. Counting only the patch models, the WRN-50 test was Carpet's third scoring.
  - **Grid:** the Phase 1 ConvAE and the ResNet-18 model-family study, before WRN-50. Counting only the patch
    models, the WRN-50 test was Grid's second scoring.
  - **The other seven** had each been scored by a ConvAE and, except for Bottle and Hazelnut, by the ResNet-18
    model-family study.
- **Grid full320 (Addendum 1).** Grid's best normal-data synthetic recall in the locked study was 0.853, the
  lowest of the nine. A third input mode (full320) was therefore added as a separate addendum
  (`ADDENDUM_1_GRID_FULL320.md`). The addendum used normal data only and was committed before any WRN-50 test
  scoring. Its 6 rows were pooled with Grid's 12 existing rows, a full320 row won, and Grid's test set was scored
  once with that model. The original crop224 Grid lock was never scored. The addendum records that it was an
  author-initiated change, added after the original plan without outside approval, and disclosed for that reason.
- **Image-level AP for the six ResNet-18 categories was computed afterwards.** Their final-test reports store no
  AP. On 2026-10-07 it was computed from the per-image scores those reports saved, with no re-scoring and no use
  in any selection (`backend/app/ai/ap_addendum.json`).
- **The localization evaluation is a second scoring of all 15 test sets, for localization only.** The rules
  were declared before scoring (`localization_map/PROTOCOL.md`), and no model, threshold, lock or box rule was
  changed. See [Localization](#localization).
- **Overall leakage-check flag false for Capsule, Carpet, Pill and Wood.** In the leakage checks of these four
  WRN-50 final tests, the only failed item is `metrics_match_project_metrics_from_predictions`. It compares the runner's F1 with the F1
  from the project's metrics function using exact floating-point equality, and the two values differ only in the
  last digit (for example 0.8775510204081634 against 0.8775510204081632 for Capsule). Every data-access check for
  those four categories passed: the lock was verified and the sentinel written before listing, no mask path was built or
  opened, and every dataset access stayed inside `<category>/test/`.

## Results

Final-test point estimates. Recall, F1, FPR, AUROC and image AP come from each category's single final test.
Box AP@0.5 and pixel AUROC come from the localization evaluation. AUROC and AP are rounded to 4 decimals, box AP
to 3.

| Category | Model | Recall | F1 | FPR | AUROC | Image AP | Box AP@0.5 | Pixel AUROC | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| bottle | WRN-50 crop224 | 100.0% | 100.0% | 0.0% | 1.0000 | 1.0000 | 0.707 | 0.9834 | EXCELLENT |
| cable | ResNet-18 kNN 128 | 85.9% | 91.9% | 1.7% | 0.9880 | 0.9929 | 0.230 | 0.9770 | GOOD |
| capsule | WRN-50 crop224 | 78.9% | 87.8% | 4.3% | 0.9781 | 0.9952 | 0.095 | 0.9877 | ACCEPTABLE |
| carpet | WRN-50 crop224 | 98.9% | 94.6% | 32.1% | 0.9892 | 0.9968 | 0.029 | 0.9887 | NOT PRODUCTION READY |
| grid | WRN-50 full320 | 98.2% | 98.2% | 4.8% | 0.9908 | 0.9971 | 0.050 | 0.9859 | EXCELLENT |
| hazelnut | WRN-50 crop224 | 100.0% | 100.0% | 0.0% | 1.0000 | 1.0000 | 0.137 | 0.9843 | EXCELLENT |
| leather | ResNet-18 Gaussian 128 | 100.0% | 98.9% | 6.3% | 1.0000 | 1.0000 | 0.000 | 0.9706 | EXCELLENT |
| metal_nut | ResNet-18 kNN 256 | 96.8% | 97.8% | 4.5% | 0.9912 | 0.9982 | 0.504 | 0.9816 | EXCELLENT |
| pill | WRN-50 crop224 | 66.0% | 79.1% | 3.8% | 0.9479 | 0.9899 | 0.097 | 0.9762 | NOT PRODUCTION READY |
| screw | WRN-50 full256 | 70.6% | 82.8% | 0.0% | 0.9508 | 0.9835 | 0.057 | 0.9912 | NOT PRODUCTION READY |
| tile | ResNet-18 kNN 256 | 91.7% | 95.7% | 0.0% | 0.9971 | 0.9989 | 0.604 | 0.9442 | EXCELLENT |
| toothbrush | ResNet-18 Gaussian 256 | 90.0% | 93.1% | 8.3% | 0.9722 | 0.9885 | 0.059 | 0.9876 | EXCELLENT |
| transistor | ResNet-18 kNN 256 | 92.5% | 96.1% | 0.0% | 0.9863 | 0.9854 | 0.290 | 0.9524 | EXCELLENT |
| wood | WRN-50 full256 | 98.3% | 93.7% | 36.8% | 0.9868 | 0.9962 | 0.052 | 0.9410 | NOT PRODUCTION READY |
| zipper | WRN-50 full256 | 99.2% | 97.9% | 12.5% | 0.9892 | 0.9970 | 0.082 | 0.9870 | ACCEPTABLE |

Unweighted means:

| Group | Recall | F1 | FPR | AUROC | Image AP | Box AP@0.5 | Pixel AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| All 15 | 91.1% | 93.8% | 7.7% | 0.9845 | 0.9946 | 0.200 | 0.9759 |
| 9 WRN-50 PatchCore | 90.0% | 92.7% | 10.5% | 0.9814 | 0.9951 | 0.145 | 0.9806 |
| 6 ResNet-18 | 92.8% | 95.6% | 3.5% | 0.9891 | 0.9940 | 0.281 | 0.9689 |

Gate counts: EXCELLENT 8, GOOD 1, ACCEPTABLE 2, NOT PRODUCTION READY 4. The WRN-50 categories have 3 EXCELLENT,
2 ACCEPTABLE and 4 NOT PRODUCTION READY; the ResNet-18 categories have 5 EXCELLENT and 1 GOOD.

Confusion counts (TP / TN / FP / FN): bottle 63/20/0/0, cable 79/57/1/13, capsule 86/22/1/23, carpet
88/19/9/1, grid 56/20/1/1 (recall 56/57 = 98.2%), hazelnut 70/40/0/0, leather 92/30/2/0, metal_nut 90/21/1/3,
pill 93/25/1/48, screw 84/41/0/35, tile 77/33/0/7, toothbrush 27/11/1/3, transistor 37/60/0/3, wood 59/12/7/1,
zipper 118/28/4/1.

How to read the failures:

- **Wood and Carpet fail on FPR** (7 of 19 and 9 of 28 good images flagged), even though their recall and AUROC
  are high. The thresholds came from normal training data and did not transfer to the test set's good images.
- **Pill and Screw fail on recall** (48 and 35 missed defects) at FPR 3.8% and 0%. AUROC is 0.948 and 0.951, so
  the scores rank most defects above normal images, but the normal-only threshold is too conservative.
- **Capsule** passes ACCEPTABLE with recall 78.9%. **Zipper** passes ACCEPTABLE with recall 99.2% but FPR 12.5%.

The application flags every result from a NOT PRODUCTION READY category for manual review.

## Localization

### How boxes and heatmaps are made

Rule `anomaly_map_threshold_v1` in `backend/app/ai/inference/localization.py`:

1. The per-patch score grid that the image score was taken from is bilinearly upsampled to the model's input size
   and blurred with a Gaussian (sigma 4). This gives the anomaly map.
2. For a defective prediction, the mask is the set of pixels at or above the model's own threshold. When blurring
   has pulled the whole map below the threshold, pixels at or above 0.9 times the map's peak are used instead.
3. The mask is split into 8-connected components. Components under 0.25% of the original image area are dropped,
   and at most 5 are kept, ordered by peak.
4. Boxes, the flagged area and the centroid are converted to original-image coordinates. For crop224 models only
   the centre crop was analysed, so pixels outside it are outside the analysed region.
5. The heatmap is an RGBA PNG. It is transparent below half the threshold and outside the analysed region, and
   its alpha rises to 180 (of 255) at the threshold and above.

There is no trained detector, no box regressor and no defect-type classifier. A good prediction gets no boxes.

### Evaluation protocol

`backend/scripts/experiments/localization_map/PROTOCOL.md`, declared on 2026-10-09 before scoring:

- Every test image of the 15 categories is run through the served model exactly as in production. Boxes are made
  only for images predicted defective.
- **Ground-truth boxes, primary view:** each 8-connected component of the defect mask is one box, with no minimum
  area. **Merged view:** one box per defective image, around all its defect pixels.
- **Box AP@0.5:** all predicted boxes of a category are pooled and sorted by confidence (peak / threshold). A
  prediction is a true positive when its IoU with an unmatched ground-truth box of the same image is >= 0.5.
  Boxes on good images are false positives. Defective images predicted good leave their ground-truth boxes
  unmatched. AP is the all-point interpolated area under the precision-recall curve.
- **Pixel AUROC:** the mask is cropped to the analysed region and resized to the map resolution. Pixels outside
  the analysed region are excluded. AUROC is computed from a fixed 14,000-bin histogram of `map / threshold`.

### Results

| Category | GT boxes | GT boxes below min area | Predicted boxes | TP | Defective images with no box | Box AP@0.5 | Box AP@0.5, merged | Pixel AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle | 68 | 4 | 67 | 54 | 0 | 0.707 | 0.741 | 0.9834 |
| cable | 151 | 36 | 82 | 44 | 15 | 0.230 | 0.420 | 0.9770 |
| capsule | 114 | 33 | 76 | 25 | 34 | 0.095 | 0.103 | 0.9877 |
| carpet | 97 | 8 | 116 | 14 | 2 | 0.029 | 0.043 | 0.9887 |
| grid | 170 | 112 | 91 | 27 | 4 | 0.050 | 0.042 | 0.9859 |
| hazelnut | 136 | 51 | 69 | 35 | 1 | 0.137 | 0.421 | 0.9843 |
| leather | 99 | 32 | 101 | 0 | 0 | 0.000 | 0.000 | 0.9706 |
| metal_nut | 132 | 17 | 111 | 73 | 4 | 0.504 | 0.533 | 0.9816 |
| pill | 245 | 109 | 94 | 36 | 57 | 0.097 | 0.189 | 0.9762 |
| screw | 135 | 56 | 80 | 23 | 50 | 0.057 | 0.076 | 0.9912 |
| tile | 86 | 2 | 84 | 54 | 7 | 0.604 | 0.618 | 0.9442 |
| toothbrush | 66 | 40 | 41 | 8 | 3 | 0.059 | 0.148 | 0.9876 |
| transistor | 44 | 1 | 49 | 14 | 3 | 0.290 | 0.308 | 0.9524 |
| wood | 168 | 49 | 139 | 31 | 1 | 0.052 | 0.109 | 0.9410 |
| zipper | 177 | 1 | 173 | 48 | 2 | 0.082 | 0.025 | 0.9870 |

Means of box AP@0.5 (primary view): 0.200 over all 15, 0.145 over the 9 WRN-50 categories and 0.281 over the 6
ResNet-18 categories. With merged ground-truth boxes the means are 0.252, 0.194 and 0.338. The means of pixel
AUROC are 0.976, 0.981 and 0.969.

Why box AP is low while pixel AUROC is high:

- The map is blurred (sigma 4) and the mask is cut at the image-level threshold. Flagged regions are therefore
  wider than the defect, and boxes around thin or small defects rarely reach IoU 0.5. Leather has 101 predicted
  boxes and none matches.
- Ground-truth pieces under the 0.25% minimum area can never be matched by design. They are 112 of 170 GT boxes
  for Grid, 109 of 245 for Pill and 40 of 66 for Toothbrush.
- Pixel AUROC only asks whether defect pixels score higher than normal pixels. The maps do rank pixels well
  (mean 0.976), even where the boxes drawn from them are poor.

## History of earlier models

This section is history: none of these models is served now. Each claim was checked against the stored reports
in `backend/ai_models/<category>/`.

**Convolutional autoencoder (ConvAE).** The first model was a whole-image convolutional autoencoder scored by
reconstruction MSE. It never passed a gate in any category where it was tried:

- Phase 1 baselines (`phase1_baseline/`), test-set recall: 0% for cable, grid, metal_nut, screw, tile, toothbrush
  and wood; 3.4% zipper, 5.7% pill, 7.5% transistor, 11.9% capsule. Leather reached 69.6% recall, but at 93.8%
  FPR. Screw and Wood had AUROC 0.967 and 0.924 with 0% recall, so the ranking was there and the threshold was
  the problem.
- Bottle: the validated Phase 3 ConvAE reached recall 73.0% at FPR 30.0% and was served until WRN-50 replaced it.
  A Phase 4 optimization gave recall 66.7% at FPR 40.0% and was not adopted.
- Hazelnut: Phase 3 ConvAE recall 37.1% (FPR 0%). Phase 4 gave 42.9%, and the report records "PHASE 3 BASELINE
  RETAINED".
- Carpet: Phase 3 ConvAE recall 0%.

**Carpet frozen ResNet-18 attempts.** A Gaussian patch model (`pgauss_l23_256_top1pct_mean`) reached recall 98.9%
at FPR 57.1%. A calibration
optimization then gave recall 97.8% at FPR 39.3%. Both failed on FPR. The WRN-50 model (recall 98.9%, FPR 32.1%)
still fails on FPR.

**ResNet-18 model-family study, 12 categories.** It produced the six served ResNet-18 models (table above). It
failed for the other six, which WRN-50 later replaced: capsule (recall 26.6%), grid (22.8%), pill (55.3%), screw
(7.6%), zipper (66.4%), and wood (recall 98.3% but FPR 21.1%).

**WRN-50 PatchCore replacement.** The WRN-50 study replaced the previous final model in all nine categories,
regardless of the test result (`backend/ai_models/patchcore_wrn50_summary.md`):

| Category | Previous recall | Previous FPR | WRN-50 recall | WRN-50 FPR |
|---|---:|---:|---:|---:|
| bottle | 73.0% | 30.0% | 100.0% | 0.0% |
| capsule | 26.6% | 0.0% | 78.9% | 4.3% |
| carpet | 97.8% | 39.3% | 98.9% | 32.1% |
| grid | 22.8% | 0.0% | 98.2% | 4.8% |
| hazelnut | 37.1% | 0.0% | 100.0% | 0.0% |
| pill | 55.3% | 0.0% | 66.0% | 3.8% |
| screw | 7.6% | 0.0% | 70.6% | 0.0% |
| wood | 98.3% | 21.1% | 98.3% | 36.8% |
| zipper | 66.4% | 9.4% | 99.2% | 12.5% |

Wood's FPR was higher with WRN-50 than with the model it replaced. The replacement was kept, as the protocol
requires.
