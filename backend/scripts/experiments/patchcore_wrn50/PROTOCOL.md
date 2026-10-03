# PatchCore WRN-50-2 protocol (declared before any test-set scoring)

- **Scope:** bottle, capsule, carpet, grid, hazelnut, pill, screw, wood, zipper. The six ResNet-18 patch
  models (tile, cable, leather, metal_nut, toothbrush, transistor) are retained and are not re-scored.
- **Backbone:** WideResNet-50-2 (ImageNet, torchvision weights, SHA-256 95faca4d...50b8), layers 2+3,
  3x3 average pooling, k=1 nearest neighbour, 10% greedy coreset, seed 0.
- **Candidates per category (4):** preprocessing {crop224, full256} x image score {max, top1pct_mean}.
- **Threshold policies (3):** mean+2.5 sigma, mean+3 sigma, 99th percentile of pooled out-of-fold normal
  scores.
- **Out-of-fold scoring:** deterministic 5-fold split of train/good (sorted filenames, fold = index mod 5).
  Each fold's bank is built from the other four folds; every train/good image gets exactly one out-of-fold
  score. Threshold statistics use the pooled out-of-fold scores of all train/good images.
- **Synthetic defects:** painted on held-out normal images using the existing synthetic defect generator
  of the model-family study; test images are never used.
- **Selection:** among (candidate, policy) pairs whose worst-fold FPR on normal data is <= 10%, choose the
  highest synthetic recall; ties go to lower worst-fold FPR, then to crop224, then to max. If no pair meets
  the FPR limit, choose the lowest worst-fold FPR.
- **Lock:** the chosen configuration, threshold, bank and file hashes are written to a lock file BEFORE
  the test directory of that category is listed.
- **Final model:** bank fitted on ALL train/good images of the category with the chosen configuration;
  threshold as selected.
- **Test:** each category's test set is scored exactly once by this model. The test result is reported
  whatever it is.
- **Replacement:** the WRN model becomes the category's final model for all nine categories, regardless
  of its test result. The previous model's result stays in the documentation. No choice between old and
  new models is made using test results.
- **Disclosure:** every test set of the nine categories is scored a further time (Carpet a third time).
  Reported metrics: accuracy, precision, recall, F1, FPR, AUROC, average precision, TP/TN/FP/FN, per-defect
  recall, plus per-image scores.
- **Gates unchanged:**
  - EXCELLENT: recall >= 0.90, F1 >= 0.85, FPR <= 0.10
  - GOOD: recall >= 0.85, F1 >= 0.80, FPR <= 0.10
  - ACCEPTABLE: recall >= 0.75, F1 >= 0.70, FPR <= 0.15
