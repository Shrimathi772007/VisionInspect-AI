# Localization evaluation protocol (box AP@0.5, pixel AUROC)

Declared 2026-10-09, before any scoring. Nothing in this protocol is changed, and no metric, rule or view is added,
after results have been seen. The evaluation script is `run_box_eval.py` in this folder; its SHA-256 and the SHA-256
of this file are recorded in every output manifest.

## Status and disclosure

- **Second scoring of the final test sets, for localization only.** Every category's `dataset/<category>/test/`
  split was already scored once by its final test. This evaluation runs the served models over the same images a
  second time, only to measure where the anomaly-map boxes and maps point. It is disclosed as such wherever the
  results are shown.
- **No model, threshold, lock, box rule, quality rule or inference code is changed**, and no result is used to
  select or tune anything. Image-level predictions are recorded per image (they decide which images get boxes) but no
  new image-level metric is reported; the image-level results remain those of the final tests.
- **Anomaly-map boxes, not a trained detector.** The boxes come from thresholding the served anomaly map
  (`anomaly_map_threshold_v1`). There is no box regressor and no defect-type classification.
- The script never connects to the database and refuses to read a real `test/` split unless run with `--allow-test`.

## Data

- All images in `dataset/<category>/test/<defect_type>/*.png` for the 15 MVTec AD categories, every defect type
  including `good`, processed in sorted (defect_type, filename) order.
- Ground truth for a defective image: `dataset/<category>/ground_truth/<defect_type>/<stem>_mask.png`; a pixel is a
  defect pixel when its value is > 0. A defective image without a mask stops that category with an error. `good`
  images have no mask and are all-negative.

## Predictions (exactly as served)

For every image, `app.ai.inference.predict_image(path, category, return_patch_scores=True)` with no other argument:
the served model, locked threshold, input size/mode and aggregation, loaded through the serving loaders (artifact,
backbone and config hashes verified; Grid also verifies its study-logic hashes). Then the steps of
`app.inspections.service._localize_and_render`, without writing a heatmap:

1. `(H, W)` = shape of `app.ai.preprocessing.patchcore_preprocess.decode_image(path)`.
2. `amap = localization.anomaly_map(result.patch_scores, result.input_size[0])` (S x S, the served map resolution).
3. `region = localization.map_region(result.input_mode, W, H)`.
4. Defective prediction: `localization.localize(amap, result.threshold, region)`; good prediction:
   `localization.empty_localization()` (no boxes).

Box confidence = `box["peak"] / result.threshold`.

## Ground-truth boxes

- **Primary view:** each 8-connected component of the defect mask is one box, with no minimum area.
- **Secondary view:** one box per defective image, the bounding box of all defect pixels.
- Boxes use continuous original-image pixel coordinates: a component with left `x`, top `y`, width `w`, height `h`
  (OpenCV stats) is `[x, y, x + w, y + h]`. Predicted normalised boxes are scaled by `(W, H)` into the same space.

## Matching and AP@0.5

Per category and per view:

1. Pool all predicted boxes of all images and sort by confidence, descending. Ties keep (image order, box order).
2. Walk the list; a box is a true positive when its best IoU with a not-yet-matched GT box **of the same image** is
   >= 0.5 (that GT box is then matched); otherwise it is a false positive. One-to-one: each GT box matches at most
   one prediction, each prediction at most one GT box.
3. Boxes on `good` images are false positives. GT boxes of defective images that received no box (predicted good)
   stay unmatched, i.e. count as misses in the recall denominator.
4. AP = all-point interpolated area under the precision-recall curve (precision made monotonically non-increasing
   from the right, summed over recall steps; VOC 2010+ style). A category with GT boxes but no predicted box has
   AP = 0.

Means (unweighted): mAP over all 15 categories; over the 9 WRN-50 PatchCore categories (bottle, capsule, carpet,
grid, hazelnut, pill, screw, wood, zipper); over the 6 ResNet-18 categories (cable, leather, metal_nut, tile,
toothbrush, transistor). The same three means are reported for pixel AUROC.

Also reported per category and view: images, defective images, GT boxes, predicted boxes, true positives, defective
images that received no box ("missed images"), GT boxes smaller than the predictor's minimum component area (pixel
area < 0.25% of the image, so they can never be matched by design), and GT boxes not fully inside the analysed region
(crop224 categories only can have any).

## Pixel AUROC (secondary)

- Per image, the GT mask is cropped to the analysed region (pixel bounds `round(x0*W)`, `round(y0*H)`,
  `round(x1*W)`, `round(y1*H)`) and resized to S x S with nearest-neighbour interpolation, i.e. **the mask is
  brought to the map resolution**. Pixels outside the analysed region are excluded (the model did not see them);
  the number of GT defect pixels excluded this way is reported per category.
- `good` images contribute all-negative pixels. Every image contributes, whatever its prediction.
- Memory is bounded by a fixed histogram, declared here: the score ratio `r = amap / threshold` is binned on
  `log10(r)` over [-4, 3] in 14,000 equal bins (0.0005 decade each), plus one underflow bin (r <= 1e-4, including
  r <= 0) and one overflow bin (r > 1e3). Positive and negative pixel counts are accumulated per bin. AUROC =
  (sum over bins of pos_in_bin x neg_below_bin + 0.5 x pos_in_bin x neg_in_bin) / (P x N), i.e. pixels in the same
  bin count as ties. The result differs from the exact AUROC only through those within-bin ties.

## Outputs

- `<out>/categories/<category>.json`: per-image records (prediction, score, boxes, GT boxes of both views), the
  sparse histogram and the category metrics; written atomically when the category finishes.
- `<out>/report.json`: per-category metrics and the means. `<out>/manifest.json`: SHA-256 of `run_box_eval.py`,
  this file, every lock file used (from each serving config's provenance), and `app/ai/inference/{predict,
  localization,serving}.py`; library versions; date.
- **Resume:** a finished category file is reused only when the script, protocol and lock hashes recorded in it equal
  the current ones; a mismatch stops the run instead of mixing results. Unfinished categories are recomputed from
  their first image.
