# Milestone 2: Image processing and defect detection

## Objective

The second milestone adds the computer-vision layer. Images are preprocessed and measured, anomaly-detection models
are trained on defect-free MVTec AD images, and every inspection receives a prediction that the frontend can show
and monitor. By the end of this milestone the application should make and store its own defect predictions,
rather than only record what the dataset folders say.

## What was built

The preprocessing pipeline was committed on 5 September (`72b3f77`). It decodes an image with OpenCV, converts it to
RGB, resizes it with area interpolation and scales it to 0..1. The same commit added an image-quality module,
`backend/app/ai/analytics/quality.py`, which reports brightness (mean intensity), contrast (intensity standard
deviation) and sharpness (variance of the Laplacian), with a blur flag below a variance of 100. The training-data
layer in `backend/app/ai/training/` lists the MVTec samples and their labels. The training and evaluation code
uses this pipeline; the models served today use their own preprocessing in
`backend/app/ai/preprocessing/patchcore_preprocess.py`.

The first model was a convolutional autoencoder trained on Bottle `train/good` (`92db936`) and evaluated in
`bf1d255`. With its first threshold (mean plus 3 standard deviations of the training errors) it found 29 of 63
defective test images (recall 46.0%) and flagged 1 of 20 good images (FPR 5%). The prediction engine
(`7a61371`) turned a trained model into an `app.ai.inference.predict_image` call. `5294ed5` connected it to the
inspection workflow: every dataset import now ran inference and stored `ai_prediction`, the score, the threshold
and the model name in new columns. If inference fails, the inspection is still created and those columns stay
empty. `1acff93` added the analytics summary endpoint and the AI monitoring view on the dashboard, with counts of
analysed inspections and AI predictions.

On 15 September the Bottle model went through a validated evaluation in three steps:

- `972c203` added a held-out evaluation and threshold experiments.
- `19ddc59` retrained the autoencoder with a held-out validation split that was never used for training. This
  "Phase 3" model reached recall 73.0% at FPR 30.0% and was the served Bottle model until 4 October.
- `7b65b25` tried a controlled optimization ("Phase 4"). It gave recall 66.7% at FPR 40.0% and was not adopted.

Commit `4118bf7` (22 September) carries the title "Milestone 3: AI defect detection and model optimization", but by
content it belongs here. It added the frozen ImageNet ResNet-18 patch models (`backend/app/ai/models/patch_anomaly.py`,
`resnet18.py`), the synthetic-defect generator and the normal-data-only model-family study used to choose and lock
a model and threshold per category. `b224a77` (24 September) added the generic per-category training and
optimization engines and the serving registry (`backend/app/ai/inference/serving.py`). The registry pins each
category to exactly one locked, hash-verified model.

The autoencoder failed the project's gates in every category where it was tried. The ResNet-18 study produced
passing models for cable, leather, metal_nut, tile, toothbrush and transistor. The remaining nine categories were
later replaced by WRN-50 PatchCore models (Milestone 4). [ML_METHODOLOGY.md](../ML_METHODOLOGY.md) has the
category-by-category history.

Delivered in a later iteration:

- Tile and Cable were registered for serving on 2 October (`9a44113`).
- Uploaded images received predictions from 3 October (`363816a`, `ae74168`). The product's MVTec category now
  selects the model for its uploads.
- All 15 categories were served from 4 October (`188c903`). This added the shared backbones, the cached
  per-category models and `GET /ai/models`.
- The image-quality measurements became visible to users on 4 October (`c847154`). This added
  `GET /inspections/{id}/image-quality` and an enhancement preview with denoising and CLAHE contrast. The preview
  is computed per request, is never stored, and is never used by the models.

## Deliverables

| Task or criterion | Status | Evidence |
|---|---|---|
| Image preprocessing pipeline | DONE | `72b3f77`: `backend/app/ai/preprocessing/pipeline.py`, `backend/tests/test_ai_preprocessing.py` |
| Image quality analysis reports | DONE | `72b3f77`: `backend/app/ai/analytics/quality.py`. The endpoint was delivered in a later iteration: `c847154` (4 October). |
| Image analytics workflows | DONE | Quality metrics in the pipeline (`72b3f77`); before/after metrics of the enhancement preview (`c847154`, 4 October) |
| Train defect detection models | DONE | `92db936`, `19ddc59`, `4118bf7`; final models in Milestone 4 |
| Generate defect predictions | DONE | `7a61371`, `5294ed5`. Predictions for uploads were delivered in a later iteration: `363816a` (3 October). |
| Inspection monitoring dashboard | DONE | `1acff93`: `backend/app/inspections/analytics.py`, dashboard AI section |
| Processing and detection workflows implemented | DONE | as above |
| Defect prediction engine functional | DONE | `7a61371`: `backend/app/ai/inference/predict.py`, `backend/tests/test_ai_inference.py` |
| Inspection reports generated | DONE | The per-inspection report came with Milestone 3 (`d5a3b87`, 14 September); model evaluation reports are stored for every category |
| Computer vision models integrated | DONE | Bottle from `5294ed5`. Tile and Cable from `9a44113` (2 October) and all 15 categories from `188c903` (4 October), delivered in later iterations. |

## Commits

| Commit | Date | Content |
|---|---|---|
| `72b3f77` | 2026-09-05 | Preprocessing pipeline, image-quality analysis, training-data layer |
| `92db936` | 2026-09-05 | First convolutional autoencoder trained on Bottle |
| `bf1d255` | 2026-09-05 | Evaluation of that model |
| `7a61371` | 2026-09-05 | Inference engine (`predict_image`) |
| `5294ed5` | 2026-09-05 | AI prediction columns and inference on inspections |
| `1acff93` | 2026-09-05 | Analytics summary and AI monitoring dashboard |
| `972c203` | 2026-09-15 | Validated evaluation and threshold experiments (Bottle) |
| `19ddc59` | 2026-09-15 | Bottle model retrained with a held-out validation split |
| `7b65b25` | 2026-09-15 | Controlled optimization experiment (not adopted) |
| `4118bf7` | 2026-09-22 | ResNet-18 patch models and the model-family study (title says "Milestone 3") |
| `b224a77` | 2026-09-24 | Per-category training and optimization engines, serving registry |

Delivered in a later iteration:

| Commit | Date | Content |
|---|---|---|
| `9a44113` | 2026-10-02 | Tile and Cable patch models registered for serving |
| `363816a` | 2026-10-03 | Product MVTec category; predictions for uploads |
| `ae74168` | 2026-10-03 | Product category in the UI and AI-state messages for uploads |
| `188c903` | 2026-10-04 | All 15 categories served; shared backbones and model cache; `GET /ai/models` |
| `c847154` | 2026-10-04 | Image-quality endpoint and enhancement preview (also severity and defect form, see Milestone 3) |

## Results

At the end of this milestone's own dates, only Bottle was served, by the autoencoder at recall 73.0% and FPR 30.0%,
which does not meet any gate. The models that passed the gates in the ResNet-18 study are listed below, from
`backend/app/ai/inference/serving.py` and their stored final-test reports. The full 15-category results, with the
WRN-50 models that replaced the other nine, are in [Milestone 4](MILESTONE_4.md).

| Category | Model | Recall | FPR | AUROC | Gate |
|---|---|---:|---:|---:|---|
| cable | ResNet-18 kNN 128 | 85.9% | 1.7% | 0.9880 | GOOD |
| leather | ResNet-18 Gaussian 128 | 100.0% | 6.3% | 1.0000 | EXCELLENT |
| metal_nut | ResNet-18 kNN 256 | 96.8% | 4.5% | 0.9912 | EXCELLENT |
| tile | ResNet-18 kNN 256 | 91.7% | 0.0% | 0.9971 | EXCELLENT |
| toothbrush | ResNet-18 Gaussian 256 | 90.0% | 8.3% | 0.9722 | EXCELLENT |
| transistor | ResNet-18 kNN 256 | 92.5% | 0.0% | 0.9863 | EXCELLENT |

## Limitations

The autoencoder never reached a gate, so for most of this period the predictions had little practical value outside
the six ResNet-18 categories. Until 3 October, uploads received no prediction. The models detect anomalies; they do
not name the defect type. The enhancement preview is a display aid: it is neither stored nor used for scoring.

## How to verify

- Live deployment: http://13.206.115.50. Log in, import a dataset image on the Dataset page, then open the
  inspection to see the prediction, score, threshold and model.
- `GET /api/ai/models` (with a bearer token) lists the served model per category, as on the Models page.
- Tests, from `backend/`:
  `python -m pytest tests/test_ai_preprocessing.py tests/test_ai_inference.py tests/test_ai_serving.py tests/test_ai_all_categories_serving.py tests/test_inspection_ai_integration.py`.
