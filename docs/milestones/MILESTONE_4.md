# Milestone 4: Validation, optimization, deployment and documentation

- **Milestone:** 4 of 4
- **Period:** 2026-09-20 to 2026-10-17 (first to last commit listed under Commits)
- **Deliverables:** 7 of 8 rows DONE; see Deliverables for the full status

## Contents

1. [Objective](#objective)
2. [What was built](#what-was-built)
3. [Technical documentation](#technical-documentation)
4. [Deliverables](#deliverables)
5. [Commits](#commits)
6. [Results](#results)
7. [Limitations](#limitations)
8. [How to verify](#how-to-verify)
9. [Troubleshooting](#troubleshooting)
10. [Glossary](#glossary)

## Objective

The last milestone takes the platform from a development setup to a measured, deployed system. The detection models
are validated on held-out test data under a declared protocol, and the weak ones are replaced. Inspection
performance and the dashboard are improved. The full stack runs with Docker on a cloud VM (virtual machine), and the system is
documented for other engineers.

## What was built

**Performance instrumentation** (`82a319f`, 20 September). Every inspection now records its server-side handling
time (`processing_time_ms`) and the time of the model call (`ai_inference_time_ms`). The dashboard shows these over
the selected 7, 14 or 30-day window. The analytics summary also reports an automation rate: the share of decided
inspections whose decision was PASS or FAIL rather than `MANUAL_REVIEW` or NOT_ASSESSED. It is computed live from
the database, and no value is stored in the repository.

**Model validation and replacement** (3 to 4 October). By the end of September, nine categories (bottle, capsule,
carpet, grid, hazelnut, pill, screw, wood and zipper) had no model that met its gate. A WRN-50 (Wide ResNet-50-2) PatchCore study
replaced all nine:

- `60447b1` added the detector and declared the evaluation protocol.
- `936be59` declared the study details.
- `0506f95` selected and locked a model and threshold per category, using normal training images and synthetic
  defects only.
- `8a7bc0a` and `b84534d` declared and ran an addendum that added a 320x320 input mode for Grid, again on normal
  data only.
- `9b5c781` declared the final-test procedure.
- `e914b82` scored each of the nine test sets once with its locked model.

The protocol made the WRN-50 model the final model for all nine categories whatever its test result. `188c903`
then served all 15 categories, sharing one frozen backbone per model family and caching at most three
per-category models (`VISIONINSPECT_AI_MODEL_CACHE_SIZE`, default 3).

**Further evaluation.**
- `3bb9829` (7 October) computed image-level average precision for the six ResNet-18 categories from their saved
  final-test scores, without re-scoring.
- `531d111` and `903e54c` (9 October) declared and ran a localization evaluation: box AP@0.5 (average precision of boxes at an intersection over union of 0.5) and pixel AUROC (area under the
  receiver operating characteristic curve, computed per pixel) of
  the anomaly-map boxes. It is disclosed as a second scoring of the test sets, used for localization only.
- `c8b4706` archived the per-category experiment scripts with a SHA-256 (Secure Hash Algorithm, 256-bit) manifest.

[ML_METHODOLOGY.md](../ML_METHODOLOGY.md) describes the protocol and every disclosed exception.

**Testing.**
- `fe2b4c5` moved the backend tests to a separate test database and added the seed script for demo data
  (extended in `79804e2`).
- `fc1cab0` and `09c639e` added the Postman collection, which covers all 29 API (application programming interface) operations and has a negative-test
  folder.

**Dashboard and interface.**
- `509a6cd` refined the dashboard and added the per-category breakdown.
- `1d1ae52` fixed the mobile layout of the inspection list and the dashboard.
- `9f28c4c` and `455ee2b` fixed details of the camera and severity views.

**Deployment.**
- `d0f72e0` (6 October) added the Dockerfiles, `docker-compose.yml` and [DEPLOY.md](../DEPLOY.md).
  `docker-compose.yml` runs three services: `web` (nginx and the built frontend), `backend` and `db`.
- `99f2656` recorded a tested local Docker build.
- `7a507ec` (17 October) recorded the deployment on an AWS (Amazon Web Services) EC2 (Elastic Compute Cloud) VM in ap-south-1 (Mumbai) and the redeploy
  procedure.

**Documentation.**
- `2b57933`, `5b28e85` and `e013c0c` added [README.md](../../README.md), [ARCHITECTURE.md](../ARCHITECTURE.md),
  [ML_METHODOLOGY.md](../ML_METHODOLOGY.md) and the licence.
- `df7499d` removed an absolute path from the experiments README.

## Technical documentation

### Scope

This milestone covers the validation of the 15 served models, the replacement study for nine categories, the
localization evaluation, the test suites, and the Docker deployment on AWS. The machine-learning (ML) protocol is in
[ML_METHODOLOGY.md](../ML_METHODOLOGY.md#evaluation-protocol) and the deployment steps are in
[DEPLOY.md](../DEPLOY.md); this section adds the milestone's experiments, analysis, tests and safeguards.

### Experiments

The table below lists the evaluation runs of this milestone in commit order, with the protocol documents in
`backend/scripts/experiments/` that declared each one before it ran.

| Run | Settings | Result | Decision |
|---|---|---|---|
| WRN-50 study, 9 categories (`0506f95`) | 4 candidates (crop224 or full256, max or top-1% mean) x 3 threshold policies; 5-fold out-of-fold scores on `train/good`; synthetic defects | one model and threshold locked per category (`patchcore_wrn50/PROTOCOL.md`) | Final test |
| Grid full320 addendum (`b84534d`) | 6 extra rows at 320x320 pooled with Grid's 12 rows; normal data only | a full320 row won (`ADDENDUM_1_GRID_FULL320.md`) | Grid scored with full320 |
| WRN-50 final test (`e914b82`) | each test set scored once with its locked model | 3 EXCELLENT, 2 ACCEPTABLE, 4 NOT PRODUCTION READY (`serving.py`) | Served from `188c903`, whatever the result, as declared |
| Image-level AP addendum (`3bb9829`) | AP from the saved per-image scores of the six ResNet-18 final tests; no re-scoring | `backend/app/ai/ap_addendum.json` | Shown on the Models page |
| Localization evaluation (`903e54c`) | all 15 test sets scored a second time, for localization only (`localization_map/PROTOCOL.md`) | `backend/app/ai/box_eval_addendum.json` | Shown on the Models page |

### How to read box AP@0.5

Box AP@0.5 measures how well the boxes match the defects. The boxes are cut from the anomaly map of a model trained
on defect-free images only, at that model's image-level threshold. They are scored against boxes drawn around each
connected region of the ground-truth masks of the official MVTec AD (MVTec Anomaly Detection) test split; a box counts as a match when its IoU
(intersection over union) with a ground-truth box is at least 0.5. The figure therefore combines three things: the
image-level decision (only images predicted defective get boxes), the extent of the blurred map, and the minimum box
area of 0.25% of the image (`backend/app/ai/inference/localization.py`). Pixel AUROC instead ranks every pixel of
the map against the mask and does not depend on the threshold.

### Localization analysis

The figures below come from `backend/app/ai/box_eval_addendum.json` (tracked). Per-defect-type counts and false
positives come from the final-test reports in the models ZIP (`final_test_result.json` and `per_image.csv` under
`backend/ai_models/<category>/`).

**Pixel AUROC.** The mean is 0.9759 over 15 categories, ranging from 0.9410 for wood to 0.9912 for screw.

**Counts behind box overlap.** The table lists, per category, the ground-truth (GT) boxes, those below the minimum
box area, those not fully inside the analysed area (only crop224 models analyse a centre crop), and the defective
images that received no box.

| Category | GT boxes | Below minimum area | Outside analysed area | Defective images without a box |
|---|---:|---:|---:|---:|
| grid | 170 | 112 | 0 | 4 of 57 |
| pill | 245 | 109 | 5 | 57 of 141 |
| toothbrush | 66 | 40 | 0 | 3 of 30 |
| hazelnut | 136 | 51 | 0 | 1 of 70 |
| screw | 135 | 56 | 0 | 50 of 119 |
| capsule | 114 | 33 | 3 | 34 of 109 |
| carpet | 97 | 8 | 11 | 2 of 89 |
| bottle | 68 | 4 | 4 | 0 of 63 |

**Per-defect-type recall.** The table shows the defect types with the lowest recall in pill, screw and capsule (the
categories whose recall is below 80%) and in cable, as detected defective images out of that type's test images.

| Category | Defect types with the lowest recall |
|---|---|
| pill | crack 8/26, color 16/25, faulty_imprint 13/19, contamination 15/21 |
| screw | thread_side 12/23, manipulated_front 13/24, scratch_head 14/24 |
| capsule | scratch 13/23, faulty_imprint 16/22, poke 16/21 |
| cable | poke_insulation 4/10, missing_wire 7/10 |

**False positives.** The table gives the good test images flagged as defective, with the Wilson 95% interval of the
false-positive rate (FPR) stored in each report, and the range of those images' scores divided by the locked
threshold.

| Category | Flagged good images | Wilson 95% interval of FPR | Score / threshold of those images |
|---|---:|---|---|
| wood | 7 of 19 | 19.1% to 59.0% | 1.003 to 1.194 |
| carpet | 9 of 28 | 17.9% to 50.7% | 1.049 to 1.246 |
| zipper | 4 of 32 | 5.0% to 28.1% | 1.001 to 1.326 |

Hypothesis, not proven by the data: since every flagged good image of wood and carpet scored within 1.25 times the
threshold, threshold placement explains more of their false-positive rate than the features do.

### Tests

The table shows the backend tests per area, from `pytest --collect-only` on 17 October 2026 (1712 tests in 60
modules), and the frontend tests per folder from the Vitest JSON report (324 tests in 42 files).

| Suite | Area | Tests | Modules or files |
|---|---|---:|---:|
| Backend | Model training, evaluation and study logic | 670 | 30 |
| Backend | Localization, confidence, severity, defect form, quality, report | 458 | 8 |
| Backend | Model serving and inference | 172 | 6 |
| Backend | Analytics, trends, performance | 148 | 5 |
| Backend | Products, uploads, dataset, health | 108 | 6 |
| Backend | Demo data seed script | 68 | 1 |
| Backend | Authentication, users, login limit | 66 | 3 |
| Backend | Batch upload and enhancement preview | 22 | 1 |
| Frontend | Pages (`frontend/src/pages/`) | 159 | 15 |
| Frontend | Components (`frontend/src/components/`) | 88 | 15 |
| Frontend | Utilities, hooks, API clients, constants, app routes | 77 | 12 |

The backend suite runs against its own `{POSTGRES_DB}_test` database. `backend/tests/conftest.py` stops the run if
that resolves to the development database, and it writes uploads to a temporary folder. The Postman collection adds
requests against a running backend for all 29 operations, with negative tests; requests that write run only when
`allowWrites` is `true` ([postman/README.md](../../postman/README.md)).

### Guarantees and where they are enforced

The table lists the statements the code enforces, each with the test or code path that checks it.

| Statement | Enforced by |
|---|---|
| A changed model, configuration or backbone file is refused at load | `test_wrong_pinned_hash_refuses_to_load`, `test_wrong_pinned_backbone_hash_refuses_to_load`, `test_tampered_wrn50_bank_in_place_is_refused` |
| A category without a registry entry loads no model; there is no fallback to another category | `test_unknown_category_has_no_model_and_loads_nothing`; `SERVING_CONFIGS` has no default entry |
| A product without a category gets no AI (artificial intelligence) result | `test_upload_for_product_without_category_never_calls_predict_image` |
| A failing or changed model does not fail the upload | `test_upload_survives_a_raising_predictor_without_leaking_details`, `test_tampered_model_never_fails_an_upload_and_leaks_nothing` |
| Ground truth and AI output stay independent | `test_ground_truth_and_ai_prediction_remain_independent`, `test_false_positive_import_gets_severity_but_keeps_its_ground_truth` |
| Self-registration cannot create a quality engineer; the last quality engineer cannot be demoted | `test_register_cannot_choose_quality_engineer_role`, `test_demoting_the_last_quality_engineer_returns_409_and_changes_nothing` |
| Password hashes do not appear in responses | `test_register_login_and_me_never_expose_password_hash` |
| A locked login stays locked with the right password, and an unknown email gets the same 429 | `test_lockout_after_max_failures_even_with_the_right_password`, `test_unknown_email_gets_the_same_429_as_a_real_account` |
| Dataset and stored paths cannot leave their root | the seven path tests in `test_dataset.py`, `test_delete_inspection_rejects_path_traversal_in_stored_image_path` |
| At most three category models are cached, least recently used evicted first | `test_lru_cache_evicts_the_least_recently_used_model` |
| The enhancement preview is not used for scoring | code path: `service._resolve_absolute_path` gives inference the original file |

### Security measures

The table lists each threat the code addresses, the measure, and the scope of that measure.

| Threat | Measure | Scope and notes |
|---|---|---|
| Self-assigned role | Registration sets the role to factory supervisor | Quality engineers are created by `create_quality_engineer.py` or promoted by another quality engineer |
| Password guessing | Limiter: 5 failures per client IP (Internet Protocol) address and email, 20 per IP, within 900 seconds | Counters are in memory per worker and reset on restart ([DEPLOY.md](../DEPLOY.md#login-rate-limiting)) |
| Stored passwords | bcrypt hashes; passwords over 72 bytes refused | No response includes the hash (`test_register_login_and_me_never_expose_password_hash`) |
| Forged or old tokens | HS256 (keyed SHA-256) signature with `JWT_SECRET_KEY`; 60-minute expiry; user reloaded per request | Logout clears the token in the browser; the token stays valid until it expires |
| Harmful uploads | Extension, size, emptiness and Pillow checks; random file names | Limits in [Milestone 1](MILESTONE_1.md#upload-validation-and-storage) |
| Path traversal | Validated dataset paths; `resolve_image_path` containment | Covered by the traversal tests above |
| Changed model files | SHA-256 pins in `serving.py` and the backbone loaders | Checked on first load of each model |
| Forged client IP | nginx overwrites `X-Forwarded-For`; only `web` publishes a port | Holds while the backend port stays unpublished ([DEPLOY.md](../DEPLOY.md)) |
| Secrets in the repository | `.env.docker` and `backend/.env` are Git-ignored; the example files hold values to be replaced | Secrets are generated on the VM (DEPLOY.md section 5) |
| Traffic in transit | Plain HTTP (Hypertext Transfer Protocol) on the demo VM | DEPLOY.md section 9 describes HTTPS (encrypted HTTP) through a reverse proxy |

### End-to-end flows

These three flows connect the milestones on the live deployment; each step names the page or endpoint involved.

1. **Upload by a quality engineer.** Log in, set the product's MVTec category on the Products page, upload on the
   Upload page (`POST /inspections/upload`), then open the inspection to see the prediction, heatmap, boxes, severity
   and decision, and its report (`GET /inspections/{id}/report`).
2. **Import or camera simulation by a quality engineer.** Pick a dataset image on the Dataset page
   (`POST /inspections/import`), or start a run on the Camera simulation page, which imports images on a timer.
3. **Review by a factory supervisor.** Log in, follow the dashboard and the per-category breakdown, open inspections
   flagged for manual review, and compare categories on the Models page (`GET /ai/models`).

### Design rationale

- **WRN-50 for nine categories.** None of the nine had a model that met its gate after the earlier studies, and the
  protocol made the WRN-50 model final whatever its test result, so no choice depended on test data.
- **Grid at 320x320.** Grid's best synthetic recall in the locked study was 0.853, the lowest of the nine; the
  addendum tested a larger input on normal data only, before any WRN-50 test scoring.
- **One backend worker.** The login counters are correct only with one worker, and each worker keeps its own
  backbones and model cache ([DEPLOY.md](../DEPLOY.md#known-limits)).

### Design challenges and how they were resolved

- **Tests on the development database.** Before `fe2b4c5` (4 October), the tests used the application's own
  database session. `fe2b4c5` moved them to a separate test database with a guard against the development database.
- **Grid memory.** Holding the full320 features of 264 images at once needs about 2.6 GB, so the addendum built each
  fold's bank from a buffer and the final bank in two streaming passes, with an equivalence check
  (`ADDENDUM_1_GRID_FULL320.md`).
- **Dataset permissions on Linux.** The MVTec archives extract with directories other users cannot read; DEPLOY.md
  records `chmod -R a+rX dataset` as the fix.
- **Download links.** On 17 October the full-dataset link and 9 of the 15 per-category links returned 404; the
  categories were copied from a verified copy, checked file by file (6,642 files, DEPLOY.md).
- **Models outside Git.** The models ZIP (2.14 GB) was moved to the VM in 500 MB parts, each SHA-256-checked, and its
  files' write permissions were removed after extraction (DEPLOY.md).
- **Client IP behind nginx.** The backend logs on AWS showed each client's real public IP, so the per-IP limit
  applies per user (DEPLOY.md).

### Future work

These are possible directions, each tied to a measured result above.

- Calibrate the thresholds of wood and carpet, whose flagged good images all scored within 1.25 times the threshold.
- Try larger inputs or other feature layers for the pill and screw defect types with the lowest recall.
- Check that an upload matches its product's category; today the product's category is used as given.
- Serve HTTPS through a reverse proxy with the real-IP setup in DEPLOY.md section 9.
- Move the login counters to a shared store so that several workers can run.
- Try a finer map or a smaller minimum box area for the many small defect pieces of grid, pill and toothbrush.

## Deliverables

| Task or criterion | Status | Evidence |
|---|---|---|
| Validate detection and classification | PARTIAL | Detection: final tests of all 15 categories and the localization evaluation (`e914b82`, `3bb9829`, `903e54c`). Classification: no defect-type classifier exists, so there is no classification accuracy to validate. The defect form has not been compared with ground truth. |
| Optimize models and inspection performance | DONE | WRN-50 study and Grid addendum (`0506f95` to `e914b82`); shared backbones and model cache (`188c903`); timing columns (`82a319f`) |
| Improve dashboard and reporting | DONE | `82a319f`, `509a6cd`, `1d1ae52` |
| Docker and AWS deployment | DONE | `d0f72e0`, `99f2656`, `7a507ec`; "Tested on AWS" in [DEPLOY.md](../DEPLOY.md) |
| Technical documentation | DONE | [README.md](../../README.md), [ARCHITECTURE.md](../ARCHITECTURE.md), [ML_METHODOLOGY.md](../ML_METHODOLOGY.md), [DEPLOY.md](../DEPLOY.md), [postman/README.md](../../postman/README.md) |
| End-to-end checks on the live deployment | DONE | [DEPLOY.md](../DEPLOY.md), "Tested on AWS": health, login for both roles, uploads, heatmap, analytics, dataset and camera listings, upload size limits and the login limit, all through the public address |
| Frontend and backend fully deployed | DONE | http://13.206.115.50 |
| Model testing and validation completed | DONE | Final tests of all 15 categories; 1712 backend and 324 frontend tests passing |

## Commits

| Commit | Date | Content |
|---|---|---|
| `82a319f` | 2026-09-20 | Processing-time columns, performance overview and dashboard reporting |
| `c8b4706` | 2026-10-03 | Experiment scripts archived with a SHA-256 manifest |
| `60447b1` | 2026-10-03 | WRN-50 PatchCore detector and evaluation protocol |
| `936be59` | 2026-10-03 | WRN-50 study details declared before running |
| `0506f95` | 2026-10-04 | Nine categories locked on normal data |
| `8a7bc0a` | 2026-10-04 | Grid 320x320 addendum declared |
| `b84534d` | 2026-10-04 | Grid addendum run on normal data |
| `9b5c781` | 2026-10-04 | Final-test procedure declared |
| `e914b82` | 2026-10-04 | Nine test sets scored once |
| `188c903` | 2026-10-04 | All 15 categories served (also listed in Milestone 2) |
| `fe2b4c5` | 2026-10-04 | Separate test database; seed script |
| `fc1cab0` | 2026-10-06 | Postman collection and environment |
| `79804e2` | 2026-10-06 | Seed script: ID reset and upload-style inspections |
| `509a6cd` | 2026-10-06 | Dashboard refinements and per-category analytics (also listed in Milestone 3) |
| `1d1ae52` | 2026-10-06 | Mobile layout fixes |
| `09c639e` | 2026-10-06 | Per-category analytics request in the Postman collection |
| `d0f72e0` | 2026-10-06 | Docker, Docker Compose and the deployment guide |
| `99f2656` | 2026-10-07 | Tested Docker configuration recorded |
| `3bb9829` | 2026-10-07 | Image-level AP for the ResNet-18 categories |
| `9f28c4c` | 2026-10-08 | Camera page explains missing dataset categories |
| `455ee2b` | 2026-10-08 | Camera banner and severity card layout fixes |
| `df7499d` | 2026-10-08 | Absolute path removed from the experiments README |
| `531d111` | 2026-10-09 | Localization evaluation protocol and script |
| `903e54c` | 2026-10-09 | Localization evaluation results |
| `7a507ec` | 2026-10-17 | AWS deployment and redeploy procedure in the deployment guide |
| `2b57933` | 2026-10-17 | README, architecture and ML methodology documents |
| `5b28e85` | 2026-10-17 | README restructured |
| `e013c0c` | 2026-10-17 | MIT licence and remaining documentation |

## Results

### Detection

Final-test point estimates (F1 is the harmonic mean of precision and recall; kNN is k-nearest-neighbour), from
`backend/app/ai/inference/serving.py`, the final-test reports,
`backend/app/ai/ap_addendum.json` and `backend/app/ai/box_eval_addendum.json`. Image AP is image-level average
precision, not detection mAP. Box AP@0.5 is the box-level measure from the localization evaluation.

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

| Group | Recall | F1 | FPR | AUROC | Image AP | Box AP@0.5 | Pixel AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| All 15 | 91.1% | 93.8% | 7.7% | 0.9845 | 0.9946 | 0.200 | 0.9759 |
| 9 WRN-50 PatchCore | 90.0% | 92.7% | 10.5% | 0.9814 | 0.9951 | 0.145 | 0.9806 |
| 6 ResNet-18 | 92.8% | 95.6% | 3.5% | 0.9891 | 0.9940 | 0.281 | 0.9689 |

Gate counts: EXCELLENT 8, GOOD 1, ACCEPTABLE 2, NOT PRODUCTION READY 4. Over the same final tests, the mean accuracy
is 91.5% and the mean precision 97.7%.

### Speed

The final-test reports store the time per image, measured on the development machine (8 CPUs, 4 PyTorch threads,
one image at a time). It covers reading and preprocessing the image, the model and, for WRN-50, the anomaly map.

| Category | ms per image | Category | ms per image | Category | ms per image |
|---|---:|---|---:|---|---:|
| toothbrush | 51 | metal_nut | 169 | zipper | 773 |
| cable | 79 | bottle | 462 | hazelnut | 806 |
| leather | 111 | capsule | 504 | wood | 809 |
| tile | 140 | pill | 589 | screw | 966 |
| transistor | 150 | carpet | 627 | grid | 1943 |

On the AWS VM (2 vCPUs), [DEPLOY.md](../DEPLOY.md) records these backend inference times and client wall times
from India:
- Tile: 0.57 s on the first request and 0.32 s warm (1.4 s and 1.2 s wall).
- Grid: 5.1 s and 3.9 s (5.7 s and 4.4 s wall).
- Wood: 1.9 s.

Inference is synchronous on one worker. DEPLOY.md expects roughly 2 inspections per second on 8 cores for the
lighter models. The repository has a concurrency check, `backend/scripts/concurrency_check.py`, but it writes its
results outside the repository and no result is stored.

### Tests

On 17 October 2026, `python -m pytest` in `backend/` passed 1712 tests with none skipped, and `npx vitest run` in
`frontend/` passed 324 tests in 42 files.

## Limitations

- **Gates:** four categories are NOT PRODUCTION READY. Wood and Carpet fail on FPR, and Pill and Screw on recall.
  Every result from them is flagged for review.
- **Box AP:** box AP@0.5 is low (mean 0.200) because the boxes come from a blurred anomaly map, even though pixel
  AUROC is high.
- **Test sets:** before the WRN-50 final test, the nine replaced categories' test sets had already been scored by
  earlier models, as the protocol discloses. Some test sets are small; Toothbrush has 12 good images and Wood 19.
- **Deployment:** the deployment is a single VM over plain HTTP, with one backend worker and an in-memory login
  limiter. The demo data's timestamps age until it is seeded again.

## How to verify

- Live deployment: http://13.206.115.50, and `curl -s http://13.206.115.50/api/health`. The Models page lists every
  category's model, gate and metrics.
- Local Docker: `cp .env.docker.example .env.docker`, then `docker compose --env-file .env.docker up -d --build`
  (see [README.md](../../README.md#4-run) and [DEPLOY.md](../DEPLOY.md)).
- Tests: `python -m pytest` in `backend/`; `npx vitest run` in `frontend/`.
- The protocols and declarations are in `backend/scripts/experiments/patchcore_wrn50/` and
  `backend/scripts/experiments/localization_map/`.

## Troubleshooting

The entries below cover problems recorded for the deployment and the dataset, with causes and fixes from
[DEPLOY.md](../DEPLOY.md) and the code.

| Symptom | Cause | Fix |
|---|---|---|
| Some categories are missing from the dataset listing, and the seed script stops with `404: Category not found` | The MVTec AD archives extract with directories that other users cannot read (`dr-xr-x---`), so the backend user (uid 10001) cannot list them | Run `chmod -R a+rX dataset` on the host; this grants read access only |
| An official MVTec AD download link returns 404 | On 17 October the full-dataset link and 9 of the 15 per-category links returned 404 | Check the official page for current links |
| The dataset browser, dataset import and camera simulation show no data, and the backend log shows `WARNING: /dataset is missing or empty; the dataset browser and import will not work.` | `DATASET_DIR` points at a missing or empty folder (`backend/docker-entrypoint.sh`) | Place the dataset at `DATASET_DIR` (default `./dataset`) and restart the backend |
| Memory use on the VM is high, especially while the seed script runs or with more workers | Each worker keeps its own backbones and up to `VISIONINSPECT_AI_MODEL_CACHE_SIZE` category models; the seed process loads its own copy of the models | Use at least 4 GB of RAM (random access memory), 8 GB recommended; lower `VISIONINSPECT_AI_MODEL_CACHE_SIZE` before adding workers. On AWS the backend used 1.18 GiB with three categories loaded |
| The first Grid inspection after a start is slow | The first prediction for a category loads and hash-checks its model and backbone | Expected: on AWS, Grid took 5.1 s on the first request and 3.9 s warm |
| The dashboard charts end several days before today | Seeded inspections get timestamps spread over the `--days` before the seed ran | Reseed with `--reset --reset-ids --yes` (deletes every inspection; back up first, DEPLOY.md section 11) |
| A large request gets 413 from nginx | nginx accepts request bodies up to 60 MB (`client_max_body_size 60m` in `frontend/nginx.conf`) | Send fewer or smaller files per request; a single upload is limited to 10 MB anyway |
| Behind a reverse proxy, all users are locked out of login together | Every request reaches nginx from the proxy's address, so all users share the per-IP limit | Configure nginx to take the client address from the proxy, as in DEPLOY.md section 9 |

## Glossary

The terms below are introduced in this milestone. Earlier terms are defined in the glossaries of
[Milestone 1](MILESTONE_1.md#glossary), [Milestone 2](MILESTONE_2.md#glossary) and
[Milestone 3](MILESTONE_3.md#glossary), among them gate, recall, FPR, AUROC, anomaly map and localization.

| Term | Meaning in this project |
|---|---|
| Out-of-fold score | The score of a normal training image from a memory bank built without that image's fold, used to set WRN-50 thresholds. |
| Image-level AP | Average precision of the image scores against the defective or good labels: the area under the image-level precision-recall curve. |
| Box AP@0.5 | Average precision of the anomaly-map boxes, where a box matches a ground-truth box at IoU 0.5 or more. |
| Pixel AUROC | AUROC of the anomaly map's pixel values against the ground-truth mask, over the analysed area. |
| IoU (intersection over union) | The overlap of two boxes divided by the area they cover together. |
| Wilson 95% interval | The confidence interval for a proportion, such as recall or FPR, stored in each WRN-50 final-test report. |
| Precision | The share of images predicted defective that are defective. |
