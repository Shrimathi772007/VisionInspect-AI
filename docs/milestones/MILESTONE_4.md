# Milestone 4: Validation, optimization, deployment and documentation

## Objective

The last milestone takes the platform from a development setup to a measured, deployed system. The detection models
are validated on held-out test data under a declared protocol, and the weak ones are replaced. Inspection
performance and the dashboard are improved. The full stack runs with Docker on a cloud VM, and the system is
documented for other engineers.

## What was built

**Performance instrumentation** (`82a319f`, 20 September). Every inspection now records its server-side handling
time (`processing_time_ms`) and the time of the model call (`ai_inference_time_ms`). The dashboard shows these over
the selected 7, 14 or 30-day window. The analytics summary also reports an automation rate: the share of decided
inspections whose decision was PASS or FAIL rather than `MANUAL_REVIEW` or NOT_ASSESSED. It is computed live from
the database, and no value is stored in the repository.

**Model validation and replacement** (3 to 4 October). By the end of September, nine categories (bottle, capsule,
carpet, grid, hazelnut, pill, screw, wood and zipper) had no model that met its gate. A WRN-50 PatchCore study
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
- `531d111` and `903e54c` (9 October) declared and ran a localization evaluation: box AP@0.5 and pixel AUROC of
  the anomaly-map boxes. It is disclosed as a second scoring of the test sets, used for localization only.
- `c8b4706` archived the per-category experiment scripts with a SHA-256 manifest.

[ML_METHODOLOGY.md](../ML_METHODOLOGY.md) describes the protocol and every disclosed exception.

**Testing.**
- `fe2b4c5` moved the backend tests to a separate test database and added the seed script for demo data
  (extended in `79804e2`).
- `fc1cab0` and `09c639e` added the Postman collection, which covers all 29 API operations and has a negative-test
  folder.

**Dashboard and interface.**
- `509a6cd` refined the dashboard and added the per-category breakdown.
- `1d1ae52` fixed the mobile layout of the inspection list and the dashboard.
- `9f28c4c` and `455ee2b` fixed details of the camera and severity views.

**Deployment.**
- `d0f72e0` (6 October) added the Dockerfiles, `docker-compose.yml` and [DEPLOY.md](../DEPLOY.md).
  `docker-compose.yml` runs three services: `web` (nginx and the built frontend), `backend` and `db`.
- `99f2656` recorded a tested local Docker build.
- `7a507ec` (17 October) recorded the deployment on an AWS EC2 VM in ap-south-1 (Mumbai) and the redeploy
  procedure.

**Documentation.**
- `2b57933`, `5b28e85` and `e013c0c` added [README.md](../../README.md), [ARCHITECTURE.md](../ARCHITECTURE.md),
  [ML_METHODOLOGY.md](../ML_METHODOLOGY.md) and the licence.
- `df7499d` removed an absolute path from the experiments README.

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

Final-test point estimates, from `backend/app/ai/inference/serving.py`, the final-test reports,
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
