# VisionInspect AI

## Overview

VisionInspect AI is a web application for visual quality inspection of manufactured parts. A quality engineer
uploads product images or imports images from the MVTec AD dataset. The backend scores each image with an
unsupervised anomaly-detection model chosen by the product's MVTec category, marks the suspicious region with a
heatmap and boxes, assigns a rule-based severity and a quality decision, and flags results that need manual
review. A dashboard aggregates the results.

The models are trained on defect-free images only (MVTec AD `train/good`). They detect "unusual compared with
normal"; they do not classify defect types.

| Layer | Technology |
|---|---|
| Frontend | React 19, React Router 7, Vite 8, lucide-react icons, CSS Modules; Vitest and Testing Library |
| Backend | Python 3.14, FastAPI, SQLAlchemy 2, Alembic, PyJWT, bcrypt, Pillow; pytest |
| ML | PyTorch 2.14 (CPU build), frozen ImageNet ResNet-18 and Wide ResNet-50-2 weights (PyTorch model files), OpenCV, NumPy, scikit-learn |
| Database | PostgreSQL (the Docker setup uses `postgres:18-alpine`) |
| Deployment | Docker Compose with three services: `web` (nginx serving the built frontend and proxying `/api`), `backend` (uvicorn), `db` |

Exact Python package versions are pinned in [backend/requirements.txt](backend/requirements.txt).

## Contents

[Overview](#overview) | [Key features](#key-features) | [Quick start](#quick-start) | [Live demo](#live-demo) |
[Results summary](#results-summary) | [Setup](#setup) | [Testing](#testing) | [Known limitations](#known-limitations) |
[Repository layout](#repository-layout) | [Documentation links](#documentation-links) |
[Dataset attribution](#dataset-attribution) | [Licence](#licence)

## Key features

### At a glance

- [Roles and authentication](#roles-and-authentication): quality engineer and factory supervisor, JWT login.
- [Upload and batch upload](#upload-and-batch-upload): JPEG/PNG, one file or up to 20 per batch.
- [Dataset import and camera simulation](#dataset-import-and-camera-simulation): inspections from labelled MVTec images.
- [Models for 15 MVTec categories](#models-for-15-mvtec-categories): one locked, hash-verified model per category.
- [Heatmap and box localization](#heatmap-and-box-localization): regions from the anomaly map, not a trained detector.
- [Confidence and manual review](#confidence-and-manual-review): margin confidence and the review rule.
- [Rule-based severity](#rule-based-severity): weighted size, location, form and confidence.
- [Defect form classification](#defect-form-classification): a rule on the shape of the flagged region; it does not identify MVTec defect types.
- [Image enhancement preview](#image-enhancement-preview): denoise and contrast preview with quality metrics.
- [Dashboard analytics](#dashboard-analytics): totals, trends and per-category breakdown over 7, 14 or 30 days.
- [Models page](#models-page): each category's model, gate and final-test metrics.
- [Postman collection](#postman-collection): all 29 API operations plus negative tests.
- [Docker Compose deployment](#docker-compose-deployment): web, backend and database on one VM.

Each section below names, in parentheses, the code or endpoint that implements the feature.

### Roles and authentication

(`backend/app/auth/`, `backend/app/users/`) JWT bearer tokens (HS256, 60 minutes by default) and bcrypt password
hashes. Two roles: quality engineer and factory supervisor. The backend enforces roles; the frontend hides
controls a role cannot use. See the role table in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#roles).

### Upload and batch upload

- **Upload** (`POST /inspections/upload`). JPEG or PNG, up to 10 MB (`MAX_UPLOAD_SIZE_MB`). The file is checked
  with Pillow and stored under a random name. The AI model is chosen by the product's MVTec category. A product
  without a category gets no AI result. The image content is not checked against the category.
- **Batch upload** (`POST /inspections/batch`). Up to 20 files and 50 MB per batch; each file goes through the
  single-upload pipeline. The response reports each file's result separately, and a bad file does not stop the
  others.

### Dataset import and camera simulation

- **Dataset import** (`POST /inspections/import`, Dataset page). Creates an inspection from an MVTec image on the
  server. The ground-truth label (`good` or `defective`) and the defect type come from the dataset folder names.
  The model is chosen by the dataset category.
- **Camera simulation** (Camera simulation page). It replays labelled MVTec images from the server's dataset
  folder as a timed sequence of dataset imports: 1 to 10 seconds between frames, at most 50 frames per run, and
  defect types taken in turn. It is not a live camera.

### Models for 15 MVTec categories

(`backend/app/ai/inference/serving.py`) Each of the 15 categories has exactly one registered model: a WRN-50
PatchCore model for 9 categories and a ResNet-18 patch model for 6, each with a locked threshold. Model files are
hash-checked on first load, and no category falls back to another category's model. Results are in
[Results summary](#results-summary); methods are in [docs/ML_METHODOLOGY.md](docs/ML_METHODOLOGY.md).

### Heatmap and box localization

(`backend/app/ai/inference/localization.py`) The per-patch anomaly scores are upsampled and blurred into an
anomaly map. Regions at or above the model's threshold become at most 5 boxes, and regions under 0.25% of the
image are discarded. The heatmap is a transparent PNG drawn over the image. The boxes come from the anomaly map;
there is no trained detector.

### Confidence and manual review

- **Confidence and reliability**. Confidence is `sigmoid(10 * |ln(score / threshold)|)`, which runs from 0.5 at
  the threshold towards 1. It is a margin heuristic, not a probability. Each category's model also carries its
  evidence gate and final-test metrics.
- **Manual-review rules**. An inspection is flagged for manual review when confidence is below 0.70, or when the
  category's model gate is NOT_PRODUCTION_READY. For uploads, a flagged inspection gets the quality decision
  `MANUAL_REVIEW`. For dataset imports the review flag is stored, but the ground-truth rules decide.

### Rule-based severity

(`backend/app/inspections/severity.py`) For an AI-defective inspection with boxes: severity = 0.30 size + 0.25
location + 0.25 form + 0.20 confidence, banded into Low, Medium, High and Critical.

### Defect form classification

(`backend/app/inspections/defect_form.py`) The form is a shape rule on the flagged region (multiple regions, large
area, linear, small spot or localized patch). It does not identify the defect type.

### Image enhancement preview

(`GET /inspections/{id}/enhanced`, `/image-quality`) Denoising and CLAHE contrast, with before/after brightness,
contrast, sharpness and noise estimates. This is a preview only: the models always score the original image, and
nothing is stored.

### Dashboard analytics

- **Dashboard** (`GET /inspections/analytics/summary`, `/analytics/by-category`). Totals, AI defect rate, quality
  decisions, severity and defect-category distributions, a daily trend, processing times and a per-category
  breakdown, over a 7, 14 or 30-day window. All values are computed in PostgreSQL.
- **Production quality report** (`GET /inspections/{id}/report`) for a single inspection.

### Models page

(`GET /ai/models`) One row per category: model, input mode, locked threshold, gate, final-test recall, FPR, AUROC,
image-level AP, box AP@0.5 and pixel AUROC.

### Postman collection

([postman/](postman/)) Covers all 29 API operations and adds a negative-test folder. Requests that write are
skipped unless `allowWrites` is set to `true`.

### Docker Compose deployment

([docker-compose.yml](docker-compose.yml)) Three services on one VM: `web`, `backend` and `db`. Only `web`
publishes a port. The models and the dataset are mounted read-only from host folders, and the backend runs the
database migrations when it starts. See [Setup, step 4](#4-run) and [docs/DEPLOY.md](docs/DEPLOY.md).

## Quick start

1. Clone: `git clone https://github.com/Shrimathi772007/VisionInspect-AI.git` ([details](#1-clone)).
2. Models: download the [models ZIP](https://drive.google.com/file/d/11ACKiX4p8R7_2AQZw_l_TAWmR_aU7KdJ/view?usp=drive_link) and extract it into `backend/` ([details](#2-models)).
3. Dataset: extract MVTec AD into `dataset/` ([details](#3-dataset)).
4. Configure: `cp .env.docker.example .env.docker`, then set `POSTGRES_PASSWORD`, `JWT_SECRET_KEY` and `FRONTEND_ORIGINS` ([details](#4-run)).
5. Start: `docker compose --env-file .env.docker up -d --build` ([details](#4-run)).
6. Open the site on `WEB_PORT` (default 8080), for example `http://localhost:8080` ([details](#4-run)).
7. First quality engineer: `docker compose --env-file .env.docker exec backend python scripts/create_quality_engineer.py --name "Quality Engineer" --email qe@example.com` ([details](#5-first-quality-engineer-and-demo-data)).

## Live demo

- URL: http://13.206.115.50 (plain HTTP, no TLS).
- Hosting: a single AWS EC2 VM in ap-south-1 (Mumbai) running the Docker Compose stack (`web`, `backend`, `db`).
  The deployment is described under "Tested on AWS" in [docs/DEPLOY.md](docs/DEPLOY.md).
- Planned availability: about 3 weeks from 17 October 2026, limited by cloud credits.
- Demo accounts:
  - `alice.qe@example.com` (quality engineer)
  - `bob.fs@example.com` (factory supervisor)

  The passwords are shared privately by the author and are not in the repository.
- `/register` always creates a factory supervisor, whatever role is sent. A quality engineer can change any user's
  role on the Users page. The last remaining quality engineer cannot be demoted.
- Failed logins are rate-limited: 5 failures per client IP and email, or 20 failures per client IP for any email,
  within a 15-minute sliding window. Further attempts get `429 Too Many Requests` with a `Retry-After` header
  until the window passes. Successful logins and registration are not limited.
- Inference runs on the CPU of a 2-vCPU VM. Backend inference times measured there were 0.3 to 0.6 s per image for
  Tile and 1.9 s for Wood. Grid, the largest model, takes a few seconds per image: 3.9 s warm and 5.1 s on the
  first request.

## Results summary

All values are point estimates from each category's final test on the official MVTec AD test split, read from
`backend/app/ai/inference/serving.py`, the final-test reports and `backend/app/ai/ap_addendum.json`. The defective
class is positive. FPR is the share of good test images flagged as defective.

**Image-level AP is the area under the image-level precision-recall curve. It is not object-detection mAP.** The
box-level results are in [docs/ML_METHODOLOGY.md](docs/ML_METHODOLOGY.md#localization).

| Category | Model | Recall | F1 | FPR | AUROC | Image AP | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| bottle | WRN-50 PatchCore crop224 | 100.0% | 100.0% | 0.0% | 1.0000 | 1.0000 | EXCELLENT |
| cable | ResNet-18 kNN 128 | 85.9% | 91.9% | 1.7% | 0.9880 | 0.9929 | GOOD |
| capsule | WRN-50 PatchCore crop224 | 78.9% | 87.8% | 4.3% | 0.9781 | 0.9952 | ACCEPTABLE |
| carpet | WRN-50 PatchCore crop224 | 98.9% | 94.6% | 32.1% | 0.9892 | 0.9968 | NOT PRODUCTION READY |
| grid | WRN-50 PatchCore full320 | 98.2% | 98.2% | 4.8% | 0.9908 | 0.9971 | EXCELLENT |
| hazelnut | WRN-50 PatchCore crop224 | 100.0% | 100.0% | 0.0% | 1.0000 | 1.0000 | EXCELLENT |
| leather | ResNet-18 Gaussian 128 | 100.0% | 98.9% | 6.3% | 1.0000 | 1.0000 | EXCELLENT |
| metal_nut | ResNet-18 kNN 256 | 96.8% | 97.8% | 4.5% | 0.9912 | 0.9982 | EXCELLENT |
| pill | WRN-50 PatchCore crop224 | 66.0% | 79.1% | 3.8% | 0.9479 | 0.9899 | NOT PRODUCTION READY |
| screw | WRN-50 PatchCore full256 | 70.6% | 82.8% | 0.0% | 0.9508 | 0.9835 | NOT PRODUCTION READY |
| tile | ResNet-18 kNN 256 | 91.7% | 95.7% | 0.0% | 0.9971 | 0.9989 | EXCELLENT |
| toothbrush | ResNet-18 Gaussian 256 | 90.0% | 93.1% | 8.3% | 0.9722 | 0.9885 | EXCELLENT |
| transistor | ResNet-18 kNN 256 | 92.5% | 96.1% | 0.0% | 0.9863 | 0.9854 | EXCELLENT |
| wood | WRN-50 PatchCore full256 | 98.3% | 93.7% | 36.8% | 0.9868 | 0.9962 | NOT PRODUCTION READY |
| zipper | WRN-50 PatchCore full256 | 99.2% | 97.9% | 12.5% | 0.9892 | 0.9970 | ACCEPTABLE |

| Unweighted mean | Recall | F1 | FPR | AUROC | Image AP |
|---|---:|---:|---:|---:|---:|
| All 15 categories | 91.1% | 93.8% | 7.7% | 0.9845 | 0.9946 |
| 9 WRN-50 PatchCore categories | 90.0% | 92.7% | 10.5% | 0.9814 | 0.9951 |
| 6 ResNet-18 categories | 92.8% | 95.6% | 3.5% | 0.9891 | 0.9940 |

Gate counts: EXCELLENT 8, GOOD 1, ACCEPTABLE 2, NOT PRODUCTION READY 4. The gates are project-defined:
EXCELLENT needs recall >= 0.90, F1 >= 0.85 and FPR <= 0.10; GOOD needs 0.85 / 0.80 / 0.10; ACCEPTABLE needs
0.75 / 0.70 / 0.15.

## Setup

### 1. Clone

```bash
git clone https://github.com/Shrimathi772007/VisionInspect-AI.git
cd VisionInspect-AI
```

### 2. Models

The trained models are not in Git (`backend/ai_models/` is ignored).

1. Download the models ZIP from
   https://drive.google.com/file/d/11ACKiX4p8R7_2AQZw_l_TAWmR_aU7KdJ/view?usp=drive_link. Drive cannot preview a
   file this large; use the Download button, and choose 'Download anyway' if Google says it cannot scan the file
   for viruses. The file is:
   - file `ai_models_2026-10-09_final.zip`, 2,140,816,693 bytes, 678 files;
   - SHA-256 `57e6bf57a81cd73deaadaebdad0604d4abe6b13e3832a88771ab5ccc3335b6e7`.
2. Check the hash (`sha256sum` on Linux, `Get-FileHash -Algorithm SHA256` in PowerShell).
3. Extract it into `backend/`. The result must be `backend/ai_models/<category>/...` (15 category folders) and
   `backend/ai_models/_pretrained/`. Do not create `backend/ai_models/ai_models/`: the backend looks for the models
   exactly one level below `backend/ai_models/`.
4. On Linux, run `chmod -R go-w backend/ai_models` after extracting, because the ZIP stores its files as
   world-writable.

`backend/ai_models/_pretrained/` must contain the two frozen ImageNet backbones. Their SHA-256 values are pinned in
`backend/app/ai/models/resnet18.py` and `wide_resnet50.py`, and a file with any other hash is refused at load time:

| File | SHA-256 |
|---|---|
| `resnet18-f37072fd.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| `wide_resnet50_2-95faca4d.pth` | `95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8` |

Every per-category model file is also hash-checked against `backend/app/ai/inference/serving.py` on first load.

### 3. Dataset

Download MVTec AD from the official MVTec AD page and extract it into `dataset/`, so that each category has
`dataset/<category>/train/`, `test/` and `ground_truth/`. MVTec AD is licensed under CC BY-NC-SA 4.0: it may be
used for non-commercial purposes only, and it is not redistributed with this repository. On 17 October 2026 some
of the official per-category download links returned 404, so check the page for the current links.

On Linux, the archives extract with directories that other users cannot read. Run `chmod -R a+rX dataset` so that
the backend container (uid 10001) can read them. Without this, the affected categories are missing from the
dataset browser.

### 4. Run

**Docker Compose** (the deployed setup). Follow [docs/DEPLOY.md](docs/DEPLOY.md). In short:

```bash
cp .env.docker.example .env.docker      # set POSTGRES_PASSWORD, JWT_SECRET_KEY, FRONTEND_ORIGINS
docker compose --env-file .env.docker up -d --build
```

The site is served on `WEB_PORT` (default 8080). Migrations run automatically when the backend starts.

**Local development.** This needs Python 3.14 (the version the backend image uses), Node.js, and a local
PostgreSQL server with an empty database.

```bash
# Backend (from backend/)
cp .env.example .env                    # set POSTGRES_* to your database and choose a JWT_SECRET_KEY
python -m venv venv
venv/bin/pip install -r requirements.txt   # Windows: venv\Scripts\pip
venv/bin/alembic upgrade head
venv/bin/uvicorn app.main:app --reload     # http://127.0.0.1:8000

# Frontend (from frontend/, in a second terminal)
cp .env.example .env                    # VITE_API_BASE_URL=http://127.0.0.1:8000
npm ci
npm run dev                             # http://localhost:5173
```

The backend expects the dataset at `../dataset` relative to `backend/` (`DATASET_ROOT`) and the models at
`backend/ai_models/`. Uploads and heatmaps are written to `backend/storage/`.

### 5. First quality engineer and demo data

Registration only creates factory supervisors. Create the first quality engineer with:

```bash
python scripts/create_quality_engineer.py --name "Quality Engineer" --email qe@example.com
python scripts/create_quality_engineer.py --email existing@example.com --promote-existing
```

The password is read from `VISIONINSPECT_BOOTSTRAP_PASSWORD` when that variable is set, or prompted for twice. It is
never taken as a command-line argument.

`scripts/seed_demo_data.py` creates one quality engineer, one factory supervisor, 15 demo products and demo
inspections. Every demo inspection runs real inference, and the demo images come from `dataset/`, so set up
both the models and the dataset first. Passwords come from `VISIONINSPECT_SEED_QE_PASSWORD` and
`VISIONINSPECT_SEED_SUPERVISOR_PASSWORD`, or from a prompt. Useful options are `--dry-run`, `--days` (default 14),
`--per-category` (default 8), `--extra-uploads` (default 25), and `--reset --yes`, which deletes **every**
inspection. Run both scripts from `backend/`, or inside the backend container with
`docker compose --env-file .env.docker exec backend python scripts/...`. Section 7 of
[docs/DEPLOY.md](docs/DEPLOY.md) has the full commands.

### 6. Without the models or the dataset

The application starts without them; in Docker the backend logs a warning at startup. Login, users, products, the
dashboard and the Models page work. Without `backend/ai_models/`, inspections are still created but get no AI
prediction, heatmap, severity or confidence. Without `dataset/`, the dataset browser, dataset import, the camera
simulation and the seed script do not work.

## Testing

Both suites were run on 17 October 2026 on the development machine:

| Suite | Command | Result |
|---|---|---|
| Backend | `python -m pytest` (from `backend/`) | 1712 tests passed, none skipped (about 11 minutes) |
| Frontend | `npx vitest run` (from `frontend/`) | 42 test files, 324 tests passed |

The backend suite needs PostgreSQL. It creates and migrates a separate `<POSTGRES_DB>_test` database (or uses
`VISIONINSPECT_TEST_DATABASE_URL`) and refuses to run against the development database. Uploads go to a
temporary folder. Tests that need the real models or backbone weights are skipped when those files are missing.

Newman, 9 October 2026, commit df7499d, allowWrites=false, run against a scratch copy of the development database:
30 requests, 92 assertions, 0 failures.

The Postman collection ([postman/](postman/)) can be run in Postman or with newman against a running backend; see
[postman/README.md](postman/README.md) for the run order and what each request checks.

## Known limitations

- **Four categories are NOT PRODUCTION READY**: Wood (FPR 36.8%), Carpet (FPR 32.1%), Screw (recall 70.6%) and
  Pill (recall 66.0%). Every AI result for these categories is flagged for manual review.
- **Box AP@0.5 is low** (mean 0.200 over 15 categories), while pixel AUROC is high (mean 0.976). The boxes are
  regions of a Gaussian-blurred anomaly map, so they are often wider than the defect. Ground-truth defect pieces
  smaller than the minimum box area (0.25% of the image) can never be matched; Grid, for example, has 112 such
  pieces out of 170.
- **Severity is rule-based** and not calibrated against real severity outcomes. Its size term is measured on the
  blurred anomaly region, which overestimates the defect area (in one check, a bar covering 9.9% of the image gave
  a 41% region). The location term treats the image centre as the most critical area, as a proxy.
- **Confidence is a margin heuristic** (distance of the score from the threshold), not a calibrated probability.
- **Uploads have no ground truth.** Their label is shown as "Unlabelled"; only dataset imports have a known
  good/defective label and defect type.
- **The `ai_reconstruction_error` column holds the model's anomaly score.** For the patch models this is a
  patch-feature distance, not a reconstruction error. The column name was kept from the first autoencoder model.
- **Some test sets are small.** Toothbrush has 12 good and 30 defective test images and Wood has 19 good, so one
  image changes FPR by 8.3 or 5.3 percentage points.
- **The overall leakage-check flag is false for Capsule, Carpet, Pill and Wood.** The only failed item is a
  cross-check between two computations of F1, which compares them with exact floating-point equality. The two
  values differ only in the last digit (about 1e-16). Every data-access check for those four categories passed.
- **One backend worker; the login limiter is in memory.** The counters are per process and reset when the backend
  restarts.
- **Plain HTTP on the demo.** There is no TLS, so passwords and tokens cross the network unencrypted.
- **Throughput is CPU-bound and synchronous**: roughly 2 inspections per second for the lighter models, and a few
  seconds per image for Grid. Batches and concurrent users queue.
- **Demo timestamps age.** Seeded inspections are spread over the days before the seed ran, so the dashboard
  charts drift away from "today" until the data is reseeded.
- `/api/docs` (the interactive OpenAPI page) does not work behind the `/api` prefix; use the Postman collection.

## Repository layout

```
backend/
  app/
    main.py                  FastAPI app: CORS, routers, /health and /health/db
    auth/                    register, login, JWT, login rate limiter, role dependencies
    users/                   user list and role changes (quality engineers only)
    products/                create, list and delete products; set each product's MVTec category
    inspections/             upload, batch, import, analytics, severity, defect form, quality decision, report
    dataset/                 MVTec dataset browsing and ground-truth masks
    ai/inference/            serving.py (model registry), predict.py, localization.py (boxes, heatmap)
    ai/models/               ResNet-18, WRN-50-2, PatchCore and patch-anomaly model code
    ai/analytics/            image enhancement preview and image-quality metrics
    ai/preprocessing/, ai/evaluation/, ai/training/  image preprocessing; evaluation and training code
    ai/router.py             GET /ai/models
    ai/ap_addendum.json      image-level AP for the six ResNet-18 categories
    ai/box_eval_addendum.json  box AP@0.5 and pixel AUROC for all 15 categories
    models/, database/       SQLAlchemy models (users, products, inspections, defects), engine and session
  migrations/versions/       Alembic migrations
  scripts/
    create_quality_engineer.py  create or promote the first quality engineer
    seed_demo_data.py        demo users, products and inspections
    experiments/             archived per-category experiment runners, protocols and declarations
  tests/                     pytest suite (conftest.py sets up the isolated test database)
  docker/check_image.py      build-time checks of package pins, CPU torch and study file hashes
  Dockerfile, docker-entrypoint.sh  backend image; the entrypoint waits for PostgreSQL, migrates, starts uvicorn
  requirements.txt, .env.example    pinned Python packages; settings for local development
  ai_models/                 trained models (not in Git; see Setup)
frontend/
  src/App.jsx                routes and role-guarded pages
  src/api/                   fetch client and API calls
  src/pages/                 Login, Register, Dashboard, Products, Inspections, InspectionUpload,
                             InspectionDetail, DatasetBrowser, CameraSimulation, ModelPerformance, Users
  src/components/            shared UI, e.g. LocalizationOverlay, LocalizationPanel, SeverityCard,
                             ManualReviewBanner, BatchUploadPanel, EnhancementPreviewCard, CategoryBreakdown
  nginx.conf, Dockerfile     nginx config (serves the build, proxies /api) and the frontend image
  .env.example               VITE_API_BASE_URL for local development
dataset/                     MVTec AD (not in Git; see Setup)
docs/                        DEPLOY.md, ARCHITECTURE.md, ML_METHODOLOGY.md
postman/                     collection, local environment, sample-images/sample.png, README.md
docker-compose.yml, .env.docker.example  web, backend and db services; template for .env.docker
LICENSE.txt                  MIT licence for the code (see Licence)
```

## Documentation links

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): deployed architecture, request flows, roles, database tables and
  the API route table.
- [docs/ML_METHODOLOGY.md](docs/ML_METHODOLOGY.md): dataset, models, evaluation protocol and its exceptions, full
  results, localization evaluation and experiment history.
- [docs/DEPLOY.md](docs/DEPLOY.md): Docker Compose deployment on a Linux VM, backup, HTTPS notes and the tested
  configurations.
- [postman/README.md](postman/README.md): the Postman collection.
- [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md): the full project document, generated by
  `docs/build_full_doc.py` from the milestone reports and the documents above, with a requirements traceability
  appendix and the technology stack as built.
- Milestone reports: [Milestone 1](docs/milestones/MILESTONE_1.md) (setup, authentication, upload, dataset),
  [Milestone 2](docs/milestones/MILESTONE_2.md) (image processing and defect detection),
  [Milestone 3](docs/milestones/MILESTONE_3.md) (defect classification, severity and analytics) and
  [Milestone 4](docs/milestones/MILESTONE_4.md) (validation, optimization, deployment and documentation).

## Dataset attribution

This project uses the MVTec AD dataset:

Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, "A Comprehensive Real-World Dataset for
Unsupervised Anomaly Detection", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

MVTec AD is copyright 2019 MVTec Software GmbH and is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). It is used here for
non-commercial purposes and is not redistributed in this repository.

## Licence

The code in this repository is licensed under the MIT License; see [LICENSE.txt](LICENSE.txt).

The MIT licence does not cover:

- **The MVTec AD dataset.** It is licensed under CC BY-NC-SA 4.0 (non-commercial use only) and is not included in
  this repository.
- **The ImageNet-pretrained ResNet-18 and Wide ResNet-50-2 weights** (`resnet18-f37072fd.pth`,
  `wide_resnet50_2-95faca4d.pth`). They are the PyTorch model files from `download.pytorch.org`, the source URLs
  recorded in `backend/app/ai/models/resnet18.py` and `wide_resnet50.py`, and they keep the terms of that source.
- **The trained model files** in the models ZIP, which is distributed separately. They were trained on MVTec AD,
  so they should be used under the dataset's non-commercial terms. This is a conservative note, not legal advice.
