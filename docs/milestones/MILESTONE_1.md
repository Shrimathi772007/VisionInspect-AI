# Milestone 1: Project setup, authentication, upload and dataset integration

## Objective

The first milestone sets up a working full-stack application before any AI is added. A user can sign in with a
role, register products, upload product images, browse the MVTec AD dataset and see the inspections on a dashboard.
The data model, authentication and image intake built here are the base every later milestone extends.

## What was built

The repository started on 27 August 2026 with a React frontend (Vite) and a FastAPI backend (`942ee6b`). The next day
PostgreSQL was added through SQLAlchemy and Alembic (`ccce87e`). The first migration created the `users`,
`products`, `inspections` and `defects` tables, and this schema is still the core of the data model. JWT
authentication with bcrypt password hashes and a `require_role` dependency followed the same evening (`f7ed210`).
It introduced the two roles that the whole application still uses: quality engineer and factory supervisor.

The image intake came next (`5e582ea`). A product can be created, and an image uploaded against it is checked for
extension, size and decodability, stored under a random name, and recorded as an inspection. The same commit added
the first backend tests (`test_auth.py`, `test_inspections.py`, `test_products.py`, `test_health.py`). The frontend
gained its main workflow in `16b6b97`: login, products, the inspection list and detail views, the upload page with
a drop zone, and the shared component set. A light and dark theme followed in `4c3e30f`.

MVTec AD was integrated in `6726990`. The backend scans the dataset folder and lists categories, splits and defect
types, serves preview images, and can import any dataset image as an inspection. An import takes its ground-truth
label (`good` or `defective`) and its defect type from the folder it comes from. The Dataset page in the frontend
browses this tree. Milestone 1 closed on 5 September with `af7a252`, which added the dashboard (counts and an
activity chart), the image viewer, the product detail view and the inspection status banner.

At the end of the milestone, preprocessing meant only the display preview (RGB conversion and a bounded JPEG in
`backend/app/dataset/preprocessing.py`). The preprocessing pipeline for AI models was committed the same day as the
first Milestone 2 work (`72b3f77`) and is described there.

Several parts of this milestone were changed in later iterations:

- Until 3 October, registration took the role from the request. `6693b13` made self-registration always create a
  factory supervisor, and added QE-only role management and a script that creates the first quality engineer.
- `1b4b0ab` added the registration page and the Users page, and `0fad1e1` restricted the upload route to quality
  engineers in the frontend as well.
- `593184b` rejects passwords longer than 72 bytes (the bcrypt limit).
- `e83f94a` added a limiter for failed logins.
- `ce42d43` added batch upload and the camera simulation, which replays labelled dataset images as timed imports.

## Deliverables

| Task or criterion | Status | Evidence |
|---|---|---|
| Inspection workflows and objectives | PARTIAL | The workflows exist in code from `5e582ea` and `6726990`; no planning document from this period is in the repository. They are described in [README.md](../../README.md) and [ARCHITECTURE.md](../ARCHITECTURE.md), both written on 17 October. |
| Architecture and database schema | DONE | `ccce87e`: SQLAlchemy models and the first migration (`3dbc453c0101`). The architecture is documented in [ARCHITECTURE.md](../ARCHITECTURE.md). |
| UI wireframes and workflow planning | NOT DONE | No wireframe or design file exists in the repository or its history. |
| Frontend and backend environments | DONE | `942ee6b`, `ccce87e`, `5efe127` |
| Authentication and role-based access | DONE | `f7ed210`. Hardened in later iterations by `6693b13` (3 October), `1b4b0ab` (3 October), `593184b` (6 October) and `e83f94a` (8 October). |
| Load MVTec AD dataset | DONE | `6726990`: `backend/app/dataset/`, Dataset page |
| Image upload workflow | DONE | `5e582ea`, `16b6b97`. Batch upload delivered in a later iteration: `ce42d43` (4 October). |
| Inspection dashboard | DONE | `af7a252` |
| Initialization and architecture complete | DONE | as above |
| Authentication and upload implemented | DONE | `backend/tests/test_auth.py`, `backend/tests/test_inspections.py` |
| Dashboard functional | DONE | `af7a252` (`DashboardPage.jsx`, `ActivityChart.jsx`) |
| Dataset integration and preprocessing | DONE | Dataset integration in `6726990`. The AI preprocessing pipeline in `72b3f77` was committed on 5 September, the same day as `af7a252`. |

## Commits

| Commit | Date | Content |
|---|---|---|
| `0ebb77d` | 2026-08-27 | Repository created with a `.gitignore` |
| `942ee6b` | 2026-08-27 | React (Vite) frontend and FastAPI backend skeletons |
| `ccce87e` | 2026-08-28 | PostgreSQL, SQLAlchemy models and the first Alembic migration |
| `f7ed210` | 2026-08-28 | JWT authentication and role checks |
| `5e582ea` | 2026-08-29 | Products, image upload and storage, first backend tests |
| `16b6b97` | 2026-08-29 | Frontend workflow pages and shared components |
| `5efe127` | 2026-08-29 | Vite development server configuration fix |
| `4c3e30f` | 2026-08-30 | Light and dark theme (outside the milestone task list) |
| `6726990` | 2026-08-30 | MVTec AD dataset browsing and import |
| `af7a252` | 2026-09-05 | Dashboard, image viewer, product detail, end of Milestone 1 |

Delivered in a later iteration:

| Commit | Date | Content |
|---|---|---|
| `6693b13` | 2026-10-03 | Registration always creates a factory supervisor; QE-only role changes; first-QE script |
| `1b4b0ab` | 2026-10-03 | Registration page and Users page |
| `0fad1e1` | 2026-10-03 | Upload page restricted to quality engineers |
| `ce42d43` | 2026-10-04 | Batch upload and camera simulation (also severity and enhancement UI, see Milestone 3) |
| `593184b` | 2026-10-06 | Passwords over 72 bytes rejected |
| `e83f94a` | 2026-10-08 | Failed-login rate limiting per client IP and email |

## Limitations

The registration flaw described above existed until 3 October. Uploads carry no ground truth, so they are shown as
"Unlabelled" and only dataset imports have a known label. The camera simulation replays dataset images from the
server and is not a live camera feed. The `defects` table from the first migration is never written by any code.
The login limiter keeps its counters in memory, so they reset when the backend restarts.

## How to verify

- Live deployment: http://13.206.115.50. `/register` creates a factory supervisor, and a quality engineer can
  change roles on the Users page.
- Health checks: `curl -s http://13.206.115.50/api/health` and `curl -s http://13.206.115.50/api/health/db`.
- Tests, from `backend/`:
  `python -m pytest tests/test_auth.py tests/test_users.py tests/test_login_rate_limit.py tests/test_products.py tests/test_inspections.py tests/test_dataset.py`.
- The first quality engineer is created with `python scripts/create_quality_engineer.py` (see
  [README.md](../../README.md#5-first-quality-engineer-and-demo-data)).
