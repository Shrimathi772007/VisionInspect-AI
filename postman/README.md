# VisionInspect-AI – Postman collection

| File | What it is |
|---|---|
| `VisionInspect-AI.postman_collection.json` | Postman Collection v2.1. It has all 29 API operations, generated from the backend's `/openapi.json`, plus a **Negative tests** folder. |
| `VisionInspect-AI.local.postman_environment.json` | The **VisionInspect-AI local** environment (`baseUrl` = `http://127.0.0.1:8000`). |
| `sample-images/sample.png` | A small synthetic 64×64 PNG that the upload and batch requests attach. |

The files contain **no credentials or tokens**. `qePassword`, `supervisorPassword` and `token` are
secret variables with empty values. You type the passwords yourself, and `token` is filled in by the
login request.

## 1. Import

1. In Postman, click **Import** and select both JSON files in this folder.
2. In the environment selector (top right), choose **VisionInspect-AI local**.
3. Upload and batch requests attach `sample-images/sample.png` using a relative path. Point Postman's
   working directory at this `postman/` folder (**Settings → General → Working directory**), or pick
   any JPEG/PNG manually in each request's **Body** tab. With newman, use `--working-dir postman`.

## 2. Set the credentials

Open the **VisionInspect-AI local** environment and fill in these values. Use the *current value*
column, which stays on your machine:

| Variable | Value |
|---|---|
| `qeEmail` | Email of the demo quality engineer, **Alice** |
| `qePassword` | Alice's password (type it yourself; secret) |
| `supervisorEmail` | Email of the demo factory supervisor, **Bob** |
| `supervisorPassword` | Bob's password (type it yourself; secret) |

Leave `token` and `role` empty. **Auth → Login (quality engineer)** stores the access token in
`{{token}}` and the role in `{{role}}`, and every protected request sends
`Authorization: Bearer {{token}}`.

Do not export or share the environment after a run. It would then contain your passwords and a live token.

Other variables:

| Variable | Purpose |
|---|---|
| `productId` | The product used by upload, batch and import. **List products** fills it in if it is empty (it picks the first product that has an MVTec category). |
| `inspectionId` | The inspection used by the read-only inspection requests. **List inspections** fills it in if it is empty. |
| `allowWrites` | `false` by default. See below. |
| `createdProductId`, `createdInspectionId`, `createdInspectionIds` | Filled in only by requests that created something, so the delete requests clean up only what this collection created. |
| `targetUserId` | Must be set by hand before **Change user role** will run. |

Collection variables `datasetCategory` / `datasetSplit` / `datasetDefectType` / `datasetFilename`
(default `bottle` / `test` / `broken_large` / `000.png`) drive the Dataset and import requests.

## 3. Protecting the demo database: `allowWrites`

The collection-level pre-request script **skips every request that creates, changes or deletes data**
unless `allowWrites` is `true`. Skipped requests are logged in the Postman console. With the default
`false`, a full run only reads data.

Requests that create or change data (they only run with `allowWrites=true`):

| Request | Effect |
|---|---|
| Auth → Register (creates a user) | Creates a factory-supervisor user with a unique `postman.…@example.com` email. **Not cleaned up.** |
| Negative tests → Register with role=quality_engineer | Same: creates a supervisor user. **Not cleaned up.** |
| Users → Change user role | Changes the role of `targetUserId`. Also skipped while `targetUserId` is empty. |
| Products → Create product | Creates a `PM-<timestamp>` product (stored in `createdProductId`). |
| Products → Update product category | Changes the category of `createdProductId` only. |
| Products → Delete product | Deletes `createdProductId` only. |
| Inspections → Upload inspection | Creates 1 inspection for `productId` and runs AI. |
| Inspections → Batch upload inspections | Creates 2 inspections for `productId`. |
| Inspections → Import dataset image as inspection | Creates 1 inspection from the dataset. |
| Inspections → Delete inspection | Deletes `createdInspectionId`, then every other id in `createdInspectionIds`. It never deletes a demo inspection. |

On the clean demo database, keep `allowWrites=false`. Turn it on only against a throwaway database. If
you do run writes, run **Delete product** and **Delete inspection** too; only the two registered
users stay behind.

These requests always run, because they are rejected before anything is written:
- **Logins.**
- **Supervisor imports an inspection → 403.**
- **Batch of 21 files → 400.**
- **Invalid category → 422.**

The last three also use product id `0`, which does not exist, as a second safety net.

## 4. Recommended order ("Run collection")

Use **Run collection** in the folder order the collection already has:

1. **Health**: `/health`, `/health/db` (no login needed).
2. **Auth**: **Login (quality engineer)** first, then Current user and Register.
3. **Users**.
4. **Products**: List products fills in `productId`.
5. **Inspections**: List inspections fills in `inspectionId`; Delete inspection runs last and cleans up.
6. **Analytics**: Analytics summary, then By category.
7. **Dataset**.
8. **AI Models**.
9. **Negative tests**:
   - No token → 401.
   - Invalid token → 401.
   - Register with role=quality_engineer → a supervisor.
   - **Login as supervisor**.
   - Supervisor lists users → 403.
   - Supervisor imports → 403.
   - **Login as quality engineer (restore)**.
   - Batch of 21 files → 400.
   - Invalid category → 422.
   - By category without a token → 401.
   - By category with days=5 → 422.

For a single request, run **Auth → Login (quality engineer)** once first.

The **Negative tests** supervisor steps need `supervisorEmail` and `supervisorPassword`. If those are
empty, the supervisor login and its 403 checks are skipped.

## 5. What is tested

- **Every response**: no response body contains `password_hash` (collection-level test).
- **Login**: status 200, stores `token` and `role`, and the role is as expected.
- **Register**: status 201, `role` is `factory_supervisor` (also when `role=quality_engineer` is sent),
  and the response has no `password` or `password_hash`.
- **Protected requests**: 401 without a token or with an invalid token, and 403 for a supervisor on
  quality-engineer-only endpoints (`/users`, `/inspections/import`).
- **Successful requests**: status codes (200/201/204), and response shapes:
  - `/ai/models` returns 15 rows, one per category.
  - The analytics summary has `automation_rate`.
  - By category (`days=14`): `window_days` is 14, all 15 MVTec categories have a row, rows are ordered by
    total (largest first), and every `defect_rate` is null or between 0 and 1. Its defect rate comes
    from AI predictions, not ground truth.
  - Reports have all their sections.
  - Image endpoints return `image/*`.
- **Heatmap**: 200 with a PNG, or 404 when the inspection has no heatmap.
- **Validation**: 21 files in a batch → 400 ("at most 20"), an unknown category → 422, and a
  by-category window other than 7, 14 or 30 days → 422.

## Regenerating

The collection was generated from the running backend's `/openapi.json` (28 operations, checked one to
one). **Analytics → By category** (`GET /inspections/analytics/by-category`, the 29th) was added by hand
afterwards, together with its two negative tests. If an endpoint is added or changed, add or edit the request by hand in the matching folder, or
import `http://127.0.0.1:8000/openapi.json` into Postman to compare.
