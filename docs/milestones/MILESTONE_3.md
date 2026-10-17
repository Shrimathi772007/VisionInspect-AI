# Milestone 3: Defect classification, severity and manufacturing analytics

## Objective

The third milestone turns individual predictions into quality information that a factory can act on. Each
inspection should carry a defect category, a severity with a quality risk, and a pass or fail decision with a
recommendation. The dashboard should show how defects and decisions develop over time and across products and
categories.

## What was built

The work was committed between 12 and 14 September 2026, in six steps labelled "Milestone 3 Phase 1" to "Phase 5"
plus a final trend-monitoring commit.

**Defect category** (`f0e59e8`). A `defect_category` column stores the known defect type of an inspection. For a
dataset import it is the name of the defect folder the image came from (for example `broken_large` or `good`). For
an upload it stays empty, because nothing is known about the image. This is categorization from ground truth, not
classification: no part of the system predicts the defect type of an image.

**Severity and quality risk** (`55f6438`). `backend/app/inspections/severity.py` implements the four-factor score
of the project specification: size 30%, location 25%, defect type 25% and confidence 20%. Scores map to Critical
(80 to 100), High (60 to 79), Medium (40 to 59) and Low (0 to 39), and each level maps to a quality risk ("Critical
Risk" to "Low Risk"). As committed, the engine only scored when all four factors had real evidence. Location and
confidence had none, so every inspection was "Not assessed". The size factor could already be measured from the
MVTec ground-truth masks.

**Quality decision and recommendations** (`d505c68`). `backend/app/inspections/quality.py` derives a decision from
the ground-truth status, the AI prediction, the defect category and the severity, and adds an assessment text and a
recommendation. The rules are ordered and deterministic. A Critical or High severity fails the inspection, unless
the ground truth and the AI disagree. Disagreement between ground truth and AI gives NOT_ASSESSED, not a forced
answer. Otherwise a ground-truth defect, or an agreeing AI defect, fails, and a ground-truth good passes.

**Production quality report** (`d5a3b87`). `GET /inspections/{id}/report` returns one structured report per
inspection, with inspection, dataset, defect, severity and quality sections and a summary.

**Defect analytics** (`b481377`). The analytics summary gained defect-category, quality-decision and severity
distributions and a per-product breakdown.

**Trend monitoring** (`37ccfd7`). This added a daily series, per-category trends and deterministic insights that
compare the current window with the previous one. All values are aggregated in PostgreSQL.

Several parts were completed in later iterations:

- **Confidence, review rule and localization** (`a07b1a1`, 4 October). The margin confidence is
  `sigmoid(10 * |ln(score / threshold)|)` (`margin_confidence` in `backend/app/ai/inference/localization.py`). Its
  display bands are high at 0.95 or more, medium from 0.70, and low below 0.70 (`reliability_level`). The review
  rule (`review_rule`) flags an inspection when the confidence is below 0.70 or the category's model gate is
  NOT_PRODUCTION_READY. For uploads, which have no ground truth, a flagged result gets the decision
  `MANUAL_REVIEW`. For dataset imports the flag is stored, but the ground-truth rules above still decide. The
  same commit added the anomaly-map boxes that the severity rule needs.
- **Severity that produces scores** (`c847154`, 4 October; `fc699cb`, 7 October). `severity_v1` scores every
  inspection that the AI found defective and localized, uploads and imports alike, with the same weights and
  bands. It also introduced the rule-based defect form (`backend/app/inspections/defect_form.py`).
- **Interface and analytics** (`5070829`, `ce42d43`). These added the interface for localization, confidence,
  manual review and severity. `d2b6915` labels uploads without ground truth as "Unlabelled". `509a6cd` added the
  per-category analytics endpoint and dashboard breakdown.

## Severity as implemented, and its deviations

`severity_v1` (`assess_ai_severity` in `backend/app/inspections/severity.py`) computes four factor scores from 0 to
100 and applies the specification's weights.

| Factor | Weight | How it is computed | Code |
|---|---:|---|---|
| Size | 30% | min(100, flagged area in percent x 5) | `size_score_from_area_pct` |
| Location | 25% | 100 at the image centre, falling linearly to 40 at a corner | `location_score_from_centroid` |
| Defect type | 25% | score of the defect form | `DEFECT_FORM_SCORES` in `defect_form.py` |
| Confidence | 20% | margin confidence x 100 | `margin_confidence` in `localization.py` |

Two of these factors deviate from the specification:

- **Deviation, defect type factor.** No defect-type classifier exists, so the 25% defect-type factor is filled by
  `defect_form_v1` (`classify_defect_form` in `backend/app/inspections/defect_form.py`). This is a shape rule on the
  flagged region. It assigns, in order:
  - at least 3 regions: multiple regions, score 75;
  - flagged area at least 15%: large area, 85;
  - largest box aspect ratio at least 3: linear, 60;
  - flagged area under 2%: small spot, 30;
  - otherwise: localized patch, 50.

  The form describes what the region looks like. It does not identify the MVTec defect type.
- **Deviation, location factor.** No per-product data defines which areas are critical, so the location factor is a
  centre-weighted proxy for "critical area" (`location_score_from_centroid`). The image centre counts as most
  critical.

The size factor is measured on regions of a blurred anomaly map, which overestimates the defect area. The code
records one check in which a bar covering 9.9% of an image produced a 41% region. None of the factors is
calibrated against real severity outcomes.

## Deliverables

| Task or criterion | Status | Evidence |
|---|---|---|
| Defect categorization workflow | PARTIAL | `f0e59e8`: category taken from the dataset folder for imports; uploads have none |
| Severity scoring and risk assessment | DONE | `55f6438`; scores produced by `severity_v1`, delivered in a later iteration: `c847154` (4 October), `fc699cb` (7 October). Two factors are deviations (see above). |
| Quality assessment module | DONE | `d505c68`: `backend/app/inspections/quality.py`; `MANUAL_REVIEW` delivered in a later iteration: `a07b1a1` (4 October) |
| Production quality reports | DONE | `d5a3b87`: `GET /inspections/{id}/report`, `backend/tests/test_production_report.py` |
| Defect analytics dashboard | DONE | `b481377`, `backend/tests/test_defect_analytics.py`; per-category analytics delivered in a later iteration: `509a6cd` (6 October) |
| Trend monitoring workflow | DONE | `37ccfd7`, `backend/tests/test_defect_trends.py` |
| Classification and analytics dashboard implemented | PARTIAL | The analytics dashboard is done. Defect type classification is partial: there is no model that predicts the defect type, imports carry the ground-truth folder name, and the defect form is a shape rule. |
| Quality and severity reports functional | DONE | Reports from `d5a3b87`; severity scores from `c847154` |
| Production quality analytics generated | DONE | `b481377`, `37ccfd7` |
| Trend monitoring integrated | DONE | `37ccfd7` |

## Commits

| Commit | Date | Content |
|---|---|---|
| `f0e59e8` | 2026-09-12 | Defect category column and its use for imports |
| `55f6438` | 2026-09-12 | Severity engine and quality risk |
| `d505c68` | 2026-09-13 | Quality decision, assessment and recommendations |
| `d5a3b87` | 2026-09-14 | Production quality report per inspection |
| `b481377` | 2026-09-14 | Defect analytics in the summary and the dashboard |
| `37ccfd7` | 2026-09-14 | Trend monitoring and operational insights |

Delivered in a later iteration:

| Commit | Date | Content |
|---|---|---|
| `a07b1a1` | 2026-10-04 | Anomaly-map localization, margin confidence, review rule, AI decisions for uploads |
| `5070829` | 2026-10-04 | Localization overlay, confidence and review interface, Models page |
| `c847154` | 2026-10-04 | Defect form and `severity_v1` (also the enhancement preview, see Milestone 2) |
| `ce42d43` | 2026-10-04 | Severity and enhancement interface (also batch upload and camera simulation, see Milestone 1) |
| `509a6cd` | 2026-10-06 | Per-category analytics and dashboard refinements |
| `fc699cb` | 2026-10-07 | `severity_v1` applied to dataset imports |
| `d2b6915` | 2026-10-07 | Uploads without ground truth shown as "Unlabelled" |

## Limitations

Defect type classification is only partly met, as described above. Severity is rule-based and uncalibrated. Its
defect-type and location factors are the deviations listed above, and its size factor runs high because of the
blurred map. Confidence is a margin heuristic, not a calibrated probability. Until 4 October every severity was
"Not assessed".

## How to verify

- Live deployment: http://13.206.115.50. An inspection's detail page shows the defect category (imports), the
  severity card with its factors, the quality decision and recommendation, and the manual-review banner when it
  applies. The dashboard shows the distributions, the trend and the per-category breakdown for 7, 14 or 30 days.
- API, with a bearer token: `GET /api/inspections/{id}/report`, `GET /api/inspections/analytics/summary?days=14`
  and `GET /api/inspections/analytics/by-category?days=14`.
- Tests, from `backend/`:
  `python -m pytest tests/test_defect_categorization.py tests/test_severity_scoring.py tests/test_defect_form_severity.py tests/test_quality_assessment.py tests/test_production_report.py tests/test_defect_analytics.py tests/test_defect_trends.py`.
