# Milestone 3: Defect classification, severity and manufacturing analytics

- **Milestone:** 3 of 4
- **Period:** 2026-09-12 to 2026-09-14 (first to last commit of this milestone; the later-iteration commits listed
  under Commits run to 2026-10-07)
- **Deliverables:** 8 of 10 rows DONE; see Deliverables for the full status

## Contents

1. [Objective](#objective)
2. [What was built](#what-was-built)
3. [Technical documentation](#technical-documentation)
4. [Deliverables](#deliverables)
5. [Commits](#commits)
6. [Limitations](#limitations)
7. [Carried into the next milestone](#carried-into-the-next-milestone)
8. [How to verify](#how-to-verify)
9. [Troubleshooting](#troubleshooting)
10. [Glossary](#glossary)

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
MVTec AD (MVTec Anomaly Detection) ground-truth masks.

**Quality decision and recommendations** (`d505c68`). `backend/app/inspections/quality.py` derives a decision from
the ground-truth status, the AI (artificial intelligence) prediction, the defect category and the severity, and adds an assessment text and a
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

## Technical documentation

### Scope

This milestone covers what the system concludes from a prediction: the defect category, confidence and the review
rule, severity and quality risk, the quality decision with its recommendation, the per-inspection report, and the
analytics and trends over many inspections. The columns behind these results are listed in
[ARCHITECTURE.md](../ARCHITECTURE.md#database).

### Components

The table below gives, for each component, its module, input, output, key parameters read from the code, and the
test module that covers it.

| Component | Module | Input | Output | Key parameters | Tests |
|---|---|---|---|---|---|
| Margin confidence | `margin_confidence` in `backend/app/ai/inference/localization.py` | score, threshold | value from 0.5 up to 1 | slope 10 | `test_localization.py` (136 tests) |
| Reliability band | `reliability_level`, same module | confidence | `high`, `medium` or `low` | high at 0.95 or more, low below 0.70 | `test_localization.py` |
| Review rule | `review_rule`, same module | confidence, model gate | review flag and reason | flag when confidence < 0.70 or gate is NOT_PRODUCTION_READY | `test_localization.py` |
| Defect form | `classify_defect_form` in `backend/app/inspections/defect_form.py` | boxes, flagged area, image size | form and its score | 3 boxes, 15% area, aspect ratio 3, 2% area | `test_defect_form_severity.py` (180 tests) |
| Severity | `assess_ai_severity` in `backend/app/inspections/severity.py` | flagged area, centroid, form score, confidence | score 0 to 100, level, quality risk | weights 30/25/25/20; bands at 80, 60 and 40 | `test_defect_form_severity.py`, `test_severity_scoring.py` (39 tests) |
| Quality decision | `assess_quality` in `backend/app/inspections/quality.py` | ground-truth status, prediction, defect category, severity, review flag | decision, assessment, recommendation | ordered rules (table below) | `test_quality_assessment.py` (41 tests) |
| Production report | `build_production_quality_report` in `backend/app/inspections/report.py` | one inspection | six report sections | none | `test_production_report.py` (20 tests) |

The two scores are computed as follows, in plain text:

```
confidence = 1 / (1 + exp(-10 * |ln(score / threshold)|))
severity   = 0.30 * size + 0.25 * location + 0.25 * type + 0.20 * confidence     (each factor from 0 to 100)
```

The confidence is undefined, and stored as empty, when the score or the threshold is missing or not above 0.

### Quality decision rules

`assess_quality` applies the first matching rule. The table below lists the rules in that order, separately for
inspections with a ground-truth status (dataset imports) and without one (uploads), as documented in
`backend/app/inspections/quality.py`.

| Inspection | Rule | Decision |
|---|---|---|
| Import | Severity is Critical or High, and ground truth and AI do not disagree | FAIL |
| Import | Ground truth and AI prediction disagree | NOT_ASSESSED |
| Import | Ground truth is defective, or the AI prediction is defective | FAIL |
| Import | Ground truth is good | PASS |
| Upload | No AI prediction | NOT_ASSESSED |
| Upload | Review flag set | MANUAL_REVIEW |
| Upload | AI prediction is defective, at any severity | FAIL |
| Upload | AI prediction is good | PASS |

`test_weak_category_defective_upload_goes_to_manual_review_with_severity_stored` and
`test_false_positive_import_gets_severity_but_keeps_its_ground_truth` in `backend/tests/test_batch_enhancement_api.py`
cover the review rule for an upload and a disagreement for an import.

### Analytics and trend computation

`GET /inspections/analytics/summary` and `GET /inspections/analytics/by-category`
(`backend/app/inspections/analytics.py`) aggregate in PostgreSQL with `COUNT`, `AVG` and `FILTER`, over a trailing
window of 7, 14 or 30 days (default 14; any other value gives 422). The rules behind the trend values are:

- **Direction.** A rate change between the current and the previous window is "stable" when it is below 0.05
  (`TREND_STABLE_THRESHOLD`), otherwise "up" or "down".
- **Minimum sample.** A window comparison needs at least 5 inspections on both sides
  (`MIN_SAMPLE_SIZE_FOR_DEFECT_RATE_INSIGHT`); below that, the summary returns a single "insufficient_trend_history"
  insight.
- **Category trends.** At most 5 category trends are returned (`MAX_CATEGORY_TRENDS`).
- **Automation rate.** The share of decided inspections whose decision is PASS or FAIL.
- **Per-category defect rate.** Comes from AI predictions, not ground truth.

### Examples: report and analytics summary

The two API (application programming interface) examples below follow the schemas and the tests.
`GET /inspections/{inspection_id}/report` returns `ProductionQualityReport` (`backend/app/inspections/schemas.py`).
`backend/tests/test_production_report.py` asserts status 200 for a signed-in user, 401 without a token, 404 for an
unknown id, and that `report_summary.overall_result` equals the inspection's `quality_decision`. The table below
lists the report sections and their fields.

| Section | Fields |
|---|---|
| `inspection` | `id`, `product_id`, `product_name`, `status`, `source`, `inspection_date`, `created_at` |
| `dataset` (imports only) | `category`, `split`, `defect_type`, `filename` |
| `defect` | `category`, `ai_prediction`, `ai_reconstruction_error`, `ai_threshold`, `ai_model_name`, `ai_confidence`, `ai_reliability`, `review_required`, `review_reason`, `localization`, `has_heatmap`, `model_gate`, `defect_form`, `defect_form_label` |
| `severity` | `score`, `level`, `quality_risk`, `recommended_action` |
| `quality` | `decision`, `assessment`, `recommendation` |
| `report_summary` | `overall_result`, `report_status` (`COMPLETE`, or `PARTIAL` for a row without a quality decision), `summary` |

`GET /inspections/analytics/summary?days=14` returns `InspectionAnalyticsSummary`.
`test_analytics_summary_accessible_to_quality_engineer` and `test_analytics_summary_accessible_to_factory_supervisor`
in `backend/tests/test_inspection_analytics.py` assert status 200 for both roles, and
`test_analytics_summary_requires_authentication` asserts 401 without a token. The table below lists its top-level
fields.

| Group | Fields |
|---|---|
| Totals | `total_inspections`, `by_status`, `by_source`, `ai_analyzed_count`, `ai_prediction_counts`, `ai_defect_rate`, `avg_reconstruction_error` |
| Activity and products | `activity_by_day`, `by_product` |
| Distributions | `defect_categories`, `quality_decisions`, `severity_distribution` |
| Insights and trends | `operational_insights`, `trend_monitoring` |
| Performance and automation | `performance`, `manual_review_count`, `automation_rate`, `automation_counts` |

### Severity as implemented, and its deviations

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

### Design rationale

- **Rule-based severity.** The project has no records of real severity outcomes to learn from or calibrate against,
  as `backend/app/inspections/severity.py` notes, so the score is a declared rule with fixed weights and bands that
  can be read and tested.
- **A disagreement is NOT_ASSESSED.** When ground truth and the AI disagree on an import, the decision is
  NOT_ASSESSED instead of either side winning (rule 2 in `quality.py`), so the conflict reaches a person.
- **Review before failing an upload.** For uploads, the review rule is checked before a defective prediction fails
  the product, so a result from a NOT_PRODUCTION_READY model or with low confidence goes to manual review
  (`_assess_ai_only` in `quality.py`).
- **Category from ground truth only.** `defect_category` is the dataset folder name and is not derived from AI output,
  so it stays a ground-truth field (`backend/app/models/inspection.py`).

### Design challenges and how they were resolved

- **Severity with no scores.** The engine of `55f6438` required all four factors, and location and confidence had no
  source, so every severity was "Not assessed". `c847154` (4 October) added `severity_v1`, which takes the four
  factors from the localization, the defect form and the margin confidence; `fc699cb` (7 October) applied it to
  imports.
- **An AI false positive on a good import.** With `severity_v1`, an import with good ground truth can carry an AI
  severity. Rule 1 therefore excludes disagreements, which fall through to NOT_ASSESSED
  (`test_false_positive_import_gets_severity_but_keeps_its_ground_truth`).
- **Uploads without ground truth.** Uploads keep `status` `pending`. `d2b6915` (7 October) shows them as "Unlabelled"
  rather than as a ground-truth value.

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

## Carried into the next milestone

Milestone 4 validates and deploys what this milestone produces. The quality decision and the review rule are where
a category's evidence gate takes effect, so the final-test gates set in Milestone 4 decide which categories go to
manual review. The analytics summary carries the timing fields that Milestone 4 adds. The production report and the
dashboard are part of the end-to-end checks on the live deployment.

## How to verify

- Live deployment: http://13.206.115.50. An inspection's detail page shows the defect category (imports), the
  severity card with its factors, the quality decision and recommendation, and the manual-review banner when it
  applies. The dashboard shows the distributions, the trend and the per-category breakdown for 7, 14 or 30 days.
- API, with a bearer token: `GET /api/inspections/{id}/report`, `GET /api/inspections/analytics/summary?days=14`
  and `GET /api/inspections/analytics/by-category?days=14`.
- Tests, from `backend/`:
  `python -m pytest tests/test_defect_categorization.py tests/test_severity_scoring.py tests/test_defect_form_severity.py tests/test_quality_assessment.py tests/test_production_report.py tests/test_defect_analytics.py tests/test_defect_trends.py`.

## Troubleshooting

No setup or runtime problem specific to this milestone's components is recorded in the repository. Problems with
logins, model files, the dataset and the deployment are listed in the Troubleshooting sections of
[Milestone 1](MILESTONE_1.md#troubleshooting), [Milestone 2](MILESTONE_2.md#troubleshooting) and
[Milestone 4](MILESTONE_4.md#troubleshooting).

## Glossary

The terms below are introduced in this milestone. Earlier terms (inspection, ground truth, anomaly score, locked
threshold) are defined in the glossaries of [Milestone 1](MILESTONE_1.md#glossary) and
[Milestone 2](MILESTONE_2.md#glossary).

| Term | Meaning in this project |
|---|---|
| Confidence (margin) | How far an image's score lies from its locked threshold on a log scale, from 0.5 at the threshold towards 1; a heuristic, not a probability. |
| Reliability band | The display band of the confidence: high at 0.95 or more, medium from 0.70, low below 0.70. |
| Manual review | The flag set when confidence is below 0.70 or the category's model gate is NOT_PRODUCTION_READY; for uploads it gives the decision MANUAL_REVIEW. |
| Severity | A score from 0 to 100 with a level (Low, Medium, High, Critical) and a quality risk, computed for AI-defective inspections with boxes. |
| severity_v1 | The current severity rule: weighted size, location, defect form and confidence taken from the AI result. |
| Defect form | The shape class of the flagged region (multiple regions, large area, linear, small spot, localized patch), from `defect_form_v1`. |
| Quality decision | PASS, FAIL, MANUAL_REVIEW or NOT_ASSESSED, from the ordered rules in `quality.py`. |
| Automation rate | The share of decided inspections whose decision is PASS or FAIL, reported by the analytics summary. |
