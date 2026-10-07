"""defect_form_v1, severity_v1 and the AI-only quality precedence - pure unit tests on plain values (no model,
no database, no dataset). Also a frozen copy of the ground-truth (import) quality rules, proving imports keep
their exact decisions and texts (rule 1 skips a ground-truth/AI conflict, now that AI-defective imports
get severity_v1)."""

from itertools import product
from types import SimpleNamespace

import pytest

from app.inspections.defect_form import (
    DEFECT_FORM_VERSION,
    add_defect_form,
    classify_defect_form,
    defect_form_label,
)
from app.inspections.quality import FAIL, MANUAL_REVIEW, NOT_ASSESSED, PASS, QualityAssessment, assess_quality
from app.inspections.service import _ai_only_severity
from app.inspections.severity import (
    SEVERITY_ACTIONS,
    ai_severity_factors,
    assess_ai_severity,
    location_score_from_centroid,
    recommended_action_for_level,
    size_score_from_area_pct,
)


def box(x=0.1, y=0.1, w=0.1, h=0.1):
    return {"x": x, "y": y, "w": w, "h": h, "peak": 2.0, "area_fraction": w * h}


# ---------------------------------------------------------------------------
# defect_form_v1 - every branch and the ordering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("boxes, area_pct, size, expected", [
    ([box(), box(0.5), box(0.7)], 50.0, (100, 100), "multiple_regions"),       # >= 3 boxes beats large area
    ([box(), box(0.5), box(0.7, w=0.6, h=0.05)], 1.0, (100, 100), "multiple_regions"),  # ...and linear / small
    ([box(w=0.5, h=0.4), box(0.7)], 15.0, (100, 100), "large_area"),            # boundary 15 is large
    ([box(w=0.9, h=0.2)], 18.0, (100, 100), "large_area"),                      # large beats linear (aspect 4.5)
    ([box(w=0.6, h=0.1)], 6.0, (100, 100), "linear"),                           # pixel aspect 6
    ([box(w=0.4, h=0.1)], 1.0, (100, 100), "linear"),                           # linear beats small spot
    ([box(w=0.1, h=0.1)], 1.99, (100, 100), "small_spot"),
    ([box(w=0.2, h=0.1)], 2.0, (100, 100), "localized_patch"),                  # boundary 2 is not small
    ([box(w=0.2, h=0.2)], 4.0, (100, 100), "localized_patch"),
    ([box(w=0.2, h=0.6)], 5.0, (300, 100), "localized_patch"),                  # 60x60 px: square, not linear
    ([box(w=0.2, h=0.2)], 5.0, (100, 300), "linear"),                           # 20x60 px: aspect exactly 3
    ([box(w=0.3, h=0.3), box(0.6, w=0.25, h=0.02)], 9.5, (100, 100), "localized_patch"),  # largest box is square
])
def test_defect_form_truth_table(boxes, area_pct, size, expected):
    form = classify_defect_form(boxes, area_pct, *size)
    assert form.form == expected
    assert form.score == {"multiple_regions": 75, "large_area": 85, "linear": 60, "small_spot": 30,
                          "localized_patch": 50}[expected]


def test_no_boxes_or_unknown_size_gives_no_form():
    assert classify_defect_form([], 10.0, 100, 100) is None
    assert classify_defect_form([box()], 10.0, 0, 100) is None


def test_add_defect_form_stores_form_score_and_version_in_the_localization_only():
    located = {"boxes": [box(w=0.6, h=0.1)], "area_pct": 6.0}
    add_defect_form(located, 100, 100)
    assert located["defect_form"] == "linear"
    assert located["defect_form_score"] == 60
    assert located["defect_form_version"] == DEFECT_FORM_VERSION == "defect_form_v1"
    empty = {"boxes": [], "area_pct": 0.0}
    assert add_defect_form(empty, 100, 100) == {"boxes": [], "area_pct": 0.0}


def test_defect_form_labels():
    assert defect_form_label("localized_patch") == "Localized patch"
    assert defect_form_label("multiple_regions") == "Multiple regions"
    assert defect_form_label(None) is None
    assert defect_form_label("something_new") == "Something new"


# ---------------------------------------------------------------------------
# severity_v1 factors and formula
# ---------------------------------------------------------------------------

def test_size_score():
    assert size_score_from_area_pct(0) == 0
    assert size_score_from_area_pct(10) == 50
    assert size_score_from_area_pct(20) == 100
    assert size_score_from_area_pct(41.2) == 100


@pytest.mark.parametrize("cx, cy, expected", [
    (0.5, 0.5, 100.0), (0.0, 0.0, 40.0), (1.0, 1.0, 40.0), (1.0, 0.0, 40.0), (0.0, 0.5, 100 - 60 / 2 ** 0.5),
])
def test_location_score_centre_corner_and_edge(cx, cy, expected):
    assert location_score_from_centroid(cx, cy) == pytest.approx(expected)


def test_worked_example():
    # area 10% -> size 50; centred -> location 100; localized_patch -> type 50; confidence 0.9 -> 90
    # 0.30*50 + 0.25*100 + 0.25*50 + 0.20*90 = 15 + 25 + 12.5 + 18 = 70.5 -> High
    result = assess_ai_severity(10.0, [0.5, 0.5], 50, 0.9)
    assert result.score == pytest.approx(70.5)
    assert result.level == "High"
    assert result.quality_risk == "High Risk"


@pytest.mark.parametrize("area_pct, form_score, confidence, score, level", [
    (0.0, 30, 0.375, 40.0, "Medium"),     # 0 + 25 + 7.5 + 7.5
    (0.0, 30, 0.3125, 38.75, "Low"),       # 0 + 25 + 7.5 + 6.25
    (10.0, 50, 0.375, 60.0, "High"),      # 15 + 25 + 12.5 + 7.5
    (10.0, 50, 0.3125, 58.75, "Medium"),  # 15 + 25 + 12.5 + 6.25
    (20.0, 85, 0.1875, 80.0, "Critical"),  # 30 + 25 + 21.25 + 3.75
    (20.0, 85, 0.125, 78.75, "High"),     # 30 + 25 + 21.25 + 2.5
    (20.0, 85, 1.0, 96.25, "Critical"),
])
def test_severity_band_boundaries(area_pct, form_score, confidence, score, level):
    result = assess_ai_severity(area_pct, [0.5, 0.5], form_score, confidence)
    assert result.score == score  # exact: every term is a binary-exact value
    assert result.level == level


@pytest.mark.parametrize("args", [
    (None, [0.5, 0.5], 50, 0.9), (10.0, None, 50, 0.9), (10.0, [0.5], 50, 0.9), (10.0, [0.5, 0.5], None, 0.9),
    (10.0, [0.5, 0.5], 50, None), (float("nan"), [0.5, 0.5], 50, 0.9),
])
def test_severity_not_assessed_when_any_input_missing(args):
    assert ai_severity_factors(*args) is None
    result = assess_ai_severity(*args)
    assert result.score is None and result.level is None and result.quality_risk == "Not assessed"


def test_recommended_actions_use_the_specification_wording():
    assert SEVERITY_ACTIONS == {
        "Critical": "Reject product and trigger quality inspection workflow",
        "High": "Repair or rework recommended",
        "Medium": "Inspection review required",
        "Low": "Minor; product generally acceptable",
    }
    assert recommended_action_for_level(None) is None


# ---------------------------------------------------------------------------
# Which inspections get severity_v1 (service helper)
# ---------------------------------------------------------------------------

LOCALIZED = {"boxes": [box(0.4, 0.4, 0.2, 0.2)], "area_pct": 4.0, "centroid": [0.5, 0.5],
             "defect_form": "localized_patch", "defect_form_score": 50}


def _inspection(**overrides):
    values = dict(id=1, status="pending", ai_prediction="defective", localization=dict(LOCALIZED), ai_confidence=0.9)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ai_only_defective_localized_upload_gets_severity_v1():
    result = _ai_only_severity(_inspection())
    assert result.score == pytest.approx(0.30 * 20 + 25 + 12.5 + 18)  # 61.5
    assert result.level == "High"


@pytest.mark.parametrize("status", ["good", "defective"])
def test_ai_defective_import_gets_the_same_severity_v1_as_an_upload(status):
    # The gate depends on the AI result only; the ground-truth status is neither read nor changed.
    inspection = _inspection(status=status)
    result = _ai_only_severity(inspection)
    upload = _ai_only_severity(_inspection())
    assert (result.score, result.level) == (upload.score, upload.level) == (pytest.approx(61.5), "High")
    assert inspection.status == status


@pytest.mark.parametrize("overrides", [
    {"status": "good", "ai_prediction": "good"},          # imports the AI found good: no severity
    {"status": "defective", "ai_prediction": "good"},
    {"ai_prediction": "good"}, {"ai_prediction": None},   # good / no AI: no severity
    {"localization": None}, {"localization": {**LOCALIZED, "boxes": []}},
    {"localization": {k: v for k, v in LOCALIZED.items() if k != "defect_form_score"}},
    {"ai_confidence": None},
])
def test_no_ai_severity_otherwise(overrides):
    result = _ai_only_severity(_inspection(**overrides))
    assert result is None or result.score is None


# ---------------------------------------------------------------------------
# Quality precedence for AI-only inspections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ai, level, score, review, expected", [
    (None, None, None, None, NOT_ASSESSED),
    (None, "Critical", 90.0, True, NOT_ASSESSED),
    ("defective", "Critical", 90.0, True, MANUAL_REVIEW),   # review BEFORE a severity-driven FAIL
    ("defective", "High", 70.0, True, MANUAL_REVIEW),
    ("good", None, None, True, MANUAL_REVIEW),
    ("defective", "Critical", 90.0, False, FAIL),
    ("defective", "High", 70.0, None, FAIL),
    ("defective", "Low", 30.0, False, FAIL),
    ("defective", None, None, False, FAIL),
    ("good", None, None, False, PASS),
])
def test_ai_only_precedence_matrix(ai, level, score, review, expected):
    result = assess_quality("pending", ai, None, level, review_required=review, severity_score=score)
    assert result.decision == expected


def test_ai_only_texts_quote_severity_and_its_action():
    failed = assess_quality("pending", "defective", None, "Critical", review_required=False, severity_score=84.6)
    assert failed.assessment == "High-severity defect evidence requires review. Severity: Critical (85/100)."
    assert failed.recommendation.startswith("Reject product and trigger quality inspection workflow.")
    medium = assess_quality("pending", "defective", None, "Medium", review_required=False, severity_score=45.0)
    assert medium.assessment == "Defective product requires review. Severity: Medium (45/100)."
    assert medium.recommendation.startswith("Inspection review required.")
    review = assess_quality("pending", "defective", None, "High", review_required=True, severity_score=70.4)
    assert review.assessment.endswith("Severity: High (70/100).")
    assert len(review.assessment) <= 255 and len(failed.recommendation) <= 500  # column sizes


# ---------------------------------------------------------------------------
# Imports with ground truth: frozen copy of the rules (rule 1 skips a ground-truth/AI conflict)
# ---------------------------------------------------------------------------

def _frozen_ground_truth_quality(status, ai_prediction, defect_category, severity_level):
    def recs(category):
        if category in ("broken_large", "broken_small"):
            return ["Inspect the product for structural damage before release."]
        if category == "contamination":
            return ["Inspect the product for contamination and verify cleaning or handling procedures."]
        return []

    conflict = ai_prediction in ("good", "defective") and status != ai_prediction
    if severity_level in ("Critical", "High") and not conflict:  # a conflict is never auto-failed
        return QualityAssessment(FAIL, "High-severity defect evidence requires review.",
                                 " ".join(["Escalate the inspection for quality review.", *recs(defect_category)]))
    if conflict:
        return QualityAssessment(NOT_ASSESSED, "Conflicting inspection evidence requires review.",
                                 "Conflicting inspection evidence requires quality review.")
    if status == "defective" or ai_prediction == "defective":
        r = (["Review the detected anomaly before approving the product."] if ai_prediction == "defective" else [])
        r += recs(defect_category)
        return QualityAssessment(FAIL, "Defective product requires review.",
                                 " ".join(r or ["Escalate the inspection for quality review."]))
    return QualityAssessment(PASS, "No significant defect evidence identified.",
                             "Product can proceed to the next quality-control stage.")


@pytest.mark.parametrize("level", ["Critical", "High"])
def test_severe_false_positive_import_is_not_assessed_not_failed(level):
    # Ground truth "good", AI "defective" with a severity_v1 level: the conflict rule decides, not rule 1.
    result = assess_quality("good", "defective", "good", level, review_required=False, severity_score=90.0)
    assert result.decision == NOT_ASSESSED
    assert result.assessment == "Conflicting inspection evidence requires review."
    # Without the conflict, the same severity still fails as before.
    assert assess_quality("defective", "defective", "crack", level).decision == FAIL


@pytest.mark.parametrize("status, ai, level, category", list(product(
    ("good", "defective"), (None, "good", "defective"), (None, "Low", "Medium", "High", "Critical"),
    (None, "good", "broken_large", "contamination"),
)))
def test_imports_with_ground_truth_match_the_frozen_rules(status, ai, level, category):
    expected = _frozen_ground_truth_quality(status, ai, category, level)
    for review, score in ((None, None), (True, 90.0), (False, 30.0)):
        assert assess_quality(status, ai, category, level, review_required=review, severity_score=score) == expected
