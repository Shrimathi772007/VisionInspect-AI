"""Milestone 3 Phase 3: quality assessment and recommendations.

A deterministic, explainable PASS/FAIL/NOT_ASSESSED quality decision for one inspection,
built only from evidence that already exists elsewhere in this system:

    inspection.status        - business/ground-truth status ("good"/"defective"/"pending")
    ai_prediction             - the autoencoder's anomaly-detection guess (Phase 7/8)
    defect_category           - MVTec ground-truth defect type (Phase 1)
    severity_level            - the Phase 2 severity assessment ("Critical"/"High"/.../None)

Kept deliberately framework/DB-free (same convention as app.ai.inference and
app.inspections.severity) - this module only ever computes from plain values and never
touches the database or an ORM object, so its logic is trivially unit-testable.
app.inspections.service wires its output onto a persisted Inspection.

WHY THIS IS A SEPARATE CONCEPT FROM STATUS/AI_PREDICTION/SEVERITY
--------------------------------------------------------------------
quality_decision is NOT a copy of `status` (which is ground truth, only meaningful for
MVTec imports - generic uploads are always "pending"), NOT a copy of `ai_prediction`
(which is only the autoencoder's own guess, with known poor recall on real defects - see
frontend/src/utils/badgeMaps.js's AI_PREDICTION_* comment), and NOT a restatement of
`severity_level`/`quality_risk` (which score how bad a *known* defect is, not whether the
product should pass). It is a fourth, independent conclusion that *reasons about* the
other three without overwriting or being derived by simply copying any one of them.

EVIDENCE PRECEDENCE (in order - the first matching rule decides the outcome)
--------------------------------------------------------------------------------
1. severity_level is Critical or High
       -> FAIL. A known-severe defect must never be waved through as PASS, regardless of
          what status/ai_prediction say (in this system severity is only ever computed
          from a real defect_category, so this cannot currently contradict a "good"
          ground truth - see app.inspections.severity - but the check is kept first so it
          continues to take precedence if severity coverage improves later, per Phase 3's
          extensibility requirement). quality_risk is a deterministic function of
          severity_level in this system (see severity.quality_risk_for_level), so checking
          severity_level alone is equivalent and sufficient - there is no independent
          signal in quality_risk today.
2. status is a ground-truth value ("good"/"defective") AND ai_prediction is present AND
   they disagree
       -> NOT_ASSESSED ("conflicting evidence"). Neither value is silently trusted over
          the other, and neither is silently promoted to PASS - this is the explicit
          conflicting-evidence policy this phase requires. FAIL was deliberately not
          chosen here: asserting FAIL against a known-good ground truth is a stronger,
          less honest claim than declining to decide and flagging the disagreement for
          human review.
3. status == "defective" OR ai_prediction == "defective" (and rule 2 did not already
   apply, so if both are present they agree)
       -> FAIL. Either a real ground-truth defect or an AI-flagged anomaly is treated as
          sufficient not-PASS evidence on its own. This is intentionally asymmetric with
          rule 5 below: an AI "defective" is enough to withhold PASS, but an AI "good" is
          not enough, alone, to grant it (see rule 5's docstring for why).
4. status == "good" (regardless of whether ai_prediction agrees or is absent)
       -> PASS. Ground truth is authoritative when available and not contradicted by AI
          evidence (rule 2 already catches the disagreement case). An agreeing
          ai_prediction is not required for PASS but is compatible with it.
5. Otherwise (no ground-truth status, and either no ai_prediction or ai_prediction ==
   "good", and no disqualifying severity)
       -> NOT_ASSESSED ("insufficient evidence"). This is deliberately NOT "PASS": an
          AI "good" prediction, by itself, with no ground truth to corroborate it, is not
          treated as sufficient evidence for PASS, given this autoencoder's known poor
          recall on real defects (~46% - see the AI Prediction card in
          InspectionDetailPage.jsx / badgeMaps.js). Concretely, today every generic
          upload has status="pending", defect_category=None, and ai_prediction=None (only
          mvtec_ad inspections currently ever get an AI prediction at all - see
          app.inspections.service._resolve_category), so every generic upload lands here.
"""

from dataclasses import dataclass
from typing import Optional

from app.ai.inference import DEFECTIVE_PREDICTION, GOOD_PREDICTION
from app.inspections.severity import CRITICAL, HIGH

PASS = "PASS"
FAIL = "FAIL"
NOT_ASSESSED = "NOT_ASSESSED"

_GROUND_TRUTH_STATUSES = (GOOD_PREDICTION, DEFECTIVE_PREDICTION)  # "good"/"defective" - not "pending"

# Recommendation sentences - fixed, deterministic wording. Each states only what this
# system actually knows; none claims a capability (physical defect location, AI
# classification of defect type, or a human inspection) this system does not have.
_ESCALATE_RECOMMENDATION = "Escalate the inspection for quality review."
_AI_REVIEW_RECOMMENDATION = "Review the detected anomaly before approving the product."
_STRUCTURAL_RECOMMENDATION = "Inspect the product for structural damage before release."
_CONTAMINATION_RECOMMENDATION = "Inspect the product for contamination and verify cleaning or handling procedures."
_CONFLICT_RECOMMENDATION = "Conflicting inspection evidence requires quality review."
_PASS_RECOMMENDATION = "Product can proceed to the next quality-control stage."
_INSUFFICIENT_EVIDENCE_RECOMMENDATION = "Additional inspection evidence is required before making a quality decision."

# Defect-category -> recommendation sentence(s), for defect_category values with a known,
# documented real-world meaning (Phase 1 MVTec ground truth). This is an explicit
# implementation/domain mapping, not a specification-defined value - see
# app.inspections.severity's _DEFECT_TYPE_SCORES comment for the same caveat about
# defect_category-keyed judgment calls. Categories not listed here (including future
# non-Bottle MVTec categories) get no defect-specific recommendation, never a guessed one.
_STRUCTURAL_DEFECT_CATEGORIES = ("broken_large", "broken_small")
_CONTAMINATION_DEFECT_CATEGORIES = ("contamination",)


def _defect_category_recommendations(defect_category: Optional[str]) -> list[str]:
    if defect_category in _STRUCTURAL_DEFECT_CATEGORIES:
        return [_STRUCTURAL_RECOMMENDATION]
    if defect_category in _CONTAMINATION_DEFECT_CATEGORIES:
        return [_CONTAMINATION_RECOMMENDATION]
    return []


@dataclass
class QualityAssessment:
    """The persisted result of a Phase 3 quality assessment for one inspection."""

    decision: str  # PASS / FAIL / NOT_ASSESSED
    assessment: str  # short, human-readable explanation of the decision
    recommendation: str  # one or more deterministic sentences, highest-priority first


def _combine(*sentences: str) -> str:
    return " ".join(s for s in sentences if s)


def assess_quality(
    status: str,
    ai_prediction: Optional[str],
    defect_category: Optional[str],
    severity_level: Optional[str],
) -> QualityAssessment:
    """Deterministic PASS/FAIL/NOT_ASSESSED quality decision - see module docstring for
    the exact, ordered precedence rules this implements."""

    # Rule 1: a known-severe defect always wins, regardless of status/ai_prediction.
    if severity_level in (CRITICAL, HIGH):
        return QualityAssessment(
            decision=FAIL,
            assessment="High-severity defect evidence requires review.",
            recommendation=_combine(_ESCALATE_RECOMMENDATION, *_defect_category_recommendations(defect_category)),
        )

    has_ground_truth = status in _GROUND_TRUTH_STATUSES
    ai_available = ai_prediction in (GOOD_PREDICTION, DEFECTIVE_PREDICTION)

    # Rule 2: ground truth and AI evidence both present but disagree - never silently
    # resolved either way.
    if has_ground_truth and ai_available and status != ai_prediction:
        return QualityAssessment(
            decision=NOT_ASSESSED,
            assessment="Conflicting inspection evidence requires review.",
            recommendation=_CONFLICT_RECOMMENDATION,
        )

    # Rule 3: a real defect, from ground truth and/or the AI, with nothing contradicting it.
    if status == DEFECTIVE_PREDICTION or ai_prediction == DEFECTIVE_PREDICTION:
        recommendations = []
        if ai_prediction == DEFECTIVE_PREDICTION:
            recommendations.append(_AI_REVIEW_RECOMMENDATION)
        recommendations.extend(_defect_category_recommendations(defect_category))
        if not recommendations:
            recommendations.append(_ESCALATE_RECOMMENDATION)
        return QualityAssessment(
            decision=FAIL,
            assessment="Defective product requires review.",
            recommendation=_combine(*recommendations),
        )

    # Rule 4: ground truth confirms good, and rule 2 already ruled out disagreement.
    if status == GOOD_PREDICTION:
        return QualityAssessment(
            decision=PASS,
            assessment="No significant defect evidence identified.",
            recommendation=_PASS_RECOMMENDATION,
        )

    # Rule 5: nothing sufficient to decide - includes every generic upload today, and an
    # AI "good" prediction with no ground truth to corroborate it (see module docstring).
    return QualityAssessment(
        decision=NOT_ASSESSED,
        assessment="Insufficient evidence.",
        recommendation=_INSUFFICIENT_EVIDENCE_RECOMMENDATION,
    )
