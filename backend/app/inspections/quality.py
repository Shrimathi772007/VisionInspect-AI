"""Milestone 3 Phase 3: quality assessment and recommendations.

A deterministic, explainable PASS/FAIL/MANUAL_REVIEW/NOT_ASSESSED quality decision for one
inspection, built only from evidence that already exists elsewhere in this system:

    inspection.status        - business/ground-truth status ("good"/"defective"/"pending")
    ai_prediction             - the category's served anomaly model's prediction
    defect_category           - MVTec ground-truth defect type (Phase 1)
    severity_level            - the Phase 2 severity assessment ("Critical"/"High"/.../None)
    review_required           - the declared manual-review rule (app.ai.inference.localization.
                                review_rule: margin confidence < 0.70 or a NOT_PRODUCTION_READY
                                category model); only consulted when there is no ground truth

Kept deliberately framework/DB-free (same convention as app.ai.inference and
app.inspections.severity) - this module only ever computes from plain values and never
touches the database or an ORM object, so its logic is trivially unit-testable.
app.inspections.service wires its output onto a persisted Inspection.

WHY THIS IS A SEPARATE CONCEPT FROM STATUS/AI_PREDICTION/SEVERITY
--------------------------------------------------------------------
quality_decision is NOT a copy of `status` (which is ground truth, only meaningful for
MVTec imports - generic uploads are always "pending"), NOT a copy of `ai_prediction`
(which is only the served model's own guess - its reliability differs per category, see the
`gate` in app.ai.inference.serving), and NOT a restatement of
`severity_level`/`quality_risk` (which score how bad a *known* defect is, not whether the
product should pass). It is a fourth, independent conclusion that *reasons about* the
other three without overwriting or being derived by simply copying any one of them.

EVIDENCE PRECEDENCE (in order - the first matching rule decides the outcome)
--------------------------------------------------------------------------------
1. severity_level is Critical or High, AND rule 2's conflict does not apply
       -> FAIL. A known-severe defect must never be waved through as PASS. Imports the AI found
          defective now get severity_v1 from the AI's own localization (app.inspections.severity),
          so for an AI false positive (ground truth "good", AI "defective") the severity is the
          AI's claim, not known evidence - that conflict falls through to rule 2 (NOT_ASSESSED)
          instead of a FAIL against the ground truth. quality_risk is a deterministic function of
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
   apply, so if both are present they agree) - with a ground-truth status
       -> FAIL. Either a real ground-truth defect or an AI-flagged anomaly is treated as
          sufficient not-PASS evidence on its own.
4. status == "good" (regardless of whether ai_prediction agrees or is absent)
       -> PASS. Ground truth is authoritative when available and not contradicted by AI
          evidence (rule 2 already catches the disagreement case). An agreeing
          ai_prediction is not required for PASS but is compatible with it.
Rules 2-4 (inspections WITH a ground-truth status, i.e. MVTec imports) are unchanged by the
AI-only rules below; review_required is stored for them but never changes their decision.

AI-ONLY INSPECTIONS (no ground-truth status - every upload) use their own order; rule 1 does
NOT come first for them (severity_v1, app.inspections.severity, now scores AI-defective uploads
from their localization, and a weak model's result must reach a person rather than be auto-failed):
5. no AI prediction
       -> NOT_ASSESSED ("insufficient evidence"), as before.
6. review_required (confidence < 0.70, and/or the category's model is NOT_PRODUCTION_READY)
       -> MANUAL_REVIEW, BEFORE any severity-driven FAIL. The AI result exists but is not
          reliable enough to decide on, so a person must inspect the product. Applies to an
          AI "good" and "defective" alike; the severity, if any, is still stored and quoted.
7. ai_prediction == "defective" (any severity, Critical/High included)
       -> FAIL, with the severity and its recommended action in the texts when present.
8. ai_prediction == "good"
       -> PASS. Every category is now served by its own locked model with a static evidence
          gate (app.ai.inference.serving); a NOT_PRODUCTION_READY model or a low-margin score
          never reaches this rule (rule 6), so an AI "good" from a production-ready model with
          a clear margin is accepted as PASS. (This used to be NOT_ASSESSED because the only
          model then served, the Bottle ConvAE, had poor recall - 73.02% on its final test.)
"""

from dataclasses import dataclass
from typing import Optional

from app.ai.inference import DEFECTIVE_PREDICTION, GOOD_PREDICTION
from app.inspections.severity import CRITICAL, HIGH, recommended_action_for_level

PASS = "PASS"
FAIL = "FAIL"
NOT_ASSESSED = "NOT_ASSESSED"
MANUAL_REVIEW = "MANUAL_REVIEW"
DECISIONS = (PASS, FAIL, MANUAL_REVIEW, NOT_ASSESSED)

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
_MANUAL_REVIEW_ASSESSMENT = "AI result is low-reliability for this category/confidence; manual inspection required."
_MANUAL_REVIEW_RECOMMENDATION = "Inspect the product manually before making a quality decision."

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

    decision: str  # PASS / FAIL / MANUAL_REVIEW / NOT_ASSESSED
    assessment: str  # short, human-readable explanation of the decision
    recommendation: str  # one or more deterministic sentences, highest-priority first


def _combine(*sentences: str) -> str:
    return " ".join(s for s in sentences if s)


def assess_quality(
    status: str,
    ai_prediction: Optional[str],
    defect_category: Optional[str],
    severity_level: Optional[str],
    review_required: Optional[bool] = None,
    severity_score: Optional[float] = None,
) -> QualityAssessment:
    """Deterministic PASS/FAIL/MANUAL_REVIEW/NOT_ASSESSED quality decision - see the module
    docstring for the exact, ordered precedence rules this implements. `review_required` and
    `severity_score` are only consulted for inspections without a ground-truth status;
    review_required None ("not computed") is treated like False."""

    # AI-only inspections (uploads) have their own precedence - see _assess_ai_only.
    if status not in _GROUND_TRUTH_STATUSES:
        return _assess_ai_only(ai_prediction, severity_level, severity_score, review_required)

    has_ground_truth = status in _GROUND_TRUTH_STATUSES
    ai_available = ai_prediction in (GOOD_PREDICTION, DEFECTIVE_PREDICTION)
    conflicting = has_ground_truth and ai_available and status != ai_prediction

    # Rule 1: a known-severe defect wins, unless the ground truth contradicts the AI it came from.
    if severity_level in (CRITICAL, HIGH) and not conflicting:
        return QualityAssessment(
            decision=FAIL,
            assessment="High-severity defect evidence requires review.",
            recommendation=_combine(_ESCALATE_RECOMMENDATION, *_defect_category_recommendations(defect_category)),
        )

    # Rule 2: ground truth and AI evidence both present but disagree - never silently
    # resolved either way.
    if conflicting:
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
    if has_ground_truth and status == GOOD_PREDICTION:
        return QualityAssessment(
            decision=PASS,
            assessment="No significant defect evidence identified.",
            recommendation=_PASS_RECOMMENDATION,
        )

    # Not reachable through the rules above (every status/ai_prediction combination is
    # decided); kept as the safe default.
    return _insufficient_evidence()


def _severity_note(severity_level: Optional[str], severity_score: Optional[float]) -> str:
    if not severity_level:
        return ""
    if severity_score is None:
        return f"Severity: {severity_level}."
    return f"Severity: {severity_level} ({round(severity_score)}/100)."


def _assess_ai_only(
    ai_prediction: Optional[str],
    severity_level: Optional[str],
    severity_score: Optional[float],
    review_required: Optional[bool],
) -> QualityAssessment:
    """Rules 5-8 for inspections without ground truth (uploads), in this order:
    no AI result -> NOT_ASSESSED; review_required -> MANUAL_REVIEW (BEFORE any severity-driven FAIL, so a
    defective result from a not-production-ready model is never auto-failed); AI defective (any severity,
    Critical/High included) -> FAIL; AI good -> PASS. Severity is quoted in the assessment when present."""
    if ai_prediction not in (GOOD_PREDICTION, DEFECTIVE_PREDICTION):
        return _insufficient_evidence()
    note = _severity_note(severity_level, severity_score)
    if review_required:
        return QualityAssessment(
            decision=MANUAL_REVIEW,
            assessment=_combine(_MANUAL_REVIEW_ASSESSMENT, note),
            recommendation=_MANUAL_REVIEW_RECOMMENDATION,
        )
    if ai_prediction == DEFECTIVE_PREDICTION:
        severe = severity_level in (CRITICAL, HIGH)
        action = recommended_action_for_level(severity_level)
        return QualityAssessment(
            decision=FAIL,
            assessment=_combine(
                "High-severity defect evidence requires review." if severe else "Defective product requires review.",
                note,
            ),
            recommendation=_combine(f"{action}." if action else "", _AI_REVIEW_RECOMMENDATION),
        )
    return QualityAssessment(
        decision=PASS,
        assessment="No significant defect evidence identified.",
        recommendation=_PASS_RECOMMENDATION,
    )


def _insufficient_evidence() -> QualityAssessment:
    return QualityAssessment(
        decision=NOT_ASSESSED,
        assessment="Insufficient evidence.",
        recommendation=_INSUFFICIENT_EVIDENCE_RECOMMENDATION,
    )
