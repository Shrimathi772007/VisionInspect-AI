"""Milestone 3 Phase 2: severity scoring and quality risk assessment.

Implements the project specification's four-factor weighted severity score:

    Defect Size            30%
    Defect Location         25%
    Defect Type              25%
    Detection Confidence    20%

    severity_score = size_score + location_score + defect_type_score + confidence_score
    (each factor pre-scaled to its own weight, so the sum is already out of 100)

Kept deliberately framework/DB-free (same convention as app.ai.inference) - this module
only ever computes from plain values and never touches the database or an ORM object, so
its logic is trivially unit-testable. app.inspections.service wires its output onto a
persisted Inspection.

PHASE 2 CORRECTION - WHERE EACH FACTOR NOW STANDS
----------------------------------------------------
A follow-up review inspected the actual MVTec AD dataset (dataset/bottle/ground_truth/)
rather than assuming no real evidence existed. Findings (verified against every real mask
file currently in the dataset, not just spot-checked):

- Defect Size: MVTec DOES ship a real, per-image ground-truth segmentation mask for every
  non-"good" test image (dataset/<category>/ground_truth/<defect_type>/<stem>_mask.png -
  binary 0/255, same pixel dimensions as the source image, one mask per defective image,
  1:1 with every defective Bottle test image with no exceptions). That is real, measured
  evidence of defect size, not fabrication. app.dataset.ground_truth reads it and computes
  a real defect_area_ratio (defect pixels / total pixels); resolve_size_score below maps
  that ratio to a 0-30 score using an explicit, documented, *implementation* threshold
  (SIZE_SATURATION_RATIO) - see that function's docstring. "good" images have no mask
  because there is no defect to segment; that absence is itself real ground truth (area
  ratio = 0.0), not missing evidence - see app.dataset.ground_truth.compute_defect_area_ratio.
- Defect Location: the same masks could, in principle, yield a centroid or bounding box
  (real geometric data). However, the project specification says only that Location
  contributes 25% - it defines no rule for turning a position into a severity score, and
  there is no defensible domain basis here for "center is worse" or "edge is worse"
  (unlike, say, a seal or thread region on a specific part, which would need real domain
  input this project does not have). Inventing such a rule just to populate the field
  would be exactly the fabrication this phase must avoid. Defect Location therefore stays
  unavailable (None), by explicit decision, not oversight.
- Detection Confidence: unchanged. The autoencoder's reconstruction_error/threshold (see
  app.ai.inference) are real, measured values, but they are NOT a calibrated
  classification confidence. Converting them into a fake confidence percentage is exactly
  the kind of fabrication this phase must avoid, so this factor is never derived from them.

Defect Type continues to use Inspection.defect_category (Phase 1 ground truth for MVTec
imports) via a documented domain-assumption mapping - see _DEFECT_TYPE_SCORES below for
why it is labeled that way rather than as a specification-defined value.

POLICY: a severity score is produced ONLY when all four factors are available
-------------------------------------------------------------------------------
Scoring on partial evidence (e.g. only Defect Type, or Defect Type + Defect Size) would
silently understate severity for every inspection with a real defect - even with Size now
available, a broken_large bottle would cap out at 55/100 ("Medium" at best) purely because
Location and Confidence are structurally missing, which is factually misleading. Reporting
"not assessed" until every factor has a real value remains the honest choice. Concretely,
this means severity_score is still NULL for every inspection today (Location and
Confidence are unavailable for all of them) - that is the correct, non-fabricated answer
given what evidence this project actually has, not a bug. As real evidence sources are
added in later phases (a defensible location rule, a calibrated classifier), inspections
will start receiving real scores automatically, with zero changes to the scoring engine.
"""

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Specification weights
# ---------------------------------------------------------------------------

SIZE_WEIGHT = 30.0
LOCATION_WEIGHT = 25.0
DEFECT_TYPE_WEIGHT = 25.0
CONFIDENCE_WEIGHT = 20.0
MAX_SEVERITY_SCORE = SIZE_WEIGHT + LOCATION_WEIGHT + DEFECT_TYPE_WEIGHT + CONFIDENCE_WEIGHT  # 100.0

# ---------------------------------------------------------------------------
# Severity levels (score -> level) and quality risk (level -> risk)
# ---------------------------------------------------------------------------

CRITICAL = "Critical"
HIGH = "High"
MEDIUM = "Medium"
LOW = "Low"

NOT_ASSESSED = "Not assessed"

_RISK_BY_LEVEL = {
    CRITICAL: "Critical Risk",
    HIGH: "High Risk",
    MEDIUM: "Medium Risk",
    LOW: "Low Risk",
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def calculate_severity_score(
    size_score: Optional[float],
    location_score: Optional[float],
    defect_type_score: Optional[float],
    confidence_score: Optional[float],
) -> Optional[float]:
    """Sum the four weighted factor scores into a single 0-100 severity score.

    Returns None - never a fabricated number - if any factor is unavailable (None), per
    this module's all-four-required policy (see module docstring).

    Each factor is clamped to its own [0, weight] range before summing (so a caller
    passing an out-of-range value cannot push the total outside [0, 100]), and the final
    total is clamped to [0, 100] as a second safeguard.
    """
    if size_score is None or location_score is None or defect_type_score is None or confidence_score is None:
        return None

    total = (
        _clamp(size_score, 0.0, SIZE_WEIGHT)
        + _clamp(location_score, 0.0, LOCATION_WEIGHT)
        + _clamp(defect_type_score, 0.0, DEFECT_TYPE_WEIGHT)
        + _clamp(confidence_score, 0.0, CONFIDENCE_WEIGHT)
    )
    return _clamp(total, 0.0, MAX_SEVERITY_SCORE)


def severity_level_for_score(score: Optional[float]) -> Optional[str]:
    """Map a 0-100 severity score to Critical / High / Medium / Low.

    Boundaries per the project specification:
        80-100 -> Critical
        60-79  -> High
        40-59  -> Medium
        0-39   -> Low
    None in, None out - there is no "unassessed" severity level, only a missing one.
    """
    if score is None:
        return None
    if score >= 80:
        return CRITICAL
    if score >= 60:
        return HIGH
    if score >= 40:
        return MEDIUM
    return LOW


def quality_risk_for_level(level: Optional[str]) -> str:
    """Map a severity level to its quality risk label, per the specification's mapping.

    Returns "Not assessed" (never a guessed risk level) when `level` is None.
    """
    if level is None:
        return NOT_ASSESSED
    return _RISK_BY_LEVEL.get(level, NOT_ASSESSED)


# ---------------------------------------------------------------------------
# Factor resolution - real evidence only, no fabricated defaults
# ---------------------------------------------------------------------------

# Defect Type scores (out of DEFECT_TYPE_WEIGHT), keyed by Inspection.defect_category
# (Phase 1 MVTec ground truth). IMPORTANT: these numbers are NOT defined by the project
# specification PDF - the spec only says Defect Type contributes 25% of the total, it does
# not say how to rank defect *categories* against each other. This is a documented,
# deterministic implementation/domain assumption (option A of the Phase 2 correction
# review, not option B or C): "good" carries no defect (0); among the three real Bottle
# defect types, broken_large is treated as the most severe structural failure (max
# weight), broken_small a smaller instance of the same structural failure mode, and
# contamination as typically a surface/cosmetic defect rather than a structural one. This
# ranking was kept rather than replaced because (a) it is at least as defensible as any
# other static ranking without real severity-outcome data to calibrate against, and (b) it
# is now a genuinely different signal from Defect Size (which measures THIS instance's
# actual mask area, computed below) rather than a stand-in for it - the two factors are no
# longer redundant. It should be revisited if real severity-outcome data ever becomes
# available. Any defect_category not listed here (including future non-Bottle MVTec
# categories) is intentionally left unavailable (None) rather than guessed - see
# resolve_defect_type_score.
_DEFECT_TYPE_SCORES = {
    "good": 0.0,
    "broken_large": DEFECT_TYPE_WEIGHT,
    "broken_small": 15.0,
    "contamination": 10.0,
}


def resolve_defect_type_score(defect_category: Optional[str]) -> Optional[float]:
    """Defect Type factor score from ground-truth defect_category, or None if unknown."""
    if defect_category is None:
        return None
    return _DEFECT_TYPE_SCORES.get(defect_category)


# Defect Size saturation point: the real defect_area_ratio (see app.dataset.ground_truth)
# at or above which the Defect Size factor is treated as maximal (full SIZE_WEIGHT). This
# is NOT a project-specification value - the spec defines no area-ratio-to-score formula
# at all. 0.30 (30% of image area) is an explicit, documented, *implementation* threshold,
# chosen as a round number comfortably above the largest ratio actually observed across
# every real ground-truth mask in the current Bottle dataset (broken_large: up to ~0.274 -
# verified by inspecting all 63 Bottle ground-truth masks, see the Phase 2 correction
# report), so the mapping stays monotonic and does not saturate prematurely for real data.
# Kept as a module-level constant specifically so it can be reconsidered later without
# touching the scoring logic around it.
SIZE_SATURATION_RATIO = 0.30


def resolve_size_score(defect_area_ratio: Optional[float]) -> Optional[float]:
    """Defect Size factor score from a real MVTec ground-truth mask area ratio.

    `defect_area_ratio` must come from app.dataset.ground_truth.compute_defect_area_ratio
    (defect pixels / total pixels in the actual ground-truth mask for this inspection's
    image) - real, measured evidence, never a guess. None in (no mask available - e.g. a
    generic upload, or an MVTec image/category with no matching ground-truth file) means
    None out: this factor is unavailable, not zero.

    The ratio is linearly scaled to [0, SIZE_WEIGHT], saturating at SIZE_SATURATION_RATIO
    (see that constant's docstring for why 0.30 was chosen and that it is an
    implementation threshold, not a specification value).
    """
    if defect_area_ratio is None:
        return None
    clamped_ratio = _clamp(defect_area_ratio, 0.0, SIZE_SATURATION_RATIO)
    return (clamped_ratio / SIZE_SATURATION_RATIO) * SIZE_WEIGHT


def resolve_location_score() -> Optional[float]:
    """Defect Location factor - unavailable by explicit decision, not oversight.

    MVTec's ground-truth masks could yield a real centroid or bounding box, but the
    project specification defines no rule for converting a position into a severity
    score, and there is no defensible domain basis in this project for a rule like
    "center is worse" or "edge is worse". Inventing one just to populate this field would
    be fabricated severity logic, which this phase must avoid - see the module docstring's
    Phase 2 correction notes. Deliberately takes no arguments: there is currently no input
    this function could honestly use.
    """
    return None


def resolve_confidence_score() -> Optional[float]:
    """Detection Confidence factor - always unavailable today (see module docstring).

    Deliberately takes no arguments: it must never be derived from
    ai_reconstruction_error/ai_threshold, which are not a calibrated confidence.
    """
    return None


@dataclass
class SeverityAssessment:
    """The persisted result of a severity assessment for one inspection."""

    score: Optional[float]
    level: Optional[str]
    quality_risk: str


def assess_severity(
    defect_category: Optional[str],
    defect_area_ratio: Optional[float] = None,
) -> SeverityAssessment:
    """Assess severity from the evidence currently available for an inspection.

    Takes plain values rather than an Inspection ORM object to stay framework/DB-free -
    `defect_area_ratio` must already be computed by the caller (see
    app.dataset.ground_truth.compute_defect_area_ratio), never derived here from a path.

    Defect Type is resolved from defect_category and Defect Size from defect_area_ratio
    when given; Defect Location and Detection Confidence are always None (see their
    resolver docstrings for why). Since this all-four-required scorer needs every factor,
    it currently always returns score=None, level=None, quality_risk="Not assessed" for
    every real inspection - honestly, not by omission (see the module docstring).
    """
    score = calculate_severity_score(
        size_score=resolve_size_score(defect_area_ratio),
        location_score=resolve_location_score(),
        defect_type_score=resolve_defect_type_score(defect_category),
        confidence_score=resolve_confidence_score(),
    )
    level = severity_level_for_score(score)
    return SeverityAssessment(score=score, level=level, quality_risk=quality_risk_for_level(level))
