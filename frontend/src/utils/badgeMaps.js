export const ROLE_LABELS = {
  quality_engineer: "Quality Engineer",
  factory_supervisor: "Factory Supervisor",
};

export const ROLE_TONES = {
  quality_engineer: "accent",
  factory_supervisor: "info",
};

// Display text only: the stored/API status value stays "pending" (an upload without a ground-truth label).
export const STATUS_LABELS = {
  pending: "Unlabelled",
  good: "Good",
  defective: "Defective",
};

export const UNLABELLED_STATUS_HINT = "No ground-truth label. The AI prediction is shown separately.";

export const STATUS_HINTS = {
  pending: UNLABELLED_STATUS_HINT,
};

export const STATUS_TONES = {
  pending: "warning",
  good: "success",
  defective: "danger",
};

export const SOURCE_LABELS = {
  upload: "User Upload",
  mvtec_ad: "MVTec AD",
};

export const SOURCE_TONES = {
  upload: "accent",
  mvtec_ad: "info",
};

export function roleLabel(role) {
  return ROLE_LABELS[role] || role;
}

export function roleTone(role) {
  return ROLE_TONES[role] || "neutral";
}

export function statusLabel(status) {
  return STATUS_LABELS[status] || status;
}

export function statusTone(status) {
  return STATUS_TONES[status] || "neutral";
}

/** Tooltip for a status badge, or undefined when the label needs none. */
export function statusHint(status) {
  return STATUS_HINTS[status];
}

export function sourceLabel(source) {
  return SOURCE_LABELS[source] || source;
}

export function sourceTone(source) {
  return SOURCE_TONES[source] || "neutral";
}

// AI prediction (Phase 8) - intentionally NOT success/danger like STATUS_TONES.
// ai_prediction is a model guess, independent of and never reconciled with the
// ground-truth `status`; reusing green/red here would visually imply the two
// values agree (each category's served model has its own measured recall and
// false-positive rate, below perfect - see the Model Performance page).
export const AI_PREDICTION_LABELS = {
  good: "AI: Good",
  defective: "AI: Defective",
};

export const AI_PREDICTION_TONES = {
  good: "info",
  defective: "warning",
};

export function aiPredictionLabel(prediction) {
  return AI_PREDICTION_LABELS[prediction] || prediction;
}

export function aiPredictionTone(prediction) {
  return AI_PREDICTION_TONES[prediction] || "neutral";
}

// Defect category (Milestone 3 Phase 1) - the known defect category/type, e.g. from an
// MVTec import's ground-truth defect_type. Kept visually and conceptually distinct from
// STATUS_LABELS/STATUS_TONES (business status) and AI_PREDICTION_* (anomaly detector
// guess). Only "good" has an explicit label/tone below because the set of MVTec defect
// types varies by category; any other value is titleized from its snake_case name so new
// categories (e.g. "metal_contamination") render sensibly without code changes.
export const DEFECT_CATEGORY_LABELS = {
  good: "Good",
};

function titleizeSnakeCase(value) {
  return value
    .split("_")
    .filter(Boolean)
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

export function defectCategoryLabel(category) {
  if (!category) return "Not categorized";
  return DEFECT_CATEGORY_LABELS[category] || titleizeSnakeCase(category);
}

export function defectCategoryTone(category) {
  if (!category) return "neutral";
  return category === "good" ? "success" : "danger";
}

// Severity / quality risk (Milestone 3 Phase 2) - from the backend severity scoring
// engine (app.inspections.severity). Deliberately a distinct tone set/badge from
// AI_PREDICTION_* and STATUS_*: severity is a derived risk assessment, not the AI's raw
// guess or the business status, and must never look like either. `null`/"Not assessed"
// means current evidence is insufficient to score - never render a fabricated value.
export const SEVERITY_LEVEL_TONES = {
  Critical: "danger",
  High: "warning",
  Medium: "info",
  Low: "success",
};

export const QUALITY_RISK_TONES = {
  "Critical Risk": "danger",
  "High Risk": "warning",
  "Medium Risk": "info",
  "Low Risk": "success",
};

export function severityLabel(level) {
  return level || "Not assessed";
}

export function severityTone(level) {
  return SEVERITY_LEVEL_TONES[level] || "neutral";
}

export function qualityRiskLabel(risk) {
  return risk || "Not assessed";
}

export function qualityRiskTone(risk) {
  return QUALITY_RISK_TONES[risk] || "neutral";
}

// Quality decision (Milestone 3 Phase 3) - from the backend quality assessment engine
// (app.inspections.quality). A fourth, independent conclusion - deliberately its own tone
// set, not reused from AI_PREDICTION_*, STATUS_*, or SEVERITY_LEVEL_TONES, so it never
// visually implies it is a copy of any of those.
// MANUAL_REVIEW: an AI-only result the declared manual-review rule flagged (low confidence and/or
// a category model that is not production ready) - its own tone, distinct from all three others.
export const QUALITY_DECISION_LABELS = {
  PASS: "PASS",
  FAIL: "FAIL",
  NOT_ASSESSED: "NOT ASSESSED",
  MANUAL_REVIEW: "MANUAL REVIEW",
};

export const QUALITY_DECISION_TONES = {
  PASS: "success",
  FAIL: "danger",
  NOT_ASSESSED: "warning",
  MANUAL_REVIEW: "accent",
};

export function qualityDecisionLabel(decision) {
  if (!decision) return "NOT ASSESSED";
  return QUALITY_DECISION_LABELS[decision] || decision;
}

export function qualityDecisionTone(decision) {
  return QUALITY_DECISION_TONES[decision] || "neutral";
}

// AI confidence (ai_confidence) - a MARGIN-BASED HEURISTIC (how far the anomaly score is from the
// decision threshold), not a calibrated probability. ai_reliability is the backend's display band.
export const CONFIDENCE_NOTE =
  "Confidence is a margin-based heuristic from the distance between the anomaly score and the decision threshold. It is not a calibrated probability.";

export const RELIABILITY_LABELS = {
  high: "High reliability",
  medium: "Medium reliability",
  low: "Low reliability",
};

export const RELIABILITY_TONES = {
  high: "success",
  medium: "info",
  low: "warning",
};

export function formatConfidence(confidence) {
  return Number.isFinite(confidence) ? `${(confidence * 100).toFixed(1)}%` : null;
}

export function reliabilityLabel(reliability) {
  if (!reliability) return "Not available";
  return RELIABILITY_LABELS[reliability] || reliability;
}

export function reliabilityTone(reliability) {
  return RELIABILITY_TONES[reliability] || "neutral";
}

// Model gate (model_gate) - the static evidence gate of the category's served model, from the API.
// Which categories are strong or weak is never hard-coded here: only the gate values are mapped.
export const MODEL_GATE_LABELS = {
  EXCELLENT: "Production ready (Excellent)",
  GOOD: "Good",
  ACCEPTABLE: "Acceptable",
  NOT_PRODUCTION_READY: "Not production ready",
};

export const MODEL_GATE_TONES = {
  EXCELLENT: "success",
  GOOD: "info",
  ACCEPTABLE: "warning",
  NOT_PRODUCTION_READY: "danger",
};

export function modelGateLabel(gate) {
  if (!gate) return "Not available";
  return MODEL_GATE_LABELS[gate] || gate;
}

export function modelGateTone(gate) {
  return MODEL_GATE_TONES[gate] || "neutral";
}

// review_reason is "low confidence", "category model not production ready", or both joined by "; ".
export const REVIEW_REASON_LABELS = {
  "low confidence": "Low confidence",
  "category model not production ready": "Category model not production ready",
};

export function reviewReasonLabels(reason) {
  if (typeof reason !== "string" || !reason.trim()) return [];
  return reason
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => REVIEW_REASON_LABELS[part] || part[0].toUpperCase() + part.slice(1));
}

// Production quality report completeness (Milestone 3 Phase 4 `report_summary.report_status`).
// This is whether the quality assessment behind the report exists - NOT a product quality
// outcome - so it deliberately uses its own vocabulary and tones, never PASS/FAIL's.
export const REPORT_STATUS_LABELS = {
  COMPLETE: "Complete",
  PARTIAL: "Partial",
};

export const REPORT_STATUS_TONES = {
  COMPLETE: "success",
  PARTIAL: "warning",
};

export function reportStatusLabel(status) {
  return REPORT_STATUS_LABELS[status] || status || "Unknown";
}

export function reportStatusTone(status) {
  return REPORT_STATUS_TONES[status] || "neutral";
}

// Dashboard outcome vocabulary - one colour per meaning across the dashboard's charts and badges:
// Pass / Good green, Fail / Defective red, Manual review amber, Not assessed / Unlabelled grey. Dashboard-only
// on purpose: the maps above keep their own tones on the other pages.
export const OUTCOME_TONES = {
  PASS: "success",
  FAIL: "danger",
  MANUAL_REVIEW: "warning",
  NOT_ASSESSED: "neutral",
};

export const PREDICTION_OUTCOME_TONES = {
  good: "success",
  defective: "danger",
  pending: "neutral",
};

// "MANUAL REVIEW" -> "Manual review": the readable form of QUALITY_DECISION_LABELS. An unknown value is
// shown as sent (never hidden), a missing one as "Not assessed".
export function qualityDecisionDisplayLabel(decision) {
  if (!decision) return "Not assessed";
  const label = QUALITY_DECISION_LABELS[decision];
  if (!label) return String(decision);
  return label[0] + label.slice(1).toLowerCase();
}

export function outcomeTone(decision) {
  if (!decision) return OUTCOME_TONES.NOT_ASSESSED;
  return OUTCOME_TONES[decision] || "neutral";
}

export function predictionOutcomeTone(value) {
  return PREDICTION_OUTCOME_TONES[value] || "neutral";
}

// Short explanations shown as info tooltips on the dashboard.
export const AUTOMATION_RATE_INFO = "Share of inspections that received an automatic Pass or Fail decision.";
export const MANUAL_REVIEW_INFO = "The AI result is low reliability, so a person must check it.";

// By-category analytics (GET /inspections/analytics/by-category): defect_rate is derived from AI predictions.
export const DEFECT_RATE_NOTE =
  "Defect rate is the share of AI-analysed inspections predicted defective. It is based on AI predictions, not ground truth.";

// A 0..1 rate as a percentage with one decimal; an em dash when there is no rate (null or not a number).
export function formatRate(rate) {
  return typeof rate === "number" && Number.isFinite(rate) ? `${(rate * 100).toFixed(1)}%` : "—";
}
