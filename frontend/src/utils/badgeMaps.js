export const ROLE_LABELS = {
  quality_engineer: "Quality Engineer",
  factory_supervisor: "Factory Supervisor",
};

export const ROLE_TONES = {
  quality_engineer: "accent",
  factory_supervisor: "info",
};

export const STATUS_LABELS = {
  pending: "Pending",
  good: "Good",
  defective: "Defective",
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

export function sourceLabel(source) {
  return SOURCE_LABELS[source] || source;
}

export function sourceTone(source) {
  return SOURCE_TONES[source] || "neutral";
}

// AI prediction (Phase 8) - intentionally NOT success/danger like STATUS_TONES.
// ai_prediction is a model guess, independent of and never reconciled with the
// ground-truth `status`; reusing green/red here would visually imply the two
// values agree (Phase 5: this model only has ~46% recall on real defects).
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
