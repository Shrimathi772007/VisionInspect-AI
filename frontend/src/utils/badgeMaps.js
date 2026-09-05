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
