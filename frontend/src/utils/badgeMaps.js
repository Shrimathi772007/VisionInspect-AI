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
