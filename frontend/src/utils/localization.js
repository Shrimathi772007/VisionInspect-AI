// Anomaly-based defect localization (backend localization method "anomaly_map_threshold_v1"):
// regions are derived from the model's anomaly heatmap - this is not an object-detection model.
//
// Backend shape (app.ai.inference.localization + app.inspections.service), all coordinates
// normalised 0..1 from the top-left of the ORIGINAL image:
//   boxes:           [{ x, y, w, h, peak, area_fraction }]  (at most 5, highest peak first)
//   area_pct:        percentage of the image covered by the kept regions
//   centroid:        [cx, cy] of the largest region, or null
//   mask_rule:       "threshold" | "relative_peak" | null
//   analysed_region: [x0, y0, x1, y1] - the part of the image the model saw ([0, 0, 1, 1] unless
//                    the model centre-crops, e.g. crop224 sees the central 87.5% of a square image)

export const LOCALIZATION_NOTE =
  "Anomaly-based: regions are derived from the model's anomaly heatmap. Not an object-detection model.";

// Heatmap overlay opacity: a simple 3-step control.
export const HEATMAP_OPACITY_STEPS = [
  { label: "Low", value: 0.35 },
  { label: "Medium", value: 0.65 },
  { label: "High", value: 1 },
];

const FULL_REGION = [0, 0, 1, 1];
const EPSILON = 1e-4;

function toPercent(value) {
  return `${Number((value * 100).toFixed(4))}%`;
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/** Valid boxes only (anything malformed is skipped rather than drawn in the wrong place). */
export function localizationBoxes(localization) {
  const boxes = Array.isArray(localization?.boxes) ? localization.boxes : [];
  return boxes.filter((box) => ["x", "y", "w", "h"].every((key) => isFiniteNumber(box?.[key])));
}

/** Absolute-position style of one box, as percentages of the displayed image. */
export function boxStyle(box) {
  return { left: toPercent(box.x), top: toPercent(box.y), width: toPercent(box.w), height: toPercent(box.h) };
}

/** [x0, y0, x1, y1] the model analysed; the whole image when missing or malformed. */
export function analysedRegion(localization) {
  const region = localization?.analysed_region;
  if (!Array.isArray(region) || region.length !== 4 || !region.every(isFiniteNumber)) return FULL_REGION;
  return region;
}

/** True when part of the image (its edges) was outside the region the model analysed. */
export function hasUnanalysedEdges(localization) {
  return analysedRegion(localization).some((value, index) => Math.abs(value - FULL_REGION[index]) > EPSILON);
}

export function regionStyle(region) {
  const [x0, y0, x1, y1] = region;
  return { left: toPercent(x0), top: toPercent(y0), width: toPercent(x1 - x0), height: toPercent(y1 - y0) };
}

export function formatAreaPct(areaPct) {
  return isFiniteNumber(areaPct) ? `${areaPct.toFixed(1)}%` : null;
}

/** "upper left", "centre", "lower right", ... for a normalised point. */
export function describePosition(cx, cy) {
  const vertical = cy < 1 / 3 ? "upper" : cy > 2 / 3 ? "lower" : "middle";
  const horizontal = cx < 1 / 3 ? "left" : cx > 2 / 3 ? "right" : "centre";
  if (vertical === "middle" && horizontal === "centre") return "centre";
  if (vertical === "middle") return `middle ${horizontal}`;
  return `${vertical} ${horizontal === "centre" ? "centre" : horizontal}`;
}

export function describeCentroid(centroid) {
  if (!Array.isArray(centroid) || centroid.length !== 2 || !centroid.every(isFiniteNumber)) return null;
  const [cx, cy] = centroid;
  return `Centred in the ${describePosition(cx, cy)} of the image (${Math.round(cx * 100)}% from the left, ${Math.round(cy * 100)}% from the top)`;
}
