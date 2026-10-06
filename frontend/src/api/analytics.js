import { apiFetch } from "./client";

/**
 * @param {{ days?: number }} [options] `days` selects the trailing window for the
 *   time-windowed sections (7, 14 or 30 - the backend rejects anything else). Omitted, the
 *   backend uses its default (14), exactly as before windows were selectable.
 */
export function getInspectionAnalyticsSummary({ days } = {}) {
  const query = days ? `?days=${encodeURIComponent(days)}` : "";
  return apiFetch(`/inspections/analytics/summary${query}`);
}

/**
 * GET /inspections/analytics/by-category - per-category counts for all 15 MVTec categories over the
 * trailing `days` window (7, 14 or 30; omitted, the backend default of 14). defect_rate is derived from
 * AI predictions, not ground truth.
 * @param {{ days?: number }} [options]
 */
export function getCategoryAnalytics({ days } = {}) {
  const query = days ? `?days=${encodeURIComponent(days)}` : "";
  return apiFetch(`/inspections/analytics/by-category${query}`);
}
