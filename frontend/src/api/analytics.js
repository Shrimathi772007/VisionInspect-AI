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
