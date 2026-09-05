import { apiFetch } from "./client";

export function getInspectionAnalyticsSummary() {
  return apiFetch("/inspections/analytics/summary");
}
