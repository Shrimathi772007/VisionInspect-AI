import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./client", () => ({ apiFetch: vi.fn() }));

import { apiFetch } from "./client";
import { getCategoryAnalytics, getInspectionAnalyticsSummary } from "./analytics";

describe("analytics API", () => {
  beforeEach(() => {
    apiFetch.mockReset().mockResolvedValue({});
  });

  it("asks for the by-category figures of a window", () => {
    getCategoryAnalytics({ days: 7 });
    expect(apiFetch).toHaveBeenLastCalledWith("/inspections/analytics/by-category?days=7");
  });

  it("leaves the window to the backend default when none is given", () => {
    getCategoryAnalytics();
    expect(apiFetch).toHaveBeenLastCalledWith("/inspections/analytics/by-category");
  });

  it("keeps the summary request as it was", () => {
    getInspectionAnalyticsSummary({ days: 30 });
    expect(apiFetch).toHaveBeenLastCalledWith("/inspections/analytics/summary?days=30");
  });
});
