import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../auth/useAuth", () => ({
  useAuth: () => ({ user: { name: "Ada Lovelace", role: "quality_engineer" } }),
}));
vi.mock("../hooks/useProducts", () => ({ useProducts: vi.fn() }));
vi.mock("../hooks/useInspections", () => ({ useInspections: vi.fn() }));
vi.mock("../hooks/useAnalyticsSummary", () => ({ useAnalyticsSummary: vi.fn() }));

import { useProducts } from "../hooks/useProducts";
import { useInspections } from "../hooks/useInspections";
import { useAnalyticsSummary } from "../hooks/useAnalyticsSummary";
import { DashboardPage } from "./DashboardPage";

const PRODUCT = { id: 1, product_name: "Bottle A", product_code: "BOT-1" };

const stats = (overrides = {}) => ({ count: 0, avg_ms: null, median_ms: null, min_ms: null, max_ms: null, ...overrides });

// 14 days, oldest -> newest. The first 7 ("last week") total 10 inspections; the newest 7 total 15.
function makeActivity() {
  return Array.from({ length: 14 }, (_, index) => {
    const total = index === 0 ? 10 : index === 13 ? 15 : 0;
    return {
      date: `2026-09-${String(index + 1).padStart(2, "0")}`,
      total,
      good: total,
      defective: 0,
      pending: 0,
    };
  });
}

function makeAnalytics(overrides = {}) {
  const daily = makeActivity().map((day) => ({
    ...day,
    ai_analyzed: 0,
    ai_defective: 0,
    quality_pass: day.total,
    quality_fail: 0,
    quality_not_assessed: 0,
  }));
  return {
    total_inspections: 120,
    by_status: { pending: 5, good: 60, defective: 55 },
    by_source: { upload: 20, mvtec_ad: 100 },
    ai_analyzed_count: 80,
    ai_prediction_counts: { good: 50, defective: 30 },
    ai_defect_rate: 0.375,
    avg_reconstruction_error: 0.003,
    activity_by_day: makeActivity(),
    by_product: [
      {
        product_id: 1,
        product_name: "Bottle A",
        total: 100,
        ai_defective: 30,
        defective: 55,
        defect_rate: 0.55,
        recent_defect_rate: 0.5,
        recent_trend: "up",
      },
    ],
    defect_categories: [{ category: "broken_large", count: 20, percentage: 16.7 }],
    quality_decisions: [{ decision: "PASS", count: 60, percentage: 50 }],
    severity_distribution: [{ level: "Not assessed", count: 120, percentage: 100 }],
    operational_insights: [{ type: "highest_volume_product", message: "Highest inspection volume: Bottle A." }],
    trend_monitoring: {
      period_days: 14,
      daily,
      category_trends: [],
      insights: [{ type: "insufficient_trend_history", message: "Insufficient historical data yet." }],
    },
    performance: {
      window_days: 14,
      inspections_in_window: 100,
      ai_analyzed_in_window: 60,
      ai_analyzed_rate: 0.6,
      processing_time: stats({ count: 40, avg_ms: 1240, median_ms: 1100, min_ms: 300, max_ms: 4200 }),
      ai_inference_time: stats({ count: 30, avg_ms: 9800, median_ms: 9500, min_ms: 8000, max_ms: 12000 }),
    },
    ...overrides,
  };
}

const RECENT = [
  { id: 7, product_id: 1, status: "good", created_at: new Date().toISOString() },
  { id: 6, product_id: 1, status: "defective", created_at: new Date().toISOString() },
];

let refetchAnalytics;
let refetchRecent;

function mockData({ products, recent, analytics } = {}) {
  useProducts.mockReturnValue({
    products: [PRODUCT],
    isLoading: false,
    error: null,
    getProductById: (id) => (id === 1 ? PRODUCT : undefined),
    ...products,
  });
  useInspections.mockReturnValue({
    inspections: RECENT,
    isLoading: false,
    error: null,
    refetch: refetchRecent,
    ...recent,
  });
  useAnalyticsSummary.mockReturnValue({
    data: makeAnalytics(),
    isLoading: false,
    isRefreshing: false,
    error: null,
    refetch: refetchAnalytics,
    ...analytics,
  });
}

function renderDashboard() {
  return render(
    <MemoryRouter>
      <DashboardPage />
    </MemoryRouter>
  );
}

// A stat card: the label's parent holds that card's value and note.
const card = (label) => within(screen.getByText(label, { selector: "p" }).parentElement);

describe("DashboardPage", () => {
  beforeEach(() => {
    refetchAnalytics = vi.fn();
    refetchRecent = vi.fn();
    useProducts.mockReset();
    useInspections.mockReset();
    useAnalyticsSummary.mockReset();
  });

  describe("data loading", () => {
    it("asks for only the newest few inspections instead of downloading every one", () => {
      mockData();
      renderDashboard();

      expect(useInspections).toHaveBeenCalledWith({ limit: 5 });
    });

    it("starts on the default 14-day window and asks for a different one when the range is changed", () => {
      mockData();
      renderDashboard();

      expect(useAnalyticsSummary).toHaveBeenLastCalledWith({ days: 14 });
      expect(screen.getByRole("button", { name: "14 days" })).toHaveAttribute("aria-pressed", "true");

      fireEvent.click(screen.getByRole("button", { name: "30 days" }));

      expect(useAnalyticsSummary).toHaveBeenLastCalledWith({ days: 30 });
      expect(screen.getByRole("button", { name: "30 days" })).toHaveAttribute("aria-pressed", "true");
    });
  });

  describe("rendering with data", () => {
    it("shows the aggregated counts with their scope, taken from the analytics summary", () => {
      mockData();
      renderDashboard();

      expect(card("Inspections").getByText("120")).toBeInTheDocument();
      expect(card("Pending review").getByText("5")).toBeInTheDocument();
      expect(card("AI analyzed").getByText("80")).toBeInTheDocument();
      expect(card("AI defect rate").getByText("37.5%")).toBeInTheDocument();
      expect(card("AI defect rate").getByText("Of AI-analyzed inspections")).toBeInTheDocument();
    });

    it("computes the week-over-week chip from the backend's per-day series", () => {
      mockData();
      renderDashboard();

      // previous 7 days = 10 inspections, newest 7 days = 15  ->  +50%
      expect(card("Inspections").getByText("+50% vs last week")).toBeInTheDocument();
    });

    it("shows the performance section with real figures", () => {
      mockData();
      renderDashboard();

      expect(screen.getByRole("heading", { name: "Inspection performance" })).toBeInTheDocument();
      expect(card("Avg processing time").getByText("1.2 s")).toBeInTheDocument();
      expect(card("Timing coverage").getByText("40 / 100")).toBeInTheDocument();
    });

    it("lists the recent inspections by product name", () => {
      mockData();
      renderDashboard();

      expect(screen.getAllByText("Bottle A").length).toBeGreaterThan(0);
      expect(screen.getAllByText(/BOT-1/)).toHaveLength(RECENT.length);
    });

    it("labels the sections with the window the data actually covers", () => {
      mockData();
      renderDashboard();

      expect(screen.getByText(/Last 14 days, compared with the previous 14-day period/)).toBeInTheDocument();
    });
  });

  describe("loading state", () => {
    it("shows no numbers and no error while the analytics are loading", () => {
      mockData({ analytics: { data: null, isLoading: true } });
      renderDashboard();

      expect(screen.queryByText("120")).not.toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.queryByText("Unavailable")).not.toBeInTheDocument();
    });
  });

  describe("when analytics fail", () => {
    beforeEach(() => {
      mockData({ analytics: { data: null, error: new Error("Analytics exploded") } });
    });

    it("shows ONE alert with the real reason and a working Retry", () => {
      renderDashboard();

      const alerts = screen.getAllByRole("alert");
      expect(alerts).toHaveLength(1);
      expect(alerts[0]).toHaveTextContent("Analytics couldn't be loaded");
      expect(alerts[0]).toHaveTextContent("Analytics exploded");

      fireEvent.click(within(alerts[0]).getByRole("button", { name: "Retry" }));
      expect(refetchAnalytics).toHaveBeenCalledTimes(1);
    });

    it("never presents a failure as zero", () => {
      renderDashboard();

      const inspections = card("Inspections");
      expect(inspections.getByText("—")).toBeInTheDocument();
      expect(inspections.getByText("Unavailable")).toBeInTheDocument();
      expect(inspections.queryByText("0")).not.toBeInTheDocument();
      expect(card("Pending review").getByText("Unavailable")).toBeInTheDocument();
    });

    it("marks every analytics-backed section as unavailable", () => {
      renderDashboard();

      expect(screen.getAllByText("Unavailable").length).toBeGreaterThanOrEqual(10);
    });

    it("leaves the sections that do not depend on analytics working", () => {
      renderDashboard();

      // Products and Recent inspections come from their own requests and are unaffected.
      expect(card("Products").getByText("1")).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "Recent inspections" })).toBeInTheDocument();
      expect(screen.getAllByText(/BOT-1/)).toHaveLength(RECENT.length);
    });
  });

  describe("when the recent inspections fail", () => {
    it("shows the failure with Retry in that card only - not an empty list", () => {
      mockData({ recent: { inspections: [], error: new Error("Inspections request failed") } });
      renderDashboard();

      const alert = screen.getByRole("alert");
      expect(alert).toHaveTextContent("Inspections request failed");
      expect(screen.queryByText("No inspections yet")).not.toBeInTheDocument();

      fireEvent.click(within(alert).getByRole("button", { name: "Retry" }));
      expect(refetchRecent).toHaveBeenCalledTimes(1);
    });

    it("leaves the analytics untouched", () => {
      mockData({ recent: { inspections: [], error: new Error("Inspections request failed") } });
      renderDashboard();

      expect(card("Inspections").getByText("120")).toBeInTheDocument();
      expect(screen.getAllByRole("alert")).toHaveLength(1);
    });
  });

  describe("when the products fail", () => {
    it("marks only the Products card as unavailable", () => {
      mockData({ products: { products: [], error: new Error("Products request failed") } });
      renderDashboard();

      expect(card("Products").getByText("Unavailable")).toBeInTheDocument();
      expect(card("Inspections").getByText("120")).toBeInTheDocument();
    });
  });

  describe("empty state", () => {
    it("explains each empty section instead of showing bare zeros", () => {
      mockData({
        recent: { inspections: [] },
        analytics: {
          data: makeAnalytics({
            total_inspections: 0,
            ai_analyzed_count: 0,
            ai_prediction_counts: { good: 0, defective: 0 },
            ai_defect_rate: null,
            by_status: { pending: 0, good: 0, defective: 0 },
            by_product: [],
            defect_categories: [],
            quality_decisions: [],
            severity_distribution: [],
            operational_insights: [],
            activity_by_day: makeActivity().map((day) => ({ ...day, total: 0, good: 0 })),
            performance: {
              window_days: 14,
              inspections_in_window: 0,
              ai_analyzed_in_window: 0,
              ai_analyzed_rate: null,
              processing_time: stats(),
              ai_inference_time: stats(),
            },
          }),
        },
      });
      renderDashboard();

      expect(screen.getByRole("heading", { name: "No AI predictions yet" })).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "No product data yet" })).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "No inspections yet" })).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "No inspections in this range" })).toBeInTheDocument();
      expect(card("AI defect rate").getByText("No data")).toBeInTheDocument();
    });

    it("shows 'No data' for timings when nothing in the range has been timed", () => {
      mockData({
        analytics: {
          data: makeAnalytics({
            performance: {
              window_days: 14,
              inspections_in_window: 100,
              ai_analyzed_in_window: 60,
              ai_analyzed_rate: 0.6,
              processing_time: stats(),
              ai_inference_time: stats(),
            },
          }),
        },
      });
      renderDashboard();

      expect(card("Avg processing time").getByText("No data")).toBeInTheDocument();
      expect(card("Timing coverage").getByText("0 / 100")).toBeInTheDocument();
    });
  });

  describe("changing the range", () => {
    it("keeps the current data visible, marked as updating, while the new range loads", () => {
      mockData({ analytics: { isRefreshing: true } });
      renderDashboard();

      expect(screen.getByRole("status")).toHaveTextContent("Updating");
      expect(document.querySelector('[aria-busy="true"]')).not.toBeNull();
      expect(card("Inspections").getByText("120")).toBeInTheDocument();
    });

    it("does not show the updating state when nothing is loading", () => {
      mockData();
      renderDashboard();

      expect(screen.queryByText(/Updating/)).not.toBeInTheDocument();
    });
  });
});
