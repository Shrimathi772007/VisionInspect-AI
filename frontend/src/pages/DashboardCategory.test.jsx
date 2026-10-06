import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../auth/useAuth", () => ({
  useAuth: () => ({ user: { name: "Ada Lovelace", role: "quality_engineer" } }),
}));
vi.mock("../hooks/useProducts", () => ({ useProducts: vi.fn() }));
vi.mock("../hooks/useInspections", () => ({ useInspections: vi.fn() }));
vi.mock("../hooks/useAnalyticsSummary", () => ({ useAnalyticsSummary: vi.fn() }));
vi.mock("../hooks/useCategoryAnalytics", () => ({ useCategoryAnalytics: vi.fn() }));

import { useProducts } from "../hooks/useProducts";
import { useInspections } from "../hooks/useInspections";
import { useAnalyticsSummary } from "../hooks/useAnalyticsSummary";
import { useCategoryAnalytics } from "../hooks/useCategoryAnalytics";
import { DashboardPage } from "./DashboardPage";
import { makeCategoryRows } from "../test/categoryRowsFixture";
import { AUTOMATION_RATE_INFO, MANUAL_REVIEW_INFO } from "../utils/badgeMaps";

const stats = () => ({ count: 0, avg_ms: null, median_ms: null, min_ms: null, max_ms: null });

function makeDays(total) {
  return Array.from({ length: 14 }, (_, index) => ({
    date: `2026-09-${String(index + 1).padStart(2, "0")}`,
    total,
    good: total,
    defective: 0,
    pending: 0,
    ai_analyzed: 0,
    ai_defective: 0,
    quality_pass: total,
    quality_fail: 0,
    quality_not_assessed: 0,
  }));
}

function makeAnalytics(overrides = {}) {
  return {
    total_inspections: 1234,
    by_status: { pending: 5, good: 60, defective: 55 },
    by_source: { upload: 20, mvtec_ad: 100 },
    ai_analyzed_count: 80,
    ai_prediction_counts: { good: 30, defective: 50 },
    ai_defect_rate: 0.625,
    activity_by_day: makeDays(1),
    by_product: [],
    defect_categories: [],
    quality_decisions: [
      { decision: "NOT_ASSESSED", count: 3, percentage: 4 },
      { decision: "PASS", count: 60, percentage: 80 },
      { decision: "MANUAL_REVIEW", count: 7, percentage: 9.3 },
      { decision: "FAIL", count: 5, percentage: 6.7 },
    ],
    severity_distribution: [],
    operational_insights: [],
    trend_monitoring: { period_days: 14, daily: makeDays(1), category_trends: [], insights: [] },
    performance: {
      window_days: 14,
      inspections_in_window: 14,
      ai_analyzed_in_window: 0,
      ai_analyzed_rate: 0,
      processing_time: stats(),
      ai_inference_time: stats(),
    },
    manual_review_count: 7,
    automation_rate: 0.8,
    automation_counts: { automatic: 65, manual_review: 7, not_assessed: 3 },
    ...overrides,
  };
}

let refetchCategories;

function mockData({ analytics, categories } = {}) {
  useProducts.mockReturnValue({ products: [], isLoading: false, error: null, getProductById: () => undefined });
  useInspections.mockReturnValue({ inspections: [], isLoading: false, error: null, refetch: vi.fn() });
  useAnalyticsSummary.mockReturnValue({
    data: makeAnalytics(),
    isLoading: false,
    isRefreshing: false,
    error: null,
    refetch: vi.fn(),
    ...analytics,
  });
  useCategoryAnalytics.mockReturnValue({
    data: { window_days: 14, categories: makeCategoryRows() },
    isLoading: false,
    isRefreshing: false,
    error: null,
    refetch: refetchCategories,
    ...categories,
  });
}

function renderDashboard() {
  return render(
    <MemoryRouter>
      <DashboardPage />
    </MemoryRouter>
  );
}

const section = (title) => within(screen.getByRole("heading", { name: title, level: 2 }).closest("section"));
const cardOf = (title) => within(screen.getByRole("heading", { name: title }).closest("div").parentElement);

describe("Dashboard layout and presentation", () => {
  beforeEach(() => {
    refetchCategories = vi.fn();
    [useProducts, useInspections, useAnalyticsSummary, useCategoryAnalytics].forEach((hook) => hook.mockReset());
  });

  it("groups the content under section headings in the agreed order", () => {
    mockData();
    renderDashboard();
    const titles = [...document.querySelectorAll("section > div > h2")].map((h) => h.textContent);
    expect(titles).toEqual(["Overview", "Quality outcomes", "Trend", "Defects", "Performance", "By category"]);
  });

  it("says which sections follow the time range and that Inspection activity is always 14 days", () => {
    mockData();
    renderDashboard();
    expect(
      screen.getByText(
        "The time range applies to the Trend charts (except Inspection activity, which always shows the last 14 days), the Defect category trend, Performance and By category. Other figures are all-time."
      )
    ).toBeInTheDocument();
  });

  it("keeps every existing chart and card", () => {
    mockData();
    renderDashboard();
    [
      "Inspection activity",
      "AI prediction distribution",
      "Defect Category Distribution",
      "Quality Decision Distribution",
      "Severity & Risk Distribution",
      "Product Quality Overview",
      "Inspection & Quality Trend",
      "Defect Category Trend",
      "Quality Decision Trend",
      "Trend Insights",
      "Operational Insights",
      "Inspection performance",
      "Recent inspections",
      "Quick actions",
    ].forEach((name) => expect(screen.getByRole("heading", { name })).toBeInTheDocument());
    ["Products", "Inspections", "Pending review", "AI analyzed", "AI defect rate", "Good (ground truth)",
      "Defective (ground truth)", "Pending (no ground truth yet)", "Manual review", "Automation rate"].forEach((label) =>
      expect(screen.getByText(label, { selector: "p" })).toBeInTheDocument()
    );
  });

  it("formats counts with thousands separators", () => {
    mockData();
    renderDashboard();
    expect(section("Overview").getByText("1,234")).toBeInTheDocument();
  });

  it("uses readable decision labels, sorted largest first, with count and percentage", () => {
    mockData();
    renderDashboard();
    const legend = cardOf("Quality Decision Distribution");
    const texts = legend.getAllByText(/ · \d+/).map((el) => el.textContent);
    expect(texts).toEqual(["Pass · 60(80.0%)", "Manual review · 7(9.3%)", "Fail · 5(6.7%)", "Not assessed · 3(4.0%)"]);
    expect(legend.queryByText(/MANUAL_REVIEW|NOT_ASSESSED|MANUAL REVIEW/)).not.toBeInTheDocument();
  });

  it("colours each decision by its meaning: Pass green, Fail red, Manual review amber, Not assessed grey", () => {
    mockData();
    const { container } = renderDashboard();
    const bar = (title) => container.querySelector(`[title="${title}"]`).className;
    expect(bar("Pass: 60")).toMatch(/success/);
    expect(bar("Fail: 5")).toMatch(/danger/);
    expect(bar("Manual review: 7")).toMatch(/warning/);
    expect(bar("Not assessed: 3")).toMatch(/neutral/);
    expect(bar("AI good: 30")).toMatch(/success/);
    expect(bar("AI defective: 50")).toMatch(/danger/);
  });

  it("sorts the AI prediction bar largest first with percentages", () => {
    mockData();
    renderDashboard();
    const texts = cardOf("AI prediction distribution").getAllByText(/ · \d+/).map((el) => el.textContent);
    expect(texts).toEqual(["AI defective · 50(62.5%)", "AI good · 30(37.5%)"]);
  });

  it("shows readable labels in the quality decision trend", () => {
    mockData();
    renderDashboard();
    const trend = cardOf("Quality Decision Trend");
    expect(trend.getByText("Pass / Fail / Not assessed")).toBeInTheDocument();
    ["Pass", "Fail", "Not assessed"].forEach((label) => expect(trend.getByText(label)).toBeInTheDocument());
  });

  it("explains Automation rate and Manual review with tooltips", () => {
    mockData();
    renderDashboard();
    expect(screen.getByRole("img", { name: AUTOMATION_RATE_INFO })).toHaveAttribute("title", AUTOMATION_RATE_INFO);
    expect(screen.getByRole("img", { name: MANUAL_REVIEW_INFO })).toHaveAttribute("title", MANUAL_REVIEW_INFO);
  });

  it("says why a windowed chart is empty", () => {
    mockData({
      analytics: {
        data: makeAnalytics({ trend_monitoring: { period_days: 7, daily: makeDays(0), category_trends: [], insights: [] } }),
      },
    });
    renderDashboard();
    expect(screen.getAllByRole("heading", { name: "No inspections in the last 7 days" })).toHaveLength(2);
    expect(screen.getByRole("heading", { name: "No defects in the last 7 days" })).toBeInTheDocument();
  });

  it("never shows YOLO", () => {
    mockData();
    renderDashboard();
    expect(document.body.textContent).not.toMatch(/yolo/i);
  });
});

describe("Dashboard by-category section", () => {
  beforeEach(() => {
    refetchCategories = vi.fn();
    [useProducts, useInspections, useAnalyticsSummary, useCategoryAnalytics].forEach((hook) => hook.mockReset());
  });

  it("renders the 15 rows of the response, sorted by total, with an em dash for a null rate and the note", () => {
    mockData();
    renderDashboard();
    const byCategory = section("By category");
    const rows = byCategory.getAllByTestId("category-row");
    expect(rows).toHaveLength(15);
    expect(within(rows[0]).getByText("Zipper")).toBeInTheDocument();
    expect(within(rows[14]).getByText("Bottle")).toBeInTheDocument();
    expect(within(rows[14]).getByText("—")).toBeInTheDocument();
    expect(byCategory.getByText(/It is based on AI predictions, not ground truth\./)).toBeInTheDocument();
    expect(byCategory.getByText("Inspections per MVTec category over the last 14 days.")).toBeInTheDocument();
  });

  it("follows the range selector: both requests ask for the new window", () => {
    mockData();
    renderDashboard();
    expect(useCategoryAnalytics).toHaveBeenLastCalledWith({ days: 14 });

    fireEvent.click(screen.getByRole("button", { name: "30 days" }));

    expect(useAnalyticsSummary).toHaveBeenLastCalledWith({ days: 30 });
    expect(useCategoryAnalytics).toHaveBeenLastCalledWith({ days: 30 });
  });

  it("shows a skeleton while loading", () => {
    mockData({ categories: { data: null, isLoading: true } });
    renderDashboard();
    expect(section("By category").queryAllByTestId("category-row")).toHaveLength(0);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("isolates a failure: its own error with Retry, the rest of the dashboard unaffected", () => {
    mockData({ categories: { data: null, error: new Error("By-category exploded") } });
    renderDashboard();

    const alerts = screen.getAllByRole("alert");
    expect(alerts).toHaveLength(1);
    expect(section("By category").getByRole("alert")).toHaveTextContent("By-category exploded");
    fireEvent.click(within(alerts[0]).getByRole("button", { name: "Retry" }));
    expect(refetchCategories).toHaveBeenCalledTimes(1);

    expect(section("Overview").getByText("1,234")).toBeInTheDocument();
    expect(screen.queryByText("Analytics couldn't be loaded")).not.toBeInTheDocument();
  });

  it("still renders when the summary fails", () => {
    mockData({ analytics: { data: null, error: new Error("Summary exploded") } });
    renderDashboard();
    expect(section("By category").getAllByTestId("category-row")).toHaveLength(15);
  });

  it("explains an empty window", () => {
    mockData({ categories: { data: { window_days: 7, categories: makeCategoryRows().map((r) => ({ ...r, total: 0 })) } } });
    renderDashboard();
    expect(section("By category").getByRole("heading", { name: "No inspections in the last 7 days" })).toBeInTheDocument();
  });

  it("is safe with an unexpected response", () => {
    mockData({ categories: { data: { window_days: "?", categories: "nope" } } });
    renderDashboard();
    expect(section("By category").getByRole("heading", { name: "No inspections in the last 14 days" })).toBeInTheDocument();
  });

  it("shows the updating state while only the by-category request refreshes", () => {
    mockData({ categories: { isRefreshing: true } });
    renderDashboard();
    expect(screen.getByRole("status")).toHaveTextContent("Updating");
    expect(screen.getByRole("heading", { name: "By category", level: 2 }).closest("section")).toHaveAttribute(
      "aria-busy",
      "true"
    );
  });
});
