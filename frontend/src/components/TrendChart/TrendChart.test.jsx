import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { Activity } from "lucide-react";
import { TrendChart } from "./TrendChart";
import { ActivityChart } from "../ActivityChart/ActivityChart";

const SERIES = [
  { key: "good", label: "Good", tone: "success" },
  { key: "defective", label: "Defective", tone: "danger" },
];

function makeDays(count, valuesFor = (index) => ({ good: index + 1, defective: index % 2 })) {
  return Array.from({ length: count }, (_, index) => ({
    date: `2026-09-${String(index + 1).padStart(2, "0")}`,
    ...valuesFor(index),
  }));
}

function renderChart(days, props = {}) {
  return render(
    <TrendChart
      days={days}
      series={SERIES}
      emptyIcon={Activity}
      emptyTitle="No trend data"
      emptyDescription="Nothing recorded."
      {...props}
    />
  );
}

describe("TrendChart", () => {
  it("shows the empty state, not an empty plot, when every value is zero", () => {
    renderChart(makeDays(10, () => ({ good: 0, defective: 0 })));

    expect(screen.getByRole("heading", { name: "No trend data" })).toBeInTheDocument();
    expect(screen.queryAllByTestId("trend-bar")).toHaveLength(0);
  });

  it("renders one bar per day", () => {
    renderChart(makeDays(14));

    expect(screen.getAllByTestId("trend-bar")).toHaveLength(14);
  });

  it("gives the plot a text alternative that summarises each series", () => {
    renderChart(makeDays(4, () => ({ good: 2, defective: 1 })), { ariaLabel: "Inspection trend" });

    const label = screen.getByRole("img").getAttribute("aria-label");
    expect(label).toContain("Inspection trend");
    expect(label).toContain("last 4 days");
    expect(label).toContain("Good 8");
    expect(label).toContain("Defective 4");
  });

  it("shows a day's values when its bar is tapped - not only on mouse hover", () => {
    renderChart(makeDays(10, () => ({ good: 3, defective: 2 })));

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    fireEvent.click(screen.getAllByTestId("trend-bar")[4]);

    const tooltip = screen.getByRole("status");
    expect(tooltip).toHaveTextContent("Good: 3");
    expect(tooltip).toHaveTextContent("Defective: 2");
  });

  it("still shows the tooltip on mouse hover, and hides it on leave", () => {
    renderChart(makeDays(10));
    const bar = screen.getAllByTestId("trend-bar")[6];

    fireEvent.mouseEnter(bar);
    expect(screen.getByRole("status")).toBeInTheDocument();

    fireEvent.mouseLeave(bar);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("anchors the tooltip to the left edge for the first bar and the right edge for the last, so it cannot overflow", () => {
    renderChart(makeDays(14));
    const bars = screen.getAllByTestId("trend-bar");

    fireEvent.click(bars[0]);
    const first = screen.getByRole("status");
    expect(first.style.left).toBe("0%");
    expect(first.style.transform).toBe("");

    fireEvent.click(bars[13]);
    const last = screen.getByRole("status");
    expect(last.style.right).toBe("0%");
    expect(last.style.left).toBe("");
  });

  it("centres the tooltip over a bar in the middle of the chart", () => {
    renderChart(makeDays(14));

    fireEvent.click(screen.getAllByTestId("trend-bar")[7]);

    expect(screen.getByRole("status").style.transform).toBe("translateX(-50%)");
  });

  it("labels the middle of the axis only when there are enough days for it to be useful", () => {
    const { unmount } = renderChart(makeDays(9));
    expect(screen.queryByText("Sep 5")).not.toBeInTheDocument();
    unmount();

    renderChart(makeDays(10));
    expect(screen.getByText("Sep 5")).toBeInTheDocument();
    expect(screen.getByText("Today")).toBeInTheDocument();
  });

  it("supports a 30-day window", () => {
    renderChart(makeDays(30));

    expect(screen.getAllByTestId("trend-bar")).toHaveLength(30);
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("last 30 days");
  });
});

describe("ActivityChart", () => {
  it("renders the backend's per-day series with the Passed / Failed / Pending legend", () => {
    const days = makeDays(14, () => ({ total: 4, good: 2, defective: 1, pending: 1 }));

    render(<ActivityChart days={days} />);

    expect(screen.getByText("Passed")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("Pending")).toBeInTheDocument();
    expect(screen.getAllByTestId("trend-bar")).toHaveLength(14);
  });

  it("shows its own empty state when there is no activity", () => {
    render(<ActivityChart days={makeDays(14, () => ({ total: 0, good: 0, defective: 0, pending: 0 }))} />);

    expect(screen.getByRole("heading", { name: "Not enough activity yet" })).toBeInTheDocument();
  });
});
