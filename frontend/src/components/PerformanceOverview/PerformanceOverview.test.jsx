import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { PerformanceOverview } from "./PerformanceOverview";

const stats = (overrides = {}) => ({ count: 0, avg_ms: null, median_ms: null, min_ms: null, max_ms: null, ...overrides });

const PERFORMANCE = {
  window_days: 14,
  inspections_in_window: 100,
  ai_analyzed_in_window: 60,
  ai_analyzed_rate: 0.6,
  processing_time: stats({ count: 40, avg_ms: 1240, median_ms: 1100, min_ms: 300, max_ms: 4200 }),
  ai_inference_time: stats({ count: 30, avg_ms: 9800, median_ms: 9500, min_ms: 8000, max_ms: 12000 }),
};

// The tile for a label: the label's parent holds that tile's value and note.
const tile = (label) => within(screen.getByText(label, { selector: "p" }).parentElement);

describe("PerformanceOverview", () => {
  it("shows formatted timings with their sample sizes", () => {
    render(<PerformanceOverview performance={PERFORMANCE} />);

    expect(tile("Avg processing time").getByText("1.2 s")).toBeInTheDocument();
    expect(tile("Avg processing time").getByText("Median 1.1 s · 40 timed")).toBeInTheDocument();
    expect(tile("Avg AI inference time").getByText("9.8 s")).toBeInTheDocument();
    expect(tile("Avg AI inference time").getByText("Median 9.5 s · 30 timed")).toBeInTheDocument();
  });

  it("shows the fastest-to-slowest range", () => {
    render(<PerformanceOverview performance={PERFORMANCE} />);

    expect(tile("Fastest – slowest").getByText("300 ms – 4.2 s")).toBeInTheDocument();
  });

  it("shows how much of the range actually has a timing, and the AI-analyzed rate", () => {
    render(<PerformanceOverview performance={PERFORMANCE} />);

    expect(tile("Timing coverage").getByText("40 / 100")).toBeInTheDocument();
    expect(tile("AI-analyzed").getByText("60")).toBeInTheDocument();
    expect(tile("AI-analyzed").getByText("60.0% of 100 inspections")).toBeInTheDocument();
  });

  it("says 'No data' - never 0 ms - when no inspection in the range was timed", () => {
    render(
      <PerformanceOverview
        performance={{ ...PERFORMANCE, processing_time: stats(), ai_inference_time: stats() }}
      />
    );

    expect(tile("Avg processing time").getByText("No data")).toBeInTheDocument();
    expect(tile("Avg processing time").getByText("No timed inspections in this range")).toBeInTheDocument();
    expect(tile("Avg AI inference time").getByText("No data")).toBeInTheDocument();
    expect(tile("Fastest – slowest").getByText("No data")).toBeInTheDocument();
    expect(screen.queryByText(/0 ms/)).not.toBeInTheDocument();
    expect(tile("Timing coverage").getByText("0 / 100")).toBeInTheDocument();
  });

  it("treats a measured 0 ms as a real value, not as missing", () => {
    render(
      <PerformanceOverview
        performance={{
          ...PERFORMANCE,
          processing_time: stats({ count: 1, avg_ms: 0, median_ms: 0, min_ms: 0, max_ms: 0 }),
        }}
      />
    );

    expect(tile("Avg processing time").getByText("0 ms")).toBeInTheDocument();
  });

  it("shows an empty state when there are no inspections in the range", () => {
    render(<PerformanceOverview performance={{ ...PERFORMANCE, inspections_in_window: 0, ai_analyzed_rate: null }} />);

    expect(screen.getByRole("heading", { name: "No inspections in this range" })).toBeInTheDocument();
  });

  it("shows placeholders, not numbers, while loading", () => {
    render(<PerformanceOverview isLoading />);

    expect(screen.queryByText("No data")).not.toBeInTheDocument();
    expect(screen.queryByText("0 / 0")).not.toBeInTheDocument();
  });

  it("shows 'Unavailable' on every tile when the data failed to load", () => {
    render(<PerformanceOverview error />);

    expect(screen.getAllByText("Unavailable")).toHaveLength(5);
  });

  it("explains what the timings do and do not include", () => {
    render(<PerformanceOverview performance={PERFORMANCE} />);

    expect(screen.getByText(/never estimated/)).toBeInTheDocument();
    expect(screen.getByText(/including model loading and threshold derivation/)).toBeInTheDocument();
  });
});
