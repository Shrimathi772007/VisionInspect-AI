import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DistributionBar } from "./DistributionBar";

const SEGMENTS = [
  { key: "a", label: "Small", count: 1, tone: "neutral" },
  { key: "b", label: "Big", count: 6, tone: "success" },
  { key: "c", label: "Middle", count: 3, tone: "danger" },
  { key: "d", label: "Zero", count: 0, tone: "warning" },
];

const legendTexts = (container) =>
  [...container.querySelectorAll("span")]
    .filter((el) => el.textContent.includes("·") && el.querySelector("span"))
    .map((el) => el.textContent);

describe("DistributionBar", () => {
  it("lists segments largest first with count and percentage", () => {
    const { container } = render(<DistributionBar segments={SEGMENTS} emptyTitle="Nothing" />);
    expect(legendTexts(container)).toEqual(["Big · 6(60.0%)", "Middle · 3(30.0%)", "Small · 1(10.0%)"]);
    expect(screen.queryByText(/Zero/)).not.toBeInTheDocument();
  });

  it("keeps the given order for equal counts and colours each segment by its tone", () => {
    const { container } = render(
      <DistributionBar
        segments={[
          { key: "x", label: "First", count: 2, tone: "success" },
          { key: "y", label: "Second", count: 2, tone: "danger" },
        ]}
        emptyTitle="Nothing"
      />
    );
    const bars = container.querySelectorAll("[title]");
    expect([...bars].map((el) => el.getAttribute("title"))).toEqual(["First: 2", "Second: 2"]);
    expect(bars[0].className).toMatch(/success/);
    expect(bars[1].className).toMatch(/danger/);
  });

  it("shows its empty state when every count is zero", () => {
    render(<DistributionBar segments={[{ key: "z", label: "Z", count: 0, tone: "neutral" }]} emptyTitle="No data here" />);
    expect(screen.getByRole("heading", { name: "No data here" })).toBeInTheDocument();
  });
});
