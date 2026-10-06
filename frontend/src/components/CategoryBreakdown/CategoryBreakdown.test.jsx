import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { CategoryBreakdown } from "./CategoryBreakdown";
import { DEFECT_RATE_NOTE, formatRate } from "../../utils/badgeMaps";
import { makeCategoryRows } from "../../test/categoryRowsFixture";

const rowTexts = () => screen.getAllByTestId("category-row").map((row) => row.textContent);

describe("CategoryBreakdown", () => {
  it("renders all 15 rows largest first with count, share, defect rate and review count", () => {
    render(<CategoryBreakdown rows={makeCategoryRows()} windowDays={14} />);
    const rows = screen.getAllByTestId("category-row");
    expect(rows).toHaveLength(15);
    expect(within(rows[0]).getByText("Zipper")).toBeInTheDocument(); // total 14, the largest
    expect(within(rows[14]).getByText("Bottle")).toBeInTheDocument(); // total 0, the smallest
    // zipper: 14 of 105 inspections = 13.3%, defect rate 1/14 = 7.1%
    expect(rows[0]).toHaveTextContent("14 (13.3%)");
    expect(rows[0]).toHaveTextContent("7.1%");
    const wood = rows.find((row) => within(row).queryByText("Wood"));
    expect(wood.textContent).toMatch(/2$/);
    expect(rows.some((row) => within(row).queryByText("Metal Nut"))).toBe(true);
  });

  it("shows an em dash for a null defect rate", () => {
    render(<CategoryBreakdown rows={makeCategoryRows()} windowDays={14} />);
    const bottle = screen.getAllByTestId("category-row")[14];
    expect(bottle).toHaveTextContent("0 (0.0%)");
    expect(within(bottle).getByText("—")).toBeInTheDocument();
  });

  it("shows the defect-rate note", () => {
    render(<CategoryBreakdown rows={makeCategoryRows()} windowDays={14} />);
    expect(screen.getByText(DEFECT_RATE_NOTE)).toBeInTheDocument();
    expect(DEFECT_RATE_NOTE).toBe(
      "Defect rate is the share of AI-analysed inspections predicted defective. It is based on AI predictions, not ground truth."
    );
  });

  it("explains an empty window", () => {
    render(<CategoryBreakdown rows={makeCategoryRows().map((row) => ({ ...row, total: 0 }))} windowDays={7} />);
    expect(screen.getByRole("heading", { name: "No inspections in the last 7 days" })).toBeInTheDocument();
    expect(screen.queryAllByTestId("category-row")).toHaveLength(0);
  });

  it("renders unknown values safely", () => {
    render(
      <CategoryBreakdown
        rows={[
          { category: "uncategorised", total: 3, defect_rate: null, manual_review: undefined },
          { category: null, total: "x", defect_rate: Number.NaN },
          null,
          { category: "tile", total: 1, defect_rate: 0, manual_review: -2 },
        ]}
        windowDays={30}
      />
    );
    expect(rowTexts()).toEqual(["Uncategorised3 (75.0%)—0", "Tile1 (25.0%)0.0%0", "Unknown0 (0.0%)—0"]);
    expect(document.body.textContent).not.toMatch(/NaN|undefined/);
  });

  it("is empty-safe when the rows array is missing", () => {
    render(<CategoryBreakdown rows={undefined} windowDays={14} />);
    expect(screen.getByRole("heading", { name: "No inspections in the last 14 days" })).toBeInTheDocument();
  });

  it("formats rates", () => {
    expect(formatRate(0.375)).toBe("37.5%");
    expect(formatRate(null)).toBe("—");
    expect(formatRate(undefined)).toBe("—");
    expect(formatRate(Number.POSITIVE_INFINITY)).toBe("—");
  });
});
