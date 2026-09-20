import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { RangeSelector } from "./RangeSelector";

const OPTIONS = [
  { value: 7, label: "7 days" },
  { value: 14, label: "14 days" },
  { value: 30, label: "30 days" },
];

describe("RangeSelector", () => {
  it("marks only the current option as pressed", () => {
    render(<RangeSelector label="Time range" value={14} options={OPTIONS} onChange={() => {}} />);

    expect(screen.getByRole("button", { name: "14 days" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "7 days" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "30 days" })).toHaveAttribute("aria-pressed", "false");
  });

  it("is exposed as a labelled group", () => {
    render(<RangeSelector label="Time range" value={14} options={OPTIONS} onChange={() => {}} />);

    expect(screen.getByRole("group", { name: "Time range" })).toBeInTheDocument();
  });

  it("reports the newly chosen value", () => {
    const onChange = vi.fn();
    render(<RangeSelector label="Time range" value={14} options={OPTIONS} onChange={onChange} />);

    fireEvent.click(screen.getByRole("button", { name: "30 days" }));

    expect(onChange).toHaveBeenCalledWith(30);
  });

  it("does not re-report the option that is already selected", () => {
    const onChange = vi.fn();
    render(<RangeSelector label="Time range" value={14} options={OPTIONS} onChange={onChange} />);

    fireEvent.click(screen.getByRole("button", { name: "14 days" }));

    expect(onChange).not.toHaveBeenCalled();
  });
});
