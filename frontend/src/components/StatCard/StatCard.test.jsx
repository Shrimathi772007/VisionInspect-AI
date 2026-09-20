import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Box } from "lucide-react";
import { StatCard } from "./StatCard";

describe("StatCard", () => {
  it("shows the label, value and scope note", () => {
    render(<StatCard icon={Box} label="Inspections" value={42} note="All time" />);

    expect(screen.getByText("Inspections")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("All time")).toBeInTheDocument();
  });

  it("hides the value while loading", () => {
    render(<StatCard icon={Box} label="Inspections" value={42} loading />);

    expect(screen.queryByText("42")).not.toBeInTheDocument();
  });

  it("shows a dash and 'Unavailable' - never the number - when the value failed to load", () => {
    render(<StatCard icon={Box} label="Inspections" value={0} note="All time" error />);

    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    expect(screen.queryByText("0")).not.toBeInTheDocument();
    expect(screen.queryByText("All time")).not.toBeInTheDocument();
  });

  it("shows a trend chip only when the direction is not flat", () => {
    const { rerender } = render(
      <StatCard icon={Box} label="Inspections" value={5} trend={{ direction: "up", label: "+50% vs last week" }} />
    );
    expect(screen.getByText("+50% vs last week")).toBeInTheDocument();

    rerender(<StatCard icon={Box} label="Inspections" value={5} trend={{ direction: "flat", label: "" }} />);
    expect(screen.queryByText("+50% vs last week")).not.toBeInTheDocument();
  });

  it("does not show a trend chip on an errored card", () => {
    render(
      <StatCard icon={Box} label="Inspections" value={5} error trend={{ direction: "up", label: "+50% vs last week" }} />
    );

    expect(screen.queryByText("+50% vs last week")).not.toBeInTheDocument();
  });
});
