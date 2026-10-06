import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Bot } from "lucide-react";
import { InfoTip } from "./InfoTip";
import { StatCard } from "../StatCard/StatCard";

describe("InfoTip", () => {
  it("exposes the text as tooltip and accessible name, and can be focused", () => {
    render(<InfoTip text="Explains the metric." />);
    const tip = screen.getByRole("img", { name: "Explains the metric." });
    expect(tip).toHaveAttribute("title", "Explains the metric.");
    expect(tip).toHaveAttribute("tabindex", "0");
  });

  it("renders nothing without text", () => {
    const { container } = render(<InfoTip text="" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("sits in a StatCard label without changing the label text", () => {
    render(<StatCard icon={Bot} label="Automation rate" value="81.3%" info="Share of automatic decisions." />);
    expect(screen.getByText("Automation rate", { selector: "p" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Share of automatic decisions." })).toBeInTheDocument();
  });
});
