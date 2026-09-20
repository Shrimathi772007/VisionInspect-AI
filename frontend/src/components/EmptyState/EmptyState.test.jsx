import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Layers } from "lucide-react";
import { EmptyState } from "./EmptyState";

describe("EmptyState", () => {
  it("renders the title and description", () => {
    render(<EmptyState icon={Layers} title="No data yet" description="Come back later." />);

    expect(screen.getByRole("heading", { name: "No data yet" })).toBeInTheDocument();
    expect(screen.getByText("Come back later.")).toBeInTheDocument();
  });

  it("omits the description and action when not given", () => {
    render(<EmptyState title="Nothing here" />);

    expect(screen.getByRole("heading", { name: "Nothing here" })).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
