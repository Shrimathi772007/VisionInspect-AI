import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatusBadge } from "./StatusBadge";
import { STATUS_LABELS, UNLABELLED_STATUS_HINT, statusHint, statusLabel } from "../../utils/badgeMaps";

const HINT = "No ground-truth label. The AI prediction is shown separately.";

describe("StatusBadge", () => {
  it("shows the stored 'pending' status as Unlabelled, with the ground-truth tooltip", () => {
    render(<StatusBadge status="pending" />);
    const badge = screen.getByText("Unlabelled");
    expect(badge).toHaveAttribute("title", HINT);
    expect(badge.className).toMatch(/warning/);
    expect(screen.queryByText(/pending/i)).not.toBeInTheDocument();
  });

  it.each([
    ["good", "Good", /success/],
    ["defective", "Defective", /danger/],
  ])("keeps %s as %s, without a tooltip", (status, label, tone) => {
    render(<StatusBadge status={status} />);
    const badge = screen.getByText(label);
    expect(badge).not.toHaveAttribute("title");
    expect(badge.className).toMatch(tone);
  });

  it("shows the fallback when there is no status", () => {
    render(<StatusBadge status={undefined} fallback="—" />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});

describe("status label map", () => {
  it("maps only the display text; the value it is keyed by stays 'pending'", () => {
    expect(Object.keys(STATUS_LABELS)).toEqual(["pending", "good", "defective"]);
    expect(statusLabel("pending")).toBe("Unlabelled");
    expect(statusHint("pending")).toBe(HINT);
    expect(UNLABELLED_STATUS_HINT).toBe(HINT);
    expect(statusHint("good")).toBeUndefined();
    expect(statusLabel("something_new")).toBe("something_new");
  });
});
