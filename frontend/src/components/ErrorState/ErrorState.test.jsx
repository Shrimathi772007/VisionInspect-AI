import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ErrorState, Unavailable } from "./ErrorState";

describe("ErrorState", () => {
  it("announces the message as an alert", () => {
    render(<ErrorState title="Analytics couldn't be loaded" message="Server returned 500" />);

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Analytics couldn't be loaded");
    expect(alert).toHaveTextContent("Server returned 500");
  });

  it("calls onRetry when Retry is clicked", () => {
    const onRetry = vi.fn();
    render(<ErrorState message="Failed" onRetry={onRetry} />);

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("shows no Retry button when there is nothing to retry", () => {
    render(<ErrorState message="Failed" />);

    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});

describe("Unavailable", () => {
  it("is a quiet placeholder, not an alert, so many of them don't each announce the same failure", () => {
    render(<Unavailable />);

    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
