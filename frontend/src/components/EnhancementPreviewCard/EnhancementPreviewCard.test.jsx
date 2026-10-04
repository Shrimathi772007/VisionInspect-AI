import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";

vi.mock("../../api/inspections", () => ({
  getInspectionEnhancedBlob: vi.fn(),
  getInspectionImageQuality: vi.fn(),
}));

import { getInspectionEnhancedBlob, getInspectionImageQuality } from "../../api/inspections";
import { EnhancementPreviewCard } from "./EnhancementPreviewCard";

const METRICS = {
  preview_only: true,
  note: "Preview only. The AI models do not use the enhanced image.",
  resolution: { width: 900, height: 300 },
  analysed_resolution: { width: 640, height: 213 },
  before: { sharpness: 120.5, contrast: 30.25, noise: 4.5, brightness: 128 },
  after: { sharpness: 150.125, contrast: 45, noise: 1.75, brightness: 131.5 },
};
const NOTE = "Preview only. Noise removal and contrast enhancement are not used by the AI models.";

describe("EnhancementPreviewCard", () => {
  const realCreate = URL.createObjectURL;
  const realRevoke = URL.revokeObjectURL;
  beforeAll(() => {
    URL.createObjectURL = vi.fn(() => "blob:enhanced");
    URL.revokeObjectURL = vi.fn();
  });
  afterAll(() => {
    URL.createObjectURL = realCreate;
    URL.revokeObjectURL = realRevoke;
  });
  beforeEach(() => {
    URL.createObjectURL.mockClear();
    URL.revokeObjectURL.mockClear();
    getInspectionEnhancedBlob.mockReset().mockResolvedValue(new Blob(["png"], { type: "image/png" }));
    getInspectionImageQuality.mockReset().mockResolvedValue(METRICS);
  });
  afterEach(() => {
    delete globalThis.IntersectionObserver;
  });

  it("loads nothing until 'Load preview', then shows both images and the before/after table", async () => {
    const { unmount } = render(<EnhancementPreviewCard inspectionId="5" originalUrl="blob:original" />);
    expect(screen.getByText(NOTE)).toBeInTheDocument();
    expect(getInspectionEnhancedBlob).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Load preview" }));
    const enhanced = await screen.findByTestId("enhanced-image");
    expect(getInspectionEnhancedBlob).toHaveBeenCalledWith("5");
    expect(getInspectionImageQuality).toHaveBeenCalledWith("5");
    expect(enhanced).toHaveAttribute("src", "blob:enhanced");
    expect(screen.getByAltText("Original")).toHaveAttribute("src", "blob:original");

    const row = (label) => within(screen.getByRole("rowheader", { name: new RegExp(label) }).closest("tr"));
    expect(row("Sharpness").getByText("120.50")).toBeInTheDocument();
    expect(row("Sharpness").getByText("150.13")).toBeInTheDocument();
    expect(row("Contrast").getByText("45.00")).toBeInTheDocument();
    expect(row("Noise").getByText("4.50")).toBeInTheDocument();
    expect(row("Noise").getByText("1.75")).toBeInTheDocument();
    expect(row("Brightness").getByText("131.50")).toBeInTheDocument();
    expect(screen.getByText(/Original 900 × 300 px; metrics measured at 640 × 213 px/)).toBeInTheDocument();

    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:enhanced");
  });

  it("loads when the card scrolls into view", async () => {
    let trigger;
    globalThis.IntersectionObserver = class {
      constructor(callback) {
        trigger = callback;
      }
      observe() {}
      disconnect() {}
    };
    render(<EnhancementPreviewCard inspectionId="7" originalUrl={null} />);
    expect(getInspectionEnhancedBlob).not.toHaveBeenCalled();

    act(() => trigger([{ isIntersecting: true }]));
    await screen.findByTestId("enhanced-image");
    expect(getInspectionEnhancedBlob).toHaveBeenCalledWith("7");
    expect(screen.getByText("Original not available")).toBeInTheDocument();
  });

  it("a failed preview shows an error with Retry and keeps the metrics", async () => {
    getInspectionEnhancedBlob.mockRejectedValueOnce(new Error("boom"));
    render(<EnhancementPreviewCard inspectionId="5" originalUrl="blob:original" />);
    fireEvent.click(screen.getByRole("button", { name: "Load preview" }));

    expect(await screen.findByText("The enhancement preview couldn't be loaded.")).toBeInTheDocument();
    expect(screen.getByText("Enhanced preview not available")).toBeInTheDocument();
    expect(screen.getByText("120.50")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByTestId("enhanced-image")).toBeInTheDocument();
    expect(getInspectionEnhancedBlob).toHaveBeenCalledTimes(2);
  });

  it("failed metrics leave the image and show an error", async () => {
    getInspectionImageQuality.mockRejectedValueOnce(new Error("boom"));
    render(<EnhancementPreviewCard inspectionId="5" originalUrl="blob:original" />);
    fireEvent.click(screen.getByRole("button", { name: "Load preview" }));

    expect(await screen.findByTestId("enhanced-image")).toBeInTheDocument();
    expect(screen.getByText("The enhancement preview couldn't be loaded.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/yolo/i);
  });
});
