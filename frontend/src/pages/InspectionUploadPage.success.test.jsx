import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../hooks/useProducts", () => ({ useProducts: vi.fn() }));
vi.mock("../api/inspections", () => ({ uploadInspection: vi.fn() }));
vi.mock("../components/Toast/ToastProvider", () => ({ useToast: () => ({ showToast: vi.fn() }) }));
vi.mock("../components/ImagePreview/ImagePreview", () => ({ ImagePreview: () => <div>preview</div> }));

import { useProducts } from "../hooks/useProducts";
import { uploadInspection } from "../api/inspections";
import { InspectionUploadPage } from "./InspectionUploadPage";

const WOOD = { id: 31, product_name: "Wood panel", product_code: "WOOD-1", category: "wood" };

const created = (overrides = {}) => ({
  id: 77,
  ai_prediction: "good",
  product_category: "wood",
  ai_confidence: 0.9833,
  ai_reliability: "high",
  review_required: true,
  review_reason: "category model not production ready",
  quality_decision: "MANUAL_REVIEW",
  has_heatmap: true,
  ...overrides,
});

async function upload(result) {
  uploadInspection.mockResolvedValue(result);
  const { container } = render(
    <MemoryRouter>
      <InspectionUploadPage />
    </MemoryRouter>
  );
  fireEvent.change(screen.getByLabelText("Product"), { target: { value: String(WOOD.id) } });
  const input = container.querySelector('input[type="file"]');
  fireEvent.change(input, { target: { files: [new File(["x"], "wood.png", { type: "image/png" })] } });
  fireEvent.click(screen.getByRole("button", { name: "Create Inspection" }));
  await screen.findByText("Inspection recorded");
}

describe("InspectionUploadPage success card", () => {
  beforeEach(() => {
    useProducts.mockReturnValue({ products: [WOOD], isLoading: false });
    uploadInspection.mockReset();
  });

  it("shows confidence, reliability, the review line and a link to the localization", async () => {
    await upload(created());

    expect(screen.getByText("Confidence 98.3%")).toBeInTheDocument();
    expect(screen.getByText("High reliability")).toBeInTheDocument();
    const banner = screen.getByTestId("manual-review-banner");
    expect(banner).toHaveTextContent("Manual review required");
    expect(banner).toHaveTextContent("Reason: Category model not production ready");
    expect(banner).toHaveTextContent("The AI result is shown for guidance only.");
    expect(screen.getByRole("link", { name: "View localization" })).toHaveAttribute("href", "/inspections/77#localization");
    expect(screen.getByRole("link", { name: /View Inspection/ })).toHaveAttribute("href", "/inspections/77");
  });

  it("shows no review line when review is not required", async () => {
    await upload(created({ review_required: false, review_reason: null, quality_decision: "PASS" }));

    expect(screen.getByText("Confidence 98.3%")).toBeInTheDocument();
    expect(screen.queryByTestId("manual-review-banner")).not.toBeInTheDocument();
  });

  it("without an AI result shows no confidence and no localization link", async () => {
    await upload(created({ ai_prediction: null, ai_confidence: null, ai_reliability: null, review_required: null }));

    expect(screen.queryByText(/Confidence/)).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "View localization" })).not.toBeInTheDocument();
  });
});
