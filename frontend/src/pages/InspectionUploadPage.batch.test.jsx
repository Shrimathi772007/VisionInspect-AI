import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../hooks/useProducts", () => ({ useProducts: vi.fn() }));
vi.mock("../api/inspections", () => ({ uploadInspection: vi.fn(), batchUploadInspections: vi.fn() }));
vi.mock("../components/Toast/ToastProvider", () => ({ useToast: () => ({ showToast: vi.fn() }) }));

import { useProducts } from "../hooks/useProducts";
import { batchUploadInspections } from "../api/inspections";
import { InspectionUploadPage } from "./InspectionUploadPage";

const TILE = { id: 11, product_name: "Tile product", product_code: "TILE-001", category: "tile" };
const CAUTION =
  "Upload an image of the same product type as the selected category. A different object type gives unreliable results.";

function renderPage() {
  return render(
    <MemoryRouter>
      <InspectionUploadPage />
    </MemoryRouter>
  );
}

describe("InspectionUploadPage modes", () => {
  beforeEach(() => {
    useProducts.mockReturnValue({ products: [TILE], isLoading: false });
    batchUploadInspections.mockReset();
  });

  it("starts in single-image mode", () => {
    renderPage();
    expect(screen.getByRole("button", { name: "Single image" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Create Inspection" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Upload batch" })).not.toBeInTheDocument();
  });

  it("batch mode keeps the product note and the caution, and sends the selected product", async () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Batch (up to 20)" }));
    expect(screen.getByRole("button", { name: "Batch (up to 20)" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("button", { name: "Create Inspection" })).not.toBeInTheDocument();
    expect(screen.getByText(CAUTION)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Product"), { target: { value: "11" } });
    expect(screen.getByText("AI analysis will run using the Tile model if one is available.")).toBeInTheDocument();

    batchUploadInspections.mockResolvedValue({ total: 1, succeeded: 1, failed: 0, items: [{ filename: "a.png", inspection: { id: 9, quality_decision: "PASS" }, error: null }] });
    const file = new File(["x"], "a.png", { type: "image/png" });
    fireEvent.change(screen.getByTestId("batch-file-input"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Upload batch" }));

    expect(await screen.findByText("1 of 1 succeeded")).toBeInTheDocument();
    expect(batchUploadInspections).toHaveBeenCalledWith({ productId: "11", files: [file] });
  });
});
