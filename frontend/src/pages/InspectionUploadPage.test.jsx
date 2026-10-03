import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../hooks/useProducts", () => ({ useProducts: vi.fn() }));
vi.mock("../api/inspections", () => ({ uploadInspection: vi.fn() }));
vi.mock("../components/Toast/ToastProvider", () => ({ useToast: () => ({ showToast: vi.fn() }) }));

import { useProducts } from "../hooks/useProducts";
import { InspectionUploadPage } from "./InspectionUploadPage";

const TILE = { id: 11, product_name: "Tile product", product_code: "TILE-001", category: "tile" };
const PLAIN = { id: 12, product_name: "Plain product", product_code: "PLAIN-001", category: null };

const CAUTION =
  "Upload an image of the same product type as the selected category. A different object type gives unreliable results.";
const NO_CATEGORY_NOTE =
  "No category set. The image will be stored but not analysed by AI. Ask a Quality Engineer to set a category on this product.";

function renderPage() {
  return render(
    <MemoryRouter>
      <InspectionUploadPage />
    </MemoryRouter>
  );
}

const productSelect = () => screen.getByLabelText("Product");

describe("InspectionUploadPage category messaging", () => {
  beforeEach(() => {
    useProducts.mockReturnValue({ products: [TILE, PLAIN], isLoading: false });
  });

  it("shows each product's category in the product selector", () => {
    renderPage();
    expect(screen.getByRole("option", { name: "Tile product (TILE-001) - Tile" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Plain product (PLAIN-001) - No category" })).toBeInTheDocument();
  });

  it("always shows the same-object-type caution", () => {
    renderPage();
    expect(screen.getByText(CAUTION)).toBeInTheDocument();
    fireEvent.change(productSelect(), { target: { value: "11" } });
    expect(screen.getByText(CAUTION)).toBeInTheDocument();
  });

  it("changes the AI note with the selected product", () => {
    renderPage();
    expect(screen.queryByText(NO_CATEGORY_NOTE)).not.toBeInTheDocument();

    fireEvent.change(productSelect(), { target: { value: "11" } });
    expect(screen.getByText("AI analysis will run using the Tile model if one is available.")).toBeInTheDocument();
    expect(screen.queryByText(NO_CATEGORY_NOTE)).not.toBeInTheDocument();

    fireEvent.change(productSelect(), { target: { value: "12" } });
    expect(screen.getByText(NO_CATEGORY_NOTE)).toBeInTheDocument();
    expect(screen.queryByText(/AI analysis will run/)).not.toBeInTheDocument();
  });
});
