import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

const auth = vi.hoisted(() => ({ user: { id: 1, name: "Ada", role: "quality_engineer" } }));
vi.mock("../auth/useAuth", () => ({ useAuth: () => auth }));
vi.mock("../hooks/useProducts", () => ({ useProducts: vi.fn() }));
vi.mock("../hooks/useInspections", () => ({
  useInspections: () => ({ inspections: [], isLoading: false, error: null }),
}));
vi.mock("../api/products", () => ({
  createProduct: vi.fn(),
  deleteProduct: vi.fn(),
  updateProductCategory: vi.fn(),
}));

import { useProducts } from "../hooks/useProducts";
import { createProduct, updateProductCategory } from "../api/products";
import { ApiError } from "../api/client";
import { ToastProvider } from "../components/Toast/ToastProvider";
import { ProductsPage } from "./ProductsPage";

const NOW = "2026-10-01T10:00:00Z";
const TILE = { id: 11, product_name: "Tile Line", product_code: "TILE-1", category: "tile", created_at: NOW };
const PLAIN = { id: 12, product_name: "Plain Line", product_code: "PLAIN-1", category: null, created_at: NOW };

let refetch;

function renderPage() {
  return render(
    <ToastProvider>
      <MemoryRouter>
        <ProductsPage />
      </MemoryRouter>
    </ToastProvider>
  );
}

const card = (name) => within(screen.getByRole("button", { name: `View details for ${name}` }));

function openCreateModal() {
  fireEvent.click(screen.getByRole("button", { name: "Create Product" }));
  return screen.getByRole("dialog");
}

async function openCategoryModal(name) {
  fireEvent.click(screen.getByRole("button", { name: `Change category for ${name}` }));
  return screen.findByRole("dialog");
}

describe("ProductsPage categories", () => {
  beforeEach(() => {
    auth.user = { id: 1, name: "Ada", role: "quality_engineer" };
    refetch = vi.fn();
    useProducts.mockReturnValue({ products: [TILE, PLAIN], isLoading: false, error: null, refetch });
    createProduct.mockReset().mockResolvedValue({});
    updateProductCategory.mockReset().mockResolvedValue({});
  });

  it("shows each product's category, or 'Not set'", () => {
    renderPage();
    expect(card("Tile Line").getByText("Tile")).toBeInTheDocument();
    expect(card("Plain Line").getByText("Not set")).toBeInTheDocument();
  });

  it("offers an optional MVTec category in the create modal", () => {
    renderPage();
    const dialog = openCreateModal();
    const select = within(dialog).getByLabelText("MVTec category");
    const options = within(select).getAllByRole("option");
    expect(options).toHaveLength(16);
    expect(options[0]).toHaveTextContent("None (no AI analysis)");
    expect(options[0]).toHaveValue("");
    expect(within(select).getByRole("option", { name: "Metal Nut" })).toHaveValue("metal_nut");
    expect(select).toHaveValue("");
  });

  it.each([
    ["with a category", "tile", "tile"],
    ["without a category", "", ""],
  ])("creates a product %s", async (_, chosen, expected) => {
    renderPage();
    const dialog = openCreateModal();
    fireEvent.change(within(dialog).getByLabelText("Product name"), { target: { value: "New Line" } });
    fireEvent.change(within(dialog).getByLabelText("Product code"), { target: { value: "NEW-1" } });
    fireEvent.change(within(dialog).getByLabelText("MVTec category"), { target: { value: chosen } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create Product" }));

    await waitFor(() => expect(createProduct).toHaveBeenCalledTimes(1));
    // The page passes "" for None; the API module omits the key (see api/products.test.js).
    expect(createProduct).toHaveBeenCalledWith({ productName: "New Line", productCode: "NEW-1", category: expected });
  });

  it("lets a quality engineer change a category after confirming, then refetches", async () => {
    renderPage();
    const dialog = await openCategoryModal("Plain Line");
    const save = within(dialog).getByRole("button", { name: "Save Category" });
    expect(save).toBeDisabled();

    fireEvent.change(within(dialog).getByLabelText("MVTec category"), { target: { value: "cable" } });
    expect(updateProductCategory).not.toHaveBeenCalled();
    fireEvent.click(save);

    await waitFor(() => expect(updateProductCategory).toHaveBeenCalledWith(12, "cable"));
    expect(await screen.findByText("Category updated")).toBeInTheDocument();
    expect(refetch).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("clears a category with None (sends null)", async () => {
    renderPage();
    const dialog = await openCategoryModal("Tile Line");
    expect(within(dialog).getByLabelText("MVTec category")).toHaveValue("tile");

    fireEvent.change(within(dialog).getByLabelText("MVTec category"), { target: { value: "" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save Category" }));

    await waitFor(() => expect(updateProductCategory).toHaveBeenCalledWith(11, null));
    expect(refetch).toHaveBeenCalled();
  });

  it("shows 'Product not found' on 404 and refetches", async () => {
    updateProductCategory.mockRejectedValue(new ApiError(404, "Product not found"));
    renderPage();
    const dialog = await openCategoryModal("Tile Line");
    fireEvent.change(within(dialog).getByLabelText("MVTec category"), { target: { value: "cable" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save Category" }));

    expect(await screen.findByText("Product not found")).toBeInTheDocument();
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  it("shows the normalized message for other errors", async () => {
    updateProductCategory.mockRejectedValue(new ApiError(422, "Input should be 'bottle', 'cable' or 'tile'"));
    renderPage();
    const dialog = await openCategoryModal("Tile Line");
    fireEvent.change(within(dialog).getByLabelText("MVTec category"), { target: { value: "cable" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save Category" }));

    expect(await screen.findByText("Couldn't update category")).toBeInTheDocument();
    expect(refetch).not.toHaveBeenCalled();
  });

  it("shows categories to a supervisor without any edit control", () => {
    auth.user = { id: 2, name: "Bob", role: "factory_supervisor" };
    renderPage();
    expect(card("Tile Line").getByText("Tile")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Change category for/ })).not.toBeInTheDocument();
  });
});
