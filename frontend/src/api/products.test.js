import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./client", () => ({ apiFetch: vi.fn() }));

import { apiFetch } from "./client";
import { createProduct, updateProductCategory } from "./products";

describe("products API", () => {
  beforeEach(() => {
    apiFetch.mockReset().mockResolvedValue({});
  });

  it("sends the category when one is chosen", () => {
    createProduct({ productName: "Tile A", productCode: "TILE-1", category: "tile" });
    expect(apiFetch).toHaveBeenCalledWith("/products", {
      method: "POST",
      json: { product_name: "Tile A", product_code: "TILE-1", category: "tile" },
    });
  });

  it.each([[""], [null], [undefined]])("omits the category key when none is chosen (%s)", (category) => {
    createProduct({ productName: "Plain", productCode: "PLAIN-1", category });
    const { json } = apiFetch.mock.calls[0][1];
    expect(json).toEqual({ product_name: "Plain", product_code: "PLAIN-1" });
    expect(Object.keys(json)).not.toContain("category");
  });

  it("patches the category, sending null to clear it", () => {
    updateProductCategory(5, "cable");
    expect(apiFetch).toHaveBeenLastCalledWith("/products/5/category", { method: "PATCH", json: { category: "cable" } });
    updateProductCategory(5, "");
    expect(apiFetch).toHaveBeenLastCalledWith("/products/5/category", { method: "PATCH", json: { category: null } });
    updateProductCategory(5, null);
    expect(apiFetch).toHaveBeenLastCalledWith("/products/5/category", { method: "PATCH", json: { category: null } });
  });
});
