import { apiFetch } from "./client";

export function listProducts() {
  return apiFetch("/products");
}

export function createProduct({ productName, productCode, category }) {
  const json = { product_name: productName, product_code: productCode };
  // Optional MVTec category: the key is omitted (never an empty string) when none is chosen.
  if (category) json.category = category;
  return apiFetch("/products", { method: "POST", json });
}

// QE only. `category` null clears it; only inspections created afterwards are affected.
export function updateProductCategory(productId, category) {
  return apiFetch(`/products/${productId}/category`, {
    method: "PATCH",
    json: { category: category || null },
  });
}

export function deleteProduct(id) {
  return apiFetch(`/products/${id}`, { method: "DELETE", responseType: "none" });
}
