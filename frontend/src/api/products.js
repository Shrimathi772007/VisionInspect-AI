import { apiFetch } from "./client";

export function listProducts() {
  return apiFetch("/products");
}

export function createProduct({ productName, productCode }) {
  return apiFetch("/products", {
    method: "POST",
    json: { product_name: productName, product_code: productCode },
  });
}

export function deleteProduct(id) {
  return apiFetch(`/products/${id}`, { method: "DELETE", responseType: "none" });
}
