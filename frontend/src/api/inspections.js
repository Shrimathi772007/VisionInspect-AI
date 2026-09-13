import { apiFetch } from "./client";

export function listInspections() {
  return apiFetch("/inspections");
}

export function getInspection(id) {
  return apiFetch(`/inspections/${id}`);
}

export function getInspectionReport(id) {
  return apiFetch(`/inspections/${id}/report`);
}

export function uploadInspection({ productId, file }) {
  const formData = new FormData();
  formData.append("product_id", String(productId));
  formData.append("file", file);
  return apiFetch("/inspections/upload", { method: "POST", formData });
}

export async function getInspectionImageObjectUrl(id) {
  const blob = await apiFetch(`/inspections/${id}/image`, { responseType: "blob" });
  return URL.createObjectURL(blob);
}

export function deleteInspection(id) {
  return apiFetch(`/inspections/${id}`, { method: "DELETE", responseType: "none" });
}
