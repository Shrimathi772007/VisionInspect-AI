import { apiFetch } from "./client";

/**
 * @param {{ limit?: number }} [options] `limit` asks the backend for only the newest N
 *   inspections (a SQL LIMIT). Omitted, every inspection is returned, as before.
 */
export function listInspections({ limit } = {}) {
  const query = limit ? `?limit=${encodeURIComponent(limit)}` : "";
  return apiFetch(`/inspections${query}`);
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

/**
 * The inspection's anomaly heatmap (RGBA PNG in the original image's geometry) as a Blob. The
 * endpoint needs the Bearer token, so an <img src> cannot load it directly - the caller makes an
 * object URL and must revoke it. Rejects with ApiError 404 when the inspection has no heatmap.
 */
export function getInspectionHeatmapBlob(id) {
  return apiFetch(`/inspections/${id}/heatmap`, { responseType: "blob" });
}

/**
 * POST /inspections/batch (quality engineers): one product, up to 20 images (50 MB in total), sent as
 * repeated "files" fields. Resolves to { total, succeeded, failed, items: [{ filename, inspection, error }] }
 * - with HTTP 200 even when some files failed.
 */
export function batchUploadInspections({ productId, files }) {
  const formData = new FormData();
  formData.append("product_id", String(productId));
  for (const file of files) formData.append("files", file);
  return apiFetch("/inspections/batch", { method: "POST", formData });
}

/** Enhanced (denoised + contrast) PNG preview as a Blob. Preview only - not used by the AI models. */
export function getInspectionEnhancedBlob(id) {
  return apiFetch(`/inspections/${id}/enhanced`, { responseType: "blob" });
}

/** Before/after image-quality metrics of the enhancement preview. */
export function getInspectionImageQuality(id) {
  return apiFetch(`/inspections/${id}/image-quality`);
}
