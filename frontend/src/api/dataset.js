import { apiFetch } from "./client";

export function listDatasetCategories() {
  return apiFetch("/dataset/categories");
}

export function getDatasetCategoryDetail(category) {
  return apiFetch(`/dataset/categories/${encodeURIComponent(category)}`);
}

export function listDatasetImages({ category, split, defectType }) {
  const params = new URLSearchParams({ split, defect_type: defectType });
  return apiFetch(`/dataset/categories/${encodeURIComponent(category)}/images?${params.toString()}`);
}

export async function getDatasetPreviewObjectUrl({ category, split, defectType, filename, maxDim }) {
  const params = new URLSearchParams({ split, defect_type: defectType, filename });
  if (maxDim) params.set("max_dim", String(maxDim));
  const blob = await apiFetch(
    `/dataset/categories/${encodeURIComponent(category)}/preview?${params.toString()}`,
    { responseType: "blob" }
  );
  return URL.createObjectURL(blob);
}

export function importDatasetInspection({ productId, category, split, defectType, filename }) {
  return apiFetch("/inspections/import", {
    method: "POST",
    json: {
      product_id: Number(productId),
      category,
      split,
      defect_type: defectType,
      filename,
    },
  });
}
