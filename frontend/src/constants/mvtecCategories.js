// The 15 MVTec AD categories, exactly as the backend accepts them (app.dataset.categories).
// This is only the list of VALID values for a product's category. It says nothing about
// which categories have a deployed AI model - that is only ever inferred from API data.
export const MVTEC_CATEGORIES = [
  "bottle",
  "cable",
  "capsule",
  "carpet",
  "grid",
  "hazelnut",
  "leather",
  "metal_nut",
  "pill",
  "screw",
  "tile",
  "toothbrush",
  "transistor",
  "wood",
  "zipper",
];

// "metal_nut" -> "Metal Nut"
export function categoryLabel(value) {
  if (!value) return "";
  return value
    .split("_")
    .filter(Boolean)
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}
