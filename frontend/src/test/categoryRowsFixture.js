// Test fixture: one by-category row per MVTec category, totals 0 (bottle) .. 14 (zipper), wood with 2 reviews.
const CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill",
  "screw", "tile", "toothbrush", "transistor", "wood", "zipper"];

export function makeCategoryRows() {
  return CATEGORIES.map((category, index) => ({
    category,
    total: index,
    ai_analysed: index,
    ai_defective: index === 0 ? 0 : 1,
    ai_good: index === 0 ? 0 : index - 1,
    defect_rate: index === 0 ? null : 1 / index,
    manual_review: category === "wood" ? 2 : 0,
    pass_count: 0,
    fail_count: 0,
    manual_review_decisions: 0,
    not_assessed: index,
  }));
}
