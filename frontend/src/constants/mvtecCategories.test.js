import { describe, expect, it } from "vitest";
import { MVTEC_CATEGORIES, categoryLabel } from "./mvtecCategories";

describe("mvtecCategories", () => {
  it("has exactly 15 unique lowercase categories", () => {
    expect(MVTEC_CATEGORIES).toHaveLength(15);
    expect(new Set(MVTEC_CATEGORIES).size).toBe(15);
    for (const category of MVTEC_CATEGORIES) {
      expect(category).toBe(category.trim().toLowerCase());
    }
  });

  it("formats labels", () => {
    expect(categoryLabel("metal_nut")).toBe("Metal Nut");
    expect(categoryLabel("tile")).toBe("Tile");
    expect(categoryLabel(null)).toBe("");
  });
});
