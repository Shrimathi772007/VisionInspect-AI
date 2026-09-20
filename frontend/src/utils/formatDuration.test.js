import { describe, expect, it } from "vitest";
import { formatDuration } from "./formatDuration";

describe("formatDuration", () => {
  it("formats sub-10ms values with one decimal", () => {
    expect(formatDuration(2.61)).toBe("2.6 ms");
    expect(formatDuration(0.4)).toBe("0.4 ms");
  });

  it("formats values under a second as whole milliseconds", () => {
    expect(formatDuration(845.4)).toBe("845 ms");
    expect(formatDuration(10)).toBe("10 ms");
  });

  it("formats seconds with one decimal", () => {
    expect(formatDuration(12300)).toBe("12.3 s");
    expect(formatDuration(1000)).toBe("1 s");
  });

  it("formats minutes", () => {
    expect(formatDuration(125000)).toBe("2 min 5 s");
    expect(formatDuration(120000)).toBe("2 min");
  });

  it("treats a measured zero as a real value", () => {
    expect(formatDuration(0)).toBe("0 ms");
  });

  it("returns null for missing or invalid values instead of a made-up figure", () => {
    expect(formatDuration(null)).toBeNull();
    expect(formatDuration(undefined)).toBeNull();
    expect(formatDuration(Number.NaN)).toBeNull();
    expect(formatDuration(-5)).toBeNull();
  });
});
