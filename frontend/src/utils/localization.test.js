import { describe, expect, it } from "vitest";
import {
  analysedRegion,
  boxStyle,
  describeCentroid,
  describePosition,
  formatAreaPct,
  hasUnanalysedEdges,
  localizationBoxes,
  regionStyle,
} from "./localization";
import { qualityDecisionLabel, qualityDecisionTone, reviewReasonLabels } from "./badgeMaps";

describe("localization helpers", () => {
  it("positions a box by its normalised top-left corner and size", () => {
    expect(boxStyle({ x: 0.125, y: 0.5, w: 0.3333333, h: 0.01 })).toEqual({
      left: "12.5%",
      top: "50%",
      width: "33.3333%",
      height: "1%",
    });
  });

  it("skips malformed boxes and tolerates a missing localization", () => {
    expect(localizationBoxes(null)).toEqual([]);
    expect(localizationBoxes({ boxes: "nope" })).toEqual([]);
    expect(localizationBoxes({ boxes: [{ x: 0.1, y: 0.1, w: 0.1, h: 0.1 }, { x: 0.1, y: null, w: 0.1, h: 0.1 }] })).toHaveLength(1);
  });

  it("detects a partially analysed image (crop224) and only that", () => {
    expect(hasUnanalysedEdges({ analysed_region: [0.0625, 0.0625, 0.9375, 0.9375] })).toBe(true);
    expect(hasUnanalysedEdges({ analysed_region: [0, 0, 1, 1] })).toBe(false);
    expect(hasUnanalysedEdges({})).toBe(false);
    expect(hasUnanalysedEdges(null)).toBe(false);
    expect(analysedRegion({ analysed_region: [0, 0, 1] })).toEqual([0, 0, 1, 1]);
    expect(regionStyle([0.17, 0.0625, 0.83, 0.9375])).toEqual({ left: "17%", top: "6.25%", width: "66%", height: "87.5%" });
  });

  it("describes the centroid and area in words", () => {
    expect(describePosition(0.1, 0.1)).toBe("upper left");
    expect(describePosition(0.5, 0.5)).toBe("centre");
    expect(describePosition(0.9, 0.5)).toBe("middle right");
    expect(describePosition(0.5, 0.9)).toBe("lower centre");
    expect(describeCentroid([0.8, 0.85])).toBe("Centred in the lower right of the image (80% from the left, 85% from the top)");
    expect(describeCentroid(null)).toBeNull();
    expect(formatAreaPct(3.21)).toBe("3.2%");
    expect(formatAreaPct(undefined)).toBeNull();
  });
});

describe("quality decision and review labels", () => {
  it.each([
    ["PASS", "PASS", "success"],
    ["FAIL", "FAIL", "danger"],
    ["NOT_ASSESSED", "NOT ASSESSED", "warning"],
    ["MANUAL_REVIEW", "MANUAL REVIEW", "accent"],
    ["SOMETHING_NEW", "SOMETHING_NEW", "neutral"],
    [null, "NOT ASSESSED", "neutral"],
  ])("%s renders as %s with tone %s", (decision, label, tone) => {
    expect(qualityDecisionLabel(decision)).toBe(label);
    expect(qualityDecisionTone(decision)).toBe(tone);
  });

  it("turns review reasons into plain words", () => {
    expect(reviewReasonLabels("low confidence; category model not production ready")).toEqual([
      "Low confidence",
      "Category model not production ready",
    ]);
    expect(reviewReasonLabels("something else")).toEqual(["Something else"]);
    expect(reviewReasonLabels(null)).toEqual([]);
  });
});
