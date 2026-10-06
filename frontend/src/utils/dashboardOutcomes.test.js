import { describe, expect, it } from "vitest";
import {
  AUTOMATION_RATE_INFO,
  MANUAL_REVIEW_INFO,
  outcomeTone,
  predictionOutcomeTone,
  qualityDecisionDisplayLabel,
  qualityDecisionLabel,
  qualityDecisionTone,
} from "./badgeMaps";

describe("dashboard outcome labels and tones", () => {
  it.each([
    ["PASS", "Pass"],
    ["FAIL", "Fail"],
    ["MANUAL_REVIEW", "Manual review"],
    ["NOT_ASSESSED", "Not assessed"],
    [null, "Not assessed"],
    [undefined, "Not assessed"],
    ["SOMETHING_NEW", "SOMETHING_NEW"],
  ])("labels %s as %s", (decision, label) => {
    expect(qualityDecisionDisplayLabel(decision)).toBe(label);
  });

  it("gives one colour per meaning", () => {
    expect(outcomeTone("PASS")).toBe("success");
    expect(outcomeTone("FAIL")).toBe("danger");
    expect(outcomeTone("MANUAL_REVIEW")).toBe("warning");
    expect(outcomeTone("NOT_ASSESSED")).toBe("neutral");
    expect(outcomeTone(null)).toBe("neutral");
    expect(outcomeTone("SOMETHING_NEW")).toBe("neutral");
    expect(predictionOutcomeTone("good")).toBe(outcomeTone("PASS"));
    expect(predictionOutcomeTone("defective")).toBe(outcomeTone("FAIL"));
    expect(predictionOutcomeTone("pending")).toBe(outcomeTone("NOT_ASSESSED"));
    expect(predictionOutcomeTone(undefined)).toBe("neutral");
  });

  it("leaves the existing label and tone maps used by other pages unchanged", () => {
    expect(qualityDecisionLabel("MANUAL_REVIEW")).toBe("MANUAL REVIEW");
    expect(qualityDecisionTone("MANUAL_REVIEW")).toBe("accent");
    expect(qualityDecisionTone("NOT_ASSESSED")).toBe("warning");
  });

  it("explains the terms in short sentences", () => {
    expect(AUTOMATION_RATE_INFO).toBe("Share of inspections that received an automatic Pass or Fail decision.");
    expect(MANUAL_REVIEW_INFO).toBe("The AI result is low reliability, so a person must check it.");
  });
});
