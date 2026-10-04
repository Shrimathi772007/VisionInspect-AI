import { describe, expect, it } from "vitest";
import { buildFrameSequence, clampInterval, clampMaxFrames, frameAt } from "./cameraSimulation";

describe("camera simulation helpers", () => {
  it.each([
    [0, 1], [1, 1], [3, 3], [10, 10], [11, 10], [-5, 1], [2.6, 3], ["7", 7], ["", 3], ["abc", 3], [undefined, 3],
  ])("clamps the interval %s to %s s", (input, expected) => {
    expect(clampInterval(input)).toBe(expected);
  });

  it.each([[0, 1], [1, 1], [10, 10], [50, 50], [51, 50], [999, 50], ["x", 10], [null, 10], [" ", 10]])(
    "clamps max frames %s to %s",
    (input, expected) => {
      expect(clampMaxFrames(input)).toBe(expected);
    }
  );

  it("orders frames deterministically: sorted types, sorted files, round-robin", () => {
    const sequence = buildFrameSequence([
      { defectType: "good", filenames: ["002.png", "000.png", "001.png"] },
      { defectType: "crack", filenames: ["001.png", "000.png"] },
      { defectType: "empty", filenames: [] },
    ]);
    expect(sequence.map((f) => `${f.defectType}/${f.filename}`)).toEqual([
      "crack/000.png", "good/000.png", "crack/001.png", "good/001.png", "good/002.png",
    ]);
  });

  it("cycles back to the start", () => {
    const sequence = buildFrameSequence([{ defectType: "good", filenames: ["b.png", "a.png"] }]);
    expect([0, 1, 2, 3].map((i) => frameAt(sequence, i).filename)).toEqual(["a.png", "b.png", "a.png", "b.png"]);
    expect(frameAt([], 0)).toBeNull();
  });
});
