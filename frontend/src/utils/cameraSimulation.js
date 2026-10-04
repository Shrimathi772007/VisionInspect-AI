// Safety limits and frame order of the simulated camera (pages/CameraSimulationPage.jsx).
export const MIN_INTERVAL_S = 1;
export const MAX_INTERVAL_S = 10;
export const DEFAULT_INTERVAL_S = 3;
export const MIN_FRAMES = 1;
export const MAX_FRAMES = 50;
export const DEFAULT_FRAMES = 10;
export const ALL_DEFECT_TYPES = "all";

function clampInt(value, min, max, fallback) {
  if (value === null || value === undefined || (typeof value === "string" && value.trim() === "")) return fallback;
  const number = Math.round(Number(value));
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

/** Seconds between frames, an integer in [1, 10]; anything unparsable becomes the default 3. */
export function clampInterval(value) {
  return clampInt(value, MIN_INTERVAL_S, MAX_INTERVAL_S, DEFAULT_INTERVAL_S);
}

/** Frames per run, an integer in [1, 50]; anything unparsable becomes the default 10. */
export function clampMaxFrames(value) {
  return clampInt(value, MIN_FRAMES, MAX_FRAMES, DEFAULT_FRAMES);
}

/**
 * The deterministic frame order: `lists` is [{ defectType, filenames }]. Defect types are sorted by name and
 * each list's filenames sorted; with more than one type the frames take turns (round-robin) so a short run
 * still shows every type: a0, b0, c0, a1, b1, ... A shorter list simply drops out once exhausted.
 */
export function buildFrameSequence(lists) {
  const sorted = [...lists]
    .filter((list) => Array.isArray(list.filenames) && list.filenames.length > 0)
    .sort((a, b) => a.defectType.localeCompare(b.defectType))
    .map((list) => ({ defectType: list.defectType, filenames: [...list.filenames].sort() }));
  const sequence = [];
  const longest = Math.max(0, ...sorted.map((list) => list.filenames.length));
  for (let i = 0; i < longest; i += 1) {
    for (const list of sorted) {
      if (i < list.filenames.length) sequence.push({ defectType: list.defectType, filename: list.filenames[i] });
    }
  }
  return sequence;
}

/** The frame for the n-th tick (0-based), cycling back to the start when the sequence is exhausted. */
export function frameAt(sequence, index) {
  return sequence.length ? sequence[index % sequence.length] : null;
}
