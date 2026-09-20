/**
 * Human-readable form of a duration in milliseconds, e.g. 2.6 ms, 845 ms, 12.3 s, 2 min 5 s.
 *
 * Returns null - never "0 ms" or a placeholder - for a missing or non-finite value, so the
 * caller decides how to say "not recorded". A measured 0 is a real value and is formatted.
 */
export function formatDuration(ms) {
  if (ms === null || ms === undefined || !Number.isFinite(ms) || ms < 0) return null;

  if (ms < 10) return `${Number(ms.toFixed(1))} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;

  const seconds = ms / 1000;
  if (seconds < 60) return `${Number(seconds.toFixed(1))} s`;

  const wholeSeconds = Math.round(seconds);
  const minutes = Math.floor(wholeSeconds / 60);
  const remainder = wholeSeconds % 60;
  return remainder === 0 ? `${minutes} min` : `${minutes} min ${remainder} s`;
}
