// Client-side checks for POST /inspections/batch, mirroring the backend limits (the backend re-checks
// everything; these only stop an obviously invalid batch before it is sent).
export const MAX_BATCH_FILES = 20;
export const MAX_BATCH_BYTES = 50 * 1024 * 1024;
export const MAX_FILE_BYTES = 10 * 1024 * 1024; // the single-upload limit, applied to every file
export const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"];

export function fileExtension(name) {
  const dot = typeof name === "string" ? name.lastIndexOf(".") : -1;
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** { errors: string[], totalBytes } for a list of File objects; no errors means the batch may be sent. */
export function validateBatch(files) {
  const errors = [];
  const totalBytes = files.reduce((sum, file) => sum + (file.size || 0), 0);
  if (files.length > MAX_BATCH_FILES) {
    errors.push(`Too many files: ${files.length} selected, at most ${MAX_BATCH_FILES} per batch.`);
  }
  if (totalBytes > MAX_BATCH_BYTES) {
    errors.push(`Total size ${formatBytes(totalBytes)} exceeds the 50 MB batch limit.`);
  }
  const badType = files.filter((file) => !ALLOWED_EXTENSIONS.includes(fileExtension(file.name)));
  if (badType.length > 0) {
    errors.push(`Unsupported file type (JPEG or PNG only): ${badType.map((file) => file.name).join(", ")}.`);
  }
  const tooLarge = files.filter((file) => file.size > MAX_FILE_BYTES);
  if (tooLarge.length > 0) {
    errors.push(`Larger than 10 MB: ${tooLarge.map((file) => file.name).join(", ")}.`);
  }
  return { errors, totalBytes };
}
