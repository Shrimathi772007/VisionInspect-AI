import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { AlertCircle, Images, RotateCcw, X } from "lucide-react";
import { batchUploadInspections } from "../../api/inspections";
import { ApiError } from "../../api/client";
import { Button } from "../Button/Button";
import { Badge } from "../Badge/Badge";
import {
  aiPredictionLabel,
  aiPredictionTone,
  formatConfidence,
  qualityDecisionLabel,
  qualityDecisionTone,
} from "../../utils/badgeMaps";
import { MAX_BATCH_FILES, formatBytes, validateBatch } from "../../utils/batchUpload";
import styles from "./BatchUploadPanel.module.css";

/**
 * Batch mode of the upload page: up to 20 JPEG/PNG images (50 MB in total) for the selected product, sent in
 * one POST /inspections/batch. Each file goes through the normal upload pipeline on the server; the results
 * table shows every file's outcome, including failures.
 */
export function BatchUploadPanel({ productId, onUploaded }) {
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [requestError, setRequestError] = useState(null);
  const [result, setResult] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef(null);

  const { errors, totalBytes } = validateBatch(files);
  const canSubmit = Boolean(productId) && files.length > 0 && errors.length === 0 && !isUploading;

  const addFiles = (list) => {
    const incoming = Array.from(list || []);
    if (incoming.length) setFiles((current) => [...current, ...incoming]);
  };

  const removeFile = (index) => setFiles((current) => current.filter((_file, i) => i !== index));

  const reset = () => {
    setFiles([]);
    setResult(null);
    setRequestError(null);
  };

  const submit = async () => {
    if (!canSubmit) return;
    setIsUploading(true);
    setRequestError(null);
    try {
      const response = await batchUploadInspections({ productId, files });
      setResult(response);
      onUploaded?.(response);
    } catch (err) {
      setRequestError(err instanceof ApiError ? err.message : "Batch upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  if (result) {
    const items = Array.isArray(result.items) ? result.items : [];
    return (
      <div className={styles.results}>
        <p className={styles.summary} role="status">
          {result.succeeded ?? 0} of {result.total ?? items.length} succeeded
        </p>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col">File</th>
                <th scope="col">Result</th>
                <th scope="col">AI prediction</th>
                <th scope="col">Confidence</th>
                <th scope="col">Review</th>
                <th scope="col">
                  <span className="visually-hidden">Link</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {items.map((item, index) => {
                const inspection = item.inspection;
                return (
                  <tr key={`${item.filename}-${index}`} data-testid="batch-result-row">
                    <td className={styles.fileCell}>{item.filename}</td>
                    {inspection ? (
                      <>
                        <td>
                          <Badge tone={qualityDecisionTone(inspection.quality_decision)}>
                            {qualityDecisionLabel(inspection.quality_decision)}
                          </Badge>
                        </td>
                        <td>
                          {inspection.ai_prediction ? (
                            <Badge tone={aiPredictionTone(inspection.ai_prediction)}>
                              {aiPredictionLabel(inspection.ai_prediction)}
                            </Badge>
                          ) : (
                            <span className={styles.muted}>No AI result</span>
                          )}
                        </td>
                        <td className={styles.mono}>{formatConfidence(inspection.ai_confidence) ?? "—"}</td>
                        <td>{inspection.review_required === true ? <Badge tone="accent">Review</Badge> : "—"}</td>
                        <td>
                          <Link to={`/inspections/${inspection.id}`} className={styles.link}>
                            View #{inspection.id}
                          </Link>
                        </td>
                      </>
                    ) : (
                      <td colSpan={5} className={styles.errorCell}>
                        <Badge tone="danger">Failed</Badge> <span>{item.error || "Upload failed."}</span>
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <Button variant="secondary" leftIcon={<RotateCcw size={16} />} onClick={reset}>
          Upload another batch
        </Button>
      </div>
    );
  }

  return (
    <div className={styles.panel}>
      <div
        className={`${styles.dropzone} ${isDragOver ? styles.dragOver : ""}`}
        role="button"
        tabIndex={isUploading ? -1 : 0}
        aria-label="Add images to the batch: click to browse or drag and drop files"
        onClick={() => !isUploading && inputRef.current?.click()}
        onKeyDown={(event) => {
          if ((event.key === "Enter" || event.key === " ") && !isUploading) {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragOver(false);
          if (!isUploading) addFiles(event.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".jpg,.jpeg,.png,image/jpeg,image/png"
          className="visually-hidden"
          data-testid="batch-file-input"
          onChange={(event) => {
            addFiles(event.target.files);
            event.target.value = "";
          }}
          disabled={isUploading}
          tabIndex={-1}
        />
        <Images size={26} strokeWidth={1.5} aria-hidden="true" />
        <p className={styles.dropText}>
          <span className={styles.link}>Click to add images</span> or drag and drop
        </p>
        <p className={styles.dropHint}>JPEG or PNG &middot; up to {MAX_BATCH_FILES} files &middot; 50 MB in total</p>
      </div>

      {files.length > 0 && (
        <>
          <ul className={styles.fileList} aria-label="Selected files">
            {files.map((file, index) => (
              <li key={`${file.name}-${index}`} className={styles.fileItem}>
                <span className={styles.fileName}>{file.name}</span>
                <span className={styles.fileSize}>{formatBytes(file.size)}</span>
                <button
                  type="button"
                  className={styles.removeButton}
                  onClick={() => removeFile(index)}
                  aria-label={`Remove ${file.name}`}
                  disabled={isUploading}
                >
                  <X size={14} />
                </button>
              </li>
            ))}
          </ul>
          <p className={styles.totals}>
            {files.length} file{files.length === 1 ? "" : "s"} &middot; {formatBytes(totalBytes)}
          </p>
        </>
      )}

      {errors.length > 0 && (
        <div className={styles.validation} role="alert">
          {errors.map((message) => (
            <p key={message}>
              <AlertCircle size={13} aria-hidden="true" /> {message}
            </p>
          ))}
        </div>
      )}

      {requestError && (
        <div className={styles.validation} role="alert">
          <p>
            <AlertCircle size={13} aria-hidden="true" /> {requestError}
          </p>
        </div>
      )}

      {isUploading && (
        <div className={styles.progressWrap} aria-live="polite">
          <div className={styles.progressTrack}>
            <div className={styles.progressBar} />
          </div>
          <span className={styles.progressLabel}>
            Uploading and analysing {files.length} image{files.length === 1 ? "" : "s"}&hellip;
          </span>
        </div>
      )}

      <Button type="button" size="lg" fullWidth loading={isUploading} disabled={!canSubmit} onClick={submit}>
        Upload batch
      </Button>
    </div>
  );
}
