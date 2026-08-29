import { useCallback, useRef, useState } from "react";
import { UploadCloud } from "lucide-react";
import styles from "./FileDropzone.module.css";

const DEFAULT_ACCEPT = ["image/jpeg", "image/png"];

export function FileDropzone({
  onFileAccepted,
  onFileRejected,
  accept = DEFAULT_ACCEPT,
  maxSizeBytes = 10 * 1024 * 1024,
  disabled = false,
}) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef(null);
  const dragCounter = useRef(0);

  const validateAndAccept = useCallback(
    (file) => {
      if (!file) return;
      if (!accept.includes(file.type)) {
        onFileRejected?.("Unsupported file type. Please choose a JPEG or PNG image.");
        return;
      }
      if (file.size > maxSizeBytes) {
        onFileRejected?.(`File is too large. Maximum size is ${Math.round(maxSizeBytes / (1024 * 1024))} MB.`);
        return;
      }
      onFileAccepted(file);
    },
    [accept, maxSizeBytes, onFileAccepted, onFileRejected]
  );

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      dragCounter.current = 0;
      setIsDragOver(false);
      if (disabled) return;
      const file = event.dataTransfer.files?.[0];
      validateAndAccept(file);
    },
    [disabled, validateAndAccept]
  );

  const handleDragEnter = useCallback(
    (event) => {
      event.preventDefault();
      if (disabled) return;
      dragCounter.current += 1;
      setIsDragOver(true);
    },
    [disabled]
  );

  const handleDragLeave = useCallback((event) => {
    event.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setIsDragOver(false);
    }
  }, []);

  const handleBrowseClick = () => {
    if (!disabled) inputRef.current?.click();
  };

  const handleInputChange = (event) => {
    const file = event.target.files?.[0];
    validateAndAccept(file);
    event.target.value = "";
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handleBrowseClick();
    }
  };

  return (
    <div
      className={`${styles.dropzone} ${isDragOver ? styles.dragOver : ""} ${disabled ? styles.disabled : ""}`}
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onClick={handleBrowseClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label="Upload inspection image: click to browse or drag and drop a file"
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept.join(",")}
        className="visually-hidden"
        onChange={handleInputChange}
        disabled={disabled}
        tabIndex={-1}
      />
      <div className={styles.iconWrap}>
        <UploadCloud size={28} strokeWidth={1.5} aria-hidden="true" />
      </div>
      <p className={styles.primaryText}>
        <span className={styles.link}>Click to upload</span> or drag and drop
      </p>
      <p className={styles.secondaryText}>JPEG or PNG &middot; up to 10 MB</p>
    </div>
  );
}
