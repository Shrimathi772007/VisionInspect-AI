import { useEffect, useState } from "react";
import { FileImage, X } from "lucide-react";
import { Badge } from "../Badge/Badge";
import styles from "./ImagePreview.module.css";

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function ImagePreview({ file, onRemove, disabled = false }) {
  const [objectUrl, setObjectUrl] = useState(null);

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  if (!file) return null;

  return (
    <div className={styles.preview}>
      <div className={styles.thumbWrap}>
        {objectUrl && <img src={objectUrl} alt={`Preview of ${file.name}`} className={styles.thumb} />}
      </div>
      <div className={styles.info}>
        <div className={styles.nameRow}>
          <FileImage size={16} className={styles.fileIcon} aria-hidden="true" />
          <span className={styles.name} title={file.name}>
            {file.name}
          </span>
        </div>
        <div className={styles.meta}>
          <Badge tone="neutral">{file.type.split("/")[1]?.toUpperCase() || "IMAGE"}</Badge>
          <span className={styles.size}>{formatFileSize(file.size)}</span>
        </div>
      </div>
      <button
        type="button"
        className={styles.removeButton}
        onClick={onRemove}
        disabled={disabled}
        aria-label={`Remove ${file.name}`}
      >
        <X size={16} />
      </button>
    </div>
  );
}
