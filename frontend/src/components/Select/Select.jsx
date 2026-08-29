import { useId } from "react";
import { ChevronDown } from "lucide-react";
import styles from "./Select.module.css";

export function Select({ label, error, helperText, className = "", id, children, ...rest }) {
  const generatedId = useId();
  const selectId = id || generatedId;

  return (
    <div className={`${styles.field} ${className}`}>
      {label && (
        <label htmlFor={selectId} className={styles.label}>
          {label}
        </label>
      )}
      <div className={`${styles.selectWrap} ${error ? styles.hasError : ""}`}>
        <select id={selectId} className={styles.select} aria-invalid={Boolean(error)} {...rest}>
          {children}
        </select>
        <ChevronDown size={16} className={styles.chevron} aria-hidden="true" />
      </div>
      {error && (
        <p className={styles.errorText} role="alert">
          {error}
        </p>
      )}
      {!error && helperText && <p className={styles.helperText}>{helperText}</p>}
    </div>
  );
}
