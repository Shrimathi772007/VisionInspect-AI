import { useId } from "react";
import styles from "./Input.module.css";

export function Input({
  label,
  error,
  helperText,
  leftIcon = null,
  rightSlot = null,
  className = "",
  id,
  ...rest
}) {
  const generatedId = useId();
  const inputId = id || generatedId;
  const describedBy = error ? `${inputId}-error` : helperText ? `${inputId}-helper` : undefined;

  return (
    <div className={`${styles.field} ${className}`}>
      {label && (
        <label htmlFor={inputId} className={styles.label}>
          {label}
        </label>
      )}
      <div className={`${styles.inputWrap} ${error ? styles.hasError : ""}`}>
        {leftIcon && <span className={styles.leftIcon}>{leftIcon}</span>}
        <input
          id={inputId}
          className={`${styles.input} ${leftIcon ? styles.withLeftIcon : ""} ${
            rightSlot ? styles.withRightSlot : ""
          }`}
          aria-invalid={Boolean(error)}
          aria-describedby={describedBy}
          {...rest}
        />
        {rightSlot && <span className={styles.rightSlot}>{rightSlot}</span>}
      </div>
      {error && (
        <p id={`${inputId}-error`} className={styles.errorText} role="alert">
          {error}
        </p>
      )}
      {!error && helperText && (
        <p id={`${inputId}-helper`} className={styles.helperText}>
          {helperText}
        </p>
      )}
    </div>
  );
}
