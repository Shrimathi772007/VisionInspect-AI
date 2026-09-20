import styles from "./RangeSelector.module.css";

/**
 * A small segmented control for choosing one of a few fixed options (e.g. a time range).
 * Rendered as a group of toggle buttons so it works with keyboard, touch and screen readers
 * without a dependency: each button reports its state through aria-pressed.
 *
 * `options`: [{ value, label }]
 */
export function RangeSelector({ label, value, options, onChange, disabled = false }) {
  return (
    <div className={styles.wrap}>
      <span className={styles.label}>{label}</span>
      <div role="group" aria-label={label} className={styles.group}>
        {options.map((option) => {
          const isActive = option.value === value;
          return (
            <button
              key={option.value}
              type="button"
              className={`${styles.option} ${isActive ? styles.active : ""}`}
              aria-pressed={isActive}
              disabled={disabled}
              onClick={() => {
                if (!isActive) onChange(option.value);
              }}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
