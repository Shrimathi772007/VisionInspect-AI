import { Moon, Sun } from "lucide-react";
import { useTheme } from "../../theme/useTheme";
import styles from "./ThemeToggle.module.css";

export function ThemeToggle({ compact = false, className = "" }) {
  const { theme, toggleTheme } = useTheme();
  const isLight = theme === "light";
  const label = isLight ? "Switch to dark mode" : "Switch to light mode";

  if (compact) {
    return (
      <button
        type="button"
        className={`${styles.iconButton} ${className}`}
        onClick={toggleTheme}
        role="switch"
        aria-checked={isLight}
        aria-label={label}
        title={label}
      >
        {isLight ? <Sun size={17} strokeWidth={1.9} aria-hidden="true" /> : <Moon size={17} strokeWidth={1.9} aria-hidden="true" />}
      </button>
    );
  }

  return (
    <button
      type="button"
      className={`${styles.toggle} ${className}`}
      onClick={toggleTheme}
      role="switch"
      aria-checked={isLight}
      aria-label={label}
      title={label}
    >
      <Moon size={13} strokeWidth={2} className={styles.trackIcon} aria-hidden="true" />
      <Sun size={13} strokeWidth={2} className={styles.trackIcon} aria-hidden="true" />
      <span className={styles.thumb}>
        {isLight ? <Sun size={13} strokeWidth={2} aria-hidden="true" /> : <Moon size={13} strokeWidth={2} aria-hidden="true" />}
      </span>
    </button>
  );
}
