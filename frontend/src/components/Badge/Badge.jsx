import styles from "./Badge.module.css";

export function Badge({ tone = "neutral", dot = false, className = "", children }) {
  const classes = [styles.badge, styles[tone], className].filter(Boolean).join(" ");
  return (
    <span className={classes}>
      {dot && <span className={styles.dot} aria-hidden="true" />}
      {children}
    </span>
  );
}
