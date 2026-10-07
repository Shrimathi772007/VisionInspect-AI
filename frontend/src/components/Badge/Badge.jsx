import styles from "./Badge.module.css";

export function Badge({ tone = "neutral", dot = false, className = "", title, children }) {
  const classes = [styles.badge, styles[tone], className].filter(Boolean).join(" ");
  return (
    <span className={classes} title={title}>
      {dot && <span className={styles.dot} aria-hidden="true" />}
      {children}
    </span>
  );
}
