import styles from "./Skeleton.module.css";

export function Skeleton({ variant = "text", width, height, className = "", style = {} }) {
  const classes = [styles.skeleton, styles[variant], className].filter(Boolean).join(" ");
  return <div className={classes} style={{ width, height, ...style }} aria-hidden="true" />;
}
