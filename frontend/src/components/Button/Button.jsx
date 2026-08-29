import styles from "./Button.module.css";

export function Button({
  as: Component = "button",
  variant = "primary",
  size = "md",
  loading = false,
  leftIcon = null,
  rightIcon = null,
  fullWidth = false,
  className = "",
  children,
  disabled,
  type,
  ...rest
}) {
  const classes = [
    styles.btn,
    styles[variant],
    styles[size],
    fullWidth ? styles.fullWidth : "",
    loading ? styles.loading : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  const isNativeButton = Component === "button";

  return (
    <Component
      className={classes}
      disabled={isNativeButton ? disabled || loading : undefined}
      aria-busy={loading || undefined}
      type={isNativeButton ? type || "button" : undefined}
      {...rest}
    >
      {loading && <span className={styles.spinner} aria-hidden="true" />}
      {!loading && leftIcon && <span className={styles.icon}>{leftIcon}</span>}
      <span className={styles.label}>{children}</span>
      {!loading && rightIcon && <span className={styles.icon}>{rightIcon}</span>}
    </Component>
  );
}
