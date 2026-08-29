import styles from "./Card.module.css";

export function Card({ as: Component = "div", hoverable = false, className = "", children, ...rest }) {
  const classes = [styles.card, hoverable ? styles.hoverable : "", className].filter(Boolean).join(" ");
  return (
    <Component className={classes} {...rest}>
      {children}
    </Component>
  );
}
