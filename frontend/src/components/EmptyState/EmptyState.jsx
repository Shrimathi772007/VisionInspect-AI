import styles from "./EmptyState.module.css";

export function EmptyState({ icon: Icon, title, description, action = null }) {
  return (
    <div className={styles.wrap}>
      {Icon && (
        <div className={styles.iconWrap}>
          <Icon size={26} strokeWidth={1.5} aria-hidden="true" />
        </div>
      )}
      <h3 className={styles.title}>{title}</h3>
      {description && <p className={styles.description}>{description}</p>}
      {action && <div className={styles.action}>{action}</div>}
    </div>
  );
}
