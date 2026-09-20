import { AlertCircle } from "lucide-react";
import { Button } from "../Button/Button";
import styles from "./ErrorState.module.css";

/**
 * A failed-request message with an optional Retry - the counterpart of EmptyState/Skeleton, so
 * every data section can show "loading", "empty" and "failed" as three distinct things (a
 * failure must never look like "no data").
 *
 * variant="inline"  a compact row for the inside of a card.
 * variant="banner"  a full-width notice, for a failure that affects several sections at once.
 */
export function ErrorState({ message, title, onRetry, variant = "inline", retryLabel = "Retry" }) {
  return (
    <div className={`${styles.wrap} ${styles[variant] ?? ""}`} role="alert">
      <AlertCircle size={variant === "banner" ? 18 : 15} className={styles.icon} aria-hidden="true" />
      <div className={styles.text}>
        {title && <p className={styles.title}>{title}</p>}
        <p className={styles.message}>{message}</p>
      </div>
      {onRetry && (
        <Button size="sm" variant="secondary" onClick={onRetry}>
          {retryLabel}
        </Button>
      )}
    </div>
  );
}

/**
 * Muted placeholder for a section whose data could not be loaded, when the reason and the
 * Retry are already shown once elsewhere (e.g. an ErrorState banner). Deliberately not an
 * alert, so a page with many affected sections doesn't announce the same failure many times.
 */
export function Unavailable({ children = "Unavailable" }) {
  return <p className={styles.unavailable}>{children}</p>;
}
