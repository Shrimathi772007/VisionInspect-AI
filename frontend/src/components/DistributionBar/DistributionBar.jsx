import { EmptyState } from "../EmptyState/EmptyState";
import styles from "./DistributionBar.module.css";

/**
 * A horizontal, segmented distribution bar + legend - the same visualization idiom
 * already used for AI prediction distribution on the Dashboard (Milestone 2), generalized
 * to any labeled set of counts (defect category, quality decision, severity level).
 * Deliberately not a new chart library: consistent with the existing project approach.
 *
 * `segments`: [{ key, label, count, tone }] - `tone` matches Badge's tone vocabulary
 * (success/danger/warning/info/accent/neutral) so colors stay consistent app-wide.
 */
export function DistributionBar({ segments, emptyIcon, emptyTitle, emptyDescription }) {
  const total = segments.reduce((sum, segment) => sum + segment.count, 0);

  if (total === 0) {
    return <EmptyState icon={emptyIcon} title={emptyTitle} description={emptyDescription} />;
  }

  const visibleSegments = segments.filter((segment) => segment.count > 0);

  return (
    <div className={styles.wrap}>
      <div className={styles.bar}>
        {visibleSegments.map((segment) => (
          <div
            key={segment.key}
            className={`${styles.segment} ${styles[segment.tone] || styles.neutral}`}
            style={{ width: `${(segment.count / total) * 100}%` }}
            title={`${segment.label}: ${segment.count}`}
          />
        ))}
      </div>
      <div className={styles.legend}>
        {visibleSegments.map((segment) => (
          <span key={segment.key} className={styles.legendItem}>
            <span className={`${styles.dot} ${styles[segment.tone] || styles.neutral}`} aria-hidden="true" />
            {segment.label} &middot; {segment.count}
            <span className={styles.legendPercentage}>
              ({((segment.count / total) * 100).toFixed(1)}%)
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
