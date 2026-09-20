import { TrendingUp, TrendingDown } from "lucide-react";
import { Card } from "../Card/Card";
import { Skeleton } from "../Skeleton/Skeleton";
import styles from "./StatCard.module.css";

/**
 * `note`   optional short line under the value that says what the number covers
 *          (e.g. "All time") - so a metric is never ambiguous about its scope.
 * `error`  the value could not be loaded: shows a dash and "Unavailable" instead of a
 *          number, so a failed request is never displayed as a real 0.
 */
export function StatCard({
  icon: Icon,
  label,
  value,
  loading = false,
  tone = "accent",
  trend = null,
  note = null,
  error = false,
}) {
  const TrendIcon = trend?.direction === "down" ? TrendingDown : TrendingUp;
  const footnote = error ? "Unavailable" : note;

  return (
    <Card className={styles.card}>
      <div className={`${styles.iconWrap} ${styles[tone]}`}>
        <Icon size={20} strokeWidth={1.75} aria-hidden="true" />
      </div>
      <div className={styles.content}>
        <p className={styles.label}>{label}</p>
        {loading ? (
          <Skeleton width={56} height={26} />
        ) : (
          <>
            <div className={styles.valueRow}>
              <p className={`${styles.value} ${error ? styles.muted : ""}`}>{error ? "—" : value}</p>
              {!error && trend && trend.direction !== "flat" && (
                <span className={styles.trend}>
                  <TrendIcon size={12} strokeWidth={2.25} aria-hidden="true" />
                  {trend.label}
                </span>
              )}
            </div>
            {footnote && <p className={styles.note}>{footnote}</p>}
          </>
        )}
      </div>
    </Card>
  );
}
