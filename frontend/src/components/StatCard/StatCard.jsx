import { TrendingUp, TrendingDown } from "lucide-react";
import { Card } from "../Card/Card";
import { Skeleton } from "../Skeleton/Skeleton";
import styles from "./StatCard.module.css";

export function StatCard({ icon: Icon, label, value, loading = false, tone = "accent", trend = null }) {
  const TrendIcon = trend?.direction === "down" ? TrendingDown : TrendingUp;

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
          <div className={styles.valueRow}>
            <p className={styles.value}>{value}</p>
            {trend && trend.direction !== "flat" && (
              <span className={styles.trend}>
                <TrendIcon size={12} strokeWidth={2.25} aria-hidden="true" />
                {trend.label}
              </span>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}
