import { Card } from "../Card/Card";
import { Skeleton } from "../Skeleton/Skeleton";
import styles from "./StatCard.module.css";

export function StatCard({ icon: Icon, label, value, loading = false, tone = "accent" }) {
  return (
    <Card className={styles.card}>
      <div className={`${styles.iconWrap} ${styles[tone]}`}>
        <Icon size={20} strokeWidth={1.75} aria-hidden="true" />
      </div>
      <div className={styles.content}>
        <p className={styles.label}>{label}</p>
        {loading ? <Skeleton width={56} height={26} /> : <p className={styles.value}>{value}</p>}
      </div>
    </Card>
  );
}
