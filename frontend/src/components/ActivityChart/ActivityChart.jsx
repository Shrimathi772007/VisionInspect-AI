import { useMemo, useState } from "react";
import styles from "./ActivityChart.module.css";

const DAYS = 14;
const CHART_HEIGHT = 120;

function buildBuckets(inspections) {
  const buckets = [];
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  for (let i = DAYS - 1; i >= 0; i -= 1) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    buckets.push({ date, good: 0, defective: 0, pending: 0 });
  }

  const startTime = buckets[0].date.getTime();

  inspections.forEach((inspection) => {
    const created = new Date(inspection.created_at);
    created.setHours(0, 0, 0, 0);
    const dayIndex = Math.round((created.getTime() - startTime) / 86400000);
    if (dayIndex < 0 || dayIndex >= DAYS) return;
    const bucket = buckets[dayIndex];
    if (inspection.status === "good") bucket.good += 1;
    else if (inspection.status === "defective") bucket.defective += 1;
    else bucket.pending += 1;
  });

  return buckets;
}

export function ActivityChart({ inspections }) {
  const [hoverIndex, setHoverIndex] = useState(null);

  const buckets = useMemo(() => buildBuckets(inspections), [inspections]);
  const totalInspections = buckets.reduce((sum, b) => sum + b.good + b.defective + b.pending, 0);
  const maxTotal = Math.max(1, ...buckets.map((b) => b.good + b.defective + b.pending));

  if (totalInspections === 0) {
    return (
      <div className={styles.emptyState}>
        <p>Not enough activity yet to chart a trend. Upload more inspections to see volume over time.</p>
      </div>
    );
  }

  const barWidth = 100 / DAYS;
  const hovered = hoverIndex !== null ? buckets[hoverIndex] : null;

  return (
    <div className={styles.chart}>
      <div className={styles.legend}>
        <span className={styles.legendItem}>
          <span className={`${styles.dot} ${styles.good}`} /> Passed
        </span>
        <span className={styles.legendItem}>
          <span className={`${styles.dot} ${styles.defective}`} /> Failed
        </span>
        <span className={styles.legendItem}>
          <span className={`${styles.dot} ${styles.pending}`} /> Pending
        </span>
      </div>

      <div className={styles.plot} style={{ height: CHART_HEIGHT }}>
        <div className={styles.gridlines} aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
        </div>

        {hovered && (
          <div
            className={styles.tooltip}
            style={{ left: `${(hoverIndex + 0.5) * barWidth}%` }}
            role="status"
          >
            <p className={styles.tooltipDate}>
              {hovered.date.toLocaleDateString(undefined, { month: "short", day: "numeric" })}
            </p>
            <p>
              {hovered.good + hovered.defective + hovered.pending} inspection
              {hovered.good + hovered.defective + hovered.pending === 1 ? "" : "s"}
            </p>
          </div>
        )}

        <div className={styles.bars}>
          {buckets.map((bucket, index) => {
            const total = bucket.good + bucket.defective + bucket.pending;
            const totalHeight = (total / maxTotal) * 100;
            return (
              <div
                key={bucket.date.toISOString()}
                className={styles.barGroup}
                style={{ width: `${barWidth}%` }}
                onMouseEnter={() => setHoverIndex(index)}
                onMouseLeave={() => setHoverIndex((prev) => (prev === index ? null : prev))}
              >
                <div className={styles.barStack} style={{ height: `${totalHeight}%` }}>
                  {bucket.defective > 0 && (
                    <div
                      className={`${styles.segment} ${styles.defective}`}
                      style={{ height: `${(bucket.defective / total) * 100}%` }}
                    />
                  )}
                  {bucket.pending > 0 && (
                    <div
                      className={`${styles.segment} ${styles.pending}`}
                      style={{ height: `${(bucket.pending / total) * 100}%` }}
                    />
                  )}
                  {bucket.good > 0 && (
                    <div
                      className={`${styles.segment} ${styles.good}`}
                      style={{ height: `${(bucket.good / total) * 100}%` }}
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className={styles.axisLabels}>
        <span>{buckets[0].date.toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span>
        <span>Today</span>
      </div>
    </div>
  );
}
