import { useMemo, useState } from "react";
import { EmptyState } from "../EmptyState/EmptyState";
import styles from "./TrendChart.module.css";

/**
 * A stacked daily bar chart driven by pre-aggregated backend data - the same stacked-bar
 * visual idiom as ActivityChart (Milestone 2), generalized to any set of named series over
 * any pre-bucketed `days` array, so it can render the Milestone 3 Phase 6 trend_monitoring
 * series (ground truth, AI, quality decisions, defect categories) without a new chart
 * library. Unlike ActivityChart, this never buckets raw inspections itself - the backend
 * has already done that (see app.inspections.analytics), so the same day boundaries the
 * API used are the ones rendered here.
 *
 * `days`: [{ date: "YYYY-MM-DD", [seriesKey]: number, ... }]
 * `series`: [{ key, label, tone }] - `tone` matches Badge/DistributionBar's tone
 * vocabulary (success/danger/warning/info/accent/neutral).
 */
export function TrendChart({ days, series, emptyIcon, emptyTitle, emptyDescription }) {
  const [hoverIndex, setHoverIndex] = useState(null);

  const dayTotals = useMemo(
    () => days.map((day) => series.reduce((sum, s) => sum + (day[s.key] || 0), 0)),
    [days, series]
  );
  const grandTotal = dayTotals.reduce((sum, value) => sum + value, 0);

  if (grandTotal === 0) {
    return <EmptyState icon={emptyIcon} title={emptyTitle} description={emptyDescription} />;
  }

  const maxTotal = Math.max(1, ...dayTotals);
  const barWidth = 100 / days.length;
  const hovered = hoverIndex !== null ? days[hoverIndex] : null;

  return (
    <div className={styles.chart}>
      <div className={styles.legend}>
        {series.map((s) => (
          <span key={s.key} className={styles.legendItem}>
            <span className={`${styles.dot} ${styles[s.tone] || styles.neutral}`} aria-hidden="true" />
            {s.label}
          </span>
        ))}
      </div>

      <div className={styles.plot} style={{ height: 120 }}>
        <div className={styles.gridlines} aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
        </div>

        {hovered && (
          <div className={styles.tooltip} style={{ left: `${(hoverIndex + 0.5) * barWidth}%` }} role="status">
            <p className={styles.tooltipDate}>
              {new Date(`${hovered.date}T00:00:00Z`).toLocaleDateString(undefined, {
                month: "short",
                day: "numeric",
                timeZone: "UTC",
              })}
            </p>
            {series.map((s) => (
              <p key={s.key}>
                {s.label}: {hovered[s.key] || 0}
              </p>
            ))}
          </div>
        )}

        <div className={styles.bars}>
          {days.map((day, index) => {
            const dayTotal = dayTotals[index];
            const totalHeight = (dayTotal / maxTotal) * 100;
            return (
              <div
                key={day.date}
                className={styles.barGroup}
                style={{ width: `${barWidth}%` }}
                onMouseEnter={() => setHoverIndex(index)}
                onMouseLeave={() => setHoverIndex((prev) => (prev === index ? null : prev))}
              >
                <div className={styles.barStack} style={{ height: `${totalHeight}%` }}>
                  {dayTotal > 0 &&
                    series.map((s) => {
                      const value = day[s.key] || 0;
                      if (value === 0) return null;
                      return (
                        <div
                          key={s.key}
                          className={`${styles.segment} ${styles[s.tone] || styles.neutral}`}
                          style={{ height: `${(value / dayTotal) * 100}%` }}
                        />
                      );
                    })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className={styles.axisLabels}>
        <span>
          {new Date(`${days[0].date}T00:00:00Z`).toLocaleDateString(undefined, {
            month: "short",
            day: "numeric",
            timeZone: "UTC",
          })}
        </span>
        <span>Today</span>
      </div>
    </div>
  );
}
