import { useMemo, useState } from "react";
import { EmptyState } from "../EmptyState/EmptyState";
import styles from "./TrendChart.module.css";

const formatDay = (isoDate) =>
  new Date(`${isoDate}T00:00:00Z`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });

/**
 * Keeps the tooltip inside the chart: near either end it is anchored to that end instead of
 * being centred on its bar, which would push it past the card edge on a narrow screen.
 */
function tooltipPosition(index, count) {
  const barWidth = 100 / count;
  if (index < count * 0.2) return { left: `${index * barWidth}%` };
  if (index >= count * 0.8) return { right: `${(count - index - 1) * barWidth}%` };
  return { left: `${(index + 0.5) * barWidth}%`, transform: "translateX(-50%)" };
}

/**
 * A stacked daily bar chart driven by pre-aggregated backend data - the same stacked-bar
 * visual idiom as ActivityChart (Milestone 2), generalized to any set of named series over
 * any pre-bucketed `days` array, so it can render the Milestone 3 Phase 6 trend_monitoring
 * series (ground truth, AI, quality decisions, defect categories) without a new chart
 * library. It never buckets raw inspections itself - the backend has already done that (see
 * app.inspections.analytics), so the same day boundaries the API used are the ones rendered.
 *
 * `days`: [{ date: "YYYY-MM-DD", [seriesKey]: number, ... }]
 * `series`: [{ key, label, tone }] - `tone` matches Badge/DistributionBar's tone
 * vocabulary (success/danger/warning/info/accent/neutral).
 *
 * Reading a day's values works with a mouse (hover) and by touch (tap): each bar has a click
 * handler, which is also what makes iOS deliver taps as mouse events at all. The plot as a
 * whole carries a text summary for assistive technology, since the bars are plain divs.
 */
export function TrendChart({ days, series, emptyIcon, emptyTitle, emptyDescription, ariaLabel }) {
  const [activeIndex, setActiveIndex] = useState(null);

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
  const active = activeIndex !== null ? days[activeIndex] : null;
  const midIndex = Math.floor((days.length - 1) / 2);

  const seriesSummary = series
    .map((s) => `${s.label} ${days.reduce((sum, day) => sum + (day[s.key] || 0), 0)}`)
    .join(", ");
  const plotLabel = `${ariaLabel ?? "Stacked bar chart"}, last ${days.length} days. Totals: ${seriesSummary}.`;

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

      <div className={styles.plot} style={{ height: 120 }} role="img" aria-label={plotLabel}>
        <div className={styles.gridlines} aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
        </div>

        {active && (
          <div className={styles.tooltip} style={tooltipPosition(activeIndex, days.length)} role="status">
            <p className={styles.tooltipDate}>{formatDay(active.date)}</p>
            {series.map((s) => (
              <p key={s.key}>
                {s.label}: {active[s.key] || 0}
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
                data-testid="trend-bar"
                onMouseEnter={() => setActiveIndex(index)}
                onMouseLeave={() => setActiveIndex((prev) => (prev === index ? null : prev))}
                onClick={() => setActiveIndex(index)}
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
        <span>{formatDay(days[0].date)}</span>
        {days.length >= 10 && <span>{formatDay(days[midIndex].date)}</span>}
        <span>Today</span>
      </div>
    </div>
  );
}
