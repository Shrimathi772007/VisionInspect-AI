import { Activity, Clock, Cpu, Gauge, Timer } from "lucide-react";
import { StatCard } from "../StatCard/StatCard";
import { EmptyState } from "../EmptyState/EmptyState";
import { formatDuration } from "../../utils/formatDuration";
import styles from "./PerformanceOverview.module.css";

const NO_DATA = "No data";

function timedNote(stats, noneText) {
  if (!stats) return null;
  if (stats.count === 0) return noneText;
  return `Median ${formatDuration(stats.median_ms)} · ${stats.count} timed`;
}

function rangeValue(stats) {
  if (!stats || stats.count === 0) return NO_DATA;
  return `${formatDuration(stats.min_ms)} – ${formatDuration(stats.max_ms)}`;
}

/**
 * Operational performance for the selected time range, from the analytics `performance` block.
 *
 * Terminology is deliberately literal: "processing time" is the server-side handling time of
 * an inspection request; "AI inference time" is the time the AI predictor reported (including
 * model loading and threshold derivation). Neither says anything about AI accuracy.
 *
 * Only inspections that have a recorded timing are averaged - `count` on each stat says how
 * many - and a range with none shows "No data", never 0 or an estimate.
 */
export function PerformanceOverview({ performance = null, isLoading = false, error = false }) {
  const showEmpty = !isLoading && !error && performance && performance.inspections_in_window === 0;
  if (showEmpty) {
    return (
      <EmptyState
        icon={Timer}
        title="No inspections in this range"
        description="Performance figures appear once inspections are recorded in the selected range."
      />
    );
  }

  const loading = isLoading || (!error && !performance);
  const processing = performance?.processing_time;
  const ai = performance?.ai_inference_time;
  const total = performance?.inspections_in_window ?? 0;
  const aiRate = performance?.ai_analyzed_rate;

  return (
    <div>
      <div className={styles.grid}>
        <StatCard
          icon={Clock}
          label="Avg processing time"
          value={formatDuration(processing?.avg_ms) ?? NO_DATA}
          note={timedNote(processing, "No timed inspections in this range")}
          loading={loading}
          error={error}
          tone="info"
        />
        <StatCard
          icon={Gauge}
          label={"Fastest – slowest"}
          value={rangeValue(processing)}
          note="Processing time"
          loading={loading}
          error={error}
          tone="success"
        />
        <StatCard
          icon={Activity}
          label="Timing coverage"
          value={`${processing?.count ?? 0} / ${total}`}
          note="Inspections with a recorded time"
          loading={loading}
          error={error}
          tone="accent"
        />
        <StatCard
          icon={Cpu}
          label="AI-analyzed"
          value={performance?.ai_analyzed_in_window ?? 0}
          note={aiRate == null ? null : `${(aiRate * 100).toFixed(1)}% of ${total.toLocaleString()} inspections`}
          loading={loading}
          error={error}
          tone="info"
        />
        <StatCard
          icon={Timer}
          label="Avg AI inference time"
          value={formatDuration(ai?.avg_ms) ?? NO_DATA}
          note={timedNote(ai, "No AI-timed inspections in this range")}
          loading={loading}
          error={error}
          tone="warning"
        />
      </div>
      <p className={styles.footnote}>
        Processing time is the server-side handling time of an inspection (image storage, database,
        AI analysis, severity and quality assessment). AI inference time is the time spent in the AI
        predictor, including model loading and threshold derivation. Inspections recorded before
        timing was introduced have no measurement and are left out, never estimated.
      </p>
    </div>
  );
}
