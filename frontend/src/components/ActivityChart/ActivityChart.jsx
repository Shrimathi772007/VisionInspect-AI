import { Activity } from "lucide-react";
import { TrendChart } from "../TrendChart/TrendChart";

const SERIES = [
  { key: "good", label: "Passed", tone: "success" },
  { key: "defective", label: "Failed", tone: "danger" },
  { key: "pending", label: "Pending", tone: "warning" },
];

/**
 * Daily inspection volume by ground-truth status.
 *
 * `days` is the backend's own per-day series (analytics `activity_by_day`:
 * [{ date, total, good, defective, pending }]) - this component no longer downloads every
 * inspection and buckets them in the browser. It is a thin preset over TrendChart, which owns
 * the rendering (the two used to be near-duplicates, so a fix to one missed the other).
 */
export function ActivityChart({ days }) {
  return (
    <TrendChart
      days={days}
      series={SERIES}
      ariaLabel="Inspection activity by status"
      emptyIcon={Activity}
      emptyTitle="Not enough activity yet"
      emptyDescription="Upload more inspections to see volume over time."
    />
  );
}
