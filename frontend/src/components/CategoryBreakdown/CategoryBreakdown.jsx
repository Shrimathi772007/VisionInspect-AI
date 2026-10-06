import { LayoutList } from "lucide-react";
import { EmptyState } from "../EmptyState/EmptyState";
import { categoryLabel } from "../../constants/mvtecCategories";
import { DEFECT_RATE_NOTE, formatRate } from "../../utils/badgeMaps";
import styles from "./CategoryBreakdown.module.css";

const count = (value) => (Number.isFinite(value) && value > 0 ? value : 0);

function rowLabel(category) {
  if (category === "uncategorised") return "Uncategorised";
  return categoryLabel(typeof category === "string" ? category : "") || "Unknown";
}

/**
 * Inspections per MVTec category over the selected window (GET /inspections/analytics/by-category), as a
 * horizontal bar per category - largest first, with its count and share of the window's inspections beside
 * the bar, the AI defect rate as a second value (an em dash when nothing was AI-analysed) and a small
 * manual-review count. Rows arrive pre-aggregated; the backend's order (total desc, then name) is kept for
 * ties. Unknown or missing values render as 0 / "—", never as NaN.
 */
export function CategoryBreakdown({ rows, windowDays }) {
  const safeRows = (Array.isArray(rows) ? rows : [])
    .filter((row) => row && typeof row === "object")
    .map((row, index) => ({ row, index, total: count(row.total) }))
    .sort((a, b) => b.total - a.total || a.index - b.index);
  const grandTotal = safeRows.reduce((sum, { total }) => sum + total, 0);

  if (grandTotal === 0) {
    return (
      <EmptyState
        icon={LayoutList}
        title={`No inspections in the last ${windowDays} days`}
        description="Per-category counts appear once inspections are recorded in this window."
      />
    );
  }

  const maxTotal = Math.max(1, ...safeRows.map(({ total }) => total));

  return (
    <div className={styles.wrap}>
      <div className={styles.header}>
        <span className={styles.labelCol}>Category</span>
        <span className={styles.barCol}>Inspections</span>
        <span className={styles.rateCol}>Defect rate</span>
        <span className={styles.reviewCol} title="Inspections flagged for manual review">
          Review
        </span>
      </div>
      <ul className={styles.list} aria-label={`Inspections per category, last ${windowDays} days`}>
        {safeRows.map(({ row, total }) => {
          const label = rowLabel(row.category);
          const share = ((total / grandTotal) * 100).toFixed(1);
          const review = count(row.manual_review);
          return (
            <li key={String(row.category)} className={styles.row} data-testid="category-row">
              <span className={styles.labelCol} title={label}>
                {label}
              </span>
              <span className={styles.barCol}>
                <span className={styles.track} aria-hidden="true">
                  <span className={styles.bar} style={{ width: `${(total / maxTotal) * 100}%` }} />
                </span>
                <span className={styles.value}>
                  {total} <span className={styles.share}>({share}%)</span>
                </span>
              </span>
              <span className={styles.rateCol}>{formatRate(row.defect_rate)}</span>
              <span className={`${styles.reviewCol} ${review > 0 ? styles.reviewSome : ""}`}>{review}</span>
            </li>
          );
        })}
      </ul>
      <p className={styles.note}>{DEFECT_RATE_NOTE}</p>
    </div>
  );
}
