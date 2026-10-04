import { UserCheck } from "lucide-react";
import { reviewReasonLabels } from "../../utils/badgeMaps";
import styles from "./ManualReviewBanner.module.css";

const GUIDANCE_ONLY_TEXT = "The AI result is shown for guidance only.";

/**
 * Shown only when the inspection's review_required is exactly true (the backend's manual-review rule:
 * low confidence and/or a category model that is not production ready). `compact` is a one-line
 * variant for the upload success card.
 */
export function ManualReviewBanner({ reviewRequired, reviewReason, compact = false }) {
  if (reviewRequired !== true) return null;
  const reasons = reviewReasonLabels(reviewReason);

  return (
    <div className={`${styles.banner} ${compact ? styles.compact : ""}`} role="status" data-testid="manual-review-banner">
      <span className={styles.iconWrap}>
        <UserCheck size={compact ? 16 : 20} strokeWidth={2} aria-hidden="true" />
      </span>
      <div>
        <p className={styles.title}>Manual review required</p>
        {reasons.length > 0 && <p className={styles.reason}>Reason: {reasons.join("; ")}</p>}
        <p className={styles.guidance}>{GUIDANCE_ONLY_TEXT}</p>
      </div>
    </div>
  );
}
