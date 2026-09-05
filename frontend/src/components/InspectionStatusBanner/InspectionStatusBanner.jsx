import { CheckCircle2, XCircle, CircleDashed } from "lucide-react";
import { statusLabel, statusTone } from "../../utils/badgeMaps";
import styles from "./InspectionStatusBanner.module.css";

const STATUS_ICONS = {
  good: CheckCircle2,
  defective: XCircle,
  pending: CircleDashed,
};

const STATUS_DESCRIPTIONS = {
  good: "No defects were recorded for this capture.",
  defective: "This capture was flagged as defective.",
  pending: "This capture is awaiting quality review.",
};

export function InspectionStatusBanner({ status }) {
  const tone = statusTone(status);
  const Icon = STATUS_ICONS[status] || CircleDashed;

  return (
    <div className={`${styles.banner} ${styles[tone] || ""}`} role="status">
      <span className={styles.iconWrap}>
        <Icon size={20} strokeWidth={2} aria-hidden="true" />
      </span>
      <div>
        <p className={styles.label}>{statusLabel(status)}</p>
        <p className={styles.description}>{STATUS_DESCRIPTIONS[status] || "Status details are unavailable."}</p>
      </div>
    </div>
  );
}
