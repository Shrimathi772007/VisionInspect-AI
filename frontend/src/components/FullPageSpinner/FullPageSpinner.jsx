import { ScanLine } from "lucide-react";
import styles from "./FullPageSpinner.module.css";

export function FullPageSpinner() {
  return (
    <div className={styles.wrap} role="status" aria-label="Loading">
      <div className={styles.mark}>
        <ScanLine size={22} aria-hidden="true" />
      </div>
      <div className={styles.spinner} aria-hidden="true" />
    </div>
  );
}
