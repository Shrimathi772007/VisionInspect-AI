import { Card } from "../Card/Card";
import { Badge } from "../Badge/Badge";
import { severityLabel, severityTone, qualityRiskLabel, qualityRiskTone } from "../../utils/badgeMaps";
import styles from "./SeverityCard.module.css";

const DEFECT_FORM_NOTE = "Shape-based form derived from the anomaly region; not a trained defect-type classifier.";
const SEVERITY_NOTE =
  "Severity is rule-based: size, location (centre-weighted proxy), form and confidence. Region size is approximate.";
const NO_SEVERITY_AI_ONLY = "No severity: the inspection is good or has no localized region.";
const NO_SEVERITY_GROUND_TRUTH = "Not assessed - insufficient evidence available";

const GROUND_TRUTH_STATUSES = ["good", "defective"];

/**
 * Severity assessment of one inspection. For uploads (no ground truth) severity comes from the rule-based
 * severity_v1 (size, location, form, confidence of the anomaly region); imports keep the ground-truth wording.
 */
export function SeverityCard({ inspection }) {
  const hasScore = Number.isFinite(inspection.severity_score);
  const isAiOnly = !GROUND_TRUTH_STATUSES.includes(inspection.status);

  return (
    <Card className={styles.card}>
      <h2 className={styles.title}>Severity Assessment</h2>
      {hasScore ? (
        <>
          <p className={styles.caption}>{SEVERITY_NOTE}</p>
          <dl className={styles.list}>
            <div className={styles.row}>
              <dt>Score</dt>
              <dd className={styles.mono}>{Math.round(inspection.severity_score)} / 100</dd>
            </div>
            <div className={styles.row}>
              <dt>Level</dt>
              <dd>
                <Badge tone={severityTone(inspection.severity_level)}>{severityLabel(inspection.severity_level)}</Badge>
              </dd>
            </div>
            <div className={styles.row}>
              <dt>Quality Risk</dt>
              <dd>
                <Badge tone={qualityRiskTone(inspection.quality_risk)}>{qualityRiskLabel(inspection.quality_risk)}</Badge>
              </dd>
            </div>
            {inspection.severity_action && (
              <div className={styles.row}>
                <dt>Recommended action</dt>
                <dd>{inspection.severity_action}</dd>
              </div>
            )}
            {inspection.defect_form_label && (
              <div className={styles.row}>
                <dt>Defect form</dt>
                <dd>{inspection.defect_form_label}</dd>
              </div>
            )}
          </dl>
          {inspection.defect_form_label && <p className={styles.note}>{DEFECT_FORM_NOTE}</p>}
        </>
      ) : (
        <p className={styles.fallback}>{isAiOnly ? NO_SEVERITY_AI_ONLY : NO_SEVERITY_GROUND_TRUTH}</p>
      )}
    </Card>
  );
}
