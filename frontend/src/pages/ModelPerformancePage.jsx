import { Gauge } from "lucide-react";
import { useAiModels } from "../hooks/useAiModels";
import { PageHeader } from "../components/PageHeader/PageHeader";
import { Card } from "../components/Card/Card";
import { Badge } from "../components/Badge/Badge";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { ErrorState } from "../components/ErrorState/ErrorState";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { categoryLabel } from "../constants/mvtecCategories";
import { modelGateLabel, modelGateTone } from "../utils/badgeMaps";
import styles from "./ModelPerformancePage.module.css";

const METRICS_FOOTNOTE =
  "Metrics are from each category's single final test on the MVTec AD test set. Gates are project-defined " +
  "(Excellent: recall >= 0.90, F1 >= 0.85, FPR <= 0.10; Good: recall >= 0.85, F1 >= 0.80, FPR <= 0.10; " +
  "Acceptable: recall >= 0.75, F1 >= 0.70, FPR <= 0.15).";

const AP_FOOTNOTE =
  "AP is image-level (defective = positive). For the 6 ResNet-18 categories it was computed after the final test " +
  "from the saved scores; no model or threshold changed.";

export const LOCALIZATION_FOOTNOTE =
  "Second scoring of final test sets for localization only; no model or threshold changed; anomaly-map boxes, not " +
  "a trained detector. Box AP@0.5 matches each defect region of the ground-truth mask (one-box-per-image view in " +
  "brackets). Pixel AUROC excludes pixels outside the analysed area for the centre-crop 224 categories.";

const EM_DASH = "—";

const FAMILY_LABELS = {
  patchcore_wrn50: "WRN-50 PatchCore",
  patch_anomaly: "ResNet-18 patch model",
  convae: "Convolutional autoencoder",
};

function familyLabel(family) {
  return FAMILY_LABELS[family] || family || EM_DASH;
}

function inputModeLabel(model) {
  const side = Array.isArray(model.input_size) ? model.input_size[0] : null;
  switch (model.input_mode) {
    case "crop224":
      return "centre crop 224";
    case "full256":
      return "whole image 256";
    case "full320":
      return "whole image 320";
    case "resize":
      return side ? `whole image ${side}` : "whole image";
    default:
      return model.input_mode || EM_DASH;
  }
}

function percent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : EM_DASH;
}

function decimal(value) {
  return Number.isFinite(value) ? value.toFixed(3) : EM_DASH;
}

export function ModelPerformancePage() {
  const { models, isLoading, error, refetch } = useAiModels();

  return (
    <div>
      <PageHeader
        eyebrow="AI models"
        title="Model performance"
        description="The served anomaly-detection model of every MVTec category, with its gate and final-test metrics."
      />

      {isLoading && (
        <Card className={styles.tableCard}>
          <div className={styles.skeletonList} role="status" aria-label="Loading models">
            {[1, 2, 3, 4, 5].map((key) => (
              <div key={key} className={styles.skeletonRow}>
                <Skeleton width="16%" height={14} />
                <Skeleton width="28%" height={14} />
                <Skeleton width="18%" height={20} />
                <Skeleton width="24%" height={14} />
              </div>
            ))}
          </div>
        </Card>
      )}

      {!isLoading && error && (
        <Card className={styles.tableCard}>
          <ErrorState message={`Failed to load models. ${error.message}`} onRetry={refetch} />
        </Card>
      )}

      {!isLoading && !error && models.length === 0 && (
        <Card>
          <EmptyState icon={Gauge} title="No models registered" description="Served models appear here once registered." />
        </Card>
      )}

      {!isLoading && !error && models.length > 0 && (
        <Card className={styles.tableCard}>
          <div className={styles.tableWrap}>
            <table className={styles.table} aria-label="Served models">
              <thead>
                <tr>
                  <th scope="col">Category</th>
                  <th scope="col">Model</th>
                  <th scope="col">Gate</th>
                  <th scope="col" className={styles.numeric}>Recall</th>
                  <th scope="col" className={styles.numeric}>FPR</th>
                  <th scope="col" className={styles.numeric}>AUROC</th>
                  <th scope="col" className={styles.numeric}>AP</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.category}>
                    <td className={styles.categoryCell}>{categoryLabel(model.category)}</td>
                    <td>
                      <span className={styles.family}>{familyLabel(model.family)}</span>
                      <span className={styles.mode}>{inputModeLabel(model)}</span>
                    </td>
                    <td>
                      <Badge tone={modelGateTone(model.gate)}>{modelGateLabel(model.gate)}</Badge>
                    </td>
                    <td className={styles.numeric}>{percent(model.final_test_recall)}</td>
                    <td className={styles.numeric}>{percent(model.final_test_fpr)}</td>
                    <td className={styles.numeric}>{decimal(model.final_test_auroc)}</td>
                    <td className={styles.numeric}>{decimal(model.final_test_average_precision)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className={styles.footnote}>{METRICS_FOOTNOTE}</p>
          <p className={styles.footnote}>{AP_FOOTNOTE}</p>
          <p className={styles.footnote}>
            Inspections from a category whose model is not production ready always go to manual review.
          </p>
        </Card>
      )}

      {!isLoading && !error && models.length > 0 && (
        <Card className={styles.tableCard}>
          <h2 className={styles.sectionTitle}>Localization (evaluation only)</h2>
          <div className={styles.tableWrap}>
            <table className={`${styles.table} ${styles.compactTable}`} aria-label="Localization evaluation">
              <thead>
                <tr>
                  <th scope="col">Category</th>
                  <th scope="col" className={styles.numeric}>Box AP@0.5</th>
                  <th scope="col" className={styles.numeric}>Pixel AUROC</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.category}>
                    <td className={styles.categoryCell}>{categoryLabel(model.category)}</td>
                    <td className={styles.numeric}>
                      {decimal(model.box_ap50)}
                      {Number.isFinite(model.box_ap50_merged) && (
                        <span className={styles.secondary}> ({decimal(model.box_ap50_merged)})</span>
                      )}
                    </td>
                    <td className={styles.numeric}>{decimal(model.pixel_auroc)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className={styles.footnote}>{LOCALIZATION_FOOTNOTE}</p>
        </Card>
      )}
    </div>
  );
}
