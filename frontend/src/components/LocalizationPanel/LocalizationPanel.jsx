import { Crosshair } from "lucide-react";
import { Card } from "../Card/Card";
import { Skeleton } from "../Skeleton/Skeleton";
import { ErrorState } from "../ErrorState/ErrorState";
import {
  HEATMAP_OPACITY_STEPS,
  LOCALIZATION_NOTE,
  describeCentroid,
  formatAreaPct,
  hasUnanalysedEdges,
} from "../../utils/localization";
import styles from "./LocalizationPanel.module.css";

/**
 * "Defect localization" section of the inspection detail page: the anomaly-based note, the overlay
 * toggles and the region summary. The overlays themselves are drawn over the image by
 * LocalizationOverlay; this panel only holds the controls and the words.
 */
export function LocalizationPanel({
  inspection,
  boxes,
  heatmap,
  showHeatmap,
  onShowHeatmapChange,
  heatmapOpacity,
  onHeatmapOpacityChange,
  showBoxes,
  onShowBoxesChange,
}) {
  const localization = inspection.localization;
  const hasHeatmap = inspection.has_heatmap === true;
  const isGood = inspection.ai_prediction === "good";
  const areaText = formatAreaPct(localization?.area_pct);
  const centroidText = describeCentroid(localization?.centroid);

  return (
    <Card className={styles.card} id="localization">
      <h2 className={styles.title}>
        <Crosshair size={14} aria-hidden="true" /> Defect localization
      </h2>
      <p className={styles.note}>{LOCALIZATION_NOTE}</p>

      {localization && isGood && <p className={styles.summary}>No anomalous region above the threshold.</p>}

      {localization && !isGood && boxes.length > 0 && (
        <ul className={styles.summaryList}>
          {areaText && <li>Affected area {areaText} of the image</li>}
          <li>
            {boxes.length} region{boxes.length === 1 ? "" : "s"} marked (numbered by peak anomaly score)
          </li>
          {centroidText && <li>{centroidText}</li>}
        </ul>
      )}

      {localization && !isGood && boxes.length === 0 && (
        <p className={styles.summary}>No region large enough to mark.</p>
      )}

      {localization && hasUnanalysedEdges(localization) && (
        <p className={styles.edgeNote} data-testid="analysed-region-note">
          Edges outside the analysed region (shaded) were not examined by the model.
        </p>
      )}

      {hasHeatmap ? (
        <div className={styles.controls}>
          <label className={styles.toggle}>
            <input type="checkbox" checked={showHeatmap} onChange={(event) => onShowHeatmapChange(event.target.checked)} />
            Heatmap
          </label>
          {boxes.length > 0 && (
            <label className={styles.toggle}>
              <input type="checkbox" checked={showBoxes} onChange={(event) => onShowBoxesChange(event.target.checked)} />
              Defect regions
            </label>
          )}
          <div className={styles.opacity} role="group" aria-label="Heatmap opacity">
            <span className={styles.opacityLabel}>Opacity</span>
            {HEATMAP_OPACITY_STEPS.map((step) => (
              <button
                key={step.label}
                type="button"
                className={`${styles.step} ${heatmapOpacity === step.value ? styles.stepActive : ""}`}
                aria-pressed={heatmapOpacity === step.value}
                onClick={() => onHeatmapOpacityChange(step.value)}
                disabled={!showHeatmap}
              >
                {step.label}
              </button>
            ))}
          </div>
          {heatmap.isLoading && (
            <span className={styles.loading} role="status" aria-label="Loading heatmap">
              <Skeleton width={90} height={12} />
            </span>
          )}
          {heatmap.error && (
            <ErrorState message="Heatmap couldn't be loaded." onRetry={heatmap.retry} />
          )}
        </div>
      ) : (
        <p className={styles.fallback}>Heatmap not available for this inspection.</p>
      )}
    </Card>
  );
}
