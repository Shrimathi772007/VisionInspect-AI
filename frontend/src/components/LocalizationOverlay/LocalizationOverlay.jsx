import { analysedRegion, boxStyle, hasUnanalysedEdges, regionStyle } from "../../utils/localization";
import styles from "./LocalizationOverlay.module.css";

/**
 * Drawn inside ImageViewer's overlay layer, which has exactly the displayed image's box, so every
 * position is a percentage of the image: the heatmap PNG (already in original-image geometry) is
 * stretched over it, each box is placed at its normalised x/y/w/h, and the area outside the
 * analysed region (if any) is lightly shaded.
 */
export function LocalizationOverlay({ heatmapUrl = null, showHeatmap, heatmapOpacity, boxes, showBoxes, localization }) {
  const partial = hasUnanalysedEdges(localization);

  return (
    <div className={styles.layer} data-testid="localization-overlay">
      {showHeatmap && heatmapUrl && (
        <img
          src={heatmapUrl}
          alt=""
          aria-hidden="true"
          className={styles.heatmap}
          style={{ opacity: heatmapOpacity }}
          data-testid="heatmap-overlay"
        />
      )}
      {partial && (
        <div
          className={styles.analysedRegion}
          style={regionStyle(analysedRegion(localization))}
          data-testid="analysed-region"
        />
      )}
      {showBoxes &&
        boxes.map((box, index) => (
          <div key={index} className={styles.box} style={boxStyle(box)} data-testid="defect-region">
            {/* A box touching the top edge gets its label inside, so the layer never clips it. */}
            <span className={`${styles.boxLabel} ${box.y < 0.06 ? styles.labelInside : ""}`}>
              {index + 1}
              {Number.isFinite(box.peak) ? ` · ${box.peak.toFixed(2)}` : ""}
            </span>
          </div>
        ))}
    </div>
  );
}
