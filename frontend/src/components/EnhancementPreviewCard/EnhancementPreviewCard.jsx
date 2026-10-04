import { useEffect, useRef, useState } from "react";
import { WandSparkles } from "lucide-react";
import { Card } from "../Card/Card";
import { Button } from "../Button/Button";
import { Skeleton } from "../Skeleton/Skeleton";
import { ErrorState } from "../ErrorState/ErrorState";
import { useEnhancementPreview } from "../../hooks/useEnhancementPreview";
import styles from "./EnhancementPreviewCard.module.css";

const PREVIEW_NOTE = "Preview only. Noise removal and contrast enhancement are not used by the AI models.";

const METRICS = [
  { key: "sharpness", label: "Sharpness", hint: "variance of the Laplacian" },
  { key: "contrast", label: "Contrast", hint: "intensity standard deviation" },
  { key: "noise", label: "Noise", hint: "robust estimate" },
  { key: "brightness", label: "Brightness", hint: "mean intensity" },
];

function formatMetric(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "—";
}

/**
 * Original vs enhanced (denoised + contrast) preview with before/after image-quality metrics. Fetched
 * lazily: when the card scrolls into view (where IntersectionObserver exists) or on "Load preview". Its
 * failures stay inside this card.
 */
export function EnhancementPreviewCard({ inspectionId, originalUrl }) {
  const [requested, setRequested] = useState(false);
  const cardRef = useRef(null);
  const preview = useEnhancementPreview(inspectionId, requested);

  useEffect(() => {
    const node = cardRef.current;
    if (requested || !node || typeof IntersectionObserver === "undefined") return undefined;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setRequested(true);
        observer.disconnect();
      }
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [requested]);

  const resolution = preview.metrics?.resolution;
  const analysed = preview.metrics?.analysed_resolution;

  return (
    <Card className={styles.card} ref={cardRef}>
      <h2 className={styles.title}>
        <WandSparkles size={14} aria-hidden="true" /> Image enhancement preview
      </h2>
      <p className={styles.note}>{PREVIEW_NOTE}</p>

      {!requested && (
        <Button size="sm" variant="secondary" onClick={() => setRequested(true)}>
          Load preview
        </Button>
      )}

      {requested && (
        <>
          <div className={styles.pair}>
            <figure className={styles.figure}>
              {originalUrl ? (
                <img src={originalUrl} alt="Original" className={styles.image} />
              ) : (
                <div className={styles.placeholder}>Original not available</div>
              )}
              <figcaption>Original</figcaption>
            </figure>
            <figure className={styles.figure}>
              {preview.isLoading && <Skeleton variant="block" height={180} />}
              {preview.url && <img src={preview.url} alt="Enhanced preview" className={styles.image} data-testid="enhanced-image" />}
              {!preview.isLoading && !preview.url && <div className={styles.placeholder}>Enhanced preview not available</div>}
              <figcaption>Enhanced (preview)</figcaption>
            </figure>
          </div>

          {preview.isLoading && (
            <div role="status" aria-label="Loading enhancement metrics">
              <Skeleton height={14} width="60%" />
            </div>
          )}

          {(preview.urlError || preview.metricsError) && (
            <ErrorState message="The enhancement preview couldn't be loaded." onRetry={preview.retry} />
          )}

          {preview.metrics && (
            <>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th scope="col">Metric</th>
                    <th scope="col">Before</th>
                    <th scope="col">After</th>
                  </tr>
                </thead>
                <tbody>
                  {METRICS.map((metric) => (
                    <tr key={metric.key}>
                      <th scope="row">
                        {metric.label}
                        <span className={styles.hint}>{metric.hint}</span>
                      </th>
                      <td>{formatMetric(preview.metrics.before?.[metric.key])}</td>
                      <td>{formatMetric(preview.metrics.after?.[metric.key])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {resolution && (
                <p className={styles.resolution}>
                  Original {resolution.width} &times; {resolution.height} px
                  {analysed ? `; metrics measured at ${analysed.width} × ${analysed.height} px` : ""}
                </p>
              )}
            </>
          )}
        </>
      )}
    </Card>
  );
}
