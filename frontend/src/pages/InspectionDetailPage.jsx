import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Box, Hash, ImageOff, AlertCircle } from "lucide-react";
import { getInspection, getInspectionImageObjectUrl } from "../api/inspections";
import { useProducts } from "../hooks/useProducts";
import { ApiError } from "../api/client";
import { Card } from "../components/Card/Card";
import { Badge } from "../components/Badge/Badge";
import { Button } from "../components/Button/Button";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { statusLabel, statusTone, sourceLabel, sourceTone } from "../utils/badgeMaps";
import { formatDateTime } from "../utils/formatDate";
import styles from "./InspectionDetailPage.module.css";

export function InspectionDetailPage() {
  const { id } = useParams();
  const { getProductById, isLoading: productsLoading } = useProducts();

  const [inspection, setInspection] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [imageError, setImageError] = useState(null);

  useEffect(() => {
    let cancelledImageUrl = null;
    let isCancelled = false;

    async function load() {
      setIsLoading(true);
      setError(null);
      setImageError(null);
      try {
        const data = await getInspection(id);
        if (isCancelled) return;
        setInspection(data);

        try {
          const url = await getInspectionImageObjectUrl(id);
          if (isCancelled) {
            URL.revokeObjectURL(url);
            return;
          }
          cancelledImageUrl = url;
          setImageUrl(url);
        } catch (imgErr) {
          if (!isCancelled) {
            setImageError(imgErr instanceof ApiError ? imgErr.message : "Failed to load image.");
          }
        }
      } catch (err) {
        if (!isCancelled) {
          setError(err instanceof ApiError ? err.message : "Failed to load inspection.");
        }
      } finally {
        if (!isCancelled) setIsLoading(false);
      }
    }

    load();

    return () => {
      isCancelled = true;
      if (cancelledImageUrl) URL.revokeObjectURL(cancelledImageUrl);
    };
  }, [id]);

  const product = inspection ? getProductById(inspection.product_id) : null;

  return (
    <div>
      <Link to="/inspections" className={styles.backLink}>
        <ArrowLeft size={15} /> Back to Inspections
      </Link>

      {isLoading && (
        <div className={styles.layout}>
          <Skeleton variant="block" height={420} />
          <div className={styles.sideColumn}>
            <Skeleton variant="block" height={180} />
            <Skeleton variant="block" height={180} />
          </div>
        </div>
      )}

      {!isLoading && error && (
        <Card className={styles.errorCard}>
          <AlertCircle size={20} />
          <p>{error}</p>
          <Button as={Link} to="/inspections" variant="secondary" size="sm">
            Back to Inspections
          </Button>
        </Card>
      )}

      {!isLoading && !error && inspection && (
        <div className={styles.layout}>
          <Card className={styles.imageCard}>
            {/* Overlay layer intentionally empty for now — reserved so future
                anomaly heatmaps / bounding boxes / segmentation masks can be
                absolutely positioned over the image without restructuring. */}
            <div className={styles.imageViewer}>
              {imageUrl && <img src={imageUrl} alt={`Inspection ${inspection.id} capture`} className={styles.image} />}
              {!imageUrl && !imageError && <Skeleton variant="block" height="100%" />}
              {imageError && (
                <div className={styles.imageError}>
                  <ImageOff size={24} />
                  <p>{imageError}</p>
                </div>
              )}
              <div className={styles.overlayLayer} aria-hidden="true" />
            </div>
          </Card>

          <div className={styles.sideColumn}>
            <Card className={styles.metaCard}>
              <h2 className={styles.metaCardTitle}>Inspection</h2>
              <dl className={styles.metaList}>
                <div className={styles.metaRow}>
                  <dt>ID</dt>
                  <dd className={styles.mono}>#{inspection.id}</dd>
                </div>
                <div className={styles.metaRow}>
                  <dt>Status</dt>
                  <dd>
                    <Badge tone={statusTone(inspection.status)}>{statusLabel(inspection.status)}</Badge>
                  </dd>
                </div>
                <div className={styles.metaRow}>
                  <dt>Source</dt>
                  <dd>
                    <Badge tone={sourceTone(inspection.source)}>{sourceLabel(inspection.source)}</Badge>
                  </dd>
                </div>
                <div className={styles.metaRow}>
                  <dt>Inspection date</dt>
                  <dd>{formatDateTime(inspection.inspection_date)}</dd>
                </div>
                <div className={styles.metaRow}>
                  <dt>Recorded</dt>
                  <dd>{formatDateTime(inspection.created_at)}</dd>
                </div>
              </dl>
            </Card>

            <Card className={styles.metaCard}>
              <h2 className={styles.metaCardTitle}>Product</h2>
              {productsLoading && <Skeleton height={16} width="70%" />}
              {!productsLoading && product && (
                <div className={styles.productInfo}>
                  <div className={styles.productIcon}>
                    <Box size={16} />
                  </div>
                  <div>
                    <p className={styles.productName}>{product.product_name}</p>
                    <p className={styles.productCode}>
                      <Hash size={11} /> {product.product_code}
                    </p>
                  </div>
                </div>
              )}
              {!productsLoading && !product && (
                <p className={styles.metaFallback}>Product #{inspection.product_id}</p>
              )}
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
