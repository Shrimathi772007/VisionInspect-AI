import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Box, Hash, ImageOff, AlertCircle, Trash2 } from "lucide-react";
import { deleteInspection, getInspection, getInspectionImageObjectUrl } from "../api/inspections";
import { useProducts } from "../hooks/useProducts";
import { ApiError } from "../api/client";
import { useToast } from "../components/Toast/ToastProvider";
import { Card } from "../components/Card/Card";
import { Badge } from "../components/Badge/Badge";
import { Button } from "../components/Button/Button";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { Modal } from "../components/Modal/Modal";
import { RoleGate } from "../components/RoleGate/RoleGate";
import { ImageViewer } from "../components/ImageViewer/ImageViewer";
import { InspectionStatusBanner } from "../components/InspectionStatusBanner/InspectionStatusBanner";
import {
  statusLabel,
  statusTone,
  sourceLabel,
  sourceTone,
  aiPredictionLabel,
  aiPredictionTone,
  defectCategoryLabel,
  defectCategoryTone,
  severityLabel,
  severityTone,
  qualityRiskLabel,
  qualityRiskTone,
  qualityDecisionLabel,
  qualityDecisionTone,
} from "../utils/badgeMaps";
import { formatDateTime } from "../utils/formatDate";
import styles from "./InspectionDetailPage.module.css";

export function InspectionDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { getProductById, isLoading: productsLoading } = useProducts();
  const { showToast } = useToast();

  const [inspection, setInspection] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [imageError, setImageError] = useState(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

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

  const closeDeleteModal = () => {
    if (isDeleting) return;
    setIsDeleteModalOpen(false);
  };

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await deleteInspection(id);
      showToast({ type: "success", title: "Inspection deleted", message: `Inspection #${id} was removed.` });
      navigate("/inspections");
    } catch (err) {
      setIsDeleting(false);
      setIsDeleteModalOpen(false);
      if (err instanceof ApiError && err.status === 404) {
        showToast({ type: "error", title: "Inspection not found", message: "It may have already been deleted." });
        navigate("/inspections");
        return;
      }
      showToast({
        type: "error",
        title: "Couldn't delete inspection",
        message: err instanceof ApiError ? err.message : "Something went wrong.",
      });
    }
  };

  return (
    <div>
      <div className={styles.topRow}>
        <Link to="/inspections" className={styles.backLink}>
          <ArrowLeft size={15} /> Back to Inspections
        </Link>
        {!isLoading && !error && inspection && (
          <RoleGate allow={["quality_engineer"]}>
            <Button
              variant="danger"
              size="sm"
              leftIcon={<Trash2 size={14} />}
              onClick={() => setIsDeleteModalOpen(true)}
            >
              Delete Inspection
            </Button>
          </RoleGate>
        )}
      </div>

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
          <div className={styles.imageColumn}>
            <InspectionStatusBanner status={inspection.status} />
            <Card className={styles.imageCard}>
              {imageUrl && <ImageViewer src={imageUrl} alt={`Inspection ${inspection.id} capture`} />}
              {!imageUrl && !imageError && <Skeleton variant="block" height={420} />}
              {imageError && (
                <div className={styles.imageError}>
                  <ImageOff size={24} />
                  <p>{imageError}</p>
                </div>
              )}
            </Card>
          </div>

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
                  <dt>Defect Category</dt>
                  <dd>
                    <Badge tone={defectCategoryTone(inspection.defect_category)}>
                      {defectCategoryLabel(inspection.defect_category)}
                    </Badge>
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
              <h2 className={styles.metaCardTitle}>AI Prediction</h2>
              {inspection.ai_prediction ? (
                <>
                  <p className={styles.aiCaption}>AI-generated result, independent of inspection status.</p>
                  <dl className={styles.metaList}>
                    <div className={styles.metaRow}>
                      <dt>AI Prediction</dt>
                      <dd>
                        <Badge tone={aiPredictionTone(inspection.ai_prediction)}>
                          {aiPredictionLabel(inspection.ai_prediction)}
                        </Badge>
                      </dd>
                    </div>
                    <div className={styles.metaRow}>
                      <dt>Reconstruction error</dt>
                      <dd className={styles.mono}>{inspection.ai_reconstruction_error.toFixed(6)}</dd>
                    </div>
                    <div className={styles.metaRow}>
                      <dt>Threshold</dt>
                      <dd className={styles.mono}>{inspection.ai_threshold.toFixed(6)}</dd>
                    </div>
                    <div className={styles.metaRow}>
                      <dt>Model</dt>
                      <dd className={styles.mono}>{inspection.ai_model_name}</dd>
                    </div>
                  </dl>
                </>
              ) : (
                <p className={styles.metaFallback}>Not yet analyzed</p>
              )}
            </Card>

            <Card className={styles.metaCard}>
              <h2 className={styles.metaCardTitle}>Severity Assessment</h2>
              {Number.isFinite(inspection.severity_score) ? (
                <>
                  <p className={styles.aiCaption}>
                    Deterministic score from currently available defect evidence.
                  </p>
                  <dl className={styles.metaList}>
                    <div className={styles.metaRow}>
                      <dt>Score</dt>
                      <dd className={styles.mono}>{Math.round(inspection.severity_score)} / 100</dd>
                    </div>
                    <div className={styles.metaRow}>
                      <dt>Level</dt>
                      <dd>
                        <Badge tone={severityTone(inspection.severity_level)}>
                          {severityLabel(inspection.severity_level)}
                        </Badge>
                      </dd>
                    </div>
                    <div className={styles.metaRow}>
                      <dt>Quality Risk</dt>
                      <dd>
                        <Badge tone={qualityRiskTone(inspection.quality_risk)}>
                          {qualityRiskLabel(inspection.quality_risk)}
                        </Badge>
                      </dd>
                    </div>
                  </dl>
                </>
              ) : (
                <p className={styles.metaFallback}>Not assessed - insufficient evidence available</p>
              )}
            </Card>

            <Card className={styles.metaCard}>
              <h2 className={styles.metaCardTitle}>Quality Assessment</h2>
              <p className={styles.aiCaption}>
                Deterministic decision derived from available inspection evidence.
              </p>
              <dl className={styles.metaList}>
                <div className={styles.metaRow}>
                  <dt>Decision</dt>
                  <dd>
                    <Badge tone={qualityDecisionTone(inspection.quality_decision)}>
                      {qualityDecisionLabel(inspection.quality_decision)}
                    </Badge>
                  </dd>
                </div>
                {inspection.quality_assessment && (
                  <div className={styles.metaRow}>
                    <dt>Assessment</dt>
                    <dd>{inspection.quality_assessment}</dd>
                  </div>
                )}
                {inspection.quality_recommendation && (
                  <div className={styles.metaRow}>
                    <dt>Recommendation</dt>
                    <dd>{inspection.quality_recommendation}</dd>
                  </div>
                )}
              </dl>
            </Card>

            {inspection.source === "mvtec_ad" && (
              <Card className={styles.metaCard}>
                <h2 className={styles.metaCardTitle}>Dataset reference</h2>
                <dl className={styles.metaList}>
                  <div className={styles.metaRow}>
                    <dt>Category</dt>
                    <dd className={styles.mono}>{inspection.dataset_category}</dd>
                  </div>
                  <div className={styles.metaRow}>
                    <dt>Split</dt>
                    <dd className={styles.mono}>{inspection.dataset_split}</dd>
                  </div>
                  <div className={styles.metaRow}>
                    <dt>Defect type</dt>
                    <dd className={styles.mono}>{inspection.dataset_defect_type}</dd>
                  </div>
                  <div className={styles.metaRow}>
                    <dt>Filename</dt>
                    <dd className={styles.mono}>{inspection.dataset_filename}</dd>
                  </div>
                </dl>
              </Card>
            )}

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

      {inspection && (
        <Modal
          open={isDeleteModalOpen}
          onClose={closeDeleteModal}
          title={`Delete inspection #${inspection.id}?`}
          description={
            inspection.source === "mvtec_ad"
              ? "This removes the inspection record only. The original MVTec dataset image is not deleted and will remain available for future imports."
              : "This permanently removes the inspection record and its uploaded image. This action cannot be undone."
          }
          footer={
            <>
              <Button variant="ghost" onClick={closeDeleteModal} disabled={isDeleting}>
                Cancel
              </Button>
              <Button variant="danger" onClick={handleDelete} loading={isDeleting}>
                Delete Inspection
              </Button>
            </>
          }
        >
          <p className={styles.deleteConfirmText}>
            Product: <strong>{product ? product.product_name : `#${inspection.product_id}`}</strong>
          </p>
        </Modal>
      )}
    </div>
  );
}
