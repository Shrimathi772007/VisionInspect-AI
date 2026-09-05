import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Box, CheckCircle2, XCircle, ScanEye, ArrowRight, AlertCircle } from "lucide-react";
import { Modal } from "../Modal/Modal";
import { Card } from "../Card/Card";
import { Badge } from "../Badge/Badge";
import { Button } from "../Button/Button";
import { Skeleton } from "../Skeleton/Skeleton";
import { EmptyState } from "../EmptyState/EmptyState";
import { statusLabel, statusTone, sourceLabel, sourceTone } from "../../utils/badgeMaps";
import { formatDateTime } from "../../utils/formatDate";
import styles from "./ProductDetailModal.module.css";

function computeStats(inspections) {
  const total = inspections.length;
  const good = inspections.filter((i) => i.status === "good").length;
  const defective = inspections.filter((i) => i.status === "defective").length;
  const goodPct = total ? Math.round((good / total) * 100) : 0;
  const defectivePct = total ? Math.round((defective / total) * 100) : 0;
  return { total, good, defective, goodPct, defectivePct };
}

export function ProductDetailModal({ product, inspections, isLoading, error, open, onClose }) {
  const navigate = useNavigate();
  const stats = useMemo(() => computeStats(inspections), [inspections]);

  const handleViewInspection = (id) => {
    onClose();
    navigate(`/inspections/${id}`);
  };

  if (!product) return null;

  return (
    <Modal
      open={open}
      onClose={onClose}
      size="lg"
      title={product.product_name}
      description={`${product.product_code} · Created ${formatDateTime(product.created_at)}`}
    >
      <div className={styles.statsGrid}>
        <Card className={styles.statTile}>
          <div className={`${styles.statIcon} ${styles.accent}`}>
            <Box size={18} strokeWidth={1.75} aria-hidden="true" />
          </div>
          <div>
            <p className={styles.statLabel}>Total Inspections</p>
            <p className={styles.statValue}>{stats.total}</p>
          </div>
        </Card>
        <Card className={styles.statTile}>
          <div className={`${styles.statIcon} ${styles.success}`}>
            <CheckCircle2 size={18} strokeWidth={1.75} aria-hidden="true" />
          </div>
          <div>
            <p className={styles.statLabel}>Good</p>
            <p className={styles.statValue}>
              {stats.good} <span className={styles.statPct}>({stats.goodPct}%)</span>
            </p>
          </div>
        </Card>
        <Card className={styles.statTile}>
          <div className={`${styles.statIcon} ${styles.danger}`}>
            <XCircle size={18} strokeWidth={1.75} aria-hidden="true" />
          </div>
          <div>
            <p className={styles.statLabel}>Defective</p>
            <p className={styles.statValue}>
              {stats.defective} <span className={styles.statPct}>({stats.defectivePct}%)</span>
            </p>
          </div>
        </Card>
      </div>

      {stats.total > 0 && (
        <div className={styles.ratioBar} role="img" aria-label={`${stats.goodPct}% good, ${stats.defectivePct}% defective`}>
          {stats.goodPct > 0 && <div className={styles.ratioGood} style={{ width: `${stats.goodPct}%` }} />}
          {stats.defectivePct > 0 && <div className={styles.ratioDefective} style={{ width: `${stats.defectivePct}%` }} />}
        </div>
      )}

      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>Recent Inspections</h3>
      </div>

      {isLoading && (
        <div className={styles.list}>
          {[1, 2, 3].map((key) => (
            <div key={key} className={styles.row}>
              <Skeleton variant="circle" width={32} height={32} />
              <div style={{ flex: 1 }}>
                <Skeleton width="50%" height={12} />
                <Skeleton width="30%" height={10} style={{ marginTop: 6 }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {!isLoading && error && (
        <div className={styles.errorState}>
          <AlertCircle size={18} aria-hidden="true" />
          <p>Failed to load inspections. {error.message}</p>
        </div>
      )}

      {!isLoading && !error && inspections.length === 0 && (
        <EmptyState
          icon={ScanEye}
          title="No inspections yet"
          description="Inspections captured for this product will appear here."
        />
      )}

      {!isLoading && !error && inspections.length > 0 && (
        <div className={styles.list}>
          {inspections.map((inspection) => (
            <div key={inspection.id} className={styles.row}>
              <div className={styles.rowIcon}>
                <ScanEye size={15} aria-hidden="true" />
              </div>
              <div className={styles.rowInfo}>
                <p className={styles.rowTitle}>
                  Inspection <span className={styles.mono}>#{inspection.id}</span>
                </p>
                <p className={styles.rowMeta}>
                  {formatDateTime(inspection.inspection_date)}
                  {inspection.dataset_category && (
                    <>
                      {" "}
                      · <span className={styles.mono}>{inspection.dataset_category}</span>
                      {inspection.dataset_defect_type && <span className={styles.mono}>/{inspection.dataset_defect_type}</span>}
                    </>
                  )}
                </p>
              </div>
              <div className={styles.rowBadges}>
                <Badge tone={sourceTone(inspection.source)}>{sourceLabel(inspection.source)}</Badge>
                <Badge tone={statusTone(inspection.status)}>{statusLabel(inspection.status)}</Badge>
              </div>
              <Button
                size="sm"
                variant="ghost"
                rightIcon={<ArrowRight size={13} />}
                onClick={() => handleViewInspection(inspection.id)}
              >
                View
              </Button>
            </div>
          ))}
        </div>
      )}
    </Modal>
  );
}
