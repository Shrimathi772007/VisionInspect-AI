import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Search, ScanEye, UploadCloud, ChevronRight, Trash2 } from "lucide-react";
import { useInspections } from "../hooks/useInspections";
import { useProducts } from "../hooks/useProducts";
import { deleteInspection } from "../api/inspections";
import { ApiError } from "../api/client";
import { useToast } from "../components/Toast/ToastProvider";
import { PageHeader } from "../components/PageHeader/PageHeader";
import { Card } from "../components/Card/Card";
import { Button } from "../components/Button/Button";
import { Input } from "../components/Input/Input";
import { Badge } from "../components/Badge/Badge";
import { Modal } from "../components/Modal/Modal";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { RoleGate } from "../components/RoleGate/RoleGate";
import { statusLabel, statusTone, sourceLabel, sourceTone } from "../utils/badgeMaps";
import { formatDateTime } from "../utils/formatDate";
import styles from "./InspectionsPage.module.css";

export function InspectionsPage() {
  const { inspections, isLoading, error, refetch } = useInspections();
  const { getProductById } = useProducts();
  const { showToast } = useToast();
  const [searchTerm, setSearchTerm] = useState("");
  const [pendingDelete, setPendingDelete] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const filteredInspections = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    if (!term) return inspections;
    return inspections.filter((inspection) => {
      const product = getProductById(inspection.product_id);
      return (
        product?.product_name.toLowerCase().includes(term) ||
        product?.product_code.toLowerCase().includes(term) ||
        String(inspection.id).includes(term)
      );
    });
  }, [inspections, searchTerm, getProductById]);

  const requestDelete = (event, inspection) => {
    event.preventDefault();
    event.stopPropagation();
    setPendingDelete(inspection);
  };

  const closeDeleteModal = () => {
    if (isDeleting) return;
    setPendingDelete(null);
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    setIsDeleting(true);
    try {
      await deleteInspection(pendingDelete.id);
      showToast({
        type: "success",
        title: "Inspection deleted",
        message: `Inspection #${pendingDelete.id} was removed.`,
      });
      setPendingDelete(null);
      refetch();
    } catch (err) {
      showToast({
        type: "error",
        title: "Couldn't delete inspection",
        message: err instanceof ApiError ? err.message : "Something went wrong.",
      });
      if (err instanceof ApiError && err.status === 404) {
        setPendingDelete(null);
        refetch();
      }
    } finally {
      setIsDeleting(false);
    }
  };

  const pendingDeleteProduct = pendingDelete ? getProductById(pendingDelete.product_id) : null;

  return (
    <div>
      <PageHeader
        eyebrow="Quality records"
        title="Inspections"
        description="Every inspection image captured across your product lines."
        actions={
          <RoleGate allow={["quality_engineer"]}>
            <Button as={Link} to="/inspections/upload" leftIcon={<UploadCloud size={16} />}>
              Upload Inspection
            </Button>
          </RoleGate>
        }
      />

      {inspections.length > 0 && (
        <div className={styles.searchRow}>
          <Input
            placeholder="Search by product name, code, or inspection ID..."
            leftIcon={<Search size={16} />}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            aria-label="Search inspections"
          />
        </div>
      )}

      {isLoading && (
        <Card className={styles.listCard}>
          {[1, 2, 3, 4].map((key) => (
            <div key={key} className={styles.row}>
              <Skeleton variant="circle" width={38} height={38} />
              <div style={{ flex: 1 }}>
                <Skeleton width="40%" height={13} />
                <Skeleton width="25%" height={11} style={{ marginTop: 6 }} />
              </div>
              <Skeleton width={70} height={22} />
            </div>
          ))}
        </Card>
      )}

      {!isLoading && error && (
        <Card className={styles.errorCard}>
          <p>Failed to load inspections. {error.message}</p>
          <Button size="sm" variant="secondary" onClick={refetch}>
            Retry
          </Button>
        </Card>
      )}

      {!isLoading && !error && inspections.length === 0 && (
        <Card>
          <EmptyState
            icon={ScanEye}
            title="No inspections yet"
            description="Inspection images captured for your products will appear here."
            action={
              <RoleGate allow={["quality_engineer"]}>
                <Button as={Link} to="/inspections/upload" leftIcon={<UploadCloud size={16} />}>
                  Upload Inspection
                </Button>
              </RoleGate>
            }
          />
        </Card>
      )}

      {!isLoading && !error && inspections.length > 0 && filteredInspections.length === 0 && (
        <Card>
          <EmptyState icon={Search} title="No matches" description={`No inspections match "${searchTerm}".`} />
        </Card>
      )}

      {!isLoading && !error && filteredInspections.length > 0 && (
        <Card className={styles.listCard}>
          {filteredInspections.map((inspection) => {
            const product = getProductById(inspection.product_id);
            return (
              <Link key={inspection.id} to={`/inspections/${inspection.id}`} className={styles.row}>
                <div className={styles.icon}>
                  <ScanEye size={17} />
                </div>
                <div className={styles.info}>
                  <p className={styles.productName}>
                    {product ? product.product_name : `Product #${inspection.product_id}`}
                    <span className={styles.inspectionId}>#{inspection.id}</span>
                  </p>
                  <p className={styles.meta}>
                    {product?.product_code} &middot; {formatDateTime(inspection.created_at)}
                  </p>
                </div>
                <div className={styles.badges}>
                  <Badge tone={sourceTone(inspection.source)}>{sourceLabel(inspection.source)}</Badge>
                  <Badge tone={statusTone(inspection.status)}>{statusLabel(inspection.status)}</Badge>
                </div>
                <RoleGate allow={["quality_engineer"]}>
                  <button
                    type="button"
                    className={styles.deleteButton}
                    onClick={(event) => requestDelete(event, inspection)}
                    aria-label={`Delete inspection #${inspection.id}`}
                  >
                    <Trash2 size={15} />
                  </button>
                </RoleGate>
                <ChevronRight size={16} className={styles.chevron} aria-hidden="true" />
              </Link>
            );
          })}
        </Card>
      )}

      <Modal
        open={Boolean(pendingDelete)}
        onClose={closeDeleteModal}
        title={pendingDelete ? `Delete inspection #${pendingDelete.id}?` : ""}
        description={
          pendingDelete?.source === "mvtec_ad"
            ? "This removes the inspection record only. The original MVTec dataset image is not deleted."
            : "This permanently removes the inspection record and its uploaded image. This action cannot be undone."
        }
        footer={
          <>
            <Button variant="ghost" onClick={closeDeleteModal} disabled={isDeleting}>
              Cancel
            </Button>
            <Button variant="danger" onClick={confirmDelete} loading={isDeleting}>
              Delete Inspection
            </Button>
          </>
        }
      >
        {pendingDelete && (
          <p className={styles.deleteConfirmText}>
            Product:{" "}
            <strong>
              {pendingDeleteProduct ? pendingDeleteProduct.product_name : `#${pendingDelete.product_id}`}
            </strong>
          </p>
        )}
      </Modal>
    </div>
  );
}
