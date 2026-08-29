import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Search, ScanEye, UploadCloud, ChevronRight } from "lucide-react";
import { useInspections } from "../hooks/useInspections";
import { useProducts } from "../hooks/useProducts";
import { PageHeader } from "../components/PageHeader/PageHeader";
import { Card } from "../components/Card/Card";
import { Button } from "../components/Button/Button";
import { Input } from "../components/Input/Input";
import { Badge } from "../components/Badge/Badge";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { RoleGate } from "../components/RoleGate/RoleGate";
import { statusLabel, statusTone, sourceLabel, sourceTone } from "../utils/badgeMaps";
import { formatDateTime } from "../utils/formatDate";
import styles from "./InspectionsPage.module.css";

export function InspectionsPage() {
  const { inspections, isLoading, error, refetch } = useInspections();
  const { getProductById } = useProducts();
  const [searchTerm, setSearchTerm] = useState("");

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
                <ChevronRight size={16} className={styles.chevron} aria-hidden="true" />
              </Link>
            );
          })}
        </Card>
      )}
    </div>
  );
}
