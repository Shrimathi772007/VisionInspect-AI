import { Link } from "react-router-dom";
import { Box, ScanEye, UploadCloud, ArrowRight, Clock } from "lucide-react";
import { useAuth } from "../auth/useAuth";
import { useProducts } from "../hooks/useProducts";
import { useInspections } from "../hooks/useInspections";
import { PageHeader } from "../components/PageHeader/PageHeader";
import { StatCard } from "../components/StatCard/StatCard";
import { Card } from "../components/Card/Card";
import { Badge } from "../components/Badge/Badge";
import { Button } from "../components/Button/Button";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { RoleGate } from "../components/RoleGate/RoleGate";
import { ActivityChart } from "../components/ActivityChart/ActivityChart";
import { statusLabel, statusTone } from "../utils/badgeMaps";
import { formatRelativeTime } from "../utils/formatDate";
import styles from "./DashboardPage.module.css";

const RECENT_COUNT = 5;
const WINDOW_MS = 7 * 24 * 60 * 60 * 1000;

function computeTrend(inspections, predicate) {
  const now = Date.now();
  let current = 0;
  let previous = 0;

  inspections.forEach((inspection) => {
    if (predicate && !predicate(inspection)) return;
    const age = now - new Date(inspection.created_at).getTime();
    if (age < 0) return;
    if (age <= WINDOW_MS) current += 1;
    else if (age <= WINDOW_MS * 2) previous += 1;
  });

  if (previous === 0) {
    if (current === 0) return { direction: "flat", label: "" };
    return { direction: "up", label: `+${current} this week` };
  }

  const change = Math.round(((current - previous) / previous) * 100);
  if (change === 0) return { direction: "flat", label: "" };
  return {
    direction: change > 0 ? "up" : "down",
    label: `${change > 0 ? "+" : ""}${change}% vs last week`,
  };
}

export function DashboardPage() {
  const { user } = useAuth();
  const { products, isLoading: productsLoading, getProductById } = useProducts();
  const { inspections, isLoading: inspectionsLoading } = useInspections();

  const recentInspections = inspections.slice(0, RECENT_COUNT);
  const firstName = user?.name?.split(" ")[0];

  const inspectionsTrend = computeTrend(inspections);
  const pendingTrend = computeTrend(inspections, (i) => i.status === "pending");

  return (
    <div>
      <PageHeader
        eyebrow="Overview"
        title={`Welcome back${firstName ? `, ${firstName}` : ""}`}
        description="Here's what's happening across your product lines and inspections."
        actions={
          <RoleGate allow={["quality_engineer"]}>
            <Button as={Link} to="/inspections/upload" leftIcon={<UploadCloud size={16} />}>
              Upload Inspection
            </Button>
          </RoleGate>
        }
      />

      <div className={styles.statsGrid}>
        <StatCard icon={Box} label="Products" value={products.length} loading={productsLoading} tone="accent" />
        <StatCard
          icon={ScanEye}
          label="Inspections"
          value={inspections.length}
          loading={inspectionsLoading}
          tone="info"
          trend={inspectionsLoading ? null : inspectionsTrend}
        />
        <StatCard
          icon={Clock}
          label="Pending review"
          value={inspections.filter((i) => i.status === "pending").length}
          loading={inspectionsLoading}
          tone="warning"
          trend={inspectionsLoading ? null : pendingTrend}
        />
      </div>

      <Card className={styles.activityCard}>
        <div className={styles.cardHeader}>
          <h2 className={styles.cardTitle}>Inspection activity</h2>
          <span className={styles.cardSubtitle}>Last 14 days</span>
        </div>
        {inspectionsLoading ? (
          <Skeleton variant="block" height={160} />
        ) : (
          <ActivityChart inspections={inspections} />
        )}
      </Card>

      <div className={styles.bottomGrid}>
        <Card className={styles.recentCard}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Recent inspections</h2>
            <Link to="/inspections" className={styles.viewAllLink}>
              View all <ArrowRight size={14} />
            </Link>
          </div>

          {inspectionsLoading && (
            <div className={styles.recentList}>
              {[1, 2, 3].map((key) => (
                <div key={key} className={styles.recentRow}>
                  <Skeleton variant="circle" width={36} height={36} />
                  <div style={{ flex: 1 }}>
                    <Skeleton width="60%" height={13} />
                    <Skeleton width="40%" height={11} style={{ marginTop: 6 }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {!inspectionsLoading && recentInspections.length === 0 && (
            <EmptyState
              icon={ScanEye}
              title="No inspections yet"
              description="Once inspection images are uploaded, they'll show up here."
              action={
                <RoleGate allow={["quality_engineer"]}>
                  <Button as={Link} to="/inspections/upload" size="sm" leftIcon={<UploadCloud size={14} />}>
                    Upload your first inspection
                  </Button>
                </RoleGate>
              }
            />
          )}

          {!inspectionsLoading && recentInspections.length > 0 && (
            <div className={styles.recentList}>
              {recentInspections.map((inspection) => {
                const product = getProductById(inspection.product_id);
                return (
                  <Link key={inspection.id} to={`/inspections/${inspection.id}`} className={styles.recentRow}>
                    <div className={styles.recentIcon}>
                      <ScanEye size={16} />
                    </div>
                    <div className={styles.recentInfo}>
                      <p className={styles.recentProduct}>
                        {product ? product.product_name : `Product #${inspection.product_id}`}
                      </p>
                      <p className={styles.recentMeta}>
                        {product?.product_code} &middot; {formatRelativeTime(inspection.created_at)}
                      </p>
                    </div>
                    <Badge tone={statusTone(inspection.status)}>{statusLabel(inspection.status)}</Badge>
                  </Link>
                );
              })}
            </div>
          )}
        </Card>

        <Card className={styles.quickCard}>
          <h2 className={styles.cardTitle}>Quick actions</h2>
          <div className={styles.quickActions}>
            <Link to="/products" className={styles.quickAction}>
              <div className={styles.quickActionIcon}>
                <Box size={17} />
              </div>
              <div>
                <p className={styles.quickActionTitle}>Manage products</p>
                <p className={styles.quickActionDesc}>View and register product lines</p>
              </div>
            </Link>
            <RoleGate allow={["quality_engineer"]}>
              <Link to="/inspections/upload" className={styles.quickAction}>
                <div className={styles.quickActionIcon}>
                  <UploadCloud size={17} />
                </div>
                <div>
                  <p className={styles.quickActionTitle}>Upload inspection</p>
                  <p className={styles.quickActionDesc}>Submit a new inspection image</p>
                </div>
              </Link>
            </RoleGate>
            <Link to="/inspections" className={styles.quickAction}>
              <div className={styles.quickActionIcon}>
                <ScanEye size={17} />
              </div>
              <div>
                <p className={styles.quickActionTitle}>Browse inspections</p>
                <p className={styles.quickActionDesc}>Review inspection history</p>
              </div>
            </Link>
          </div>
        </Card>
      </div>
    </div>
  );
}
