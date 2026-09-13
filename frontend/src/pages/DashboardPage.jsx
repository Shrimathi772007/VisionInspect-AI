import { Link } from "react-router-dom";
import {
  Box,
  ScanEye,
  UploadCloud,
  ArrowRight,
  Clock,
  Cpu,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Layers,
  ShieldCheck,
  Gauge,
  Lightbulb,
} from "lucide-react";
import { useAuth } from "../auth/useAuth";
import { useProducts } from "../hooks/useProducts";
import { useInspections } from "../hooks/useInspections";
import { useAnalyticsSummary } from "../hooks/useAnalyticsSummary";
import { PageHeader } from "../components/PageHeader/PageHeader";
import { StatCard } from "../components/StatCard/StatCard";
import { Card } from "../components/Card/Card";
import { Badge } from "../components/Badge/Badge";
import { Button } from "../components/Button/Button";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { RoleGate } from "../components/RoleGate/RoleGate";
import { ActivityChart } from "../components/ActivityChart/ActivityChart";
import { DistributionBar } from "../components/DistributionBar/DistributionBar";
import {
  statusLabel,
  statusTone,
  defectCategoryLabel,
  defectCategoryTone,
  qualityDecisionLabel,
  qualityDecisionTone,
  severityLabel,
  severityTone,
} from "../utils/badgeMaps";
import { formatRelativeTime } from "../utils/formatDate";
import styles from "./DashboardPage.module.css";

const MAX_VISIBLE_PRODUCTS = 6;

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

function AnalyticsErrorBlock({ onRetry }) {
  return (
    <div className={styles.aiError}>
      <p>Analytics data unavailable.</p>
      <Button size="sm" variant="secondary" onClick={onRetry}>
        Retry
      </Button>
    </div>
  );
}

export function DashboardPage() {
  const { user } = useAuth();
  const { products, isLoading: productsLoading, getProductById } = useProducts();
  const { inspections, isLoading: inspectionsLoading } = useInspections();
  const { data: analytics, isLoading: analyticsLoading, error: analyticsError, refetch: refetchAnalytics } =
    useAnalyticsSummary();

  const recentInspections = inspections.slice(0, RECENT_COUNT);
  const aiAnalyzedCount = analytics?.ai_analyzed_count ?? 0;
  const aiDefectRate = analytics?.ai_defect_rate;
  const aiGoodCount = analytics?.ai_prediction_counts?.good ?? 0;
  const aiDefectiveCount = analytics?.ai_prediction_counts?.defective ?? 0;
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
        <StatCard
          icon={Cpu}
          label="AI analyzed"
          value={aiAnalyzedCount}
          loading={analyticsLoading}
          tone="info"
        />
        <StatCard
          icon={AlertTriangle}
          label="AI defect rate"
          value={aiDefectRate == null ? "No data" : `${(aiDefectRate * 100).toFixed(1)}%`}
          loading={analyticsLoading}
          tone="warning"
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

      <Card className={styles.aiCard}>
        <div className={styles.cardHeader}>
          <h2 className={styles.cardTitle}>AI prediction distribution</h2>
          <span className={styles.cardSubtitle}>AI-analyzed inspections only</span>
        </div>

        {analyticsLoading && <Skeleton variant="block" height={56} />}

        {!analyticsLoading && analyticsError && (
          <div className={styles.aiError}>
            <p>AI monitoring data unavailable.</p>
            <Button size="sm" variant="secondary" onClick={refetchAnalytics}>
              Retry
            </Button>
          </div>
        )}

        {!analyticsLoading && !analyticsError && aiAnalyzedCount === 0 && (
          <EmptyState
            icon={Cpu}
            title="No AI predictions yet"
            description="AI predictions appear here once inspections for a supported product category (currently bottle) are analyzed."
          />
        )}

        {!analyticsLoading && !analyticsError && aiAnalyzedCount > 0 && (
          <div className={styles.aiDistribution}>
            <div className={styles.aiBar}>
              <div
                className={styles.aiBarGood}
                style={{ width: `${(aiGoodCount / aiAnalyzedCount) * 100}%` }}
              />
              <div
                className={styles.aiBarDefective}
                style={{ width: `${(aiDefectiveCount / aiAnalyzedCount) * 100}%` }}
              />
            </div>
            <div className={styles.aiLegend}>
              <Badge tone="info">AI Good &middot; {aiGoodCount}</Badge>
              <Badge tone="warning">AI Defective &middot; {aiDefectiveCount}</Badge>
            </div>
          </div>
        )}
      </Card>

      <div className={styles.analyticsSectionHeader}>
        <h2 className={styles.analyticsSectionTitle}>Manufacturing Quality Analytics</h2>
        <p className={styles.analyticsSectionSubtitle}>
          Defect categorization, quality decisions, and severity insights across all recorded inspections.
        </p>
      </div>

      <div className={styles.statsGrid}>
        <StatCard
          icon={CheckCircle2}
          label="Good (ground truth)"
          value={analytics?.by_status?.good ?? 0}
          loading={analyticsLoading}
          tone="success"
        />
        <StatCard
          icon={XCircle}
          label="Defective (ground truth)"
          value={analytics?.by_status?.defective ?? 0}
          loading={analyticsLoading}
          tone="danger"
        />
        <StatCard
          icon={Clock}
          label="Pending (no ground truth yet)"
          value={analytics?.by_status?.pending ?? 0}
          loading={analyticsLoading}
          tone="warning"
        />
      </div>

      <div className={styles.analyticsGrid}>
        <Card className={styles.analyticsCard}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Defect Category Distribution</h2>
            <span className={styles.cardSubtitle}>MVTec ground truth</span>
          </div>
          {analyticsLoading && <Skeleton variant="block" height={56} />}
          {!analyticsLoading && analyticsError && <AnalyticsErrorBlock onRetry={refetchAnalytics} />}
          {!analyticsLoading && !analyticsError && (
            <DistributionBar
              segments={(analytics?.defect_categories ?? []).map((entry) => ({
                key: entry.category ?? "uncategorized",
                label: defectCategoryLabel(entry.category),
                count: entry.count,
                tone: defectCategoryTone(entry.category),
              }))}
              emptyIcon={Layers}
              emptyTitle="No categorized defects"
              emptyDescription="Defect categories appear here once MVTec inspections are imported."
            />
          )}
        </Card>

        <Card className={styles.analyticsCard}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Quality Decision Distribution</h2>
            <span className={styles.cardSubtitle}>Phase 3 quality engine</span>
          </div>
          {analyticsLoading && <Skeleton variant="block" height={56} />}
          {!analyticsLoading && analyticsError && <AnalyticsErrorBlock onRetry={refetchAnalytics} />}
          {!analyticsLoading && !analyticsError && (
            <DistributionBar
              segments={(analytics?.quality_decisions ?? []).map((entry) => ({
                key: entry.decision,
                label: qualityDecisionLabel(entry.decision),
                count: entry.count,
                tone: qualityDecisionTone(entry.decision),
              }))}
              emptyIcon={ShieldCheck}
              emptyTitle="No quality decisions yet"
              emptyDescription="Quality decisions appear here once inspections are recorded."
            />
          )}
        </Card>

        <Card className={styles.analyticsCard}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Severity &amp; Risk Distribution</h2>
            <span className={styles.cardSubtitle}>Phase 2 severity engine</span>
          </div>
          {analyticsLoading && <Skeleton variant="block" height={56} />}
          {!analyticsLoading && analyticsError && <AnalyticsErrorBlock onRetry={refetchAnalytics} />}
          {!analyticsLoading && !analyticsError && (
            <DistributionBar
              segments={(analytics?.severity_distribution ?? []).map((entry) => ({
                key: entry.level,
                label: severityLabel(entry.level),
                count: entry.count,
                tone: severityTone(entry.level),
              }))}
              emptyIcon={Gauge}
              emptyTitle="No severity data"
              emptyDescription="Severity is assessed once all required evidence is available for an inspection."
            />
          )}
        </Card>

        <Card className={styles.analyticsCard}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Product Quality Overview</h2>
            <span className={styles.cardSubtitle}>Ground-truth defect rate</span>
          </div>
          {analyticsLoading && (
            <div className={styles.recentList}>
              {[1, 2, 3].map((key) => (
                <Skeleton key={key} height={16} width="80%" style={{ marginBottom: 10 }} />
              ))}
            </div>
          )}
          {!analyticsLoading && analyticsError && <AnalyticsErrorBlock onRetry={refetchAnalytics} />}
          {!analyticsLoading && !analyticsError && (analytics?.by_product?.length ?? 0) === 0 && (
            <EmptyState
              icon={Box}
              title="No product data yet"
              description="Product quality statistics appear once inspections are recorded."
            />
          )}
          {!analyticsLoading && !analyticsError && (analytics?.by_product?.length ?? 0) > 0 && (
            <div className={styles.productList}>
              {analytics.by_product.slice(0, MAX_VISIBLE_PRODUCTS).map((product) => (
                <div key={product.product_id} className={styles.productRow}>
                  <p className={styles.productRowName}>{product.product_name}</p>
                  <div className={styles.productRowStats}>
                    <span>{product.total} total</span>
                    <span>{product.defective} defective</span>
                    <Badge tone={product.defective > 0 ? "danger" : "success"}>
                      {(product.defect_rate * 100).toFixed(1)}%
                    </Badge>
                  </div>
                </div>
              ))}
              {analytics.by_product.length > MAX_VISIBLE_PRODUCTS && (
                <p className={styles.productListNote}>
                  Showing the {MAX_VISIBLE_PRODUCTS} highest-volume products of {analytics.by_product.length}.
                </p>
              )}
            </div>
          )}
        </Card>
      </div>

      <Card className={styles.insightsCard}>
        <div className={styles.cardHeader}>
          <h2 className={styles.cardTitle}>Operational Insights</h2>
          <span className={styles.cardSubtitle}>Deterministic, evidence-based observations</span>
        </div>
        {analyticsLoading && <Skeleton variant="block" height={56} />}
        {!analyticsLoading && analyticsError && <AnalyticsErrorBlock onRetry={refetchAnalytics} />}
        {!analyticsLoading && !analyticsError && (analytics?.operational_insights?.length ?? 0) === 0 && (
          <EmptyState
            icon={Lightbulb}
            title="No insights yet"
            description="Insights appear once there is enough inspection data to summarize."
          />
        )}
        {!analyticsLoading && !analyticsError && (analytics?.operational_insights?.length ?? 0) > 0 && (
          <ul className={styles.insightsList}>
            {analytics.operational_insights.map((insight) => (
              <li key={insight.type} className={styles.insightItem}>
                <Lightbulb size={14} strokeWidth={1.75} aria-hidden="true" />
                <span>{insight.message}</span>
              </li>
            ))}
          </ul>
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
