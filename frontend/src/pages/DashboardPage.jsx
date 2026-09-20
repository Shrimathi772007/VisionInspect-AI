import { useState } from "react";
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
  Activity,
  History,
  TrendingUp,
  TrendingDown,
  Minus,
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
import { ErrorState, Unavailable } from "../components/ErrorState/ErrorState";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { RoleGate } from "../components/RoleGate/RoleGate";
import { RangeSelector } from "../components/RangeSelector/RangeSelector";
import { PerformanceOverview } from "../components/PerformanceOverview/PerformanceOverview";
import { ActivityChart } from "../components/ActivityChart/ActivityChart";
import { DistributionBar } from "../components/DistributionBar/DistributionBar";
import { TrendChart } from "../components/TrendChart/TrendChart";
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

// The dashboard only ever shows the newest few inspections, so it asks the backend for exactly
// that many (a SQL LIMIT) instead of downloading every inspection ever recorded.
const RECENT_COUNT = 5;

// Trailing windows the backend supports for the time-windowed sections (see
// app.inspections.analytics.ALLOWED_WINDOW_DAYS); 14 is its default.
const DEFAULT_WINDOW_DAYS = 14;
const WINDOW_OPTIONS = [
  { value: 7, label: "7 days" },
  { value: 14, label: "14 days" },
  { value: 30, label: "30 days" },
];

const DAYS_PER_WEEK = 7;

// Milestone 3 Phase 6 - category-trend bars need a distinct tone per tracked category.
// "success" is deliberately excluded here: these are always real DEFECT categories (the
// backend already excludes "good"), so a success-green bar would misleadingly read as
// "no defect". Cycles if there are ever more than 5 (matches MAX_CATEGORY_TRENDS today).
const CATEGORY_TREND_TONES = ["danger", "warning", "info", "accent", "neutral"];

// Reshapes Phase 6's category_trends ([{ category, daily: [{date, count}] }]) into the
// per-day, per-series-key rows TrendChart expects ([{ date, [category]: count }]) - a
// presentation-only pivot, no new data.
function buildCategoryTrendDays(categoryTrends) {
  if (categoryTrends.length === 0) return [];
  const days = categoryTrends[0].daily.map((point) => ({ date: point.date }));
  categoryTrends.forEach((trend) => {
    trend.daily.forEach((point, index) => {
      days[index][trend.category] = point.count;
    });
  });
  return days;
}

// Week-over-week chip for a stat card, from the backend's own per-day activity series
// (oldest -> newest): the newest 7 days against the 7 before. Returns null - no chip - when the
// series is too short to compare, rather than guessing.
function computeWeeklyTrend(activityDays, key) {
  if (!activityDays || activityDays.length < DAYS_PER_WEEK * 2) return null;

  const sum = (days) => days.reduce((total, day) => total + (day[key] || 0), 0);
  const current = sum(activityDays.slice(-DAYS_PER_WEEK));
  const previous = sum(activityDays.slice(-DAYS_PER_WEEK * 2, -DAYS_PER_WEEK));

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

const errorMessage = (error, fallback) => (error && error.message ? error.message : fallback);

export function DashboardPage() {
  const { user } = useAuth();
  const [days, setDays] = useState(DEFAULT_WINDOW_DAYS);

  // Three independent data sources: each has its own loading and error state, and each section
  // below depends only on the source(s) it actually uses.
  const { products, isLoading: productsLoading, error: productsError, getProductById } = useProducts();
  const {
    inspections: recentInspections,
    isLoading: recentLoading,
    error: recentError,
    refetch: refetchRecent,
  } = useInspections({ limit: RECENT_COUNT });
  const {
    data: analytics,
    isLoading: analyticsLoading,
    isRefreshing: analyticsRefreshing,
    error: analyticsError,
    refetch: refetchAnalytics,
  } = useAnalyticsSummary({ days });

  const analyticsFailed = !analyticsLoading && Boolean(analyticsError);
  const analyticsReady = !analyticsLoading && !analyticsError && Boolean(analytics);

  const aiAnalyzedCount = analytics?.ai_analyzed_count ?? 0;
  const aiDefectRate = analytics?.ai_defect_rate;
  const aiGoodCount = analytics?.ai_prediction_counts?.good ?? 0;
  const aiDefectiveCount = analytics?.ai_prediction_counts?.defective ?? 0;
  const firstName = user?.name?.split(" ")[0];

  const activityDays = analytics?.activity_by_day ?? [];
  const inspectionsTrend = computeWeeklyTrend(activityDays, "total");
  const pendingTrend = computeWeeklyTrend(activityDays, "pending");

  const trendMonitoring = analytics?.trend_monitoring;
  const trendDaily = trendMonitoring?.daily ?? [];
  const categoryTrends = trendMonitoring?.category_trends ?? [];
  const categoryTrendDays = buildCategoryTrendDays(categoryTrends);
  const categoryTrendSeries = categoryTrends.map((trend, index) => ({
    key: trend.category,
    label: defectCategoryLabel(trend.category),
    tone: CATEGORY_TREND_TONES[index % CATEGORY_TREND_TONES.length],
  }));
  const trendInsights = trendMonitoring?.insights ?? [];
  const trendPeriodDays = trendMonitoring?.period_days ?? days;

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

      {analyticsFailed && (
        <ErrorState
          variant="banner"
          title="Analytics couldn't be loaded"
          message={errorMessage(analyticsError, "Something went wrong while loading analytics.")}
          onRetry={refetchAnalytics}
        />
      )}

      <div className={styles.statsGrid}>
        <StatCard
          icon={Box}
          label="Products"
          value={products.length}
          loading={productsLoading}
          error={Boolean(productsError)}
          note="Registered"
          tone="accent"
        />
        <StatCard
          icon={ScanEye}
          label="Inspections"
          value={analytics?.total_inspections ?? 0}
          loading={analyticsLoading}
          error={analyticsFailed}
          note="All time"
          tone="info"
          trend={analyticsReady ? inspectionsTrend : null}
        />
        <StatCard
          icon={Clock}
          label="Pending review"
          value={analytics?.by_status?.pending ?? 0}
          loading={analyticsLoading}
          error={analyticsFailed}
          note="No ground truth yet"
          tone="warning"
          trend={analyticsReady ? pendingTrend : null}
        />
        <StatCard
          icon={Cpu}
          label="AI analyzed"
          value={aiAnalyzedCount}
          loading={analyticsLoading}
          error={analyticsFailed}
          note="All time"
          tone="info"
        />
        <StatCard
          icon={AlertTriangle}
          label="AI defect rate"
          value={aiDefectRate == null ? "No data" : `${(aiDefectRate * 100).toFixed(1)}%`}
          loading={analyticsLoading}
          error={analyticsFailed}
          note="Of AI-analyzed inspections"
          tone="warning"
        />
      </div>

      <Card className={styles.activityCard}>
        <div className={styles.cardHeader}>
          <h2 className={styles.cardTitle}>Inspection activity</h2>
          <span className={styles.cardSubtitle}>Last {activityDays.length || 14} days</span>
        </div>
        {analyticsLoading && <Skeleton variant="block" height={160} />}
        {analyticsFailed && <Unavailable />}
        {analyticsReady && <ActivityChart days={activityDays} />}
      </Card>

      <Card className={styles.aiCard}>
        <div className={styles.cardHeader}>
          <h2 className={styles.cardTitle}>AI prediction distribution</h2>
          <span className={styles.cardSubtitle}>AI-analyzed inspections only</span>
        </div>

        {analyticsLoading && <Skeleton variant="block" height={56} />}

        {analyticsFailed && <Unavailable />}

        {analyticsReady && aiAnalyzedCount === 0 && (
          <EmptyState
            icon={Cpu}
            title="No AI predictions yet"
            description="AI predictions appear here once inspections for a supported product category (currently bottle) are analyzed."
          />
        )}

        {analyticsReady && aiAnalyzedCount > 0 && (
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
          error={analyticsFailed}
          tone="success"
        />
        <StatCard
          icon={XCircle}
          label="Defective (ground truth)"
          value={analytics?.by_status?.defective ?? 0}
          loading={analyticsLoading}
          error={analyticsFailed}
          tone="danger"
        />
        <StatCard
          icon={Clock}
          label="Pending (no ground truth yet)"
          value={analytics?.by_status?.pending ?? 0}
          loading={analyticsLoading}
          error={analyticsFailed}
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
          {analyticsFailed && <Unavailable />}
          {analyticsReady && (
            <DistributionBar
              segments={(analytics.defect_categories ?? []).map((entry) => ({
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
          {analyticsFailed && <Unavailable />}
          {analyticsReady && (
            <DistributionBar
              segments={(analytics.quality_decisions ?? []).map((entry) => ({
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
          {analyticsFailed && <Unavailable />}
          {analyticsReady && (
            <DistributionBar
              segments={(analytics.severity_distribution ?? []).map((entry) => ({
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
          {analyticsFailed && <Unavailable />}
          {analyticsReady && (analytics.by_product?.length ?? 0) === 0 && (
            <EmptyState
              icon={Box}
              title="No product data yet"
              description="Product quality statistics appear once inspections are recorded."
            />
          )}
          {analyticsReady && (analytics.by_product?.length ?? 0) > 0 && (
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
                    {/* Milestone 3 Phase 6 - recent (last trendPeriodDays) trend, independent
                        of the all-time defect_rate badge above. Omitted (no icon) rather than
                        guessed when there isn't enough recent history to classify a direction. */}
                    {product.recent_trend === "up" && (
                      <TrendingUp size={14} className={styles.trendUp} aria-label="Recent defect rate increasing" />
                    )}
                    {product.recent_trend === "down" && (
                      <TrendingDown size={14} className={styles.trendDown} aria-label="Recent defect rate decreasing" />
                    )}
                    {product.recent_trend === "stable" && (
                      <Minus size={14} className={styles.trendStable} aria-label="Recent defect rate stable" />
                    )}
                  </div>
                </div>
              ))}
              <p className={styles.productListNote}>
                Percentages are all-time; arrows compare the last {trendPeriodDays} days with the previous{" "}
                {trendPeriodDays}.
                {analytics.by_product.length > MAX_VISIBLE_PRODUCTS &&
                  ` Showing the ${MAX_VISIBLE_PRODUCTS} highest-volume products of ${analytics.by_product.length}.`}
              </p>
            </div>
          )}
        </Card>
      </div>

      <div className={styles.trendHeader}>
        <div>
          <h2 className={styles.analyticsSectionTitle}>Trends &amp; Performance</h2>
          <p className={styles.analyticsSectionSubtitle}>
            Last {trendPeriodDays} days, compared with the previous {trendPeriodDays}-day period.
          </p>
        </div>
        <div className={styles.trendControls}>
          {analyticsRefreshing && (
            <span className={styles.updating} role="status">
              Updating&hellip;
            </span>
          )}
          <RangeSelector label="Time range" value={days} options={WINDOW_OPTIONS} onChange={setDays} />
        </div>
      </div>

      <div className={analyticsRefreshing ? styles.refreshing : undefined} aria-busy={analyticsRefreshing}>
        <div className={styles.subsectionHeader}>
          <h3 className={styles.subsectionTitle}>Inspection performance</h3>
          <p className={styles.analyticsSectionSubtitle}>
            Server-side processing time and AI analysis coverage over the last {trendPeriodDays} days.
          </p>
        </div>
        <PerformanceOverview
          performance={analytics?.performance}
          isLoading={analyticsLoading}
          error={analyticsFailed}
        />

        <Card className={styles.activityCard}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Inspection &amp; Quality Trend</h2>
            <span className={styles.cardSubtitle}>Ground truth, last {trendPeriodDays} days</span>
          </div>
          {analyticsLoading && <Skeleton variant="block" height={160} />}
          {analyticsFailed && <Unavailable />}
          {analyticsReady && (
            <TrendChart
              days={trendDaily}
              series={[
                { key: "good", label: "Good", tone: "success" },
                { key: "defective", label: "Defective", tone: "danger" },
                { key: "pending", label: "Pending", tone: "warning" },
              ]}
              ariaLabel="Inspection ground-truth trend"
              emptyIcon={Activity}
              emptyTitle="No trend data available"
              emptyDescription="Historical trends appear once inspections are recorded."
            />
          )}
        </Card>

        <div className={styles.analyticsGrid}>
          <Card className={styles.analyticsCard}>
            <div className={styles.cardHeader}>
              <h2 className={styles.cardTitle}>Defect Category Trend</h2>
              <span className={styles.cardSubtitle}>Top {categoryTrends.length || 0} categories by volume</span>
            </div>
            {analyticsLoading && <Skeleton variant="block" height={56} />}
            {analyticsFailed && <Unavailable />}
            {analyticsReady && (
              <TrendChart
                days={categoryTrendDays}
                series={categoryTrendSeries}
                ariaLabel="Defect category trend"
                emptyIcon={Layers}
                emptyTitle="No category trend data"
                emptyDescription="Category trends appear once categorized defects are recorded in this window."
              />
            )}
          </Card>

          <Card className={styles.analyticsCard}>
            <div className={styles.cardHeader}>
              <h2 className={styles.cardTitle}>Quality Decision Trend</h2>
              <span className={styles.cardSubtitle}>PASS / FAIL / NOT ASSESSED</span>
            </div>
            {analyticsLoading && <Skeleton variant="block" height={56} />}
            {analyticsFailed && <Unavailable />}
            {analyticsReady && (
              <TrendChart
                days={trendDaily}
                series={[
                  { key: "quality_pass", label: "PASS", tone: "success" },
                  { key: "quality_fail", label: "FAIL", tone: "danger" },
                  { key: "quality_not_assessed", label: "Not assessed", tone: "warning" },
                ]}
                ariaLabel="Quality decision trend"
                emptyIcon={ShieldCheck}
                emptyTitle="No quality decision trend data"
                emptyDescription="Quality decision trends appear once inspections are recorded in this window."
              />
            )}
          </Card>
        </div>

        <Card className={styles.insightsCard}>
          <div className={styles.cardHeader}>
            <h2 className={styles.cardTitle}>Trend Insights</h2>
            <span className={styles.cardSubtitle}>
              Historical observations, last {trendPeriodDays} vs previous {trendPeriodDays} days
            </span>
          </div>
          {analyticsLoading && <Skeleton variant="block" height={56} />}
          {analyticsFailed && <Unavailable />}
          {analyticsReady && trendInsights.length === 0 && (
            <EmptyState
              icon={History}
              title="Insufficient historical data"
              description="Trend insights appear once there is enough historical data to compare periods."
            />
          )}
          {analyticsReady && trendInsights.length > 0 && (
            <ul className={styles.insightsList}>
              {trendInsights.map((insight) => (
                <li key={insight.type} className={styles.insightItem}>
                  <History size={14} strokeWidth={1.75} aria-hidden="true" />
                  <span>{insight.message}</span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <Card className={styles.insightsCard}>
        <div className={styles.cardHeader}>
          <h2 className={styles.cardTitle}>Operational Insights</h2>
          <span className={styles.cardSubtitle}>Deterministic, evidence-based observations</span>
        </div>
        {analyticsLoading && <Skeleton variant="block" height={56} />}
        {analyticsFailed && <Unavailable />}
        {analyticsReady && (analytics.operational_insights?.length ?? 0) === 0 && (
          <EmptyState
            icon={Lightbulb}
            title="No insights yet"
            description="Insights appear once there is enough inspection data to summarize."
          />
        )}
        {analyticsReady && (analytics.operational_insights?.length ?? 0) > 0 && (
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

          {recentLoading && (
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

          {!recentLoading && recentError && (
            <ErrorState
              message={errorMessage(recentError, "Couldn't load recent inspections.")}
              onRetry={refetchRecent}
            />
          )}

          {!recentLoading && !recentError && recentInspections.length === 0 && (
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

          {!recentLoading && !recentError && recentInspections.length > 0 && (
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
