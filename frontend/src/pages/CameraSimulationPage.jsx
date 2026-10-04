import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Camera, Play, Square, TriangleAlert } from "lucide-react";
import { useProducts } from "../hooks/useProducts";
import { useDatasetCategories } from "../hooks/useDatasetCategories";
import { getDatasetCategoryDetail, importDatasetInspection, listDatasetImages } from "../api/dataset";
import { ApiError } from "../api/client";
import { PageHeader } from "../components/PageHeader/PageHeader";
import { Card } from "../components/Card/Card";
import { Badge } from "../components/Badge/Badge";
import { Button } from "../components/Button/Button";
import { Select } from "../components/Select/Select";
import { Input } from "../components/Input/Input";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { ErrorState } from "../components/ErrorState/ErrorState";
import { categoryLabel } from "../constants/mvtecCategories";
import {
  aiPredictionLabel,
  aiPredictionTone,
  defectCategoryLabel,
  formatConfidence,
  qualityDecisionLabel,
  qualityDecisionTone,
  statusLabel,
  statusTone,
} from "../utils/badgeMaps";
import {
  ALL_DEFECT_TYPES,
  DEFAULT_FRAMES,
  DEFAULT_INTERVAL_S,
  MAX_FRAMES,
  MAX_INTERVAL_S,
  MIN_FRAMES,
  MIN_INTERVAL_S,
  buildFrameSequence,
  clampInterval,
  clampMaxFrames,
  frameAt,
} from "../utils/cameraSimulation";
import styles from "./CameraSimulationPage.module.css";

function errorText(error) {
  return error instanceof ApiError ? error.message : "Something went wrong.";
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/**
 * One camera frame: import the next image of `run`, record it, then schedule the next frame - only after this
 * import has settled (chained setTimeout, never setInterval). A tick that finds another request still in
 * flight (e.g. one from a run stopped mid-request) waits another interval instead of sending, so two
 * requests never overlap. Stops the run on the first error, at the frame limit, and does nothing once the
 * page has unmounted or a newer run has started.
 */
async function tick(run, loop) {
  const { runRef, timerRef, inFlightRef, mountedRef, setFrames, setRunError, stopRun } = loop;
  timerRef.current = null;
  if (!run.active || runRef.current !== run) return;
  if (inFlightRef.current) {
    timerRef.current = setTimeout(() => tick(run, loop), run.intervalMs);
    return;
  }
  const item = frameAt(run.sequence, run.index);
  run.index += 1;
  inFlightRef.current = true;
  let inspection;
  try {
    inspection = await importDatasetInspection({
      productId: run.productId,
      category: run.category,
      split: run.split,
      defectType: item.defectType,
      filename: item.filename,
    });
  } catch (error) {
    inFlightRef.current = false;
    if (mountedRef.current && runRef.current === run) {
      stopRun(null);
      setRunError(`Camera stopped: ${errorText(error)}`);
    }
    return;
  }
  inFlightRef.current = false;
  if (!mountedRef.current || runRef.current !== run) return;

  run.count += 1;
  setFrames((current) => [
    { n: run.count, time: new Date(), defectType: item.defectType, filename: item.filename, inspection },
    ...current,
  ]);
  if (run.count >= run.maxFrames) {
    stopRun(`Finished: ${run.maxFrames} frame${run.maxFrames === 1 ? "" : "s"} imported.`);
  } else if (run.active) {
    timerRef.current = setTimeout(() => tick(run, loop), run.intervalMs);
  }
}

/**
 * Camera simulation (quality engineers): imports MVTec AD dataset images one by one, as if they came from a
 * camera, through POST /inspections/import, and lists each result.
 *
 * Safety limits: interval 1-10 s, 1-50 frames, stops on the first error, on Stop, on reaching the frame
 * limit and on unmount (route change). Frames are scheduled with a chained setTimeout: the next tick is only
 * scheduled after the previous import has settled, and a tick that finds a request still in flight waits for
 * another interval instead of sending - so two requests never run at once.
 */
export function CameraSimulationPage() {
  const { products, isLoading: productsLoading } = useProducts();
  const { categories, isLoading: categoriesLoading } = useDatasetCategories();

  const [productId, setProductId] = useState("");
  const [category, setCategory] = useState("");
  const [split, setSplit] = useState("test");
  const [defectType, setDefectType] = useState(ALL_DEFECT_TYPES);
  const [intervalInput, setIntervalInput] = useState(String(DEFAULT_INTERVAL_S));
  const [framesInput, setFramesInput] = useState(String(DEFAULT_FRAMES));

  const [detailResult, setDetailResult] = useState({ category: null, detail: null, error: null });
  const [isRunning, setIsRunning] = useState(false);
  const [isPreparing, setIsPreparing] = useState(false);
  const [frames, setFrames] = useState([]);
  const [runLimit, setRunLimit] = useState(null);
  const [stopReason, setStopReason] = useState(null);
  const [runError, setRunError] = useState(null);

  const runRef = useRef(null); // the active run: { active, sequence, index, count, maxFrames, intervalMs }
  const timerRef = useRef(null);
  const inFlightRef = useRef(false);
  const mountedRef = useRef(true);

  // Category detail (splits and defect types) for the option lists; a stale response is ignored.
  useEffect(() => {
    if (!category) return undefined;
    let isCancelled = false;
    getDatasetCategoryDetail(category)
      .then((detail) => {
        if (!isCancelled) setDetailResult({ category, detail, error: null });
      })
      .catch((error) => {
        if (!isCancelled) setDetailResult({ category, detail: null, error });
      });
    return () => {
      isCancelled = true;
    };
  }, [category]);

  const detail = detailResult.category === category ? detailResult.detail : null;
  const detailError = detailResult.category === category ? detailResult.error : null;
  const splits = detail?.splits ?? [];
  const splitDefectTypes = (splits.find((s) => s.split === split)?.defect_types ?? [])
    .map((entry) => entry.defect_type)
    .sort();
  const selectedProduct = products.find((product) => String(product.id) === String(productId));
  const categoryMismatch = Boolean(selectedProduct?.category && category && selectedProduct.category !== category);

  const clearTimer = () => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const stopRun = useCallback((reason) => {
    if (runRef.current) runRef.current.active = false;
    clearTimer();
    if (mountedRef.current) {
      setIsRunning(false);
      setStopReason(reason ?? null);
    }
  }, []);

  // Stop on unmount (including route changes): no timer survives the page.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (runRef.current) runRef.current.active = false;
      clearTimer();
    };
  }, []);

  // Everything the module-level tick needs; refs and stable setters, so the object can be rebuilt freely.
  const loop = { runRef, timerRef, inFlightRef, mountedRef, setFrames, setRunError, stopRun };

  const start = async () => {
    if (isRunning || isPreparing) return;
    const intervalS = clampInterval(intervalInput);
    const maxFrames = clampMaxFrames(framesInput);
    setIntervalInput(String(intervalS));
    setFramesInput(String(maxFrames));
    setRunError(null);
    setStopReason(null);

    setIsPreparing(true);
    let sequence;
    try {
      const types = defectType === ALL_DEFECT_TYPES ? splitDefectTypes : [defectType];
      const lists = await Promise.all(
        types.map(async (type) => {
          const images = await listDatasetImages({ category, split, defectType: type });
          return { defectType: type, filenames: images.filenames ?? [] };
        })
      );
      sequence = buildFrameSequence(lists);
    } catch (error) {
      if (mountedRef.current) {
        setIsPreparing(false);
        setRunError(`Couldn't list the dataset images: ${errorText(error)}`);
      }
      return;
    }
    if (!mountedRef.current) return;
    setIsPreparing(false);
    if (sequence.length === 0) {
      setRunError("No images found for this category, split and defect type.");
      return;
    }

    const run = {
      active: true,
      sequence,
      index: 0,
      count: 0,
      maxFrames,
      intervalMs: intervalS * 1000,
      productId,
      category,
      split,
    };
    runRef.current = run;
    setFrames([]);
    setRunLimit(maxFrames);
    setIsRunning(true);
    timerRef.current = setTimeout(() => tick(run, loop), run.intervalMs);
  };

  const canStart = Boolean(productId && category && split && defectType) && !isRunning && !isPreparing;
  const controlsLocked = isRunning || isPreparing;

  return (
    <div>
      <PageHeader
        eyebrow="Image acquisition"
        title="Camera simulation"
        description="Feed MVTec AD dataset images into the inspection pipeline one at a time, like a camera on a line."
      />

      <div className={styles.banner} role="note">
        <Camera size={18} aria-hidden="true" />
        <div>
          <p className={styles.bannerTitle}>Camera simulation</p>
          <p className={styles.bannerText}>
            Images come from the MVTec AD dataset. Results here are demonstration data, not accuracy measurements.
          </p>
        </div>
      </div>

      <Card className={styles.controls}>
        <div className={styles.grid}>
          <Select label="Product" value={productId} onChange={(e) => setProductId(e.target.value)} disabled={controlsLocked || productsLoading}>
            <option value="" disabled>
              Select a product
            </option>
            {products.map((product) => (
              <option key={product.id} value={product.id}>
                {product.product_name} ({product.product_code}) - {product.category ? categoryLabel(product.category) : "No category"}
              </option>
            ))}
          </Select>
          <Select
            label="Dataset category"
            value={category}
            onChange={(e) => {
              setCategory(e.target.value);
              setDefectType(ALL_DEFECT_TYPES);
            }}
            disabled={controlsLocked || categoriesLoading}
          >
            <option value="" disabled>
              Select a category
            </option>
            {categories.map((value) => (
              <option key={value} value={value}>
                {categoryLabel(value)}
              </option>
            ))}
          </Select>
          <Select
            label="Split"
            value={split}
            onChange={(e) => {
              setSplit(e.target.value);
              setDefectType(ALL_DEFECT_TYPES);
            }}
            disabled={controlsLocked || !detail}
          >
            {(splits.length ? splits.map((s) => s.split) : ["test"]).map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </Select>
          <Select label="Defect type" value={defectType} onChange={(e) => setDefectType(e.target.value)} disabled={controlsLocked || !detail}>
            <option value={ALL_DEFECT_TYPES}>All defect types</option>
            {splitDefectTypes.map((value) => (
              <option key={value} value={value}>
                {defectCategoryLabel(value)}
              </option>
            ))}
          </Select>
          <Input
            label={`Interval (${MIN_INTERVAL_S}-${MAX_INTERVAL_S} s)`}
            type="number"
            min={MIN_INTERVAL_S}
            max={MAX_INTERVAL_S}
            step={1}
            value={intervalInput}
            onChange={(e) => setIntervalInput(e.target.value)}
            onBlur={() => setIntervalInput(String(clampInterval(intervalInput)))}
            disabled={controlsLocked}
          />
          <Input
            label={`Max frames (${MIN_FRAMES}-${MAX_FRAMES})`}
            type="number"
            min={MIN_FRAMES}
            max={MAX_FRAMES}
            step={1}
            value={framesInput}
            onChange={(e) => setFramesInput(e.target.value)}
            onBlur={() => setFramesInput(String(clampMaxFrames(framesInput)))}
            disabled={controlsLocked}
          />
        </div>

        {categoryMismatch && (
          <p className={styles.warning} role="alert">
            <TriangleAlert size={14} aria-hidden="true" /> Product category differs from the dataset category. Each
            import is still analysed by the {categoryLabel(category)} model (the dataset path decides the category).
          </p>
        )}
        {detailError && <ErrorState message={`Couldn't load the category: ${errorText(detailError)}`} />}

        <div className={styles.actions}>
          {isRunning ? (
            <Button variant="danger" leftIcon={<Square size={14} />} onClick={() => stopRun("Stopped.")}>
              Stop
            </Button>
          ) : (
            <Button leftIcon={<Play size={14} />} onClick={start} disabled={!canStart} loading={isPreparing}>
              Start simulated camera
            </Button>
          )}
          {runLimit !== null && (
            <span className={styles.counter} aria-live="polite">
              Frame {frames.length} of {runLimit}
            </span>
          )}
          {isRunning && <Badge tone="accent" dot>Running</Badge>}
          {stopReason && !isRunning && <span className={styles.stopReason}>{stopReason}</span>}
        </div>
        {runError && <ErrorState message={runError} />}
      </Card>

      <Card className={styles.framesCard}>
        {frames.length === 0 ? (
          <EmptyState icon={Camera} title="No frames yet" description="Start the simulated camera to import images." />
        ) : (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">Time</th>
                  <th scope="col">Image</th>
                  <th scope="col">Ground truth</th>
                  <th scope="col">AI prediction</th>
                  <th scope="col">Confidence</th>
                  <th scope="col">Decision</th>
                  <th scope="col">Review</th>
                  <th scope="col">
                    <span className="visually-hidden">Link</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {frames.map((frame) => {
                  const inspection = frame.inspection ?? {};
                  return (
                    <tr key={frame.n} data-testid="camera-frame">
                      <td className={styles.mono}>{frame.n}</td>
                      <td className={styles.mono}>{formatTime(frame.time)}</td>
                      <td className={styles.imageCell}>
                        {frame.defectType}/{frame.filename}
                      </td>
                      <td>
                        <Badge tone={statusTone(inspection.status)}>{statusLabel(inspection.status) || "—"}</Badge>
                      </td>
                      <td>
                        {inspection.ai_prediction ? (
                          <Badge tone={aiPredictionTone(inspection.ai_prediction)}>
                            {aiPredictionLabel(inspection.ai_prediction)}
                          </Badge>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className={styles.mono}>{formatConfidence(inspection.ai_confidence) ?? "—"}</td>
                      <td>
                        <Badge tone={qualityDecisionTone(inspection.quality_decision)}>
                          {qualityDecisionLabel(inspection.quality_decision)}
                        </Badge>
                      </td>
                      <td>{inspection.review_required === true ? <Badge tone="accent">Review</Badge> : "—"}</td>
                      <td>
                        {inspection.id != null && (
                          <Link to={`/inspections/${inspection.id}`} className={styles.link}>
                            View #{inspection.id}
                          </Link>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
