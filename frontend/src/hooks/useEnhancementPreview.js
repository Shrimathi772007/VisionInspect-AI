import { useCallback, useEffect, useState } from "react";
import { getInspectionEnhancedBlob, getInspectionImageQuality } from "../api/inspections";

/**
 * Loads the enhancement preview of one inspection - GET /{id}/enhanced (PNG, as an object URL) and
 * GET /{id}/image-quality (before/after metrics) - only once `enabled` (the card was scrolled into view or
 * "Load preview" was pressed). The object URL is revoked on id change, retry and unmount; responses that
 * arrive after any of those are discarded. The two requests fail independently.
 *
 * Returns { url, metrics, urlError, metricsError, isLoading, retry }.
 */
export function useEnhancementPreview(inspectionId, enabled) {
  const [reloadToken, setReloadToken] = useState(0);
  const requestKey = enabled ? `${inspectionId}:${reloadToken}` : null;
  const [result, setResult] = useState({ key: null, url: null, metrics: null, urlError: null, metricsError: null });

  useEffect(() => {
    if (!requestKey) return undefined;
    let isCancelled = false;
    let createdUrl = null;

    Promise.allSettled([getInspectionEnhancedBlob(inspectionId), getInspectionImageQuality(inspectionId)]).then(
      ([blob, metrics]) => {
        if (isCancelled) return;
        if (blob.status === "fulfilled") createdUrl = URL.createObjectURL(blob.value);
        setResult({
          key: requestKey,
          url: createdUrl,
          metrics: metrics.status === "fulfilled" ? metrics.value : null,
          urlError: blob.status === "rejected" ? blob.reason : null,
          metricsError: metrics.status === "rejected" ? metrics.reason : null,
        });
      }
    );

    return () => {
      isCancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [inspectionId, requestKey]);

  const retry = useCallback(() => setReloadToken((token) => token + 1), []);
  const isCurrent = requestKey !== null && result.key === requestKey;
  return {
    url: isCurrent ? result.url : null,
    metrics: isCurrent ? result.metrics : null,
    urlError: isCurrent ? result.urlError : null,
    metricsError: isCurrent ? result.metricsError : null,
    isLoading: requestKey !== null && !isCurrent,
    retry,
  };
}
