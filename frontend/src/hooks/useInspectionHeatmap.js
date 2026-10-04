import { useCallback, useEffect, useState } from "react";
import { getInspectionHeatmapBlob } from "../api/inspections";

/**
 * Loads GET /inspections/{id}/heatmap as an object URL when `enabled` (the inspection says
 * has_heatmap). The object URL is revoked when the id changes, on retry and on unmount. A
 * response that arrives after any of those is discarded (and its URL never created).
 *
 * Returns { url, error, isLoading, retry }; all null/false while disabled.
 */
export function useInspectionHeatmap(inspectionId, enabled) {
  const [reloadToken, setReloadToken] = useState(0);
  const requestKey = enabled ? `${inspectionId}:${reloadToken}` : null;

  // `result` describes the latest request that has FINISHED; `requestKey` the one wanted now.
  const [result, setResult] = useState({ key: null, url: null, error: null });

  useEffect(() => {
    if (!requestKey) return undefined;
    let isCancelled = false;
    let createdUrl = null;

    getInspectionHeatmapBlob(inspectionId)
      .then((blob) => {
        if (isCancelled) return;
        createdUrl = URL.createObjectURL(blob);
        setResult({ key: requestKey, url: createdUrl, error: null });
      })
      .catch((error) => {
        if (!isCancelled) setResult({ key: requestKey, url: null, error });
      });

    return () => {
      isCancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [inspectionId, requestKey]);

  const retry = useCallback(() => setReloadToken((token) => token + 1), []);

  const isCurrent = requestKey !== null && result.key === requestKey;
  return {
    url: isCurrent ? result.url : null,
    error: isCurrent ? result.error : null,
    isLoading: requestKey !== null && !isCurrent,
    retry,
  };
}
