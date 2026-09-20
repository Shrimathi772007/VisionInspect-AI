import { useCallback, useEffect, useState } from "react";
import { getInspectionAnalyticsSummary } from "../api/analytics";

/**
 * Loads GET /inspections/analytics/summary for a trailing window of `days`.
 *
 * Returns:
 *   data         the most recent successfully loaded summary (kept while a new range loads,
 *                so the dashboard doesn't blank out on every range change), or null
 *   error        the error from the latest finished request, or null
 *   isLoading    true only while there is nothing to show yet (first load, or a retry after
 *                a failure) - drives skeletons
 *   isRefreshing true while a new range/retry is loading and older data is still on screen
 *   refetch      reload the current range
 *
 * A response that arrives after the range changed (or the component unmounted) is ignored,
 * so a slow earlier request can never overwrite a newer one.
 */
export function useAnalyticsSummary({ days } = {}) {
  const [reloadToken, setReloadToken] = useState(0);
  const requestKey = `${days ?? "default"}:${reloadToken}`;

  // `result` describes the latest request that has FINISHED; `requestKey` the one wanted now.
  const [result, setResult] = useState({ key: null, data: null, error: null });

  useEffect(() => {
    let isCancelled = false;

    getInspectionAnalyticsSummary({ days })
      .then((data) => {
        if (!isCancelled) setResult({ key: requestKey, data, error: null });
      })
      .catch((error) => {
        if (!isCancelled) setResult({ key: requestKey, data: null, error });
      });

    return () => {
      isCancelled = true;
    };
  }, [days, requestKey]);

  const refetch = useCallback(() => setReloadToken((token) => token + 1), []);

  const isFetching = result.key !== requestKey;
  return {
    data: result.data,
    error: isFetching ? null : result.error,
    isLoading: isFetching && result.data === null,
    isRefreshing: isFetching && result.data !== null,
    refetch,
  };
}
