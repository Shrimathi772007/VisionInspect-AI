import { useCallback, useEffect, useState } from "react";
import { getCategoryAnalytics } from "../api/analytics";

/**
 * Loads GET /inspections/analytics/by-category for a trailing window of `days` - the same contract as
 * useAnalyticsSummary (data kept while a new range loads, isLoading / isRefreshing / error / refetch, a
 * stale response ignored), so the dashboard's by-category section behaves like the rest of the page but
 * loads and fails on its own.
 */
export function useCategoryAnalytics({ days } = {}) {
  const [reloadToken, setReloadToken] = useState(0);
  const requestKey = `${days ?? "default"}:${reloadToken}`;

  const [result, setResult] = useState({ key: null, data: null, error: null });

  useEffect(() => {
    let isCancelled = false;

    getCategoryAnalytics({ days })
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
