import { useCallback, useEffect, useState } from "react";
import { getAiModels } from "../api/ai";

/**
 * Loads GET /ai/models (any authenticated user) - the same return shape as useUsers. A response
 * that arrives after a newer request was started (or after unmount) is ignored.
 */
export function useAiModels() {
  const [reloadToken, setReloadToken] = useState(0);
  // `result` describes the latest request that has FINISHED; `reloadToken` the one wanted now.
  const [result, setResult] = useState({ token: null, models: null, error: null });

  useEffect(() => {
    let isCancelled = false;

    getAiModels()
      .then((models) => {
        if (!isCancelled) setResult({ token: reloadToken, models, error: null });
      })
      .catch((error) => {
        if (!isCancelled) setResult((current) => ({ token: reloadToken, models: current.models, error }));
      });

    return () => {
      isCancelled = true;
    };
  }, [reloadToken]);

  const refetch = useCallback(() => setReloadToken((token) => token + 1), []);

  const isFetching = result.token !== reloadToken;
  const error = isFetching ? null : result.error;
  return {
    models: error ? [] : (result.models ?? []),
    isLoading: isFetching && result.models === null,
    error,
    refetch,
  };
}
