import { useCallback, useEffect, useState } from "react";
import { listUsers, updateUserRole } from "../api/users";

/**
 * Loads GET /users (quality engineers only) - the same return shape as useProducts, plus
 * changeRole. The list stays on screen while a refetch is in flight, so isLoading is only
 * true when there is nothing to show yet. A response that arrives after a newer request
 * was started (or after unmount) is ignored.
 */
export function useUsers() {
  const [reloadToken, setReloadToken] = useState(0);
  // `result` describes the latest request that has FINISHED; `reloadToken` the one wanted now.
  const [result, setResult] = useState({ token: null, users: null, error: null });

  useEffect(() => {
    let isCancelled = false;

    listUsers()
      .then((users) => {
        if (!isCancelled) setResult({ token: reloadToken, users, error: null });
      })
      .catch((error) => {
        if (!isCancelled) setResult((current) => ({ token: reloadToken, users: current.users, error }));
      });

    return () => {
      isCancelled = true;
    };
  }, [reloadToken]);

  const refetch = useCallback(() => setReloadToken((token) => token + 1), []);

  // Errors (ApiError 404/409/422...) are re-thrown for the page to explain; the list is
  // refreshed only after a successful change.
  const changeRole = useCallback(
    async (userId, role) => {
      const updatedUser = await updateUserRole(userId, role);
      refetch();
      return updatedUser;
    },
    [refetch]
  );

  const isFetching = result.token !== reloadToken;
  const error = isFetching ? null : result.error;
  return {
    users: error ? [] : (result.users ?? []),
    isLoading: isFetching && result.users === null,
    error,
    refetch,
    changeRole,
  };
}
