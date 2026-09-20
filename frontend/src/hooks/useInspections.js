import { useCallback, useEffect, useState } from "react";
import { listInspections } from "../api/inspections";

/**
 * @param {{ limit?: number }} [options] `limit` loads only the newest N inspections (a SQL
 *   LIMIT on the backend). Omitted, every inspection is loaded, as before - the inspections
 *   list page still needs all of them; the dashboard only needs the latest few.
 */
export function useInspections({ limit } = {}) {
  const [inspections, setInspections] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const refetch = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listInspections({ limit });
      setInspections(data);
    } catch (err) {
      setError(err);
    } finally {
      setIsLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { inspections, isLoading, error, refetch };
}
