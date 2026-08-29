import { useCallback, useEffect, useMemo, useState } from "react";
import { listProducts } from "../api/products";

export function useProducts() {
  const [products, setProducts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const refetch = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listProducts();
      setProducts(data);
    } catch (err) {
      setError(err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  const getProductById = useMemo(() => {
    const byId = new Map(products.map((p) => [p.id, p]));
    return (id) => byId.get(id);
  }, [products]);

  return { products, isLoading, error, refetch, getProductById };
}
