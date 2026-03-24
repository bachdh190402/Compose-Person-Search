import { useState, useCallback } from "react";
import { API_BASE_URL } from "@/shared/config";

/**
 * Generic fetch hook supporting JSON and FormData POST bodies
 * with dynamic query parameters per call.
 */
export default function useApi(endpoint) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const execute = useCallback(async (options = {}, queryParams = {}) => {
    setLoading(true);
    setError(null);

    const params = new URLSearchParams(queryParams).toString();
    const url = `${API_BASE_URL}${endpoint}${params ? `?${params}` : ""}`;

    try {
      const fetchOptions = { method: "POST", mode: "cors", ...options };

      if (!(fetchOptions.body instanceof FormData)) {
        fetchOptions.headers = {
          "Content-Type": "application/json",
          Accept: "application/json",
          ...fetchOptions.headers,
        };
      } else {
        fetchOptions.headers = {
          Accept: "application/json",
          ...fetchOptions.headers,
        };
      }

      const response = await fetch(url, fetchOptions);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text}`);
      }

      const result = await response.json();
      setData(result);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [endpoint]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
}
