import { useState, useCallback } from "react";
import useApi from "./use-api";
import { API_BASE_URL } from "@/shared/config";

/**
 * Hook for the compose search flow: upload image + text -> get results.
 * Also supports evaluation submission.
 */
export default function useComposeSearch() {
  const searchApi = useApi("/compose-search/");
  const evaluateApi = useApi("/evaluate");
  const [results, setResults] = useState([]);

  const search = useCallback(async (imageFile, { topK = 10, description = "" } = {}) => {
    const formData = new FormData();
    formData.append("file", imageFile);

    const data = await searchApi.execute(
      { body: formData },
      { top_k: topK, query_text: description },
    );

    setResults(
      (data.top_k_images || []).map((item) => {
        const fp = Array.isArray(item) ? item[0] : item;
        if (typeof fp === "string" && (fp.startsWith("http://") || fp.startsWith("https://") || fp.startsWith("/images/")))
          return fp;
        return `${API_BASE_URL}/images/${String(fp).replaceAll("\\", "/").replace(/^\/+/, "")}`;
      })
    );

    return data;
  }, [searchApi]);

  const submitEvaluation = useCallback(async ({
    selectedMatches,
    evaluatorCode = "U01",
    queryId,
    description = "",
    createdAt = new Date().toISOString(),
  }) => {
    const ranked_results = results.map((url, i) => ({
      rank: i + 1,
      url,
      label: selectedMatches[i] ?? selectedMatches[String(i)] ?? "False",
    }));

    const payload = {
      evaluator_code: evaluatorCode,
      query_id: queryId ?? `q_${Date.now()}`,
      method: "compose",
      num_results: ranked_results.length,
      description,
      ranked_results,
      created_at: createdAt,
    };

    return evaluateApi.execute({
      body: JSON.stringify(payload),
      headers: { "Content-Type": "application/json" },
    });
  }, [evaluateApi, results]);

  const reset = useCallback(() => {
    setResults([]);
    searchApi.reset();
    evaluateApi.reset();
  }, [searchApi, evaluateApi]);

  return {
    results,
    loading: searchApi.loading,
    error: searchApi.error,
    evaluating: evaluateApi.loading,
    evalError: evaluateApi.error,
    search,
    submitEvaluation,
    reset,
  };
}
