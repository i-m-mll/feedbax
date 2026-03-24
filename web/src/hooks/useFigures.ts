import { useCallback, useEffect, useState } from 'react';
import type {
  FigureInfo,
  FigureDetail,
  EvaluationFigureSummary,
  FigureFilters,
} from '@/types/figures';
import {
  fetchEvaluationsWithFigures,
  fetchFigures,
  fetchFigureDetail,
  fetchFigureFile,
} from '@/api/figureAPI';

const PAGE_SIZE = 24;

export function useFigures() {
  // Filter state
  const [filters, setFilters] = useState<FigureFilters>({});
  const [evaluations, setEvaluations] = useState<EvaluationFigureSummary[]>([]);

  // Gallery state
  const [figures, setFigures] = useState<FigureInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Viewer state
  const [selectedFigure, setSelectedFigure] = useState<FigureDetail | null>(null);
  const [figureData, setFigureData] = useState<unknown>(null);
  const [viewerLoading, setViewerLoading] = useState(false);
  const [viewerError, setViewerError] = useState<string | null>(null);

  // Load evaluations on mount — treat server errors as "no data"
  useEffect(() => {
    fetchEvaluationsWithFigures()
      .then(setEvaluations)
      .catch(() => {
        // Gracefully degrade: server may not be ready or DB may be empty.
        // Show empty state instead of an error banner.
        setEvaluations([]);
      });
  }, []);

  // Load figures when filters or page change
  const loadFigures = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchFigures(filters, PAGE_SIZE, offset);
      setFigures(result.items);
      setTotal(result.total);
    } catch (err) {
      // Distinguish "no data" (500 from missing DB / empty tables) from
      // genuine unexpected failures. Server 500s when no figures exist
      // should show the empty state, not the red error banner.
      const msg = err instanceof Error ? err.message : '';
      const isServerError = msg.includes('500') || msg.includes('Internal Server Error');
      const isNotFound = msg.includes('404') || msg.includes('Not Found');
      if (isServerError || isNotFound) {
        // Treat as empty — no figures yet
        setFigures([]);
        setTotal(0);
      } else {
        setError(msg || 'Failed to load figures');
      }
    } finally {
      setLoading(false);
    }
  }, [filters, offset]);

  useEffect(() => {
    loadFigures();
  }, [loadFigures]);

  // Derive unique filter options from evaluations
  const exptNames = Array.from(
    new Set(evaluations.map((e) => e.expt_name).filter(Boolean) as string[]),
  );

  // Derive unique figure types from currently loaded figures
  // (We reload these each time the gallery loads, so they reflect available data)
  const figureTypes = Array.from(new Set(figures.map((f) => f.figure_type)));
  const pertTypes = Array.from(
    new Set(figures.map((f) => f.pert__type).filter(Boolean) as string[]),
  );

  // Update a single filter key and reset pagination
  const updateFilter = useCallback((key: keyof FigureFilters, value: string | undefined) => {
    setFilters((prev) => {
      const next = { ...prev };
      if (value) {
        next[key] = value;
      } else {
        delete next[key];
      }
      return next;
    });
    setOffset(0);
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({});
    setOffset(0);
  }, []);

  // Select a figure for viewing
  const selectFigure = useCallback(async (hash: string) => {
    setViewerLoading(true);
    setViewerError(null);
    setFigureData(null);
    try {
      const detail = await fetchFigureDetail(hash);
      setSelectedFigure(detail);

      // Load the figure file (prefer JSON for Plotly, fall back to png)
      if (detail.available_files.includes('json')) {
        const data = await fetchFigureFile(hash, 'json');
        setFigureData(data);
      } else if (detail.available_files.includes('html')) {
        const data = await fetchFigureFile(hash, 'html');
        setFigureData(data);
      } else if (detail.available_files.length > 0) {
        const fmt = detail.available_files[0];
        const data = await fetchFigureFile(hash, fmt);
        setFigureData(data);
      }
    } catch (err) {
      setViewerError(err instanceof Error ? err.message : 'Failed to load figure');
    } finally {
      setViewerLoading(false);
    }
  }, []);

  const closeFigure = useCallback(() => {
    setSelectedFigure(null);
    setFigureData(null);
    setViewerError(null);
  }, []);

  // Pagination
  const totalPages = Math.ceil(total / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  const goToPage = useCallback((page: number) => {
    setOffset((page - 1) * PAGE_SIZE);
  }, []);

  return {
    // Filters
    filters,
    evaluations,
    exptNames,
    figureTypes,
    pertTypes,
    updateFilter,
    clearFilters,

    // Gallery
    figures,
    total,
    loading,
    error,

    // Pagination
    currentPage,
    totalPages,
    goToPage,

    // Viewer
    selectedFigure,
    figureData,
    viewerLoading,
    viewerError,
    selectFigure,
    closeFigure,
  };
}
