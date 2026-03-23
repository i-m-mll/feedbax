import { useCallback } from 'react';
import clsx from 'clsx';
import { ChevronLeft, ChevronRight, Loader2, X } from 'lucide-react';
import { useFigures } from '@/hooks/useFigures';
import { FigureCard } from '@/components/panels/FigureCard';
import { FigureViewer } from '@/components/panels/FigureViewer';
import type { FigureFilters } from '@/types/figures';

function FilterSelect({
  label,
  value,
  options,
  filterKey,
  onUpdate,
}: {
  label: string;
  value: string | undefined;
  options: string[];
  filterKey: keyof FigureFilters;
  onUpdate: (key: keyof FigureFilters, value: string | undefined) => void;
}) {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      onUpdate(filterKey, e.target.value || undefined);
    },
    [filterKey, onUpdate],
  );

  if (options.length === 0) return null;

  return (
    <select
      value={value ?? ''}
      onChange={handleChange}
      className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700 bg-white min-w-[120px]"
    >
      <option value="">All {label}</option>
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
  );
}

export function FigureGalleryPanel() {
  const {
    filters,
    evaluations,
    exptNames,
    figureTypes,
    pertTypes,
    updateFilter,
    clearFilters,
    figures,
    total,
    loading,
    error,
    currentPage,
    totalPages,
    goToPage,
    selectedFigure,
    figureData,
    viewerLoading,
    viewerError,
    selectFigure,
    closeFigure,
  } = useFigures();

  const evaluationOptions = evaluations.map((e) => e.evaluation_hash);
  const hasActiveFilters = Object.values(filters).some(Boolean);

  return (
    <div className="flex flex-col h-full relative">
      {/* Top bar: filters and pagination */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-100 bg-white/80 flex-shrink-0 flex-wrap">
        {/* Evaluation filter */}
        <FilterSelect
          label="evaluations"
          value={filters.evaluation_hash}
          options={evaluationOptions}
          filterKey="evaluation_hash"
          onUpdate={updateFilter}
        />

        {/* Experiment name filter */}
        <FilterSelect
          label="experiments"
          value={filters.expt_name}
          options={exptNames}
          filterKey="expt_name"
          onUpdate={updateFilter}
        />

        {/* Figure type filter */}
        <FilterSelect
          label="types"
          value={filters.figure_type}
          options={figureTypes}
          filterKey="figure_type"
          onUpdate={updateFilter}
        />

        {/* Perturbation type filter */}
        <FilterSelect
          label="perturbations"
          value={filters.pert_type}
          options={pertTypes}
          filterKey="pert_type"
          onUpdate={updateFilter}
        />

        {/* Clear filters */}
        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="p-1 rounded hover:bg-slate-100 text-slate-400 hover:text-slate-600"
            title="Clear all filters"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Result count */}
        <span className="text-xs text-slate-400 tabular-nums">
          {total} figure{total !== 1 ? 's' : ''}
        </span>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center gap-1">
            <button
              onClick={() => goToPage(currentPage - 1)}
              disabled={currentPage <= 1}
              className="p-0.5 rounded text-slate-500 hover:text-slate-700 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed"
              title="Previous page"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-xs text-slate-500 tabular-nums min-w-[50px] text-center">
              {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => goToPage(currentPage + 1)}
              disabled={currentPage >= totalPages}
              className="p-0.5 rounded text-slate-500 hover:text-slate-700 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed"
              title="Next page"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Content area */}
      <div className="flex-1 relative min-h-0 overflow-y-auto">
        {/* Error banner */}
        {error && (
          <div className="absolute top-2 left-2 right-2 z-10 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-600">
            {error}
          </div>
        )}

        {/* Loading overlay */}
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/60">
            <Loader2 className="w-6 h-6 text-brand-500 animate-spin" />
          </div>
        )}

        {/* Empty state: no figures at all */}
        {!loading && total === 0 && !hasActiveFilters && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-sm text-slate-400">No figures yet</div>
              <div className="text-xs text-slate-300 mt-1">
                Figures will appear here after running evaluations
              </div>
            </div>
          </div>
        )}

        {/* Empty state: no results matching filters */}
        {!loading && total === 0 && hasActiveFilters && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-sm text-slate-400">No figures match the current filters</div>
              <button
                onClick={clearFilters}
                className="mt-2 text-xs text-brand-500 hover:text-brand-700"
              >
                Clear filters
              </button>
            </div>
          </div>
        )}

        {/* Gallery grid */}
        {figures.length > 0 && (
          <div className="p-4 grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
            {figures.map((fig) => (
              <FigureCard
                key={fig.hash}
                figure={fig}
                onClick={() => selectFigure(fig.hash)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Figure viewer overlay */}
      {selectedFigure && (
        <FigureViewer
          figure={selectedFigure}
          figureData={figureData}
          loading={viewerLoading}
          error={viewerError}
          onClose={closeFigure}
        />
      )}
    </div>
  );
}
