import { useCallback, useEffect, useRef } from 'react';
import { Loader2, X } from 'lucide-react';
import type { FigureDetail } from '@/types/figures';

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

type PlotlyModule = typeof import('plotly.js-dist-min');

/** Dynamically import Plotly to avoid bundling it unless needed. */
async function getPlotly(): Promise<PlotlyModule> {
  return import('plotly.js-dist-min');
}

export function FigureViewer({
  figure,
  figureData,
  loading,
  error,
  onClose,
}: {
  figure: FigureDetail;
  figureData: unknown;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}) {
  const plotRef = useRef<HTMLDivElement>(null);

  // Handle ESC to close
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  // Render Plotly figure when data is available
  useEffect(() => {
    if (!figureData || !plotRef.current) return;
    if (typeof figureData !== 'object') return;

    const plotData = figureData as { data?: unknown[]; layout?: Record<string, unknown> };
    if (!plotData.data) return;

    let cancelled = false;

    getPlotly().then((Plotly) => {
      if (cancelled || !plotRef.current) return;
      Plotly.newPlot(
        plotRef.current,
        plotData.data as import('plotly.js-dist-min').Data[],
        {
          ...((plotData.layout ?? {}) as Partial<import('plotly.js-dist-min').Layout>),
          autosize: true,
          margin: { t: 30, r: 20, b: 40, l: 50 },
        },
        { responsive: true, displayModeBar: true },
      );
    });

    return () => {
      cancelled = true;
      if (plotRef.current) {
        getPlotly().then((Plotly) => {
          if (plotRef.current) Plotly.purge(plotRef.current);
        });
      }
    };
  }, [figureData]);

  // Determine if data is a blob URL (image) rather than Plotly JSON
  const isBlobUrl = typeof figureData === 'string' && figureData.startsWith('blob:');
  const isHtml = typeof figureData === 'string' && !figureData.startsWith('blob:');
  const isPlotly = figureData && typeof figureData === 'object' && 'data' in (figureData as Record<string, unknown>);

  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) onClose();
    },
    [onClose],
  );

  return (
    <div
      className="absolute inset-0 z-20 bg-white/95 backdrop-blur-sm flex flex-col"
      onClick={handleBackdropClick}
    >
      {/* Header bar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-100 flex-shrink-0">
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-slate-100 text-slate-500 hover:text-slate-700"
          title="Close"
        >
          <X className="w-4 h-4" />
        </button>

        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-slate-800 truncate">{figure.identifier}</div>
          <div className="flex items-center gap-2 text-[10px] text-slate-400">
            <span className="font-semibold px-1.5 py-0.5 rounded-full border border-brand-200 text-brand-600 bg-brand-500/10">
              {figure.figure_type}
            </span>
            {figure.expt_name && <span>{figure.expt_name}</span>}
            {figure.pert__type && (
              <span>
                {figure.pert__type}
                {figure.pert__std != null && ` (${figure.pert__std})`}
              </span>
            )}
            <span>{formatDate(figure.created_at)}</span>
          </div>
        </div>

        {/* Available formats */}
        <div className="flex items-center gap-1">
          {figure.available_files.map((fmt) => (
            <a
              key={fmt}
              href={`/api/figures/${figure.hash}/file?format=${fmt}`}
              download
              className="text-[10px] font-semibold px-2 py-0.5 rounded-full border border-slate-200 text-slate-500 hover:border-brand-300 hover:text-brand-600"
              title={`Download as ${fmt.toUpperCase()}`}
            >
              {fmt.toUpperCase()}
            </a>
          ))}
        </div>
      </div>

      {/* Content area */}
      <div className="flex-1 relative min-h-0 overflow-hidden">
        {/* Loading */}
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/60">
            <Loader2 className="w-6 h-6 text-brand-500 animate-spin" />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="absolute top-2 left-2 right-2 z-10 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-600">
            {error}
          </div>
        )}

        {/* Plotly plot */}
        {isPlotly && (
          <div ref={plotRef} className="w-full h-full" />
        )}

        {/* Image (blob URL from png/svg/webp) */}
        {isBlobUrl && (
          <div className="flex items-center justify-center h-full p-4">
            <img
              src={figureData as string}
              alt={figure.identifier}
              className="max-w-full max-h-full object-contain"
            />
          </div>
        )}

        {/* HTML iframe fallback */}
        {isHtml && (
          <iframe
            srcDoc={figureData as string}
            className="w-full h-full border-0"
            title={figure.identifier}
            sandbox="allow-scripts"
          />
        )}

        {/* No data */}
        {!loading && !error && !figureData && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-sm text-slate-400">No displayable file found</div>
          </div>
        )}
      </div>
    </div>
  );
}
