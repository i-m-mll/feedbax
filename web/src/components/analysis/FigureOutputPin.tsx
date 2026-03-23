import { useCallback, useEffect, useRef, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { useDemandStore } from '@/stores/demandStore';
import { generateFigure, getFigureStatus, getFigureData } from '@/api/figureAPI';
import type { FigureRequestStatus } from '@/types/analysis';
import clsx from 'clsx';
import { Image, RefreshCw, Play, X, Loader2 } from 'lucide-react';

type PlotlyModule = typeof import('plotly.js-dist-min');

/** Dynamically import Plotly to avoid bundling it unless needed. */
async function getPlotly(): Promise<PlotlyModule> {
  return import('plotly.js-dist-min');
}

const STATUS_COLORS: Record<FigureRequestStatus, string> = {
  idle: 'bg-slate-300',
  running: 'bg-blue-400 animate-pulse',
  ready: 'bg-emerald-400',
  error: 'bg-red-400',
};

const STATUS_RING: Record<FigureRequestStatus, string> = {
  idle: '',
  running: 'ring-2 ring-blue-200',
  ready: 'ring-2 ring-emerald-200',
  error: 'ring-2 ring-red-200',
};

interface FigureOutputPinProps {
  nodeId: string;
  /** Vertical offset from the top of the node body, in pixels. */
  topOffset: number;
  /** Whether the node is reversed (ports swapped left/right). */
  reversed?: boolean;
}

/** Modal that renders Plotly JSON as an interactive chart instead of raw JSON. */
function FigurePreviewModal({
  data,
  loading,
  onClose,
}: {
  data: unknown;
  loading: boolean;
  onClose: () => void;
}) {
  const plotRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  useEffect(() => {
    if (!data || !plotRef.current) return;
    if (typeof data !== 'object') return;

    const plotData = data as { data?: unknown[]; layout?: Record<string, unknown> };
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
  }, [data]);

  const isPlotly = data && typeof data === 'object' && 'data' in (data as Record<string, unknown>);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-xl shadow-2xl border border-slate-200 max-w-4xl max-h-[80vh] w-full mx-4 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
          <div className="text-sm font-medium text-slate-700">Figure Preview</div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="overflow-auto max-h-[calc(80vh-52px)]">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-sm text-slate-400">
              <Loader2 className="w-5 h-5 animate-spin mr-2" />
              Loading figure...
            </div>
          ) : isPlotly ? (
            <div ref={plotRef} className="w-full" style={{ minHeight: 400 }} />
          ) : data ? (
            <pre className="p-4 text-xs font-mono text-slate-600 whitespace-pre-wrap break-words">
              {JSON.stringify(data, null, 2)}
            </pre>
          ) : (
            <div className="flex items-center justify-center py-12 text-sm text-slate-400">
              Failed to load figure data.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function FigureOutputPin({ nodeId, topOffset, reversed }: FigureOutputPinProps) {
  const status = useDemandStore((s) => s.requests[nodeId]?.status ?? 'idle');
  const figureHash = useDemandStore((s) => s.requests[nodeId]?.figureHash);
  const error = useDemandStore((s) => s.requests[nodeId]?.error);
  const requestGeneration = useDemandStore((s) => s.requestGeneration);
  const setResult = useDemandStore((s) => s.setResult);
  const setError = useDemandStore((s) => s.setError);
  const clearRequest = useDemandStore((s) => s.clearRequest);

  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [previewData, setPreviewData] = useState<unknown>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll for figure status when running
  useEffect(() => {
    if (status !== 'running') {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    const requestId = useDemandStore.getState().requests[nodeId]?.figureHash;
    if (!requestId) return;

    pollRef.current = setInterval(async () => {
      try {
        const result = await getFigureStatus(requestId);
        if (result.status === 'complete' && result.figure_hashes?.length) {
          setResult(nodeId, result.figure_hashes[0]);
        } else if (result.status === 'error') {
          setError(nodeId, result.error ?? 'Generation failed');
        }
      } catch {
        // Keep polling on transient errors
      }
    }, 2000);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [status, nodeId, setResult, setError]);

  // Close context menu on outside click
  useEffect(() => {
    if (!contextMenu) return;
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setContextMenu(null);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [contextMenu]);

  const handleGenerate = useCallback(
    async (forceRerun = false) => {
      requestGeneration(nodeId);
      try {
        const response = await generateFigure(nodeId, { forceRerun });
        // Store request_id temporarily as figureHash for polling
        useDemandStore.setState((s) => ({
          requests: {
            ...s.requests,
            [nodeId]: { ...s.requests[nodeId], figureHash: response.request_id },
          },
        }));
      } catch (err) {
        setError(nodeId, err instanceof Error ? err.message : 'Request failed');
      }
    },
    [nodeId, requestGeneration, setError]
  );

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (status === 'idle' || status === 'error') {
        handleGenerate();
      } else if (status === 'ready' && figureHash) {
        // Show preview
        setPreviewLoading(true);
        setShowPreview(true);
        getFigureData(figureHash)
          .then((data) => setPreviewData(data))
          .catch(() => setPreviewData(null))
          .finally(() => setPreviewLoading(false));
      }
    },
    [status, figureHash, handleGenerate]
  );

  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setContextMenu({ x: e.clientX, y: e.clientY });
    },
    []
  );

  return (
    <>
      {/* The pin itself */}
      <div
        className="absolute z-30 cursor-pointer group"
        style={{
          top: topOffset,
          [reversed ? 'left' : 'right']: -14,
          transform: 'translateY(-50%)',
        }}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        title={
          status === 'idle'
            ? 'Click to generate figure'
            : status === 'running'
              ? 'Generating...'
              : status === 'ready'
                ? 'Click to view figure'
                : `Error: ${error}`
        }
      >
        <div
          className={clsx(
            'w-3.5 h-3.5 rounded-full border-2 border-white shadow-soft transition-all duration-200',
            STATUS_COLORS[status],
            STATUS_RING[status],
            'group-hover:scale-125'
          )}
        />
        {/* Small icon overlay for ready state */}
        {status === 'ready' && (
          <Image className="absolute inset-0 m-auto w-2 h-2 text-white pointer-events-none" />
        )}
      </div>

      {/* Invisible React Flow handle for potential edge connections */}
      <Handle
        type="source"
        position={reversed ? Position.Left : Position.Right}
        id="__figure_out"
        style={{
          top: topOffset,
          [reversed ? 'left' : 'right']: -6,
          transform: 'translateY(-50%)',
          width: 0,
          height: 0,
          opacity: 0,
          pointerEvents: 'none',
        }}
      />

      {/* Context menu */}
      {contextMenu && (
        <div
          ref={menuRef}
          className="fixed bg-white rounded-lg shadow-lg border border-slate-200 py-1 min-w-44 z-50"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <button
            type="button"
            onClick={() => {
              handleGenerate(false);
              setContextMenu(null);
            }}
            className="w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 flex items-center gap-2"
          >
            <Play className="w-4 h-4 text-emerald-500" />
            Generate
          </button>
          <button
            type="button"
            onClick={() => {
              handleGenerate(true);
              setContextMenu(null);
            }}
            className="w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4 text-blue-500" />
            Force Re-run
          </button>
          {status === 'ready' && figureHash && (
            <>
              <div className="border-t border-slate-100 my-1" />
              <button
                type="button"
                onClick={() => {
                  setPreviewLoading(true);
                  setShowPreview(true);
                  getFigureData(figureHash)
                    .then((data) => setPreviewData(data))
                    .catch(() => setPreviewData(null))
                    .finally(() => setPreviewLoading(false));
                  setContextMenu(null);
                }}
                className="w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 flex items-center gap-2"
              >
                <Image className="w-4 h-4 text-brand-500" />
                View Figure
              </button>
            </>
          )}
          <div className="border-t border-slate-100 my-1" />
          <button
            type="button"
            onClick={() => {
              clearRequest(nodeId);
              setContextMenu(null);
            }}
            className="w-full px-3 py-2 text-left text-sm text-slate-500 hover:bg-slate-50 flex items-center gap-2"
          >
            <X className="w-4 h-4" />
            Clear
          </button>
        </div>
      )}

      {/* Figure preview modal with Plotly rendering */}
      {showPreview && (
        <FigurePreviewModal
          data={previewData}
          loading={previewLoading}
          onClose={() => setShowPreview(false)}
        />
      )}
    </>
  );
}
