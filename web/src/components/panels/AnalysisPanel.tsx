/**
 * AnalysisPanel — bottom shelf panel for the analysis DAG.
 *
 * Replaces the placeholder with a React Flow canvas showing the analysis
 * pipeline, plus a detail sidebar on the right for the selected node's
 * parameters (or page settings when no node is selected).
 *
 * Includes a compact sub-tab bar at the top for multi-page navigation.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { AnalysisCanvas } from '@/components/analysis/AnalysisCanvas';
import { AnalysisPageSettings } from '@/components/panels/AnalysisPageSettings';
import { useAnalysisStore } from '@/stores/analysisStore';
import { useDemandStore } from '@/stores/demandStore';
import { fetchAnalysisClasses } from '@/api/analysisAPI';
import { generateFigure, getFigureStatus, getFigureData } from '@/api/figureAPI';
import type { AnalysisNodeData } from '@/stores/analysisStore';
import type { AnalysisParamValue, AnalysisParamObject } from '@/types/analysis';
import { Plus, X, Play, Loader2, Image, AlertCircle } from 'lucide-react';
import clsx from 'clsx';

// ---------------------------------------------------------------------------
// Helpers for param value parsing/formatting
// ---------------------------------------------------------------------------

/** Parse a comma-separated string into an array of numbers, ignoring invalid entries. */
function parseNumberList(raw: string): number[] {
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number)
    .filter((n) => !isNaN(n));
}

/** Format an array as a comma-separated string. */
function formatArrayValue(arr: unknown[]): string {
  return arr.join(', ');
}

/** Infer the element type of an array to decide whether to parse as numbers. */
function isNumberArray(arr: unknown[]): boolean {
  return arr.length > 0 && arr.every((v) => typeof v === 'number');
}

/** Height of the page sub-tab bar in pixels. */
const PAGE_TAB_BAR_HEIGHT = 32;

export function AnalysisPanel() {
  const {
    nodes,
    selectedNodeId,
    analysisClasses,
    setAnalysisClasses,
    graphSpec,
    loadGraph,
    pages,
    activePageId,
    addPage,
    removePage,
    renamePage,
    switchPage,
    evalRunId,
  } = useAnalysisStore();

  // Load analysis classes on mount
  useEffect(() => {
    if (analysisClasses.length > 0) return;
    fetchAnalysisClasses().then(setAnalysisClasses).catch(() => {});
  }, [analysisClasses.length, setAnalysisClasses]);

  // Auto-create a first page when no pages exist yet.
  // Read from store inside the effect to avoid Strict Mode double-invocation
  // creating duplicate pages (the closure-captured `pages` is stale on re-run).
  useEffect(() => {
    if (useAnalysisStore.getState().pages.length === 0) {
      addPage('Page 1');
    }
  }, [pages.length, addPage]);

  // Initialize with a default graph (data source only) if empty
  useEffect(() => {
    if (graphSpec) return;
    loadGraph({
      dataSourceId: '__data_source__',
      nodes: {},
      wires: [],
    });
  }, [graphSpec, loadGraph]);

  // Find the selected node's data for the detail panel
  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    const node = nodes.find((n) => n.id === selectedNodeId);
    if (!node || node.type === 'dataSource') return null;
    return node;
  }, [selectedNodeId, nodes]);

  const selectedData = selectedNode?.data as AnalysisNodeData | null;

  return (
    <div className="flex flex-col h-full">
      {/* Sub-tab bar for analysis pages */}
      <AnalysisPageTabBar
        pages={pages}
        activePageId={activePageId}
        onSwitch={switchPage}
        onAdd={addPage}
        onRemove={removePage}
        onRename={renamePage}
      />

      {/* Main content area: canvas + right sidebar */}
      <div className="flex flex-1 min-h-0">
        {/* DAG canvas — fills available space */}
        <div className="relative flex-1 min-w-0">
          <ReactFlowProvider>
            <AnalysisCanvas />
          </ReactFlowProvider>
          {/* Dim overlay when no eval run is selected */}
          {!evalRunId && (
            <div className="absolute inset-0 bg-white/60 backdrop-blur-[1px] flex items-center justify-center z-10 pointer-events-none">
              <div className="text-sm text-slate-400 text-center px-8">
                Select or create an evaluation run to begin
              </div>
            </div>
          )}
        </div>

        {/* Right sidebar — node properties or page settings */}
        <div className="w-64 border-l border-slate-100 bg-white/90 overflow-y-auto shrink-0">
          {selectedData ? (
            <NodeDetailPanel selectedData={selectedData} selectedNodeId={selectedNodeId} />
          ) : (
            <AnalysisPageSettings />
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-tab bar for analysis pages
// ---------------------------------------------------------------------------

interface PageTabBarProps {
  pages: Array<{ id: string; name: string }>;
  activePageId: string | null;
  onSwitch: (id: string) => void;
  onAdd: (name: string) => void;
  onRemove: (id: string) => void;
  onRename: (id: string, name: string) => void;
}

function AnalysisPageTabBar({
  pages,
  activePageId,
  onSwitch,
  onAdd,
  onRemove,
  onRename,
}: PageTabBarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDoubleClick = useCallback((id: string, currentName: string) => {
    setEditingId(id);
    setEditValue(currentName);
  }, []);

  const commitRename = useCallback(() => {
    if (editingId && editValue.trim()) {
      onRename(editingId, editValue.trim());
    }
    setEditingId(null);
  }, [editingId, editValue, onRename]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        commitRename();
      } else if (e.key === 'Escape') {
        setEditingId(null);
      }
    },
    [commitRename],
  );

  // Focus the input when entering edit mode
  useEffect(() => {
    if (editingId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingId]);

  const handleAdd = useCallback(() => {
    const nextNum = pages.length + 1;
    onAdd(`Page ${nextNum}`);
  }, [pages.length, onAdd]);

  const handleRemove = useCallback(
    (e: React.MouseEvent, id: string) => {
      e.stopPropagation();
      // If the page has content (nodes beyond the data source), confirm
      const store = useAnalysisStore.getState();
      const page = store.pages.find((p) => p.id === id);
      const hasContent =
        page && Object.keys(page.graphSpec.nodes).length > 0;
      if (hasContent) {
        const confirmed = window.confirm(
          'This page has analysis nodes. Remove it?',
        );
        if (!confirmed) return;
      }
      onRemove(id);
    },
    [onRemove],
  );

  return (
    <div
      className="flex items-center gap-0.5 px-2 border-b border-slate-100 bg-slate-50/80 shrink-0 overflow-x-auto"
      style={{ height: PAGE_TAB_BAR_HEIGHT }}
    >
      {pages.map((page) => {
        const isActive = page.id === activePageId;
        const isEditing = page.id === editingId;
        return (
          <div
            key={page.id}
            onClick={() => {
              if (!isEditing) onSwitch(page.id);
            }}
            onDoubleClick={() => handleDoubleClick(page.id, page.name)}
            className={clsx(
              'group relative flex items-center gap-1 px-2.5 h-7 rounded-t text-xs cursor-pointer select-none transition-colors',
              'min-w-[80px] max-w-[160px]',
              isActive
                ? 'bg-white text-emerald-700 border border-b-0 border-emerald-200 font-semibold -mb-px z-10'
                : 'text-slate-400 hover:text-slate-600 hover:bg-slate-100/60',
            )}
          >
            {isEditing ? (
              <input
                ref={inputRef}
                type="text"
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                onBlur={commitRename}
                onKeyDown={handleKeyDown}
                className="text-xs bg-transparent outline-none border-b border-emerald-300 w-full min-w-0 text-emerald-700"
              />
            ) : (
              <span className="truncate">{page.name}</span>
            )}
            {/* Close button — visible on hover or when active, hidden during rename */}
            {!isEditing && (
              <button
                onClick={(e) => handleRemove(e, page.id)}
                className={clsx(
                  'shrink-0 p-0.5 rounded transition-colors',
                  isActive
                    ? 'text-emerald-400 hover:text-red-500 hover:bg-red-50'
                    : 'text-transparent group-hover:text-slate-300 hover:!text-red-500 hover:!bg-red-50',
                )}
                title="Close page"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        );
      })}

      {/* Add page button */}
      <button
        onClick={handleAdd}
        className="shrink-0 flex items-center justify-center w-7 h-7 rounded text-slate-300 hover:text-emerald-600 hover:bg-emerald-50 transition-colors"
        title="Add analysis page"
      >
        <Plus className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Node detail panel (extracted from old inline JSX)
// ---------------------------------------------------------------------------

/** Check if a node has figure/figures output ports. */
function hasFigureOutputPort(outputPorts: string[]): boolean {
  return outputPorts.some((p) => p === 'figure' || p === 'figures');
}

function NodeDetailPanel({
  selectedData,
  selectedNodeId,
}: {
  selectedData: AnalysisNodeData;
  selectedNodeId: string | null;
}) {
  const spec = selectedData.spec;
  const canGenerate = spec.role !== 'dependency' && hasFigureOutputPort(spec.outputPorts);
  const evalRunId = useAnalysisStore((s) => s.evalRunId);
  const nodeId = selectedNodeId ?? '';

  // Demand store
  const status = useDemandStore((s) => s.requests[nodeId]?.status ?? 'idle');
  const figureHash = useDemandStore((s) => s.requests[nodeId]?.figureHash);
  const requestGeneration = useDemandStore((s) => s.requestGeneration);
  const setResult = useDemandStore((s) => s.setResult);
  const setError = useDemandStore((s) => s.setError);

  // Local state for toast and figure preview
  const [showToast, setShowToast] = useState(false);
  const [previewData, setPreviewData] = useState<unknown>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const plotRef = useRef<HTMLDivElement>(null);

  // Poll for figure status when running
  useEffect(() => {
    if (status !== 'running' || !nodeId) {
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

  // Load figure data when ready
  useEffect(() => {
    if (status === 'ready' && figureHash) {
      setPreviewLoading(true);
      getFigureData(figureHash)
        .then((data) => setPreviewData(data))
        .catch(() => setPreviewData(null))
        .finally(() => setPreviewLoading(false));
    } else {
      setPreviewData(null);
    }
  }, [status, figureHash]);

  // Render Plotly figure when preview data is available
  useEffect(() => {
    if (!previewData || !plotRef.current) return;
    if (typeof previewData !== 'object') return;

    const plotData = previewData as { data?: unknown[]; layout?: Record<string, unknown> };
    if (!plotData.data) return;

    let cancelled = false;
    import('plotly.js-dist-min').then((Plotly) => {
      if (cancelled || !plotRef.current) return;
      Plotly.newPlot(
        plotRef.current,
        plotData.data as import('plotly.js-dist-min').Data[],
        {
          ...((plotData.layout ?? {}) as Partial<import('plotly.js-dist-min').Layout>),
          autosize: true,
          margin: { t: 20, r: 10, b: 30, l: 40 },
        },
        { responsive: true, displayModeBar: false },
      );
    });

    return () => {
      cancelled = true;
      if (plotRef.current) {
        import('plotly.js-dist-min').then((Plotly) => {
          if (plotRef.current) Plotly.purge(plotRef.current);
        });
      }
    };
  }, [previewData]);

  const handleGenerate = useCallback(async () => {
    if (!nodeId) return;
    if (!evalRunId) {
      setShowToast(true);
      setTimeout(() => setShowToast(false), 3000);
      return;
    }

    requestGeneration(nodeId);
    try {
      const response = await generateFigure(nodeId, { evalRunId });
      useDemandStore.setState((s) => ({
        requests: {
          ...s.requests,
          [nodeId]: { ...s.requests[nodeId], figureHash: response.request_id },
        },
      }));
    } catch (err) {
      setError(nodeId, err instanceof Error ? err.message : 'Request failed');
    }
  }, [nodeId, evalRunId, requestGeneration, setError]);

  return (
    <div className="p-4">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-400">
        {spec.role === 'dependency' ? 'Dependency' : 'Analysis'}
      </div>
      <div className="mt-1 text-base font-semibold text-slate-800">
        {spec.label}
      </div>
      <div className="mt-0.5 text-xs text-slate-500">{spec.type}</div>

      {/* Category badge */}
      <div className="mt-3">
        <span
          className={clsx(
            'inline-block rounded-full px-2 py-0.5 text-[10px] uppercase tracking-wide',
            spec.role === 'dependency'
              ? 'bg-slate-100 text-slate-500'
              : 'bg-emerald-50 text-emerald-600 border border-emerald-100',
          )}
        >
          {spec.category}
        </span>
      </div>

      {/* Ports */}
      <div className="mt-4 space-y-3">
        {spec.inputPorts.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-1">
              Inputs
            </div>
            <div className="space-y-0.5">
              {spec.inputPorts.map((port) => (
                <div key={port} className="text-xs text-slate-600 pl-2 border-l-2 border-emerald-200">
                  {port}
                </div>
              ))}
            </div>
          </div>
        )}
        {spec.outputPorts.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-1">
              Outputs
            </div>
            <div className="space-y-0.5">
              {spec.outputPorts.map((port) => (
                <div key={port} className="text-xs text-slate-600 pl-2 border-l-2 border-slate-200">
                  {port}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Parameters — editable inputs */}
      {Object.keys(spec.params).length > 0 && (
        <div className="mt-4">
          <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-2">
            Parameters
          </div>
          <div className="space-y-2">
            {Object.entries(spec.params).map(([key, value]) => (
              <ParamField
                key={key}
                nodeId={selectedNodeId!}
                paramKey={key}
                value={value}
              />
            ))}
          </div>
        </div>
      )}

      {/* Generate button for figure-producing nodes */}
      {canGenerate && (
        <div className="mt-4">
          {showToast && (
            <div className="flex items-center gap-1.5 px-2.5 py-1.5 mb-2 rounded-lg bg-amber-50 border border-amber-200 text-[11px] text-amber-700">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              Select or create an evaluation run first
            </div>
          )}
          <button
            onClick={handleGenerate}
            disabled={status === 'running'}
            className={clsx(
              'w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-colors',
              status === 'running'
                ? 'bg-blue-50 text-blue-500 cursor-wait'
                : status === 'ready'
                  ? 'bg-emerald-50 text-emerald-600 border border-emerald-200 hover:bg-emerald-100'
                  : status === 'error'
                    ? 'bg-red-50 text-red-500 hover:bg-red-100'
                    : 'bg-emerald-500 text-white hover:bg-emerald-600 shadow-sm'
            )}
            title={
              !evalRunId ? 'Select or create an evaluation run first'
                : status === 'running' ? 'Generating...'
                  : status === 'ready' ? 'Re-generate figure'
                    : status === 'error' ? 'Retry generation'
                      : 'Generate figure'
            }
          >
            {status === 'running' ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : status === 'ready' ? (
              <Image className="w-3.5 h-3.5" />
            ) : (
              <Play className="w-3.5 h-3.5" />
            )}
            <span>
              {status === 'running' ? 'Generating...'
                : status === 'ready' ? 'Re-generate'
                  : status === 'error' ? 'Retry'
                    : 'Generate Figure'}
            </span>
          </button>
        </div>
      )}

      {/* Inline figure preview */}
      {status === 'ready' && previewData && (
        <div className="mt-3">
          <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-1.5">
            Figure Preview
          </div>
          <div className="w-full bg-white rounded-lg border border-slate-200 overflow-hidden">
            {previewLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-4 h-4 text-slate-400 animate-spin" />
              </div>
            ) : (
              <div ref={plotRef} className="w-full" style={{ minHeight: 180 }} />
            )}
          </div>
        </div>
      )}

      {/* Remove button */}
      <button
        className="mt-6 w-full rounded-lg border border-red-200 bg-red-50 px-3 py-1.5 text-xs text-red-600 hover:bg-red-100 transition-colors"
        onClick={() => {
          if (selectedNodeId) {
            useAnalysisStore.getState().removeNode(selectedNodeId);
          }
        }}
      >
        Remove
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Editable parameter field — renders the appropriate input for each param type
// ---------------------------------------------------------------------------

const INPUT_CLASS =
  'w-full text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded px-2 py-1 min-h-[28px] focus:outline-none focus:border-emerald-300 focus:ring-1 focus:ring-emerald-200';

function ParamField({
  nodeId,
  paramKey,
  value,
}: {
  nodeId: string;
  paramKey: string;
  value: AnalysisParamValue;
}) {
  const updateNodeParams = useAnalysisStore((s) => s.updateNodeParams);

  const commitValue = useCallback(
    (newValue: AnalysisParamValue) => {
      updateNodeParams(nodeId, { [paramKey]: newValue });
    },
    [nodeId, paramKey, updateNodeParams],
  );

  // Boolean — toggle switch
  if (typeof value === 'boolean') {
    return (
      <div className="flex items-center justify-between gap-2 min-h-[28px]">
        <span className="text-xs text-slate-500">{paramKey}</span>
        <button
          type="button"
          role="switch"
          aria-checked={value}
          onClick={() => commitValue(!value)}
          className={clsx(
            'relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors',
            value ? 'bg-emerald-500' : 'bg-slate-200',
          )}
        >
          <span
            className={clsx(
              'pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform',
              value ? 'translate-x-4' : 'translate-x-0',
            )}
          />
        </button>
      </div>
    );
  }

  // Number — number input
  if (typeof value === 'number') {
    return (
      <div className="space-y-0.5">
        <label className="text-xs text-slate-500">{paramKey}</label>
        <input
          type="number"
          defaultValue={value}
          onBlur={(e) => {
            const parsed = Number(e.target.value);
            if (!isNaN(parsed)) commitValue(parsed);
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          }}
          className={INPUT_CLASS}
          step="any"
        />
      </div>
    );
  }

  // Array — comma-separated input
  if (Array.isArray(value)) {
    const isNums = isNumberArray(value);
    return (
      <div className="space-y-0.5">
        <label className="text-xs text-slate-500">{paramKey}</label>
        <input
          type="text"
          defaultValue={formatArrayValue(value)}
          onBlur={(e) => {
            if (isNums) {
              commitValue(parseNumberList(e.target.value));
            } else {
              commitValue(
                e.target.value
                  .split(',')
                  .map((s) => s.trim())
                  .filter((s) => s.length > 0),
              );
            }
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          }}
          placeholder={isNums ? '0.1, 0.5, 1.0' : 'a, b, c'}
          className={INPUT_CLASS}
        />
      </div>
    );
  }

  // Object — editable JSON textarea
  if (typeof value === 'object' && value !== null) {
    return (
      <ObjectParamField
        nodeId={nodeId}
        paramKey={paramKey}
        value={value as Record<string, unknown>}
        commitValue={commitValue}
      />
    );
  }

  // String / null / undefined / fallback — text input
  return (
    <div className="space-y-0.5">
      <label className="text-xs text-slate-500">{paramKey}</label>
      <input
        type="text"
        defaultValue={value == null ? '' : String(value)}
        onBlur={(e) => commitValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
        }}
        className={INPUT_CLASS}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Object parameter field — editable JSON for structured params
// ---------------------------------------------------------------------------

const TEXTAREA_CLASS =
  'w-full text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded px-2 py-1.5 font-mono leading-relaxed resize-y focus:outline-none focus:border-emerald-300 focus:ring-1 focus:ring-emerald-200';

function ObjectParamField({
  nodeId,
  paramKey,
  value,
  commitValue,
}: {
  nodeId: string;
  paramKey: string;
  value: Record<string, unknown>;
  commitValue: (v: AnalysisParamValue) => void;
}) {
  const [jsonStr, setJsonStr] = useState(() => JSON.stringify(value, null, 2));
  const [error, setError] = useState<string | null>(null);

  const handleBlur = useCallback(() => {
    try {
      const parsed = JSON.parse(jsonStr);
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        setError('Must be a JSON object');
        return;
      }
      setError(null);
      commitValue(parsed as AnalysisParamObject);
    } catch {
      setError('Invalid JSON');
    }
  }, [jsonStr, commitValue]);

  return (
    <div className="space-y-0.5">
      <label className="text-xs text-slate-500">{paramKey}</label>
      <textarea
        value={jsonStr}
        onChange={(e) => {
          setJsonStr(e.target.value);
          setError(null);
        }}
        onBlur={handleBlur}
        rows={Math.min(Math.max(jsonStr.split('\n').length, 2), 12)}
        className={clsx(TEXTAREA_CLASS, error && 'border-red-300 focus:border-red-400 focus:ring-red-200')}
      />
      {error && (
        <div className="text-[10px] text-red-500">{error}</div>
      )}
    </div>
  );
}
