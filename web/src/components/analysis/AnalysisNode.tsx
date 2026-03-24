/**
 * AnalysisNode — custom React Flow node for analysis DAG entries.
 *
 * Full analysis nodes are rendered as rounded rectangles with input/output
 * handles, matching the visual language of CustomNode in the model canvas
 * but with a distinct teal/emerald accent to differentiate the analysis
 * context from the model-editing context.
 *
 * Dependency nodes (role === 'dependency') are rendered smaller and muted.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { AnalysisNodeData } from '@/stores/analysisStore';
import { useAnalysisStore } from '@/stores/analysisStore';
import { useDemandStore } from '@/stores/demandStore';
import { generateFigure, getFigureStatus, getFigureData } from '@/api/figureAPI';
import { Play, Loader2, Image, AlertCircle } from 'lucide-react';
import clsx from 'clsx';

// Layout constants — aligned with CustomNode for consistency
const WIDTH_FULL = 200;
const WIDTH_DEP = 160;
const HEADER_HEIGHT_FULL = 52;
const HEADER_HEIGHT_DEP = 30;
const BODY_PADDING = 10;
const ROW_HEIGHT_FULL = 24;
const ROW_HEIGHT_DEP = 20;
const HANDLE_OFFSET = -6;
const LABEL_OFFSET = 18;
const GENERATE_BTN_HEIGHT = 32;

/** Check if a node has figure/figures output ports. */
function hasFigureOutputPort(outputPorts: string[]): boolean {
  return outputPorts.some((p) => p === 'figure' || p === 'figures');
}

export function AnalysisNode({ id, data, selected }: NodeProps) {
  const nodeData = data as AnalysisNodeData;
  const spec = nodeData.spec;
  const isDep = spec.role === 'dependency';
  const canGenerate = !isDep && hasFigureOutputPort(spec.outputPorts);

  // Demand store for figure generation
  const status = useDemandStore((s) => s.requests[id]?.status ?? 'idle');
  const figureHash = useDemandStore((s) => s.requests[id]?.figureHash);
  const requestGeneration = useDemandStore((s) => s.requestGeneration);
  const setResult = useDemandStore((s) => s.setResult);
  const setError = useDemandStore((s) => s.setError);

  // Eval run from analysis store
  const evalRunId = useAnalysisStore((s) => s.evalRunId);

  // Inline figure preview state
  const [showToast, setShowToast] = useState(false);
  const [previewData, setPreviewData] = useState<unknown>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
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

    const requestId = useDemandStore.getState().requests[id]?.figureHash;
    if (!requestId) return;

    pollRef.current = setInterval(async () => {
      try {
        const result = await getFigureStatus(requestId);
        if (result.status === 'complete' && result.figure_hashes?.length) {
          setResult(id, result.figure_hashes[0]);
        } else if (result.status === 'error') {
          setError(id, result.error ?? 'Generation failed');
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
  }, [status, id, setResult, setError]);

  // Auto-load preview data when figure is ready
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

  const handleGenerate = useCallback(async () => {
    if (!evalRunId) {
      setShowToast(true);
      setTimeout(() => setShowToast(false), 3000);
      return;
    }

    requestGeneration(id);
    try {
      const response = await generateFigure(id, { evalRunId });
      useDemandStore.setState((s) => ({
        requests: {
          ...s.requests,
          [id]: { ...s.requests[id], figureHash: response.request_id },
        },
      }));
    } catch (err) {
      setError(id, err instanceof Error ? err.message : 'Request failed');
    }
  }, [id, evalRunId, requestGeneration, setError]);

  const width = isDep ? WIDTH_DEP : WIDTH_FULL;
  const headerHeight = isDep ? HEADER_HEIGHT_DEP : HEADER_HEIGHT_FULL;
  const rowHeight = isDep ? ROW_HEIGHT_DEP : ROW_HEIGHT_FULL;

  const hasPreview = status === 'ready' && previewData != null;
  const PREVIEW_HEIGHT = hasPreview ? 120 : 0;

  const inputCount = spec.inputPorts.length;
  const outputCount = spec.outputPorts.length;
  const rowCount = Math.max(1, inputCount, outputCount);
  const bodyHeight = BODY_PADDING * 2 + rowCount * rowHeight;
  const totalHeight = headerHeight + bodyHeight + (canGenerate ? GENERATE_BTN_HEIGHT : 0) + PREVIEW_HEIGHT;

  const rowCenter = (index: number) => BODY_PADDING + rowHeight * (index + 0.5);

  return (
    <div
      className={clsx(
        'relative rounded-xl border shadow-soft backdrop-blur transition-all duration-150',
        isDep ? 'bg-white/70' : 'bg-white/90',
        selected
          ? 'border-brand-500 ring-1 ring-brand-500/40'
          : isDep
            ? 'border-slate-200/60'
            : 'border-emerald-200'
      )}
      style={{ width, height: totalHeight }}
    >
      {/* Header */}
      <div
        className={clsx(
          'px-3 overflow-hidden border-b rounded-t-xl',
          isDep
            ? 'py-1.5 flex items-center justify-between gap-2 bg-slate-50/60 border-slate-100/60'
            : 'py-1.5 flex flex-col justify-center bg-emerald-50/60 border-emerald-100/60'
        )}
        style={{ height: headerHeight }}
      >
        {isDep ? (
          <>
            <div className="text-xs text-slate-600 font-medium truncate" title={spec.label}>
              {spec.label}
            </div>
            <div className="text-[10px] uppercase tracking-wide shrink-0 text-slate-400">
              {spec.type}
            </div>
          </>
        ) : (
          <>
            <div className="text-sm text-slate-800 font-medium leading-tight" title={spec.label}>
              {spec.label}
            </div>
            <div className="text-[10px] uppercase tracking-wide text-emerald-500 mt-0.5">
              {spec.type}
            </div>
          </>
        )}
      </div>

      {/* Ports */}
      <div
        className={clsx('relative', isDep ? 'text-[10px]' : 'text-xs')}
        style={{ height: bodyHeight, padding: BODY_PADDING }}
      >
        {/* Input handles */}
        {spec.inputPorts.map((port, index) => (
          <Handle
            key={`in-${port}`}
            type="target"
            position={Position.Left}
            id={port}
            style={{
              top: rowCenter(index),
              left: HANDLE_OFFSET,
              transform: 'translateY(-50%)',
              clipPath: 'polygon(0% 0%, 100% 50%, 0% 100%)',
              width: isDep ? '6px' : '8px',
              height: isDep ? '6px' : '8px',
            }}
            className={clsx(
              'border border-white shadow-soft',
              isDep ? 'bg-slate-300' : 'bg-emerald-400'
            )}
          />
        ))}

        {/* Output handles */}
        {spec.outputPorts.map((port, index) => (
          <Handle
            key={`out-${port}`}
            type="source"
            position={Position.Right}
            id={port}
            style={{
              top: rowCenter(index),
              right: HANDLE_OFFSET,
              transform: 'translateY(-50%)',
              clipPath: 'polygon(0% 0%, 100% 50%, 0% 100%)',
              width: isDep ? '6px' : '8px',
              height: isDep ? '6px' : '8px',
            }}
            className={clsx(
              'border border-white shadow-soft',
              isDep ? 'bg-slate-300' : 'bg-emerald-400'
            )}
          />
        ))}

        {/* Input labels */}
        {spec.inputPorts.map((port, index) => (
          <div
            key={`label-in-${port}`}
            className={clsx(
              'absolute flex items-center',
              isDep ? 'text-slate-400' : 'text-slate-600'
            )}
            style={{
              top: rowCenter(index),
              left: LABEL_OFFSET,
              transform: 'translateY(-50%)',
            }}
          >
            {port}
          </div>
        ))}

        {/* Output labels */}
        {spec.outputPorts.map((port, index) => (
          <div
            key={`label-out-${port}`}
            className={clsx(
              'absolute flex items-center justify-end',
              isDep ? 'text-slate-400' : 'text-slate-600'
            )}
            style={{
              top: rowCenter(index),
              right: LABEL_OFFSET,
              transform: 'translateY(-50%)',
            }}
          >
            {port}
          </div>
        ))}
      </div>

      {/* Generate button for figure-producing nodes */}
      {canGenerate && (
        <div className="px-2 pb-1.5 flex justify-center">
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (status !== 'running') handleGenerate();
            }}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-1 rounded-lg text-[11px] font-medium transition-colors w-full justify-center',
              status === 'running'
                ? 'bg-blue-50 text-blue-500 cursor-wait'
                : status === 'ready'
                  ? 'bg-emerald-50 text-emerald-600 hover:bg-emerald-100'
                  : status === 'error'
                    ? 'bg-red-50 text-red-500 hover:bg-red-100'
                    : 'bg-slate-50 text-slate-600 hover:bg-emerald-50 hover:text-emerald-600'
            )}
            title={
              !evalRunId
                ? 'Select or create an evaluation run first'
                : status === 'running' ? 'Generating...'
                  : status === 'ready' ? 'Re-generate figure'
                    : status === 'error' ? 'Retry generation'
                      : 'Generate figure'
            }
          >
            {status === 'running' ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : status === 'ready' ? (
              <Image className="w-3 h-3" />
            ) : (
              <Play className="w-3 h-3" />
            )}
            <span>
              {status === 'running' ? 'Generating...'
                : status === 'ready' ? 'View Figure'
                  : status === 'error' ? 'Retry'
                    : 'Generate'}
            </span>
          </button>
        </div>
      )}

      {/* Inline figure preview thumbnail */}
      {hasPreview && (
        <div className="px-2 pb-1.5">
          <div className="w-full bg-slate-50 rounded-lg border border-slate-200 overflow-hidden"
               style={{ height: PREVIEW_HEIGHT }}>
            {previewLoading ? (
              <div className="flex items-center justify-center h-full">
                <Loader2 className="w-4 h-4 text-slate-400 animate-spin" />
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-[10px] text-emerald-600">
                <Image className="w-4 h-4 mr-1" />
                Figure ready — click View
              </div>
            )}
          </div>
        </div>
      )}

      {/* Toast: no eval run selected */}
      {showToast && (
        <div className="absolute -top-10 left-1/2 -translate-x-1/2 z-50 whitespace-nowrap">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200 shadow-lg text-[11px] text-amber-700">
            <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            Select or create an eval run first
          </div>
        </div>
      )}
    </div>
  );
}

/** Smaller variant used for dependency nodes. Same component, just aliased for clarity. */
export const AnalysisDepNode = AnalysisNode;
