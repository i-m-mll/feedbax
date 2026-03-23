/**
 * AnalysisPanel — bottom shelf panel for the analysis DAG.
 *
 * Replaces the placeholder with a React Flow canvas showing the analysis
 * pipeline, plus a detail sidebar on the right for the selected node's
 * parameters.
 */

import { useCallback, useEffect, useMemo } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { AnalysisCanvas } from '@/components/analysis/AnalysisCanvas';
import { useAnalysisStore } from '@/stores/analysisStore';
import { fetchAnalysisClasses } from '@/api/analysisAPI';
import type { AnalysisNodeData } from '@/stores/analysisStore';
import clsx from 'clsx';

export function AnalysisPanel() {
  const {
    nodes,
    selectedNodeId,
    setSelectedNode,
    analysisClasses,
    setAnalysisClasses,
    graphSpec,
    loadGraph,
  } = useAnalysisStore();

  // Load analysis classes on mount
  useEffect(() => {
    if (analysisClasses.length > 0) return;
    fetchAnalysisClasses().then(setAnalysisClasses).catch(() => {});
  }, [analysisClasses.length, setAnalysisClasses]);

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
    <div className="flex h-full">
      {/* DAG canvas — fills available space */}
      <div className="flex-1 min-w-0">
        <ReactFlowProvider>
          <AnalysisCanvas />
        </ReactFlowProvider>
      </div>

      {/* Detail panel — shows when a node is selected */}
      {selectedData && (
        <div className="w-64 border-l border-slate-100 bg-white/90 p-4 overflow-y-auto shrink-0">
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400">
            {selectedData.spec.role === 'dependency' ? 'Dependency' : 'Analysis'}
          </div>
          <div className="mt-1 text-base font-semibold text-slate-800">
            {selectedData.spec.label}
          </div>
          <div className="mt-0.5 text-xs text-slate-500">{selectedData.spec.type}</div>

          {/* Category badge */}
          <div className="mt-3">
            <span
              className={clsx(
                'inline-block rounded-full px-2 py-0.5 text-[10px] uppercase tracking-wide',
                selectedData.spec.role === 'dependency'
                  ? 'bg-slate-100 text-slate-500'
                  : 'bg-emerald-50 text-emerald-600 border border-emerald-100'
              )}
            >
              {selectedData.spec.category}
            </span>
          </div>

          {/* Ports */}
          <div className="mt-4 space-y-3">
            {selectedData.spec.inputPorts.length > 0 && (
              <div>
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-1">
                  Inputs
                </div>
                <div className="space-y-0.5">
                  {selectedData.spec.inputPorts.map((port) => (
                    <div key={port} className="text-xs text-slate-600 pl-2 border-l-2 border-emerald-200">
                      {port}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {selectedData.spec.outputPorts.length > 0 && (
              <div>
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-1">
                  Outputs
                </div>
                <div className="space-y-0.5">
                  {selectedData.spec.outputPorts.map((port) => (
                    <div key={port} className="text-xs text-slate-600 pl-2 border-l-2 border-slate-200">
                      {port}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Parameters */}
          {Object.keys(selectedData.spec.params).length > 0 && (
            <div className="mt-4">
              <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-2">
                Parameters
              </div>
              <div className="space-y-2">
                {Object.entries(selectedData.spec.params).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between gap-2">
                    <span className="text-xs text-slate-500">{key}</span>
                    <span className="text-xs text-slate-700 font-medium bg-slate-50 rounded px-1.5 py-0.5">
                      {String(value)}
                    </span>
                  </div>
                ))}
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
      )}
    </div>
  );
}
