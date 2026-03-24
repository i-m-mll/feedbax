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
import { fetchAnalysisClasses } from '@/api/analysisAPI';
import type { AnalysisNodeData } from '@/stores/analysisStore';
import { Plus, X } from 'lucide-react';
import clsx from 'clsx';

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
  } = useAnalysisStore();

  // Load analysis classes on mount
  useEffect(() => {
    if (analysisClasses.length > 0) return;
    fetchAnalysisClasses().then(setAnalysisClasses).catch(() => {});
  }, [analysisClasses.length, setAnalysisClasses]);

  // Auto-create a first page when no pages exist yet
  useEffect(() => {
    if (pages.length === 0) {
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
        <div className="flex-1 min-w-0">
          <ReactFlowProvider>
            <AnalysisCanvas />
          </ReactFlowProvider>
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

function NodeDetailPanel({
  selectedData,
  selectedNodeId,
}: {
  selectedData: AnalysisNodeData;
  selectedNodeId: string | null;
}) {
  return (
    <div className="p-4">
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
              : 'bg-emerald-50 text-emerald-600 border border-emerald-100',
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
  );
}
