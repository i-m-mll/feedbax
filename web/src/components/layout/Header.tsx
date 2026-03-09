import {
  Settings,
  Save,
  FolderOpen,
  Plus,
  Download,
  X,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useGraphsList, useSaveGraph } from '@/hooks/useGraphs';
import { fetchGraph, exportGraph } from '@/api/client';
import { useGraphStore } from '@/stores/graphStore';
import { useProjectsStore } from '@/stores/projectsStore';
import { useTrainingStore } from '@/stores/trainingStore';
import { SettingsOverlay } from '@/components/layout/SettingsOverlay';

export function Header() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [pendingTab, setPendingTab] = useState<{ name: string } | null>(null);
  const [renamingTabId, setRenamingTabId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const pendingInputRef = useRef<HTMLInputElement | null>(null);
  const renameInputRef = useRef<HTMLInputElement | null>(null);
  const saveMutation = useSaveGraph();
  const {
    graph,
    uiState,
    graphId,
    graphStack,
    isDirty,
    markSaved,
  } = useGraphStore();
  const { tabs, activeTabId, openNewTab, openProjectInTab, switchTab, closeTab, renameTab } = useProjectsStore();
  const inSubgraph = graphStack.length > 0;

  // Focus pending tab input when it appears
  useEffect(() => {
    if (pendingTab !== null && pendingInputRef.current) {
      pendingInputRef.current.focus();
    }
  }, [pendingTab]);

  // Focus rename input when it appears
  useEffect(() => {
    if (renamingTabId !== null && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingTabId]);

  const handleSave = async () => {
    if (inSubgraph) return;
    try {
      const response = await saveMutation.mutateAsync({
        graphId,
        graph,
        uiState,
      });
      if ('id' in response) {
        markSaved(response.id);
      } else if (graphId) {
        markSaved(graphId);
      }
    } catch (error) {
      console.error(error);
    }
  };

  const handleOpen = async (id: string) => {
    try {
      const data = await fetchGraph(id);
      openProjectInTab(id, data.graph, data.ui_state ?? {
        viewport: { x: 0, y: 0, zoom: 1 },
        node_states: {},
      });
      if (data.demo_training_data) {
        useTrainingStore.getState().seedDemoData(data.demo_training_data);
      }
      localStorage.setItem('feedbax:lastProjectId', id);
    } catch (error) {
      console.error(error);
    }
  };

  // Auto-load the last opened project on mount
  useEffect(() => {
    if (graphId !== null) return;
    const lastId = localStorage.getItem('feedbax:lastProjectId');
    if (lastId) {
      handleOpen(lastId);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleExport = async () => {
    if (!graphId) return;
    setExporting(true);
    try {
      const data = await exportGraph(graphId, 'json');
      const blob = new Blob([data.content], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = data.filename || 'graph.json';
      link.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  };

  return (
    <>
    {settingsOpen && <SettingsOverlay onClose={() => setSettingsOpen(false)} />}
    <header className="relative z-40 h-12 flex items-center gap-2 px-3 border-b border-slate-100 bg-white/80 backdrop-blur">
      {/* Logo — fixed width */}
      <div className="flex-none flex items-center gap-2 font-display text-sm tracking-[0.2em] text-slate-600 pr-2">
        <img src="/icon.svg" alt="feedbax studio logo" className="h-7 w-7" />
      </div>

      {/* Scrollable tab bar — fills remaining space */}
      <div className="flex-1 min-w-0 flex items-center overflow-x-auto gap-1 no-scrollbar">
        {tabs.map((tab) => {
          const isActive = tab.tabId === activeTabId;
          const isRenaming = renamingTabId === tab.tabId;
          return (
            <div
              key={tab.tabId}
              className={[
                'flex-none flex items-center gap-1.5 h-8 px-3 rounded-lg text-sm font-medium max-w-[160px] group transition-colors cursor-pointer',
                isActive
                  ? 'bg-slate-100 text-slate-900'
                  : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700',
              ].join(' ')}
              onClick={() => switchTab(tab.tabId)}
            >
              {isRenaming ? (
                <input
                  ref={renameInputRef}
                  type="text"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      const name = renameValue.trim() || tab.label;
                      renameTab(tab.tabId, name);
                      setRenamingTabId(null);
                    } else if (e.key === 'Escape') {
                      setRenamingTabId(null);
                    }
                    e.stopPropagation();
                  }}
                  onBlur={() => {
                    const name = renameValue.trim() || tab.label;
                    renameTab(tab.tabId, name);
                    setRenamingTabId(null);
                  }}
                  onClick={(e) => e.stopPropagation()}
                  className="bg-transparent outline-none text-sm text-slate-900 w-24 min-w-0"
                  autoFocus
                />
              ) : (
                <span
                  className="truncate min-w-0"
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    setRenamingTabId(tab.tabId);
                    setRenameValue(tab.label);
                  }}
                >
                  {tab.label}
                </span>
              )}
              {!isRenaming && (isActive ? isDirty : tab.graphSnapshot.isDirty) && (
                <span className="flex-none text-amber-500 text-xs leading-none">•</span>
              )}
              {!isRenaming && tabs.length > 1 && (
                <span
                  role="button"
                  tabIndex={0}
                  aria-label={`Close ${tab.label}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    closeTab(tab.tabId);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.stopPropagation();
                      closeTab(tab.tabId);
                    }
                  }}
                  className={[
                    'flex-none opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-slate-200 transition-opacity',
                    isActive ? 'opacity-60' : '',
                  ].join(' ')}
                >
                  <X className="w-3 h-3" />
                </span>
              )}
            </div>
          );
        })}
        {pendingTab !== null && (
          <div className="flex items-center px-2 py-1 rounded-lg bg-slate-100 border border-blue-400">
            <input
              ref={pendingInputRef}
              type="text"
              value={pendingTab.name}
              onChange={(e) => setPendingTab({ name: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  const name = pendingTab.name.trim() || 'Untitled';
                  openNewTab(name);
                  setPendingTab(null);
                } else if (e.key === 'Escape') {
                  setPendingTab(null);
                }
              }}
              onBlur={() => {
                // Confirm on blur too (user clicked away)
                const name = pendingTab.name.trim();
                if (name) {
                  openNewTab(name);
                }
                setPendingTab(null);
              }}
              className="bg-transparent outline-none text-sm text-slate-900 w-28 min-w-0"
              placeholder="Tab name..."
              autoFocus
            />
          </div>
        )}
        <button
          onClick={() => setPendingTab({ name: '' })}
          className="flex-none p-1.5 rounded-lg text-slate-400 hover:bg-slate-50 hover:text-slate-600"
          title="New project tab"
        >
          <Plus className="w-4 h-4" />
        </button>
      </div>

      {/* Right-side action buttons */}
      <div className="flex-none flex items-center gap-3 text-slate-500">
        <button
          className="p-1.5 rounded-full hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed"
          title={inSubgraph ? 'Return to model root to save' : 'Save'}
          onClick={handleSave}
          disabled={inSubgraph}
        >
          <Save className="w-4 h-4" />
        </button>
        <button
          className="p-1.5 rounded-full hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed"
          title={inSubgraph ? 'Return to model root to export' : 'Export JSON'}
          onClick={handleExport}
          disabled={!graphId || exporting || inSubgraph}
        >
          <Download className="w-4 h-4" />
        </button>
        <OpenProjectDropdown onOpen={handleOpen} onBeforeOpen={() => setSettingsOpen(false)} />
        <button
          className="p-1.5 rounded-full hover:bg-slate-100"
          title="Settings"
          onClick={() => setSettingsOpen((prev) => !prev)}
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </header>
    </>
  );
}

interface OpenProjectDropdownProps {
  onOpen: (id: string) => void;
  onBeforeOpen?: () => void;
}

function OpenProjectDropdown({ onOpen, onBeforeOpen }: OpenProjectDropdownProps) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const dropdownRef = useRef<HTMLDivElement | null>(null);
  const graphsQuery = useGraphsList();
  // Use a ref so addEventListener/removeEventListener always see the same function identity
  const handlerRef = useRef<((event: PointerEvent) => void) | null>(null);

  const handlePointerDown = useCallback((event: PointerEvent) => {
    const target = event.target as Node | null;
    if (!target) return;
    if (
      dropdownRef.current &&
      !dropdownRef.current.contains(target) &&
      !triggerRef.current?.contains(target)
    ) {
      if (handlerRef.current) {
        document.removeEventListener('pointerdown', handlerRef.current);
        handlerRef.current = null;
      }
      setOpen(false);
    }
  }, []);

  const toggleOpen = () => {
    if (!open) {
      onBeforeOpen?.();
      handlerRef.current = handlePointerDown;
      document.addEventListener('pointerdown', handlePointerDown);
    } else {
      if (handlerRef.current) {
        document.removeEventListener('pointerdown', handlerRef.current);
        handlerRef.current = null;
      }
    }
    setOpen((prev) => !prev);
  };

  return (
    <div className="relative">
      <button
        ref={triggerRef}
        className="p-1.5 rounded-full hover:bg-slate-100 text-slate-500"
        title="Open project in new tab"
        onClick={toggleOpen}
      >
        <FolderOpen className="w-4 h-4" />
      </button>
      {open && (
        <div
          ref={dropdownRef}
          className="absolute right-0 top-full mt-2 w-64 rounded-xl border border-slate-100 bg-white shadow-lift z-50 p-2"
        >
          <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 px-2 pb-1">
            Open in new tab
          </div>
          <div className="max-h-48 overflow-y-auto">
            {(graphsQuery.data?.graphs ?? []).map((item: any) => (
              <button
                key={item.id}
                onClick={() => {
                  onOpen(item.id);
                  if (handlerRef.current) {
                    document.removeEventListener('pointerdown', handlerRef.current);
                    handlerRef.current = null;
                  }
                  setOpen(false);
                }}
                className="w-full text-left text-sm px-2 py-2 rounded-lg hover:bg-slate-50"
              >
                <div className="font-medium text-slate-700">{item.metadata.name}</div>
                <div className="text-xs text-slate-400">{item.metadata.updated_at}</div>
              </button>
            ))}
            {graphsQuery.isLoading && (
              <div className="text-xs text-slate-400 px-2 py-2">Loading…</div>
            )}
            {!graphsQuery.isLoading && (graphsQuery.data?.graphs?.length ?? 0) === 0 && (
              <div className="text-xs text-slate-400 px-2 py-2">No saved projects yet.</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
