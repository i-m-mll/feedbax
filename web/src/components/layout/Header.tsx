import {
  Settings,
  User,
  Save,
  FolderOpen,
  Plus,
  Download,
  X,
} from 'lucide-react';
import LogoSvg from '@/assets/logo.svg?url';
import { useCallback, useRef, useState } from 'react';
import { useGraphsList, useSaveGraph } from '@/hooks/useGraphs';
import { fetchGraph, exportGraph } from '@/api/client';
import { useGraphStore } from '@/stores/graphStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { useProjectsStore } from '@/stores/projectsStore';

export function Header() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const settingsRef = useRef<HTMLDivElement | null>(null);
  const saveMutation = useSaveGraph();
  const { showMinimap, toggleMinimap } = useSettingsStore();
  const {
    graph,
    uiState,
    graphId,
    graphStack,
    isDirty,
    markSaved,
  } = useGraphStore();
  const { tabs, activeTabId, openNewTab, openProjectInTab, switchTab, closeTab } = useProjectsStore();
  const inSubgraph = graphStack.length > 0;

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
    } catch (error) {
      console.error(error);
    }
  };

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
    <header className="relative z-40 h-12 flex items-center gap-2 px-3 border-b border-slate-100 bg-white/80 backdrop-blur">
      {/* Logo — fixed width */}
      <div className="flex-none flex items-center gap-2 font-display text-sm tracking-[0.2em] text-slate-600 pr-2">
        <img src={LogoSvg} alt="feedbax studio logo" className="h-7 w-7" />
      </div>

      {/* Scrollable tab bar — fills remaining space */}
      <div className="flex-1 min-w-0 flex items-center overflow-x-auto gap-1 no-scrollbar">
        {tabs.map((tab) => {
          const isActive = tab.tabId === activeTabId;
          return (
            <button
              key={tab.tabId}
              onClick={() => switchTab(tab.tabId)}
              className={[
                'flex-none flex items-center gap-1.5 h-8 px-3 rounded-lg text-sm font-medium max-w-[160px] group transition-colors',
                isActive
                  ? 'bg-slate-100 text-slate-900'
                  : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700',
              ].join(' ')}
            >
              <span className="truncate min-w-0">
                {tab.label}
              </span>
              {isActive && isDirty && (
                <span className="flex-none text-amber-500 text-xs leading-none">•</span>
              )}
              {tabs.length > 1 && (
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
            </button>
          );
        })}
        <button
          onClick={openNewTab}
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
        <div ref={settingsRef} className="relative">
          <button
            className="p-1.5 rounded-full hover:bg-slate-100"
            title="Settings"
            onClick={() => setSettingsOpen((prev) => !prev)}
          >
            <Settings className="w-4 h-4" />
          </button>
          {settingsOpen && (
            <div className="absolute right-0 top-full mt-2 w-64 rounded-xl border border-slate-100 bg-white shadow-lift z-50 p-3">
              <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400">
                App Settings
              </div>
              <label className="mt-3 flex items-center justify-between text-sm text-slate-600">
                <span>Show minimap</span>
                <input
                  type="checkbox"
                  checked={showMinimap}
                  onChange={toggleMinimap}
                  className="h-4 w-4 rounded border-slate-300 text-brand-500 focus:ring-brand-500"
                />
              </label>
            </div>
          )}
        </div>
        <button className="p-1.5 rounded-full hover:bg-slate-100">
          <User className="w-4 h-4" />
        </button>
      </div>
    </header>
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
