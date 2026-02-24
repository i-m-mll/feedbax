import {
  LayoutPanelLeft,
  Settings,
  User,
  Save,
  FolderOpen,
  Plus,
  Download,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useGraphsList, useSaveGraph } from '@/hooks/useGraphs';
import { fetchGraph, exportGraph } from '@/api/client';
import { useGraphStore } from '@/stores/graphStore';
import { useSettingsStore } from '@/stores/settingsStore';

export function Header() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const menuTriggerRef = useRef<HTMLButtonElement | null>(null);
  const settingsRef = useRef<HTMLDivElement | null>(null);
  const graphsQuery = useGraphsList();
  const saveMutation = useSaveGraph();
  const { showMinimap, toggleMinimap } = useSettingsStore();
  const {
    graph,
    uiState,
    graphId,
    graphStack,
    isDirty,
    markSaved,
    hydrateGraph,
    resetGraph,
  } = useGraphStore();
  const inSubgraph = graphStack.length > 0;

  useEffect(() => {
    if (!menuOpen && !settingsOpen) return;
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (!target) return;
      if (
        menuOpen &&
        menuRef.current &&
        !menuRef.current.contains(target) &&
        !menuTriggerRef.current?.contains(target)
      ) {
        setMenuOpen(false);
      }
      if (settingsOpen && settingsRef.current && !settingsRef.current.contains(target)) {
        setSettingsOpen(false);
      }
    };
    document.addEventListener('pointerdown', handlePointerDown);
    return () => document.removeEventListener('pointerdown', handlePointerDown);
  }, [menuOpen, settingsOpen]);

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
      hydrateGraph(data.graph, data.ui_state ?? undefined, id);
      setMenuOpen(false);
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
    <header className="relative z-40 h-12 flex items-center justify-between px-4 border-b border-slate-100 bg-white/80 backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 font-display text-sm tracking-[0.2em] text-slate-600">
          <LayoutPanelLeft className="w-4 h-4 text-brand-500" />
          FEEDBAX
        </div>
        <div className="h-5 w-px bg-slate-200" />
        <div ref={menuRef} className="relative flex items-center gap-2">
          <button
            className="text-sm font-medium text-ink hover:text-brand-600"
            onClick={() => {
              setSettingsOpen(false);
              setMenuOpen((prev) => !prev);
            }}
          >
            Project: {graph.metadata?.name ?? 'Untitled Graph'}
          </button>
          {isDirty && <span className="text-amber-500 text-sm">•</span>}
          {menuOpen && (
            <div className="absolute left-0 top-full mt-2 w-64 rounded-xl border border-slate-100 bg-white shadow-lift z-50 p-2">
              <button
                onClick={() => {
                  resetGraph();
                  setMenuOpen(false);
                }}
                className="w-full flex items-center gap-2 text-sm px-2 py-2 rounded-lg hover:bg-slate-50"
              >
                <Plus className="w-4 h-4 text-slate-500" />
                New project
              </button>
              <div className="mt-2 border-t border-slate-100 pt-2">
                <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 px-2">
                  Saved projects
                </div>
                <div className="max-h-48 overflow-y-auto">
                  {(graphsQuery.data?.graphs ?? []).map((item: any) => (
                    <button
                      key={item.id}
                      onClick={() => handleOpen(item.id)}
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
            </div>
          )}
        </div>
      </div>
      <div className="flex items-center gap-3 text-slate-500">
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
        <button
          className="p-1.5 rounded-full hover:bg-slate-100"
          title="Open project"
          ref={menuTriggerRef}
          onClick={() => {
            setSettingsOpen(false);
            setMenuOpen((prev) => !prev);
          }}
        >
          <FolderOpen className="w-4 h-4" />
        </button>
        <div ref={settingsRef} className="relative">
          <button
            className="p-1.5 rounded-full hover:bg-slate-100"
            title="Settings"
            onClick={() => {
              setMenuOpen(false);
              setSettingsOpen((prev) => !prev);
            }}
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
