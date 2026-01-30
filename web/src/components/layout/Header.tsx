import {
  LayoutPanelLeft,
  Settings,
  User,
  Save,
  FolderOpen,
  Plus,
  Download,
  ChevronUp,
  ChevronDown,
} from 'lucide-react';
import { useState } from 'react';
import { useGraphsList, useSaveGraph } from '@/hooks/useGraphs';
import { fetchGraph, exportGraph } from '@/api/client';
import { useGraphStore } from '@/stores/graphStore';
import { useLayoutStore } from '@/stores/layoutStore';

export function Header() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const graphsQuery = useGraphsList();
  const saveMutation = useSaveGraph();
  const { graph, uiState, graphId, isDirty, markSaved, hydrateGraph, resetGraph } = useGraphStore();
  const { topCollapsed, bottomCollapsed, toggleTop, toggleBottom } = useLayoutStore();

  const handleSave = async () => {
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
    <header className="h-12 flex items-center justify-between px-4 border-b border-slate-100 bg-white/80 backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 font-display text-sm tracking-[0.2em] text-slate-600">
          <LayoutPanelLeft className="w-4 h-4 text-brand-500" />
          FEEDBAX
        </div>
        <div className="h-5 w-px bg-slate-200" />
        <div className="relative flex items-center gap-2">
          <button
            className="text-sm font-medium text-ink hover:text-brand-600"
            onClick={() => setMenuOpen((prev) => !prev)}
          >
            Project: {graph.metadata?.name ?? 'Untitled Graph'}
          </button>
          {isDirty && <span className="text-amber-500 text-sm">•</span>}
          {menuOpen && (
            <div className="absolute left-0 mt-2 w-64 rounded-xl border border-slate-100 bg-white shadow-lift z-20 p-2">
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
          className="p-1.5 rounded-full hover:bg-slate-100"
          title={topCollapsed ? 'Expand model shelf' : 'Collapse model shelf'}
          onClick={toggleTop}
        >
          <ChevronUp className={topCollapsed ? 'w-4 h-4 rotate-180' : 'w-4 h-4'} />
        </button>
        <button
          className="p-1.5 rounded-full hover:bg-slate-100"
          title={bottomCollapsed ? 'Expand workbench shelf' : 'Collapse workbench shelf'}
          onClick={toggleBottom}
        >
          <ChevronDown className={bottomCollapsed ? 'w-4 h-4 rotate-180' : 'w-4 h-4'} />
        </button>
        <button
          className="p-1.5 rounded-full hover:bg-slate-100"
          title="Save"
          onClick={handleSave}
        >
          <Save className="w-4 h-4" />
        </button>
        <button
          className="p-1.5 rounded-full hover:bg-slate-100"
          title="Export JSON"
          onClick={handleExport}
          disabled={!graphId || exporting}
        >
          <Download className="w-4 h-4" />
        </button>
        <button
          className="p-1.5 rounded-full hover:bg-slate-100"
          title="Open project"
          onClick={() => setMenuOpen((prev) => !prev)}
        >
          <FolderOpen className="w-4 h-4" />
        </button>
        <button className="p-1.5 rounded-full hover:bg-slate-100">
          <Settings className="w-4 h-4" />
        </button>
        <button className="p-1.5 rounded-full hover:bg-slate-100">
          <User className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}
