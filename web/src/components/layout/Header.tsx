import {
  Settings,
  Save,
  FolderOpen,
  FilePlus,
  Download,
  X,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import { toast } from 'sonner';
import { useGraphsList, useSaveGraph } from '@/hooks/useGraphs';
import { fetchGraph, exportGraph, createGraph, updateGraph } from '@/api/client';
import { useGraphStore, createBlankGraph } from '@/stores/graphStore';
import {
  getLastProjectId,
  persistLocalProjectTabs,
  setLastProjectId,
  useProjectsStore,
} from '@/stores/projectsStore';
import { useRunStore } from '@/stores/runStore';
import { useTrainingStore } from '@/stores/trainingStore';
import { buildWorkspaceSnapshot } from '@/stores/workspaceStore';
import { SettingsOverlay } from '@/components/layout/SettingsOverlay';
import { PROJECT_TEMPLATES } from '@/data/project-templates';
import type { AnalysisGraphSpec, AnalysisSnapshot } from '@/types/analysis';

type ProjectOverlaySection = 'projects' | 'examples' | 'import';

export function Header() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [projectOverlaySection, setProjectOverlaySection] = useState<ProjectOverlaySection | null>(null);
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
  const {
    tabs,
    activeTabId,
    hasRestoredLocalTabs,
    openNewTab,
    openProjectInTab,
    switchTab,
    closeTab,
    renameTab,
  } = useProjectsStore();
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
    persistLocalProjectTabs();
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
      persistLocalProjectTabs();
    } catch (error) {
      console.error(error);
      persistLocalProjectTabs();
      toast.error('Saved locally; backend is unreachable', { id: 'save-local-fallback' });
    }
  };

  const handleOpen = async (id: string) => {
    try {
      const data = await fetchGraph(id);
      // Build analysis snapshot from persisted pages (convert snake_case wire format)
      let analysisSnapshot: AnalysisSnapshot | null = null;
      if (data.analysis_pages && data.analysis_pages.length > 0) {
        const pages = data.analysis_pages.map((wp: any) => ({
          id: wp.id,
          name: wp.name,
          graphSpec: wp.graph_spec as unknown as AnalysisGraphSpec,
          evalParams: wp.eval_params as Record<string, unknown>,
          viewport: wp.viewport,
          evalRunId: wp.eval_run_id ?? null,
          expandedFieldPaths: (wp.expanded_field_paths as string[]) ?? [],
        }));
        // Restore the persisted active page, falling back to the first page
        const restoredActiveId = data.active_analysis_page_id;
        const activePageId = restoredActiveId && pages.some((p) => p.id === restoredActiveId)
          ? restoredActiveId
          : pages[0].id;
        analysisSnapshot = { pages, activePageId };
      }
      openProjectInTab(
        id,
        data.graph,
        data.ui_state ?? {
          viewport: { x: 0, y: 0, zoom: 1 },
          node_states: {},
        },
        data.metadata?.name ?? undefined,
        analysisSnapshot,
        data.workspace,
      );
      useRunStore.getState().hydrateFromWorkspace(data.workspace);
      if (data.demo_training_data) {
        const demo = data.demo_training_data;
        const totalBatches = demo.loss_history.length;
        const lossHistory = demo.loss_history.map((entry) => ({
          batch: entry.batch,
          total_batches: totalBatches,
          loss: entry.loss,
          loss_terms: {},
          grad_norm: 0,
          step_time_ms: 0,
          metrics: {},
          status: 'completed',
        }));
        const traj = demo.latest_trajectory;
        const latestTrajectory = traj
          ? {
              batch: totalBatches - 1,
              effector: traj.effector_pos,
              target: traj.target_pos,
              t: traj.effector_pos.map((_, i) => (i / traj.effector_pos.length) * 0.8),
            }
          : null;
        useTrainingStore.getState().seedDemoData({ lossHistory, latestTrajectory });
      }
      setLastProjectId(id);
    } catch (error) {
      console.error(error);
    }
  };

  // Auto-load the last opened project on mount
  useEffect(() => {
    if (graphId !== null) return;
    if (hasRestoredLocalTabs) return;
    const lastId = getLastProjectId();
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
    {projectOverlaySection && (
      <ProjectOpenOverlay
        initialSection={projectOverlaySection}
        onClose={() => setProjectOverlaySection(null)}
        onOpenSaved={handleOpen}
      />
    )}
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
      </div>

      {/* Right-side action buttons */}
      <div className="flex-none flex items-center gap-3 text-slate-500">
        <button
          className="p-1.5 rounded-full hover:bg-slate-100"
          title="New project"
          onClick={() => {
            setProjectOverlaySection(null);
            setSettingsOpen(false);
            setPendingTab({ name: '' });
          }}
        >
          <FilePlus className="w-4 h-4" />
        </button>
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
          className="p-1.5 rounded-full hover:bg-slate-100 text-slate-500"
          title="Open project"
          onClick={() => {
            setSettingsOpen(false);
            setProjectOverlaySection('projects');
          }}
        >
          <FolderOpen className="w-4 h-4" />
        </button>
        <button
          className="p-1.5 rounded-full hover:bg-slate-100"
          title="Settings"
          onClick={() => {
            setProjectOverlaySection(null);
            setSettingsOpen((prev) => !prev);
          }}
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </header>
    </>
  );
}
function ProjectOpenOverlay({
  initialSection,
  onClose,
  onOpenSaved,
}: {
  initialSection: ProjectOverlaySection;
  onClose: () => void;
  onOpenSaved: (id: string) => Promise<void>;
}) {
  const [activeSection, setActiveSection] = useState<ProjectOverlaySection>(initialSection);
  const graphsQuery = useGraphsList();
  const { openProjectInTab } = useProjectsStore();

  useEffect(() => {
    setActiveSection(initialSection);
  }, [initialSection]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const handleLoadTemplate = useCallback(
    async (templateIdx: number) => {
      const template = PROJECT_TEMPLATES[templateIdx];
      if (!template) return;

      let modelGraph: ReturnType<typeof createBlankGraph>;
      let uiState: any;

      if (template.createModelGraph) {
        const result = template.createModelGraph();
        modelGraph = result.graph;
        uiState = result.uiState;
      } else {
        modelGraph = createBlankGraph();
        modelGraph.metadata!.name = template.name;
        uiState = { viewport: { x: 0, y: 0, zoom: 1 }, node_states: {} };
      }

      const analysisSnapshot = template.createAnalysis();
      let workspace = buildWorkspaceSnapshot({
        workspace: null,
        graph: modelGraph,
        uiState,
        trainingSpec: useTrainingStore.getState().trainingSpec,
        taskSpec: useTrainingStore.getState().taskSpec,
        analysisSnapshot,
        projectName: template.name,
      });
      if (template.createWorkspace) {
        workspace = template.createWorkspace({
          baseWorkspace: workspace,
          graph: modelGraph,
          uiState,
          analysisSnapshot,
        });
      }

      try {
        const response = await createGraph(modelGraph, uiState, workspace);
        const graphId = response.id;
        const analysisPages = analysisSnapshot.pages.map((page) => ({
          id: page.id,
          name: page.name,
          graph_spec: page.graphSpec,
          eval_params: page.evalParams,
          viewport: page.viewport,
          eval_run_id: page.evalRunId ?? null,
          expanded_field_paths: page.expandedFieldPaths ?? [],
        }));
        await updateGraph(
          graphId,
          null,
          null,
          analysisPages,
          analysisSnapshot.activePageId,
          workspace,
        );
        openProjectInTab(
          graphId,
          modelGraph,
          uiState,
          template.name,
          analysisSnapshot,
          workspace,
        );
        useRunStore.getState().hydrateFromWorkspace(workspace);
        setLastProjectId(graphId);
      } catch (error) {
        console.error('Failed to save example project to backend:', error);
        openProjectInTab(
          '',
          modelGraph,
          uiState,
          template.name,
          analysisSnapshot,
          workspace,
        );
        useRunStore.getState().hydrateFromWorkspace(workspace);
      }
      persistLocalProjectTabs();
      onClose();
    },
    [onClose, openProjectInTab],
  );

  const navItems: Array<{ id: ProjectOverlaySection; label: string }> = [
    { id: 'projects', label: 'Projects' },
    { id: 'examples', label: 'Examples' },
    { id: 'import', label: 'Import' },
  ];

  return (
    <div className="fixed inset-x-0 bottom-0 top-12 z-50 flex bg-white">
      <nav className="w-48 shrink-0 border-r border-slate-100 bg-slate-50 px-4 py-6">
        <div className="space-y-1">
          {navItems.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setActiveSection(item.id)}
              className={clsx(
                'w-full rounded-md px-3 py-2 text-left text-sm font-medium',
                activeSection === item.id
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-500 hover:bg-white hover:text-slate-800'
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
      </nav>
      <div className="min-w-0 flex-1 overflow-y-auto p-6">
        <button
          type="button"
          onClick={onClose}
          className="absolute right-4 top-4 rounded-md p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          title="Close"
        >
          <X className="h-4 w-4" />
        </button>

        {activeSection === 'projects' && (
          <div className="max-w-3xl">
            <div className="grid gap-2">
              {(graphsQuery.data?.graphs ?? []).map((item: any) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={async () => {
                    await onOpenSaved(item.id);
                    onClose();
                  }}
                  className="rounded-lg border border-slate-200 bg-white px-4 py-3 text-left hover:border-brand-200 hover:bg-slate-50"
                >
                  <div className="font-medium text-slate-800">{item.metadata.name}</div>
                  <div className="mt-1 text-xs text-slate-400">{item.metadata.updated_at}</div>
                </button>
              ))}
              {graphsQuery.isLoading && (
                <div className="rounded-lg border border-slate-200 px-4 py-3 text-sm text-slate-400">
                  Loading projects...
                </div>
              )}
              {!graphsQuery.isLoading && (graphsQuery.data?.graphs?.length ?? 0) === 0 && (
                <div className="rounded-lg border border-dashed border-slate-200 px-4 py-6 text-sm text-slate-400">
                  No saved projects yet.
                </div>
              )}
            </div>
          </div>
        )}

        {activeSection === 'examples' && (
          <div className="grid max-w-5xl gap-3 md:grid-cols-2">
            {PROJECT_TEMPLATES.map((template, index) => (
              <button
                key={template.id}
                type="button"
                onClick={() => handleLoadTemplate(index)}
                className="rounded-lg border border-slate-200 bg-white p-4 text-left hover:border-brand-200 hover:bg-slate-50"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate font-semibold text-slate-800">{template.name}</div>
                    <div className="mt-1 text-sm leading-5 text-slate-500">{template.description}</div>
                  </div>
                  {template.createWorkspace && (
                    <span className="shrink-0 rounded-full bg-brand-50 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-brand-700">
                      runs
                    </span>
                  )}
                </div>
                <div className="mt-3 flex flex-wrap gap-1">
                  {template.pageNames.map((pageName) => (
                    <span
                      key={pageName}
                      className="rounded-full bg-emerald-50 px-1.5 py-0.5 text-[10px] text-emerald-600"
                    >
                      {pageName}
                    </span>
                  ))}
                </div>
              </button>
            ))}
          </div>
        )}

        {activeSection === 'import' && (
          <div className="max-w-2xl rounded-lg border border-dashed border-slate-200 px-5 py-8 text-sm text-slate-500">
            Bundle import is not wired yet. This surface is reserved for project bundles and local
            files once the storage/import contract is settled.
          </div>
        )}
      </div>
    </div>
  );
}
