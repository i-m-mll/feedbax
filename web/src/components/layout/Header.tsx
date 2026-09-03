import {
  Settings,
  Save,
  FolderOpen,
  FilePlus,
  Download,
  X,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import clsx from 'clsx';
import { toast } from 'sonner';
import { useGraphsList } from '@/hooks/useGraphs';
import { fetchGraph, exportGraph } from '@/api/client';
import { useGraphStore, createBlankGraph } from '@/stores/graphStore';
import { actionErrorMessage } from '@/stores/storeActions';
import {
  getLastProjectId,
  persistLocalProjectTabs,
  setLastProjectId,
  useProjectsStore,
} from '@/stores/projectsStore';
import { useRunStore } from '@/stores/runStore';
import { useTrainingStore } from '@/stores/trainingStore';
import { useCompileStatusStore } from '@/stores/compileStatusStore';
import { buildWorkspaceSnapshot, hydrateWorkspacePresentation } from '@/stores/workspaceStore';
import { SettingsOverlay } from '@/components/layout/SettingsOverlay';
import { PROJECT_TEMPLATES } from '@/data/project-templates';
import { normalizeTrainingTrajectoryPayload } from '@/features/scenario/liveTraining';
import { analysisSnapshotFromWorkspaceDocument } from '@/utils/analysisCanvasLayout';
import {
  buildDetachedStudioDocument,
  saveActiveStudioDocument,
  studioPersistence,
} from '@/services/studioPersistence';

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
  const startupAutoloadAttemptedRef = useRef(false);
  const graphsQuery = useGraphsList();
  const {
    graphId,
    graphStack,
    isDirty,
    saveStatus,
    saveError,
  } = useGraphStore(
    useShallow((state) => ({
      graphId: state.graphId,
      graphStack: state.graphStack,
      isDirty: state.isDirty,
      saveStatus: state.saveStatus,
      saveError: state.saveError,
    }))
  );
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
      const outcome = await saveActiveStudioDocument('manual');
      if (!outcome.ok) return;
      persistLocalProjectTabs();
      toast.success('Project saved.', { id: 'project-save-success' });
    } catch (error) {
      console.error(error);
      persistLocalProjectTabs();
      toast.error(actionErrorMessage(error, 'Failed to save project; changes remain local.'), {
        id: 'save-local-fallback',
      });
    }
  };

  const handleOpen = async (id: string, options?: { replaceActiveTab?: boolean }) => {
    try {
      const data = await fetchGraph(id);
      const hydratedWorkspace = hydrateWorkspacePresentation(
        data.workspace,
        data.workspace_document,
      );
      const analysisSnapshot = analysisSnapshotFromWorkspaceDocument(
        data.workspace_document,
        hydratedWorkspace,
      );
      openProjectInTab(
        id,
        data.graph,
        data.workspace_document.graph_ui_state,
        data.metadata?.name ?? undefined,
        analysisSnapshot,
        hydratedWorkspace,
        {
          ...options,
          saveRevision: data.metadata?.save_revision ?? null,
          workspaceDocument: data.workspace_document,
        },
      );
      useCompileStatusStore.getState().setReports(data.compile_reports);
      useRunStore.getState().hydrateFromWorkspace(hydratedWorkspace);
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
          ? normalizeTrainingTrajectoryPayload(
              {
                effector: traj.effector_pos,
                target: traj.target_pos,
                t: traj.effector_pos.map((_, i) => (i / traj.effector_pos.length) * 0.8),
              },
              totalBatches - 1
            )
          : null;
        useTrainingStore.getState().seedDemoData({ lossHistory, latestTrajectory });
      }
      setLastProjectId(id);
      return true;
    } catch (error) {
      console.error(error);
      toast.error(actionErrorMessage(error, 'Failed to open project.'), {
        id: `open-project-error-${id}`,
      });
      return false;
    }
  };

  // Auto-load the last opened project on mount. In a fresh browser session the
  // store creates a built-in placeholder tab first; replace it so startup does
  // not leave the old SimpleReaches model open beside the real project.
  useEffect(() => {
    if (graphId !== null) return;
    if (hasRestoredLocalTabs) return;
    if (startupAutoloadAttemptedRef.current) return;
    if (graphsQuery.isLoading) return;

    const savedGraphs = graphsQuery.data?.graphs ?? [];
    const lastId = getLastProjectId();
    const soleProjectId = savedGraphs.length === 1 ? savedGraphs[0].id : null;
    const preferredId = lastId ?? soleProjectId;
    if (!preferredId) return;

    startupAutoloadAttemptedRef.current = true;
    void (async () => {
      const opened = await handleOpen(preferredId, { replaceActiveTab: true });
      if (!opened && lastId && soleProjectId && soleProjectId !== lastId) {
        await handleOpen(soleProjectId, { replaceActiveTab: true });
      }
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphId, hasRestoredLocalTabs, graphsQuery.isLoading, graphsQuery.data]);

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
      toast.success('Project exported.', { id: 'project-export-success' });
    } catch (error) {
      toast.error(actionErrorMessage(error, 'Failed to export project.'), {
        id: 'project-export-error',
      });
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
          const tabSaveStatus = isActive ? saveStatus : tab.graphSnapshot.saveStatus;
          const tabSaveError = isActive ? saveError : tab.graphSnapshot.saveError;
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
                <span
                  className={clsx(
                    'flex-none text-xs leading-none',
                    tabSaveStatus === 'error' || tabSaveStatus === 'conflict'
                      ? 'text-red-500'
                      : 'text-amber-500',
                  )}
                  title={tabSaveError ?? 'Unsaved changes'}
                >
                  •
                </span>
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
          title={
            inSubgraph
              ? 'Return to model root to save'
              : saveError ?? (saveStatus === 'saving' ? 'Saving' : 'Save')
          }
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
  onOpenSaved: (id: string) => Promise<boolean>;
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
        const detached = buildDetachedStudioDocument({
          documentId: `template:${template.id}:${crypto.randomUUID()}`,
          label: template.name,
          graph: modelGraph,
          uiState,
          analysisSnapshot,
          workspace,
        });
        const outcome = await studioPersistence.save(detached, 'template');
        studioPersistence.reset(detached.documentId);
        if (outcome.ok === false) throw outcome.error;
        const graphId = outcome.result.graphId;
        openProjectInTab(
          graphId,
          modelGraph,
          uiState,
          template.name,
          analysisSnapshot,
          workspace,
          {
            saveRevision: outcome.result.metadata.save_revision,
            workspaceDocument: outcome.workspaceDocument,
          },
        );
        useRunStore.getState().hydrateFromWorkspace(workspace);
        setLastProjectId(graphId);
      } catch (error) {
        console.error('Failed to save example project to backend:', error);
        toast.error('Template opened locally; backend save failed.', {
          id: `template-load-error-${template.id}`,
        });
        openProjectInTab(
          '',
          modelGraph,
          uiState,
          template.name,
          analysisSnapshot,
          workspace,
        );
        useGraphStore.getState().markDirty();
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
                    const opened = await onOpenSaved(item.id);
                    if (opened) onClose();
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
