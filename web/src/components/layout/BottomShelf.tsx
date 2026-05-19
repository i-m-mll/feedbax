import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import { useLayoutStore, SHELF_HEADER_HEIGHT } from '@/stores/layoutStore';
import { AnalysisPanel } from '@/components/panels/AnalysisPanel';
import { ConsolePanel } from '@/components/panels/ConsolePanel';
import { BottomSidebar } from '@/components/layout/BottomSidebar';
import {
  EvaluateCollectionPanel,
  TrainCollectionPanel,
} from '@/components/panels/RunCollectionStagePanel';
import {
  StageDraftPanel,
  StageProvenancePanel,
} from '@/components/panels/PipelineStageWorkspace';
import { getActiveStage, getScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import { useGraphStore } from '@/stores/graphStore';
import type { StudioStageKind } from '@/types/workspace';
import { BarChart3, FileText, FlaskConical, PlayCircle, Terminal, Workflow } from 'lucide-react';

const stageIcons: Record<StudioStageKind, typeof PlayCircle> = {
  train: FlaskConical,
  eval: PlayCircle,
  analysis: BarChart3,
  report: FileText,
  import: Workflow,
  compare: Workflow,
  export: Workflow,
  protocol: Workflow,
};

type WorkspaceMode = 'stage' | 'console';

export function BottomShelf({
  height,
  availableHeight,
}: {
  height: number;
  availableHeight: number;
}) {
  const [mode, setMode] = useState<WorkspaceMode>('stage');
  const { bottomCollapsed, toggleBottom } = useLayoutStore();
  const workspace = useWorkspaceStore((state) => state.workspace);
  const stages = workspace?.stages ?? [];
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const setActiveStage = useWorkspaceStore((state) => state.setActiveStage);
  const markDirty = useGraphStore((state) => state.markDirty);
  const tabsRef = useRef<HTMLDivElement | null>(null);
  const [fadeState, setFadeState] = useState({ left: false, right: false });

  const activeContent = useMemo(() => {
    if (mode === 'console') return <ConsolePanel />;
    if (activeStage?.kind === 'train') return <TrainCollectionPanel />;
    if (activeStage?.kind === 'eval') return <EvaluateCollectionPanel />;
    if (activeStage?.kind === 'analysis') return <AnalysisPanel />;
    return <StageDraftPanel stage={activeStage} scenario={activeScenario} />;
  }, [activeStage, activeScenario, mode]);

  const selectStage = useCallback(
    (stageId: string) => {
      if (bottomCollapsed) toggleBottom(availableHeight);
      setMode('stage');
      setActiveStage(stageId);
      markDirty();
    },
    [availableHeight, bottomCollapsed, markDirty, setActiveStage, toggleBottom]
  );

  const updateFades = useCallback(() => {
    const el = tabsRef.current;
    if (!el) return;
    const left = el.scrollLeft > 4;
    const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 4;
    setFadeState({ left, right });
  }, []);

  useEffect(() => {
    updateFades();
    const el = tabsRef.current;
    if (!el) return;
    const handle = () => updateFades();
    el.addEventListener('scroll', handle);
    window.addEventListener('resize', handle);
    return () => {
      el.removeEventListener('scroll', handle);
      window.removeEventListener('resize', handle);
    };
  }, [bottomCollapsed, updateFades]);

  return (
    <section
      className="relative w-full max-w-full overflow-hidden bg-white/90 backdrop-blur-sm border-t border-slate-100"
      style={{ height }}
    >
      <div
        className="flex items-end gap-3 border-b border-slate-200 px-3"
        style={{ height: SHELF_HEADER_HEIGHT }}
      >
        <div className="relative flex-1 min-w-0">
          <div ref={tabsRef} className="flex items-end overflow-x-auto pr-6">
            {stages.map((stage) => {
              const Icon = stageIcons[stage.kind] ?? Workflow;
              const active = mode === 'stage' && stage.id === activeStage?.id;
              return (
                <button
                  key={stage.id}
                  onClick={() => selectStage(stage.id)}
                  className={clsx(
                    'inline-flex h-10 items-center gap-2 whitespace-nowrap border-b-2 px-4 text-xs font-semibold uppercase tracking-[0.12em] transition-colors',
                    active
                      ? 'border-brand-500 text-brand-600'
                      : 'border-transparent text-slate-400 hover:text-slate-600'
                  )}
                  title={`${stage.kind}: ${stage.label}`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {stage.label}
                </button>
              );
            })}
            <div className="mx-2 mb-2 h-5 w-px shrink-0 bg-slate-200" />
            <button
              onClick={() => {
                if (bottomCollapsed) toggleBottom(availableHeight);
                setMode('console');
              }}
              className={clsx(
                'inline-flex h-10 items-center gap-2 whitespace-nowrap border-b-2 px-4 text-xs font-semibold uppercase tracking-[0.12em] transition-colors',
                mode === 'console'
                  ? 'border-brand-500 text-brand-600'
                  : 'border-transparent text-slate-400 hover:text-slate-600'
              )}
            >
              <Terminal className="h-3.5 w-3.5" />
              Console
            </button>
          </div>
          {fadeState.left && (
            <div className="pointer-events-none absolute left-0 top-0 h-full w-6 bg-gradient-to-r from-white/90 to-transparent" />
          )}
          {fadeState.right && (
            <div className="pointer-events-none absolute right-0 top-0 h-full w-8 bg-gradient-to-l from-white/90 to-transparent" />
          )}
        </div>
      </div>
      {!bottomCollapsed && (
        <div
          style={{ height: Math.max(0, height - SHELF_HEADER_HEIGHT) }}
          className={clsx(
            'flex min-w-0',
            mode === 'console' || activeStage?.kind === 'analysis' ? 'overflow-hidden' : 'overflow-y-auto'
          )}
        >
          {mode === 'stage' && activeStage?.kind === 'analysis' && <BottomSidebar />}
          <div className="flex-1 min-w-0 h-full">
            {activeContent}
          </div>
          {mode === 'stage' && activeStage?.kind !== 'train' && activeStage?.kind !== 'eval' && (
            <StageProvenancePanel
              stage={activeStage}
              scenario={activeScenario}
              workspace={workspace}
            />
          )}
        </div>
      )}
    </section>
  );
}
