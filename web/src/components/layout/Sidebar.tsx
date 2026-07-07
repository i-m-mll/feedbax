// Note: Analysis palette moved to BottomSidebar. When unified graph canvas is implemented, merge back.

import { useState } from 'react';
import { ComponentLibrary } from '@/components/panels/ComponentLibrary';
import { TaskLibrary } from '@/components/panels/TaskLibrary';
import { useLayoutStore } from '@/stores/layoutStore';
import {
  getActiveStage,
  getScenario,
  getTopPaneState,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { useGraphStore } from '@/stores/graphStore';
import {
  buildScenarioEntityRegistry,
  entityKindLabel,
} from '@/features/scenario/entities';
import { objectiveProjectionItems, workspaceProjectionItems } from '@/features/scenario/projections';
import { semanticTokens } from '@/components/ui/semanticTokens';
import clsx from 'clsx';

type ActiveTab = 'components' | 'templates';

export function Sidebar() {
  const { leftSidebarWidth, leftSidebarVisible, setLeftSidebarWidth } =
    useLayoutStore();
  const [activeTab, setActiveTab] = useState<ActiveTab>('components');
  const workspace = useWorkspaceStore((state) => state.workspace);
  const topPane = getTopPaneState(workspace);

  if (!leftSidebarVisible) {
    return null;
  }

  const isModelProjection = topPane.active_projection === 'model';
  const isTaskProjection = topPane.active_projection === 'task';

  return (
    <aside
      style={{ width: leftSidebarWidth }}
      className="border-r border-slate-100 bg-white/90 backdrop-blur-sm flex flex-col relative shrink-0"
    >
      <div className="px-4 pt-4 pb-2 flex items-center">
        {isModelProjection ? (
          <div className="flex gap-1">
            {(['components', 'templates'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={clsx(
                  'text-xs uppercase tracking-[0.2em] px-2 py-1 rounded transition-colors',
                  activeTab === tab
                    ? 'bg-slate-100 text-slate-700 font-semibold'
                    : 'text-slate-400 hover:text-slate-600'
                )}
              >
                {tab === 'components' ? 'Components' : 'Templates'}
              </button>
            ))}
          </div>
        ) : isTaskProjection ? (
          <div className="text-xs uppercase tracking-[0.2em] px-2 py-1 text-emerald-700 font-semibold">
            Tasks
          </div>
        ) : (
          <div className="text-xs uppercase tracking-[0.2em] px-2 py-1 text-slate-400">
            {topPane.active_projection}
          </div>
        )}
      </div>
      {isModelProjection && <ComponentLibrary mode={activeTab} />}
      {isTaskProjection && <TaskLibrary />}
      {(topPane.active_projection === 'workspace' ||
        topPane.active_projection === 'objectives') && (
        <ProjectionSidebar projection={topPane.active_projection} />
      )}
      <div
        className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-brand-300/50 active:bg-brand-400/50"
        onPointerDown={(e) => {
          e.preventDefault();
          const startX = e.clientX;
          const startWidth = leftSidebarWidth;
          const onMove = (me: PointerEvent) => {
            setLeftSidebarWidth(startWidth + (me.clientX - startX));
          };
          const onUp = () => {
            window.removeEventListener('pointermove', onMove);
            window.removeEventListener('pointerup', onUp);
          };
          window.addEventListener('pointermove', onMove);
          window.addEventListener('pointerup', onUp);
        }}
      />
    </aside>
  );
}

function ProjectionSidebar({
  projection,
}: {
  projection: 'workspace' | 'objectives';
}) {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const graph = useGraphStore((state) => state.graph);
  const topPane = getTopPaneState(workspace);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const registry = buildScenarioEntityRegistry({ scenario: activeScenario, graph });
  const items =
    projection === 'workspace'
      ? workspaceProjectionItems(registry)
      : objectiveProjectionItems(registry);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4">
      <div className="mt-2 space-y-1">
        {items.map((item) => (
          <button
            key={item.entity_id}
            type="button"
            onClick={() => selectTopPaneEntity(item.entity_id)}
            className={clsx(
              'w-full rounded-md border px-3 py-2 text-left text-xs transition-colors',
              topPane.selected_entity_id === item.entity_id
                ? item.kind === 'objective_term'
                  ? clsx(semanticTokens.objective.border, semanticTokens.objective.background, 'text-slate-900')
                  : 'border-brand-300 bg-brand-50 text-slate-900'
                : 'border-transparent text-slate-600 hover:border-slate-200 hover:bg-slate-50'
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-medium">{item.label}</span>
              <span className="shrink-0 text-[10px] text-slate-400">
                {entityKindLabel(item.kind)}
              </span>
            </div>
            {item.summary && <div className="mt-0.5 truncate text-slate-400">{item.summary}</div>}
          </button>
        ))}
        {items.length === 0 && <div className="text-xs text-slate-400">None recorded</div>}
      </div>
    </div>
  );
}
