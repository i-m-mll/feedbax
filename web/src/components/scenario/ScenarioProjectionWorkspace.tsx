import { useMemo } from 'react';
import clsx from 'clsx';
import { GitBranch, ListChecks, Map as MapIcon, Plus, Trash2 } from 'lucide-react';
import { Canvas } from '@/components/canvas/Canvas';
import {
  buildScenarioEntityRegistry,
  entityKindLabel,
  getScenarioEntity,
} from '@/features/scenario/entities';
import {
  objectiveProjectionItems,
  relatedProjectionItems,
  workspaceProjectionItems,
  type ScenarioProjectionItem,
} from '@/features/scenario/projections';
import {
  artifactOverlaysForWorkspace,
  scenarioMetricSpecs,
  type ScenarioArtifactOverlay,
  type ScenarioMetricSpec,
} from '@/features/scenario/integration';
import {
  addObjectiveTerm,
  createObjectiveTerm,
  ensureObjectiveSpec,
  objectiveTermEnabled,
  removeObjectiveTerm,
  setObjectiveTermEnabled,
  sourceSelectorForEntity,
  targetSelectorForEntity,
  updateObjectiveTerm,
} from '@/features/scenario/objectives';
import { useGraphStore } from '@/stores/graphStore';
import {
  getActiveStage,
  getScenario,
  getTopPaneState,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type {
  StudioObjectiveSpec,
  StudioObjectiveTermSpec,
  StudioScenarioEntity,
  StudioScenarioEntityRegistry,
  StudioTopPaneProjection,
} from '@/types/workspace';

const PROJECTIONS: Array<{
  id: StudioTopPaneProjection;
  label: string;
  icon: typeof GitBranch;
}> = [
  { id: 'graph', label: 'Graph', icon: GitBranch },
  { id: 'workspace', label: 'Workspace', icon: MapIcon },
  { id: 'objectives', label: 'Objectives', icon: ListChecks },
];

function numberParam(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function taskParams(entity: StudioScenarioEntity | null): Record<string, unknown> {
  const task = entity?.metadata.task_spec;
  if (!task || typeof task !== 'object' || !('params' in task)) return {};
  const params = task.params;
  return params && typeof params === 'object' ? (params as Record<string, unknown>) : {};
}

function isSelectedOrRelated(
  item: ScenarioProjectionItem,
  selectedId: string | null,
  relatedIds: Set<string>
) {
  return item.entity_id === selectedId || relatedIds.has(item.entity_id);
}

function ProjectionTabs({
  active,
  onChange,
  canAddObjective,
  onAddObjective,
}: {
  active: StudioTopPaneProjection;
  onChange: (projection: StudioTopPaneProjection) => void;
  canAddObjective: boolean;
  onAddObjective: () => void;
}) {
  return (
    <div className="flex h-11 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-3">
      <div className="flex items-center gap-1 rounded-md border border-slate-200 bg-slate-50 p-1">
        {PROJECTIONS.map((projection) => {
          const Icon = projection.icon;
          const selected = projection.id === active;
          return (
            <button
              key={projection.id}
              type="button"
              onClick={() => onChange(projection.id)}
              className={clsx(
                'inline-flex h-8 items-center gap-2 rounded px-3 text-xs font-medium transition-colors',
                selected
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-500 hover:bg-white/70 hover:text-slate-700'
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {projection.label}
            </button>
          );
        })}
      </div>
      <button
        type="button"
        disabled={!canAddObjective}
        onClick={onAddObjective}
        className="inline-flex h-8 items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-xs font-medium text-slate-600 shadow-sm hover:border-brand-200 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Plus className="h-3.5 w-3.5" />
        Add objective
      </button>
    </div>
  );
}

function ScenarioBadge({
  stageLabel,
  scenarioLabel,
  summary,
}: {
  stageLabel: string | null;
  scenarioLabel: string | null;
  summary: string | null;
}) {
  if (!stageLabel && !scenarioLabel) return null;
  return (
    <div className="pointer-events-none absolute bottom-4 left-4 z-10 max-w-[min(28rem,calc(100%-2rem))] rounded border border-slate-200 bg-white/90 px-3 py-2 shadow-sm backdrop-blur">
      <div className="truncate text-sm font-semibold text-slate-800">
        {scenarioLabel ?? stageLabel}
      </div>
      {summary && <div className="mt-0.5 truncate text-xs text-slate-500">{summary}</div>}
    </div>
  );
}

function EntityList({
  title,
  items,
  selectedId,
  relatedIds,
  onSelect,
}: {
  title: string;
  items: ScenarioProjectionItem[];
  selectedId: string | null;
  relatedIds: Set<string>;
  onSelect: (entityId: string | null) => void;
}) {
  return (
    <section className="min-h-0">
      <div className="px-4 py-3 text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
        {title}
      </div>
      <div className="space-y-1 px-3">
        {items.map((item) => {
          const active = item.entity_id === selectedId;
          const related = relatedIds.has(item.entity_id);
          return (
            <button
              key={item.entity_id}
              type="button"
              onClick={() => onSelect(item.entity_id)}
              className={clsx(
                'w-full rounded-md border px-3 py-2 text-left text-xs transition-colors',
                active
                  ? 'border-brand-300 bg-brand-50 text-slate-900'
                  : related
                    ? 'border-slate-200 bg-white text-slate-700'
                    : 'border-transparent bg-transparent text-slate-600 hover:border-slate-200 hover:bg-white'
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
          );
        })}
        {items.length === 0 && <div className="px-1 text-xs text-slate-400">None recorded</div>}
      </div>
    </section>
  );
}

function WorkspaceProjection({
  registry,
  metrics,
  overlays,
  selectedId,
  onSelect,
}: {
  registry: StudioScenarioEntityRegistry;
  metrics: ScenarioMetricSpec[];
  overlays: ScenarioArtifactOverlay[];
  selectedId: string | null;
  onSelect: (entityId: string | null) => void;
}) {
  const items = workspaceProjectionItems(registry);
  const selectedEntity = getScenarioEntity(registry, selectedId);
  const relatedItems = relatedProjectionItems(registry, selectedId);
  const relatedIds = new Set(relatedItems.map((item) => item.entity_id));
  const taskEntity = Object.values(registry.entities).find((entity) => entity.kind === 'task_object') ?? null;
  const mechanicsEntity =
    Object.values(registry.entities).find((entity) => entity.kind === 'mechanics_object') ?? null;
  const params = taskParams(taskEntity);
  const targetCount = Math.max(1, Math.min(16, Math.round(numberParam(params.n_targets, 8))));
  const targetRadius = Math.max(2, Math.min(18, numberParam(params.target_radius, 0.02) * 420));
  const targets = Array.from({ length: targetCount }, (_, index) => {
    const theta = (Math.PI * 2 * index) / targetCount - Math.PI / 2;
    return {
      x: 250 + Math.cos(theta) * 130,
      y: 210 + Math.sin(theta) * 130,
    };
  });
  const taskSelected = taskEntity
    ? isSelectedOrRelated(
        { entity_id: taskEntity.id, kind: taskEntity.kind, label: taskEntity.label, summary: null, related_entity_ids: [] },
        selectedId,
        relatedIds
      )
    : false;
  const mechanicsSelected = mechanicsEntity
    ? isSelectedOrRelated(
        {
          entity_id: mechanicsEntity.id,
          kind: mechanicsEntity.kind,
          label: mechanicsEntity.label,
          summary: null,
          related_entity_ids: [],
        },
        selectedId,
        relatedIds
      )
    : false;
  const overlayColors = ['#0f766e', '#7c3aed', '#ea580c', '#2563eb'];
  const overlayPaths = overlays.slice(0, 4).map((overlay, index) => ({
    overlay,
    color: overlayColors[index % overlayColors.length],
    path: `M ${125 + index * 12} ${255 - index * 9} C ${190} ${120 + index * 14}, ${
      300
    } ${300 - index * 18}, ${374 - index * 10} ${160 + index * 24}`,
  }));

  return (
    <div className="grid h-full min-h-0 bg-slate-50 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="relative min-h-0 overflow-hidden">
        <svg viewBox="0 0 500 420" className="h-full w-full" role="img" aria-label="Workspace projection">
          <rect x="78" y="38" width="344" height="344" rx="8" fill="#ffffff" stroke="#dbe3ee" />
          <path d="M250 80 L250 340 M120 210 L380 210" stroke="#e2e8f0" strokeWidth="1" />
          <circle cx="250" cy="210" r="130" fill="none" stroke="#e2e8f0" strokeDasharray="6 8" />
          {targets.map((target, index) => (
            <g
              key={index}
              role="button"
              tabIndex={0}
              onClick={() => taskEntity && onSelect(taskEntity.id)}
              onKeyDown={(event) => {
                if ((event.key === 'Enter' || event.key === ' ') && taskEntity) {
                  onSelect(taskEntity.id);
                }
              }}
            >
              <circle
                cx={target.x}
                cy={target.y}
                r={targetRadius}
                fill={taskSelected ? '#10b981' : '#d1fae5'}
                stroke={taskSelected ? '#047857' : '#34d399'}
                strokeWidth="2"
              />
            </g>
          ))}
          <line
            x1="250"
            y1="250"
            x2="250"
            y2="210"
            stroke={mechanicsSelected ? '#b45309' : '#64748b'}
            strokeWidth="9"
            strokeLinecap="round"
          />
          <line
            x1="250"
            y1="210"
            x2="302"
            y2="170"
            stroke={mechanicsSelected ? '#d97706' : '#94a3b8'}
            strokeWidth="7"
            strokeLinecap="round"
          />
          <g
            role="button"
            tabIndex={0}
            onClick={() => mechanicsEntity && onSelect(mechanicsEntity.id)}
            onKeyDown={(event) => {
              if ((event.key === 'Enter' || event.key === ' ') && mechanicsEntity) {
                onSelect(mechanicsEntity.id);
              }
            }}
          >
            <circle cx="302" cy="170" r="10" fill={mechanicsSelected ? '#f59e0b' : '#475569'} />
          </g>
          {items
            .filter((item) => item.kind === 'objective_term')
            .map((item, index) => {
              const target = targets[index % targets.length] ?? { x: 250, y: 80 };
              const active = item.entity_id === selectedId || relatedIds.has(item.entity_id);
              return (
                <g
                  key={item.entity_id}
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelect(item.entity_id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      onSelect(item.entity_id);
                    }
                  }}
                >
                  <line
                    x1="302"
                    y1="170"
                    x2={target.x}
                    y2={target.y}
                    stroke={active ? '#7c3aed' : '#c4b5fd'}
                    strokeWidth={active ? 3 : 2}
                    strokeDasharray="5 5"
                  />
                </g>
              );
            })}
          {overlayPaths.map(({ overlay, color, path }) => (
            <path
              key={overlay.id}
              d={path}
              fill="none"
              stroke={color}
              strokeWidth="2.5"
              strokeOpacity="0.72"
              strokeDasharray={overlay.source === 'artifact' ? '7 5' : undefined}
            />
          ))}
          <circle cx="250" cy="210" r="4" fill="#047857" />
        </svg>
        <div className="absolute left-4 top-4 rounded border border-slate-200 bg-white/90 px-3 py-2 text-xs text-slate-600 shadow-sm">
          <div className="font-semibold text-slate-800">{taskEntity?.label ?? 'Task'}</div>
          <div className="mt-0.5 text-slate-500">{mechanicsEntity?.summary ?? 'Mechanics'}</div>
        </div>
      </div>
      <aside className="min-h-0 overflow-y-auto border-l border-slate-200 bg-white">
        <EntityList
          title="Workspace Entities"
          items={items}
          selectedId={selectedId}
          relatedIds={relatedIds}
          onSelect={onSelect}
        />
        <MetricTraceList metrics={metrics} />
        <OverlayTraceList overlays={overlays} />
        {selectedEntity && (
          <div className="border-t border-slate-100 px-4 py-3 text-xs text-slate-500">
            <div className="font-medium text-slate-700">{selectedEntity.label}</div>
            {selectedEntity.summary && <div className="mt-1">{selectedEntity.summary}</div>}
          </div>
        )}
      </aside>
    </div>
  );
}

function MetricTraceList({ metrics }: { metrics: ScenarioMetricSpec[] }) {
  return (
    <section className="border-t border-slate-100 px-4 py-3">
      <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
        Metric Specs
      </div>
      <div className="mt-2 space-y-2">
        {metrics.slice(0, 6).map((metric) => (
          <div key={`${metric.source}:${metric.sourceId}:${metric.id}`} className="text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-medium text-slate-700">{metric.label}</span>
              <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
                {metric.source}
              </span>
            </div>
            <div className="mt-0.5 truncate text-[11px] text-slate-400">
              {metric.selector ?? metric.summary ?? metric.sourceId}
            </div>
          </div>
        ))}
        {metrics.length === 0 && <div className="text-xs text-slate-400">None derived</div>}
      </div>
    </section>
  );
}

function OverlayTraceList({ overlays }: { overlays: ScenarioArtifactOverlay[] }) {
  return (
    <section className="border-t border-slate-100 px-4 py-3">
      <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
        Overlays
      </div>
      <div className="mt-2 space-y-2">
        {overlays.slice(0, 5).map((overlay) => (
          <div key={overlay.id} className="text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-medium text-slate-700">{overlay.label}</span>
              <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
                {overlay.source}
              </span>
            </div>
            <div className="mt-0.5 truncate text-[11px] text-slate-400">
              {overlay.uri ?? overlay.summary ?? overlay.role}
            </div>
          </div>
        ))}
        {overlays.length === 0 && <div className="text-xs text-slate-400">None available</div>}
      </div>
    </section>
  );
}

function ObjectivesProjection({
  registry,
  selectedId,
  objectiveSpec,
  onSelect,
  onObjectiveSpecChange,
}: {
  registry: StudioScenarioEntityRegistry;
  selectedId: string | null;
  objectiveSpec: StudioObjectiveSpec;
  onSelect: (entityId: string | null) => void;
  onObjectiveSpecChange: (spec: StudioObjectiveSpec) => void;
}) {
  const items = objectiveProjectionItems(registry);
  const relatedItems = relatedProjectionItems(registry, selectedId);
  const relatedIds = new Set(relatedItems.map((item) => item.entity_id));
  const termByEntityId = new Map(
    objectiveSpec.terms.map((term) => [`objective_term:${term.id}`, term])
  );

  const updateTerm = (termId: string, updates: Partial<StudioObjectiveTermSpec>) => {
    onObjectiveSpecChange(updateObjectiveTerm(objectiveSpec, termId, updates));
  };

  return (
    <div className="h-full overflow-y-auto bg-slate-50 p-5">
      <div className="mx-auto max-w-5xl overflow-hidden rounded-md border border-slate-200 bg-white">
        <div className="grid grid-cols-[minmax(0,1.4fr)_7rem_7rem_minmax(0,1fr)_5rem] border-b border-slate-200 bg-slate-50 px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
          <div>Term</div>
          <div>Role</div>
          <div>Weight</div>
          <div>Selector</div>
          <div>State</div>
        </div>
        {items.map((item) => {
          const term = termByEntityId.get(item.entity_id);
          const source = term?.source_selector;
          const active = item.entity_id === selectedId;
          const related = relatedIds.has(item.entity_id);
          if (!term) return null;
          return (
            <div
              key={item.entity_id}
              onClick={() => onSelect(item.entity_id)}
              className={clsx(
                'grid w-full grid-cols-[minmax(0,1.4fr)_7rem_7rem_minmax(0,1fr)_5rem] items-center gap-2 border-b border-slate-100 px-4 py-3 text-left text-xs last:border-b-0',
                active
                  ? 'bg-brand-50 text-slate-900'
                  : related
                    ? 'bg-violet-50/60 text-slate-800'
                    : 'bg-white text-slate-600 hover:bg-slate-50'
              )}
            >
              <div className="min-w-0">
                <input
                  value={term.label}
                  onChange={(event) => updateTerm(term.id, { label: event.target.value })}
                  onClick={(event) => event.stopPropagation()}
                  className="h-8 w-full rounded border border-transparent bg-transparent px-2 font-medium text-slate-800 hover:border-slate-200 focus:border-brand-300 focus:bg-white focus:outline-none"
                />
                {item.summary && <div className="mt-0.5 truncate text-slate-400">{item.summary}</div>}
              </div>
              <select
                value={term.role}
                onChange={(event) => updateTerm(term.id, { role: event.target.value })}
                onClick={(event) => event.stopPropagation()}
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs"
              >
                <option value="loss">Loss</option>
                <option value="metric">Metric</option>
                <option value="constraint">Constraint</option>
                <option value="reward">Reward</option>
                <option value="regularizer">Regularizer</option>
              </select>
              <input
                type="number"
                min={0}
                step={0.01}
                value={term.weight}
                onChange={(event) => {
                  const weight = Number.parseFloat(event.target.value);
                  if (Number.isFinite(weight)) updateTerm(term.id, { weight });
                }}
                onClick={(event) => event.stopPropagation()}
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs"
              />
              <div className="truncate text-slate-500">{source?.compact ?? 'None'}</div>
              <div className="flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={objectiveTermEnabled(term)}
                  onChange={(event) =>
                    onObjectiveSpecChange(
                      setObjectiveTermEnabled(objectiveSpec, term.id, event.target.checked)
                    )
                  }
                  onClick={(event) => event.stopPropagation()}
                  className="h-4 w-4 rounded border-slate-300"
                  title="Enabled"
                />
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onObjectiveSpecChange(removeObjectiveTerm(objectiveSpec, term.id));
                    if (selectedId === item.entity_id) onSelect(null);
                  }}
                  className="rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600"
                  title="Delete objective"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          );
        })}
        {items.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-slate-400">
            No objective terms recorded.
          </div>
        )}
      </div>
    </div>
  );
}

export function ScenarioProjectionWorkspace() {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const setTopPaneProjection = useWorkspaceStore((state) => state.setTopPaneProjection);
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const updateActiveScenarioObjectiveSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioObjectiveSpec
  );
  const graph = useGraphStore((state) => state.graph);
  const topPane = getTopPaneState(workspace);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const objectiveSpec = ensureObjectiveSpec(activeScenario?.objective_spec);
  const metrics = useMemo(() => scenarioMetricSpecs(workspace), [workspace]);
  const overlays = useMemo(() => artifactOverlaysForWorkspace(workspace), [workspace]);
  const registry = useMemo(
    () => buildScenarioEntityRegistry({ scenario: activeScenario, graph }),
    [activeScenario, graph]
  );
  const stageSummary =
    typeof activeStage?.metadata.summary === 'string' ? activeStage.metadata.summary : null;
  const selectedEntity = getScenarioEntity(registry, topPane.selected_entity_id);
  const canAddObjective = Boolean(activeScenario && selectedEntity);
  const addObjectiveFromSelection = () => {
    if (!activeScenario || !selectedEntity) return;
    const taskEntity =
      Object.values(registry.entities).find((entity) => entity.kind === 'task_object') ?? null;
    const mechanicsEntity =
      Object.values(registry.entities).find((entity) => entity.kind === 'mechanics_object') ?? null;
    const sourceSelector =
      sourceSelectorForEntity(selectedEntity, registry) ??
      sourceSelectorForEntity(mechanicsEntity, registry);
    const targetSelector =
      targetSelectorForEntity(selectedEntity) ?? targetSelectorForEntity(taskEntity);
    const term = createObjectiveTerm({
      spec: objectiveSpec,
      label: `Objective: ${selectedEntity.label}`,
      sourceSelector,
      targetSelector,
    });
    updateActiveScenarioObjectiveSpec(addObjectiveTerm(objectiveSpec, term));
    setTopPaneProjection('objectives');
    selectTopPaneEntity(`objective_term:${term.id}`);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <ProjectionTabs
        active={topPane.active_projection}
        onChange={setTopPaneProjection}
        canAddObjective={canAddObjective}
        onAddObjective={addObjectiveFromSelection}
      />
      <div className="relative min-h-0 flex-1">
        {topPane.active_projection === 'graph' && (
          <div className="absolute inset-0">
            <Canvas />
          </div>
        )}
        {topPane.active_projection === 'workspace' && (
          <WorkspaceProjection
            registry={registry}
            metrics={metrics}
            overlays={overlays}
            selectedId={topPane.selected_entity_id}
            onSelect={selectTopPaneEntity}
          />
        )}
        {topPane.active_projection === 'objectives' && (
          <ObjectivesProjection
            registry={registry}
            selectedId={topPane.selected_entity_id}
            objectiveSpec={objectiveSpec}
            onSelect={selectTopPaneEntity}
            onObjectiveSpecChange={updateActiveScenarioObjectiveSpec}
          />
        )}
        <ScenarioBadge
          stageLabel={activeStage?.label ?? null}
          scenarioLabel={activeScenario?.label ?? null}
          summary={stageSummary}
        />
      </div>
    </div>
  );
}
