import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import {
  Database,
  GitBranch,
  ListChecks,
  Map as MapIcon,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Settings2,
  Trash2,
} from 'lucide-react';
import { Canvas } from '@/components/canvas/Canvas';
import {
  retainedObservableEntityId,
  buildScenarioEntityRegistry,
} from '@/features/scenario/entities';
import {
  objectiveProjectionItems,
  relatedProjectionItems,
  type ScenarioProjectionItem,
} from '@/features/scenario/projections';
import {
  ensureObjectiveSpec,
  OBJECTIVE_PENALTY_OPTIONS,
  OBJECTIVE_TEMPORAL_MODE_OPTIONS,
  objectiveTermEnabled,
  removeObjectiveTerm,
  setObjectiveTermEnabled,
  updateObjectiveTerm,
} from '@/features/scenario/objectives';
import {
  selectorDetail,
  selectorDisplayLabel,
  selectorGroupLabel,
  selectorOptionsForRegistry,
  type StudioSelectorOption,
} from '@/features/scenario/selectors';
import {
  createRetainedObservable,
  RETENTION_POLICY_OPTIONS,
  retainedObservableSelectorPatch,
  retainedObservableTargetKindLabel,
  retentionPolicy,
  selectorToRetainedObservableTarget,
} from '@/features/scenario/observables';
import { useGraphStore } from '@/stores/graphStore';
import {
  getActiveStage,
  getScenario,
  getTopPaneState,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { useLayoutStore } from '@/stores/layoutStore';
import { useStudioSchemaRegistry } from '@/hooks/useStudioSchemas';
import type {
  RetainedObservableSpec,
  RetentionPolicySpec,
} from '@/types/graph';
import type {
  StudioObjectiveSpec,
  StudioObjectiveTermSpec,
  StudioScenarioEntity,
  StudioScenarioEntityRegistry,
  StudioSchemaRegistry,
  StudioTopPaneProjection,
} from '@/types/workspace';
import type { TimeAggregationSpec } from '@/types/training';

const PROJECTIONS: Array<{
  id: StudioTopPaneProjection;
  label: string;
  icon: typeof GitBranch;
}> = [
  { id: 'model', label: 'Model', icon: GitBranch },
  { id: 'task', label: 'Task', icon: Settings2 },
  { id: 'workspace', label: 'Workspace', icon: MapIcon },
  { id: 'observables', label: 'Observables', icon: Database },
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

export function ScenarioProjectionToolbar() {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const setTopPaneProjection = useWorkspaceStore((state) => state.setTopPaneProjection);
  const {
    leftSidebarVisible,
    rightSidebarVisible,
    toggleLeftSidebar,
    toggleRightSidebar,
  } = useLayoutStore();
  const topPane = getTopPaneState(workspace);
  const LeftIcon = leftSidebarVisible ? PanelLeftClose : PanelLeftOpen;
  const RightIcon = rightSidebarVisible ? PanelRightClose : PanelRightOpen;

  return (
    <div className="flex h-11 shrink-0 items-end justify-between border-b border-slate-200 bg-white px-3">
      <div className="flex h-full min-w-0 items-end">
        <button
          type="button"
          onClick={toggleLeftSidebar}
          className="mb-1 mr-2 inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          title={leftSidebarVisible ? 'Hide component palette' : 'Show component palette'}
        >
          <LeftIcon className="h-4 w-4" />
        </button>
        {PROJECTIONS.map((projection) => {
          const Icon = projection.icon;
          const selected = projection.id === topPane.active_projection;
          return (
            <button
              key={projection.id}
              type="button"
              onClick={() => setTopPaneProjection(projection.id)}
              className={clsx(
                'inline-flex h-10 items-center gap-2 border-b-2 px-4 text-xs font-semibold uppercase tracking-[0.12em] transition-colors',
                selected
                  ? 'border-brand-500 text-brand-600'
                  : 'border-transparent text-slate-400 hover:text-slate-600'
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
        onClick={toggleRightSidebar}
        className="mb-1 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-700"
        title={rightSidebarVisible ? 'Hide properties panel' : 'Show properties panel'}
      >
        <RightIcon className="h-4 w-4" />
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
    <div className="pointer-events-none absolute bottom-4 left-20 z-10 max-w-[min(28rem,calc(100%-6rem))] rounded border border-slate-200 bg-white/90 px-3 py-2 shadow-sm backdrop-blur">
      <div className="truncate text-sm font-semibold text-slate-800">
        {scenarioLabel ?? stageLabel}
      </div>
      {summary && <div className="mt-0.5 truncate text-xs text-slate-500">{summary}</div>}
    </div>
  );
}

function WorkspaceProjection({
  registry,
  selectedId,
  onSelect,
}: {
  registry: StudioScenarioEntityRegistry;
  selectedId: string | null;
  onSelect: (entityId: string | null) => void;
}) {
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

  return (
    <div className="h-full min-h-0 bg-slate-50">
      <div className="relative h-full min-h-0 overflow-hidden">
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
          <circle cx="250" cy="210" r="4" fill="#047857" />
        </svg>
        <div className="absolute left-4 top-4 rounded border border-slate-200 bg-white/90 px-3 py-2 text-xs text-slate-600 shadow-sm">
          <div className="font-semibold text-slate-800">{taskEntity?.label ?? 'Task'}</div>
          <div className="mt-0.5 text-slate-500">{mechanicsEntity?.summary ?? 'Mechanics'}</div>
        </div>
      </div>
    </div>
  );
}

function temporalSelector(term: StudioObjectiveTermSpec): TimeAggregationSpec {
  const value = term.temporal_selector;
  if (!value || typeof value !== 'object' || !('mode' in value)) return { mode: 'all' };
  return value as TimeAggregationSpec;
}

function updateTemporalSelector(
  term: StudioObjectiveTermSpec,
  updates: Partial<TimeAggregationSpec>
): TimeAggregationSpec {
  const current = temporalSelector(term);
  return {
    ...current,
    ...updates,
  };
}

function optionLabel(option: StudioSelectorOption): string {
  return `${selectorGroupLabel(option.group)} / ${option.label}`;
}

function ObservableSelectorSelect({
  value,
  options,
  onChange,
  className,
}: {
  value: string | null | undefined;
  options: StudioSelectorOption[];
  onChange: (option: StudioSelectorOption | null) => void;
  className?: string;
}) {
  return (
    <select
      value={value ?? ''}
      onChange={(event) => {
        const option = options.find((candidate) => candidate.selector.compact === event.target.value);
        onChange(option ?? null);
      }}
      className={clsx('h-8 rounded border border-slate-200 bg-white px-2 text-xs', className)}
    >
      <option value="">Select source</option>
      {options.map((option) => (
        <option key={option.id} value={option.selector.compact}>
          {optionLabel(option)}
        </option>
      ))}
    </select>
  );
}

function ObservablesProjection({
  registry,
  selectedId,
  graph,
  objectiveSpec,
  schemaRegistry,
  onSelect,
  onAdd,
  onUpdate,
  onRemove,
}: {
  registry: StudioScenarioEntityRegistry;
  selectedId: string | null;
  graph: { retained_observables?: RetainedObservableSpec[] | null };
  objectiveSpec: StudioObjectiveSpec;
  schemaRegistry: StudioSchemaRegistry | null;
  onSelect: (entityId: string | null) => void;
  onAdd: (observable: RetainedObservableSpec) => void;
  onUpdate: (observableId: string, updates: Partial<RetainedObservableSpec>) => void;
  onRemove: (observableId: string) => void;
}) {
  const observables = graph.retained_observables ?? [];
  const selectorOptions = useMemo(
    () => selectorOptionsForRegistry({ registry, schemaRegistry, objectiveSpec }),
    [objectiveSpec, registry, schemaRegistry]
  );
  const captureOptions = useMemo(
    () =>
      selectorOptions.filter((option) => {
        if (option.selector.namespace === 'retained_observable') return false;
        if (option.selector.namespace === 'probe') return false;
        return selectorToRetainedObservableTarget(option.selector) !== null;
      }),
    [selectorOptions]
  );
  const [draftSelector, setDraftSelector] = useState<string>(() => captureOptions[0]?.selector.compact ?? '');

  useEffect(() => {
    if (!draftSelector && captureOptions[0]) {
      setDraftSelector(captureOptions[0].selector.compact);
    }
  }, [captureOptions, draftSelector]);

  const addObservable = () => {
    const option =
      captureOptions.find((candidate) => candidate.selector.compact === draftSelector) ??
      captureOptions[0];
    if (!option) return;
    const observable = createRetainedObservable({
      selector: option.selector,
      existingIds: new Set(observables.map((item) => item.id)),
    });
    if (!observable) return;
    onAdd(observable);
    onSelect(retainedObservableEntityId(observable.id));
  };

  const updateRetentionMode = (
    observable: RetainedObservableSpec,
    mode: RetentionPolicySpec['mode']
  ) => {
    onUpdate(observable.id, { retention: retentionPolicy(mode, observable.retention) });
  };

  return (
    <div className="h-full overflow-y-auto bg-slate-50 p-5">
      <div className="mx-auto max-w-6xl space-y-4">
        <div className="rounded-md border border-slate-200 bg-white p-4">
          <div className="grid grid-cols-[minmax(12rem,1fr)_9rem] gap-3">
            <ObservableSelectorSelect
              value={draftSelector}
              options={captureOptions}
              onChange={(option) => setDraftSelector(option?.selector.compact ?? '')}
              className="w-full"
            />
            <button
              type="button"
              onClick={addObservable}
              disabled={captureOptions.length === 0}
              className="inline-flex h-8 items-center justify-center rounded-md bg-slate-900 px-3 text-xs font-medium text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              Add capture
            </button>
          </div>
        </div>

        <div className="overflow-hidden rounded-md border border-slate-200 bg-white">
          <div className="grid grid-cols-[minmax(10rem,1fr)_8rem_minmax(12rem,1.2fr)_8rem_4rem] border-b border-slate-200 bg-slate-50 px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
            <div>Observable</div>
            <div>Kind</div>
            <div>Source</div>
            <div>Retention</div>
            <div />
          </div>
          {observables.map((observable) => {
            const active = selectedId === retainedObservableEntityId(observable.id);
            const source = observable.selector ?? observable.target?.selector ?? '';
            return (
              <div
                key={observable.id}
                onClick={() => onSelect(retainedObservableEntityId(observable.id))}
                className={clsx(
                  'grid grid-cols-[minmax(10rem,1fr)_8rem_minmax(12rem,1.2fr)_8rem_4rem] items-center gap-2 border-b border-slate-100 px-4 py-3 text-xs last:border-b-0',
                  active ? 'bg-brand-50 text-slate-900' : 'bg-white text-slate-600 hover:bg-slate-50'
                )}
              >
                <div className="min-w-0">
                  <input
                    value={observable.label ?? observable.id}
                    onChange={(event) => onUpdate(observable.id, { label: event.target.value })}
                    onClick={(event) => event.stopPropagation()}
                    className="h-8 w-full rounded border border-transparent bg-transparent px-2 font-medium text-slate-800 hover:border-slate-200 focus:border-brand-300 focus:bg-white focus:outline-none"
                  />
                  <div className="mt-0.5 truncate font-mono text-[11px] text-slate-400">
                    {observable.id}
                  </div>
                </div>
                <div className="text-slate-500">
                  {retainedObservableTargetKindLabel(observable.target)}
                </div>
                <ObservableSelectorSelect
                  value={source}
                  options={captureOptions}
                  onChange={(option) => {
                    if (!option) return;
                    const patch = retainedObservableSelectorPatch(option.selector);
                    if (patch) onUpdate(observable.id, patch);
                  }}
                  className="w-full"
                />
                <select
                  value={observable.retention.mode}
                  onChange={(event) =>
                    updateRetentionMode(
                      observable,
                      event.target.value as RetentionPolicySpec['mode']
                    )
                  }
                  onClick={(event) => event.stopPropagation()}
                  className="h-8 rounded border border-slate-200 bg-white px-2 text-xs"
                >
                  {RETENTION_POLICY_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onRemove(observable.id);
                    if (active) onSelect(null);
                  }}
                  className="rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600"
                  title="Delete retained observable"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            );
          })}
          {observables.length === 0 && (
            <div className="px-4 py-8 text-center text-sm text-slate-400">
              No explicit retained observables authored.
            </div>
          )}
        </div>
      </div>
    </div>
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
        <div className="grid grid-cols-[minmax(10rem,1.4fr)_6.5rem_5.5rem_7.25rem_8.5rem_minmax(8rem,1fr)_4rem] border-b border-slate-200 bg-slate-50 px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
          <div>Term</div>
          <div>Role</div>
          <div>Weight</div>
          <div>Penalty</div>
          <div>Time</div>
          <div>Source</div>
          <div />
        </div>
        {items.map((item) => {
          const term = termByEntityId.get(item.entity_id);
          const source = term?.source_selector;
          const time = term ? temporalSelector(term) : { mode: 'all' as const };
          const active = item.entity_id === selectedId;
          const related = relatedIds.has(item.entity_id);
          if (!term) return null;
          return (
            <div
              key={item.entity_id}
              onClick={() => onSelect(item.entity_id)}
              className={clsx(
                'grid w-full grid-cols-[minmax(10rem,1.4fr)_6.5rem_5.5rem_7.25rem_8.5rem_minmax(8rem,1fr)_4rem] items-center gap-2 border-b border-slate-100 px-4 py-3 text-left text-xs last:border-b-0',
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
              <select
                value={term.penalty ?? 'squared_l2'}
                onChange={(event) => updateTerm(term.id, { penalty: event.target.value })}
                onClick={(event) => event.stopPropagation()}
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs"
              >
                {OBJECTIVE_PENALTY_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <select
                value={time.mode}
                onChange={(event) =>
                  updateTerm(term.id, {
                    temporal_selector: updateTemporalSelector(term, {
                      mode: event.target.value as TimeAggregationSpec['mode'],
                    }),
                  })
                }
                onClick={(event) => event.stopPropagation()}
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs"
              >
                {OBJECTIVE_TEMPORAL_MODE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <div className="min-w-0">
                <div className="truncate text-slate-600" title={source?.compact}>
                  {selectorDisplayLabel(source)}
                </div>
                {selectorDetail(source) && (
                  <div className="mt-0.5 truncate text-[11px] text-slate-400">
                    {selectorDetail(source)}
                  </div>
                )}
              </div>
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
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const updateActiveScenarioObjectiveSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioObjectiveSpec
  );
  const graph = useGraphStore((state) => state.graph);
  const addRetainedObservable = useGraphStore((state) => state.addRetainedObservable);
  const updateRetainedObservable = useGraphStore((state) => state.updateRetainedObservable);
  const removeRetainedObservable = useGraphStore((state) => state.removeRetainedObservable);
  const topPane = getTopPaneState(workspace);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const objectiveSpec = ensureObjectiveSpec(activeScenario?.objective_spec);
  const schemaQuery = useStudioSchemaRegistry(
    workspace,
    activeStage?.scenario_id ?? activeScenario?.id ?? null
  );
  const registry = useMemo(
    () => buildScenarioEntityRegistry({ scenario: activeScenario, graph }),
    [activeScenario, graph]
  );
  const stageSummary =
    typeof activeStage?.metadata.summary === 'string' ? activeStage.metadata.summary : null;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="relative min-h-0 flex-1">
        {(topPane.active_projection === 'model' || topPane.active_projection === 'task') && (
          <div className="absolute inset-0">
            <Canvas />
          </div>
        )}
        {topPane.active_projection === 'workspace' && (
          <WorkspaceProjection
            registry={registry}
            selectedId={topPane.selected_entity_id}
            onSelect={selectTopPaneEntity}
          />
        )}
        {topPane.active_projection === 'observables' && (
          <ObservablesProjection
            registry={registry}
            selectedId={topPane.selected_entity_id}
            graph={graph}
            objectiveSpec={objectiveSpec}
            schemaRegistry={schemaQuery.data ?? null}
            onSelect={selectTopPaneEntity}
            onAdd={addRetainedObservable}
            onUpdate={updateRetainedObservable}
            onRemove={removeRetainedObservable}
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
