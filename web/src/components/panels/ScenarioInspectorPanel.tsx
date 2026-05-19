import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import {
  buildScenarioEntityRegistry,
  getScenarioEntity,
} from '@/features/scenario/entities';
import {
  addObjectiveTerm,
  createObjectiveTerm,
  ensureObjectiveSpec,
  OBJECTIVE_DISCOUNT_OPTIONS,
  OBJECTIVE_PENALTY_OPTIONS,
  OBJECTIVE_TEMPORAL_MODE_OPTIONS,
  objectiveTermEnabled,
  removeObjectiveTerm,
  setObjectiveTermEnabled,
  sourceSelectorForEntity,
  targetSelectorForEntity,
  updateObjectiveTerm,
} from '@/features/scenario/objectives';
import {
  selectorDetail,
  selectorDisplayLabel,
  selectorGroupLabel,
  selectorAccessExpression,
  selectorWithAccessExpression,
  preferredSelectorForGraphPort,
  selectorOptionsForGraphPort,
  selectorOptionsForRegistry,
  type StudioSelectorOption,
} from '@/features/scenario/selectors';
import { useGraphStore } from '@/stores/graphStore';
import {
  getActiveStage,
  getScenario,
  getTopPaneState,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type {
  StudioObjectiveTermSpec,
  StudioScenarioEntity,
  StudioScenarioEntityRegistry,
  StudioSelectorRef,
} from '@/types/workspace';
import { PropertiesPanel } from '@/components/panels/PropertiesPanel';
import { Plus, Trash2 } from 'lucide-react';
import type { ParamValue } from '@/types/graph';
import type { TimeAggregationSpec } from '@/types/training';

const GRAPH_ENTITY_KINDS = new Set(['graph_node', 'probe']);

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return 'None';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function coerceParamValue(rawValue: string, currentValue: unknown): ParamValue {
  if (typeof currentValue === 'number') {
    const parsed = Number.parseFloat(rawValue);
    return Number.isFinite(parsed) ? parsed : currentValue;
  }
  if (typeof currentValue === 'boolean') {
    return rawValue === 'true';
  }
  if (Array.isArray(currentValue) || (currentValue && typeof currentValue === 'object')) {
    try {
      return JSON.parse(rawValue) as ParamValue;
    } catch {
      return currentValue as ParamValue;
    }
  }
  if (currentValue === null) return rawValue;
  return rawValue;
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
  return {
    ...temporalSelector(term),
    ...updates,
  };
}

function parsedOptionalNumber(value: string): number | undefined {
  if (!value.trim()) return undefined;
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function ParamValueEditor({
  name,
  value,
  onChange,
}: {
  name: string;
  value: unknown;
  onChange: (value: ParamValue) => void;
}) {
  if (typeof value === 'boolean') {
    return (
      <label className="grid grid-cols-[7rem_minmax(0,1fr)] items-center gap-3 text-xs">
        <span className="truncate font-medium text-slate-500">{name}</span>
        <input
          type="checkbox"
          checked={value}
          onChange={(event) => onChange(event.target.checked)}
          className="h-4 w-4 rounded border-slate-300"
        />
      </label>
    );
  }

  const isStructured = Array.isArray(value) || (value !== null && typeof value === 'object');
  return (
    <label className="grid grid-cols-[7rem_minmax(0,1fr)] gap-3 text-xs">
      <span className="truncate pt-2 font-medium text-slate-500">{name}</span>
      {isStructured ? (
        <textarea
          value={formatValue(value)}
          onChange={(event) => onChange(coerceParamValue(event.target.value, value))}
          className="min-h-16 rounded border border-slate-200 px-2 py-1.5 font-mono text-xs text-slate-700"
        />
      ) : (
        <input
          type={typeof value === 'number' ? 'number' : 'text'}
          value={formatValue(value)}
          step={typeof value === 'number' ? 'any' : undefined}
          onChange={(event) => onChange(coerceParamValue(event.target.value, value))}
          className="h-8 rounded border border-slate-200 px-2 text-xs text-slate-700"
        />
      )}
    </label>
  );
}

function EntityHeader({ entity }: { entity: StudioScenarioEntity }) {
  const showSummary = entity.kind !== 'graph_node' && entity.kind !== 'graph_port';
  return (
    <div className="border-b border-slate-100 px-6 py-4">
      <div className="break-words text-sm font-semibold text-slate-800">{entity.label}</div>
      {showSummary && entity.summary && (
        <div className="mt-1 break-words text-xs text-slate-500">{entity.summary}</div>
      )}
    </div>
  );
}

function RelationList({
  entity,
  registry,
}: {
  entity: StudioScenarioEntity;
  registry: StudioScenarioEntityRegistry;
}) {
  if (entity.relations.length === 0) return null;
  return (
    <section className="space-y-2 border-t border-slate-100 pt-4">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Relations</div>
      <div className="space-y-1.5">
        {entity.relations.map((item, index) => {
          const related = registry.entities[item.entity_id];
          return (
            <div
              key={`${item.kind}:${item.entity_id}:${index}`}
              className="rounded-md border border-slate-100 bg-slate-50 px-2.5 py-2 text-xs"
            >
              <div className="font-medium text-slate-600">{item.label ?? item.kind}</div>
              <div className="mt-0.5 break-words text-slate-500">
                {related ? related.label : item.entity_id}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function TaskInspector({ entity }: { entity: StudioScenarioEntity }) {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const updateActiveScenarioTaskSpec = useWorkspaceStore((state) => state.updateActiveScenarioTaskSpec);
  const markDirty = useGraphStore((state) => state.markDirty);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const task = activeScenario?.task_spec ?? entity.metadata.task_spec;
  const params =
    task && typeof task === 'object' && 'params' in task && task.params && typeof task.params === 'object'
      ? Object.entries(task.params as Record<string, unknown>)
      : [];
  const updateParam = (key: string, value: ParamValue) => {
    if (!activeScenario?.task_spec) return;
    updateActiveScenarioTaskSpec({
      ...activeScenario.task_spec,
      params: {
        ...activeScenario.task_spec.params,
        [key]: value,
      },
    });
    markDirty();
  };
  const bindingState = formatValue(entity.metadata.binding_state);
  const inheritanceState = formatValue(entity.metadata.inheritance_state);
  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Task Parameters</div>
        <div className="text-[10px] uppercase tracking-[0.18em] text-slate-400">
          {bindingState} · {inheritanceState}
        </div>
      </div>
      {params.length === 0 ? (
        <div className="text-sm text-slate-400">No task parameters recorded.</div>
      ) : (
        <div className="space-y-2">
          {params.map(([key, value]) => (
            <ParamValueEditor
              key={key}
              name={key}
              value={value}
              onChange={(nextValue) => updateParam(key, nextValue)}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function MechanicsInspector({ entity }: { entity: StudioScenarioEntity }) {
  const graph = useGraphStore((state) => state.graph);
  const updateNodeParams = useGraphStore((state) => state.updateNodeParams);
  const nodeId = typeof entity.metadata.node_id === 'string' ? entity.metadata.node_id : null;
  const componentType =
    typeof entity.metadata.component_type === 'string' ? entity.metadata.component_type : null;
  const nodeSpec = nodeId ? graph.nodes[nodeId] : null;
  const params = Object.entries(nodeSpec?.params ?? {});
  const bindingState = formatValue(entity.metadata.binding_state);
  const inheritanceState = formatValue(entity.metadata.inheritance_state);
  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Mechanics Binding</div>
        <div className="text-[10px] uppercase tracking-[0.18em] text-slate-400">
          {bindingState} · {inheritanceState}
        </div>
      </div>
      <div className="space-y-2 text-xs text-slate-600">
        <div className="grid grid-cols-[6rem_minmax(0,1fr)] gap-3">
          <div className="font-medium text-slate-500">Graph node</div>
          <div className="break-words">{nodeId ?? 'None'}</div>
        </div>
        <div className="grid grid-cols-[6rem_minmax(0,1fr)] gap-3">
          <div className="font-medium text-slate-500">Type</div>
          <div className="break-words">{componentType ?? 'Scenario mechanics'}</div>
        </div>
      </div>
      {params.length > 0 && (
        <div className="space-y-2 border-t border-slate-100 pt-3">
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Parameters</div>
          {params.map(([key, value]) => (
            <ParamValueEditor
              key={key}
              name={key}
              value={value}
              onChange={(nextValue) => {
                if (nodeId) updateNodeParams(nodeId, key, nextValue);
              }}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function SelectorPicker({
  value,
  options,
  onChange,
}: {
  value: StudioSelectorRef | null | undefined;
  options: StudioSelectorOption[];
  onChange: (selector: StudioSelectorRef | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const selectedLabel = selectorDisplayLabel(value);
  const selectedDetail = selectorDetail(value);
  const normalizedQuery = query.trim().toLowerCase();
  const visible = options.filter((option) => {
    if (!normalizedQuery) return true;
    return [
      option.label,
      option.detail,
      option.selector.compact,
      option.selector.path,
      selectorGroupLabel(option.group),
    ]
      .filter(Boolean)
      .some((part) => part!.toLowerCase().includes(normalizedQuery));
  });
  const groups = Array.from(new Set(visible.map((option) => option.group)));

  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="flex w-full items-center justify-between gap-3 rounded border border-slate-200 bg-white px-2.5 py-2 text-left hover:border-brand-200"
      >
        <span className="min-w-0">
          <span className="block truncate text-sm font-medium text-slate-800">
            {selectedLabel}
          </span>
          {selectedDetail && (
            <span className="mt-0.5 block truncate text-[11px] text-slate-400">
              {selectedDetail}
            </span>
          )}
        </span>
        <span className="shrink-0 text-[11px] font-medium text-brand-600">Browse paths</span>
      </button>
      {open && (
        <div className="rounded-md border border-slate-200 bg-white p-2 shadow-sm">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search paths, ports, task objects"
            className="mb-2 h-8 w-full rounded border border-slate-200 px-2 text-xs text-slate-700 focus:border-brand-300 focus:outline-none"
            autoFocus
          />
          <div className="max-h-72 overflow-y-auto">
            <button
              type="button"
              onClick={() => {
                onChange(null);
                setOpen(false);
              }}
              className="mb-1 flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-xs text-slate-500 hover:bg-slate-50"
            >
              None
            </button>
            {groups.map((group) => (
              <div key={group} className="mb-2 last:mb-0">
                <div className="px-2 pb-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {selectorGroupLabel(group)}
                </div>
                <div className="space-y-1">
                  {visible
                    .filter((option) => option.group === group)
                    .map((option) => (
                      <button
                        key={option.id}
                        type="button"
                        onClick={() => {
                          onChange(option.selector);
                          setOpen(false);
                          setQuery('');
                        }}
                        className="flex w-full items-start justify-between gap-3 rounded px-2 py-1.5 text-left text-xs hover:bg-brand-50"
                      >
                        <span className="min-w-0">
                          <span className="block truncate font-medium text-slate-700">
                            {option.label}
                          </span>
                          <span className="mt-0.5 block truncate text-[11px] text-slate-400">
                            {option.detail ?? option.selector.compact}
                          </span>
                        </span>
                        {option.used_by_objective_ids.length > 0 && (
                          <span className="shrink-0 rounded bg-violet-50 px-1.5 py-0.5 text-[10px] font-medium text-violet-700">
                            used
                          </span>
                        )}
                      </button>
                    ))}
                </div>
              </div>
            ))}
            {visible.length === 0 && (
              <div className="px-2 py-4 text-center text-xs text-slate-400">
                No matching selectors.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function SelectorAccessEditor({
  value,
  onChange,
}: {
  value: StudioSelectorRef | null | undefined;
  onChange: (selector: StudioSelectorRef | null) => void;
}) {
  const [expression, setExpression] = useState(selectorAccessExpression(value));

  useEffect(() => {
    setExpression(selectorAccessExpression(value));
  }, [value]);

  return (
    <div className="rounded-md border border-slate-200 bg-slate-50/70 p-2">
      <label className="block space-y-1 text-xs text-slate-500">
        <span>Subpath / slice</span>
        <div className="flex gap-2">
          <input
            value={expression}
            onChange={(event) => setExpression(event.target.value)}
            placeholder=".state[0:2]"
            className="h-8 min-w-0 flex-1 rounded border border-slate-200 bg-white px-2 font-mono text-xs text-slate-700 focus:border-brand-300 focus:outline-none"
          />
          <button
            type="button"
            onClick={() => onChange(selectorWithAccessExpression(value, expression))}
            className="shrink-0 rounded border border-slate-200 bg-white px-2 text-xs font-medium text-slate-600 hover:border-brand-200 hover:text-slate-900"
          >
            Apply
          </button>
        </div>
      </label>
      <div className="mt-1 text-[11px] leading-4 text-slate-400">
        Use field/key access and Python-style indices or slices relative to the selected source.
      </div>
    </div>
  );
}

function SelectorCandidateButtons({
  options,
  value,
  onChange,
}: {
  options: StudioSelectorOption[];
  value: StudioSelectorRef | null | undefined;
  onChange: (selector: StudioSelectorRef) => void;
}) {
  if (options.length === 0) return null;
  return (
    <div className="grid gap-1.5">
      {options.map((option) => {
        const selected = option.selector.compact === value?.compact;
        return (
          <button
            key={option.id}
            type="button"
            onClick={() => onChange(option.selector)}
            className={clsx(
              'flex min-w-0 items-center justify-between gap-2 rounded-md border px-2.5 py-2 text-left text-xs',
              selected
                ? 'border-brand-300 bg-brand-50 text-slate-800'
                : 'border-slate-200 bg-white text-slate-600 hover:border-brand-200'
            )}
          >
            <span className="min-w-0">
              <span className="block truncate font-medium">{option.label}</span>
              {option.detail && (
                <span className="mt-0.5 block truncate text-[11px] text-slate-400">
                  {option.detail}
                </span>
              )}
            </span>
            {selected && (
              <span className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.12em] text-brand-600">
                source
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

function ObjectiveInspector({
  entity,
  registry,
}: {
  entity: StudioScenarioEntity;
  registry: StudioScenarioEntityRegistry;
}) {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const updateActiveScenarioObjectiveSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioObjectiveSpec
  );
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const objectiveSpec = ensureObjectiveSpec(activeScenario?.objective_spec);
  const termId = entity.id.replace(/^objective_term:/, '');
  const term = objectiveSpec.terms.find((candidate) => candidate.id === termId);
  const selectorOptions = useMemo(
    () => selectorOptionsForRegistry({ registry, objectiveSpec }),
    [objectiveSpec, registry]
  );
  if (!term) {
    return <div className="text-sm text-slate-400">Objective term is no longer available.</div>;
  }
  const time = temporalSelector(term);

  const updateTerm = (updates: Parameters<typeof updateObjectiveTerm>[2]) => {
    updateActiveScenarioObjectiveSpec(updateObjectiveTerm(objectiveSpec, term.id, updates));
  };

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Objective Term</div>
        <button
          type="button"
          onClick={() => {
            if (!confirm('Delete this objective term?')) return;
            updateActiveScenarioObjectiveSpec(removeObjectiveTerm(objectiveSpec, term.id));
            selectTopPaneEntity(null);
          }}
          className="rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600"
          title="Delete objective"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
      <label className="block space-y-1 text-xs text-slate-500">
        <span>Label</span>
        <input
          value={term.label}
          onChange={(event) => updateTerm({ label: event.target.value })}
          className="w-full rounded border border-slate-200 px-2 py-1.5 text-sm text-slate-800"
        />
      </label>
      <div className="grid grid-cols-[minmax(0,1fr)_6rem] gap-3">
        <label className="block space-y-1 text-xs text-slate-500">
          <span>Role</span>
          <select
            value={term.role}
            onChange={(event) => updateTerm({ role: event.target.value })}
            className="w-full rounded border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-800"
          >
            <option value="loss">Loss</option>
            <option value="metric">Metric</option>
            <option value="constraint">Constraint</option>
            <option value="reward">Reward</option>
            <option value="regularizer">Regularizer</option>
          </select>
        </label>
        <label className="block space-y-1 text-xs text-slate-500">
          <span>Weight</span>
          <input
            type="number"
            min={0}
            step={0.01}
            value={term.weight}
            onChange={(event) => {
              const weight = Number.parseFloat(event.target.value);
              if (Number.isFinite(weight)) updateTerm({ weight });
            }}
            className="w-full rounded border border-slate-200 px-2 py-1.5 text-sm text-slate-800"
          />
        </label>
      </div>
      <label className="block space-y-1 text-xs text-slate-500">
        <span>Penalty</span>
        <select
          value={term.penalty ?? 'squared_l2'}
          onChange={(event) => updateTerm({ penalty: event.target.value })}
          className="w-full rounded border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-800"
        >
          {OBJECTIVE_PENALTY_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-2 text-xs text-slate-600">
        <input
          type="checkbox"
          checked={objectiveTermEnabled(term)}
          onChange={(event) =>
            updateActiveScenarioObjectiveSpec(
              setObjectiveTermEnabled(objectiveSpec, term.id, event.target.checked)
            )
          }
          className="h-4 w-4 rounded border-slate-300"
        />
        Enabled
      </label>
      <div className="block space-y-1 text-xs text-slate-500">
        <span>Objective source</span>
        <SelectorPicker
          value={term.source_selector}
          options={selectorOptions}
          onChange={(selector) => updateTerm({ source_selector: selector })}
        />
        <SelectorAccessEditor
          value={term.source_selector}
          onChange={(selector) => updateTerm({ source_selector: selector })}
        />
      </div>
      <div className="space-y-2 border-t border-slate-100 pt-3">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Time</div>
        <div className="grid grid-cols-2 gap-3">
          <label className="block space-y-1 text-xs text-slate-500">
            <span>Mode</span>
            <select
              value={time.mode}
              onChange={(event) =>
                updateTerm({
                  temporal_selector: updateTemporalSelector(term, {
                    mode: event.target.value as TimeAggregationSpec['mode'],
                  }),
                })
              }
              className="w-full rounded border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-800"
            >
              {OBJECTIVE_TEMPORAL_MODE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="block space-y-1 text-xs text-slate-500">
            <span>Discount</span>
            <select
              value={time.discount ?? 'none'}
              onChange={(event) =>
                updateTerm({
                  temporal_selector: updateTemporalSelector(term, {
                    discount: event.target.value as TimeAggregationSpec['discount'],
                  }),
                })
              }
              className="w-full rounded border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-800"
            >
              {OBJECTIVE_DISCOUNT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        {time.discount === 'power' && (
          <label className="block space-y-1 text-xs text-slate-500">
            <span>Exponent</span>
            <input
              type="number"
              step={0.1}
              value={time.discount_exp ?? ''}
              onChange={(event) =>
                updateTerm({
                  temporal_selector: updateTemporalSelector(term, {
                    discount_exp: parsedOptionalNumber(event.target.value),
                  }),
                })
              }
              className="w-full rounded border border-slate-200 px-2 py-1.5 text-sm text-slate-800"
            />
          </label>
        )}
        {time.mode === 'range' && (
          <div className="grid grid-cols-2 gap-3">
            <label className="block space-y-1 text-xs text-slate-500">
              <span>Start</span>
              <input
                type="number"
                value={time.start ?? ''}
                onChange={(event) =>
                  updateTerm({
                    temporal_selector: updateTemporalSelector(term, {
                      start: parsedOptionalNumber(event.target.value),
                    }),
                  })
                }
                className="w-full rounded border border-slate-200 px-2 py-1.5 text-sm text-slate-800"
              />
            </label>
            <label className="block space-y-1 text-xs text-slate-500">
              <span>End</span>
              <input
                type="number"
                value={time.end ?? ''}
                onChange={(event) =>
                  updateTerm({
                    temporal_selector: updateTemporalSelector(term, {
                      end: parsedOptionalNumber(event.target.value),
                    }),
                  })
                }
                className="w-full rounded border border-slate-200 px-2 py-1.5 text-sm text-slate-800"
              />
            </label>
          </div>
        )}
        {time.mode === 'segment' && (
          <label className="block space-y-1 text-xs text-slate-500">
            <span>Segment</span>
            <input
              value={time.segment_name ?? ''}
              onChange={(event) =>
                updateTerm({
                  temporal_selector: updateTemporalSelector(term, {
                    segment_name: event.target.value,
                  }),
                })
              }
              className="w-full rounded border border-slate-200 px-2 py-1.5 text-sm text-slate-800"
            />
          </label>
        )}
        {time.mode === 'custom' && (
          <label className="block space-y-1 text-xs text-slate-500">
            <span>Steps</span>
            <input
              value={(time.time_idxs ?? []).join(', ')}
              onChange={(event) =>
                updateTerm({
                  temporal_selector: updateTemporalSelector(term, {
                    time_idxs: event.target.value
                      .split(',')
                      .map((part) => Number.parseInt(part.trim(), 10))
                      .filter((value) => Number.isFinite(value)),
                  }),
                })
              }
              className="w-full rounded border border-slate-200 px-2 py-1.5 text-sm text-slate-800"
            />
          </label>
        )}
      </div>
      <div className="space-y-2 text-xs text-slate-600">
        <div className="grid grid-cols-[6rem_minmax(0,1fr)] gap-3">
          <div className="font-medium text-slate-500">Type</div>
          <div className="break-words">{formatValue(term.type_id)}</div>
        </div>
      </div>
    </section>
  );
}

function SourceInspector({
  entity,
  registry,
}: {
  entity: StudioScenarioEntity;
  registry: StudioScenarioEntityRegistry;
}) {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const updateActiveScenarioObjectiveSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioObjectiveSpec
  );
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const setTopPaneProjection = useWorkspaceStore((state) => state.setTopPaneProjection);
  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const setSelectedTap = useGraphStore((state) => state.setSelectedTap);
  const setSelectedEdge = useGraphStore((state) => state.setSelectedEdge);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const objectiveSpec = ensureObjectiveSpec(activeScenario?.objective_spec);
  const sourceSelector = sourceSelectorForEntity(entity, registry);
  const selectorOptions = useMemo(
    () => selectorOptionsForRegistry({ registry, objectiveSpec }),
    [objectiveSpec, registry]
  );
  const preferredSourceSelector = useMemo(
    () => preferredSelectorForGraphPort(sourceSelector, selectorOptions),
    [selectorOptions, sourceSelector]
  );
  const sourceCandidates = useMemo(
    () => selectorOptionsForGraphPort(sourceSelector, selectorOptions),
    [selectorOptions, sourceSelector]
  );
  const [draftSourceSelector, setDraftSourceSelector] = useState<StudioSelectorRef | null>(
    preferredSourceSelector
  );
  const taskEntity =
    Object.values(registry.entities).find((candidate) => candidate.kind === 'task_object') ?? null;

  useEffect(() => {
    setDraftSourceSelector(preferredSourceSelector);
  }, [entity.id, preferredSourceSelector]);

  const selectedSourceSelector = draftSourceSelector ?? preferredSourceSelector;

  const addObjective = () => {
    if (!activeScenario || !selectedSourceSelector) return;
    const term = createObjectiveTerm({
      spec: objectiveSpec,
      label: `Objective: ${selectorDisplayLabel(selectedSourceSelector)}`,
      sourceSelector: selectedSourceSelector,
      targetSelector: targetSelectorForEntity(taskEntity),
    });
    updateActiveScenarioObjectiveSpec(addObjectiveTerm(objectiveSpec, term));
    setSelectedNode(null);
    setSelectedTap(null);
    setSelectedEdge(null);
    setTopPaneProjection('objectives');
    selectTopPaneEntity(`objective_term:${term.id}`);
  };

  return (
    <section className="space-y-3">
      <div className="space-y-2 text-xs text-slate-500">
        <span>Objective source</span>
        <SelectorCandidateButtons
          options={sourceCandidates}
          value={selectedSourceSelector}
          onChange={setDraftSourceSelector}
        />
        <SelectorPicker
          value={selectedSourceSelector}
          options={selectorOptions}
          onChange={setDraftSourceSelector}
        />
        <SelectorAccessEditor
          value={selectedSourceSelector}
          onChange={setDraftSourceSelector}
        />
      </div>
      <button
        type="button"
        disabled={!activeScenario || !selectedSourceSelector}
        onClick={addObjective}
        className="inline-flex h-8 items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-xs font-medium text-slate-600 shadow-sm hover:border-brand-200 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Plus className="h-3.5 w-3.5" />
        Add objective from {selectorDisplayLabel(selectedSourceSelector)}
      </button>
    </section>
  );
}

function EdgeInspector({ entity }: { entity: StudioScenarioEntity }) {
  const addTapForEdge = useGraphStore((state) => state.addTapForEdge);
  const edgeId = typeof entity.metadata.edge_id === 'string' ? entity.metadata.edge_id : null;
  const edgeType = typeof entity.metadata.edge_type === 'string' ? entity.metadata.edge_type : null;
  const wire = entity.metadata.wire as
    | {
        source_node: string;
        source_port: string;
        target_node: string;
        target_port: string;
      }
    | undefined;

  if (edgeType === 'state_flow') {
    return (
      <div className="space-y-5 p-6">
        <div>
          <div className="text-sm font-medium text-slate-800">{entity.label}</div>
          <div className="mt-1 text-xs text-slate-500">Full state flow</div>
        </div>
        {edgeId && (
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 hover:text-slate-800"
              onClick={() => addTapForEdge(edgeId, 'probe')}
            >
              Add Probe Tap
            </button>
            <button
              type="button"
              className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 hover:text-slate-800"
              onClick={() => addTapForEdge(edgeId, 'intervention')}
            >
              Add Intervention Tap
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-2 p-6">
      <div className="text-sm font-medium text-slate-800">
        {wire
          ? `${wire.source_node}.${wire.source_port} → ${wire.target_node}.${wire.target_port}`
          : entity.label}
      </div>
      <div className="text-xs text-slate-400">
        Port wires are the source of truth for state merging.
      </div>
    </div>
  );
}

function EntityBody({
  entity,
  registry,
}: {
  entity: StudioScenarioEntity;
  registry: StudioScenarioEntityRegistry;
}) {
  if (GRAPH_ENTITY_KINDS.has(entity.kind)) {
    return <PropertiesPanel />;
  }

  return (
    <div className="space-y-5 p-6">
      {entity.kind === 'task_object' && <TaskInspector entity={entity} />}
      {entity.kind === 'mechanics_object' && <MechanicsInspector entity={entity} />}
      {entity.kind === 'objective_term' && (
        <ObjectiveInspector entity={entity} registry={registry} />
      )}
      {entity.kind === 'graph_port' && <SourceInspector entity={entity} registry={registry} />}
      <RelationList entity={entity} registry={registry} />
    </div>
  );
}

export function ScenarioInspectorPanel() {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const topPaneState = getTopPaneState(workspace);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const graph = useGraphStore((state) => state.graph);

  const registry = useMemo(
    () => buildScenarioEntityRegistry({ scenario: activeScenario, graph }),
    [activeScenario, graph]
  );
  const entity = getScenarioEntity(registry, topPaneState.selected_entity_id);

  if (!topPaneState.selected_entity_id) {
    return (
      <div className="p-6 text-sm text-slate-500">
        Select a scenario entity on the canvas to view properties.
      </div>
    );
  }

  if (!entity) {
    return (
      <div className="p-6 text-sm text-slate-500">
        Selected entity is no longer available.
        <div className="mt-2 break-words font-mono text-xs text-slate-400">
          {topPaneState.selected_entity_id}
        </div>
      </div>
    );
  }

  return (
    <div>
      {entity.kind !== 'graph_node' && entity.kind !== 'graph_edge' && (
        <EntityHeader entity={entity} />
      )}
      {entity.kind === 'graph_edge' ? (
        <>
          <EdgeInspector entity={entity} />
          {entity.selector && (
            <div className="border-t border-slate-100 p-6">
              <SourceInspector entity={entity} registry={registry} />
            </div>
          )}
          <div className="px-6 pb-6">
            <RelationList entity={entity} registry={registry} />
          </div>
        </>
      ) : (
      <EntityBody entity={entity} registry={registry} />
      )}
    </div>
  );
}
