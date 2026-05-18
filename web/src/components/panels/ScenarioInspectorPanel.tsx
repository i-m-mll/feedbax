import { useMemo } from 'react';
import clsx from 'clsx';
import {
  buildScenarioEntityRegistry,
  entityKindLabel,
  getScenarioEntity,
  selectorToEntityId,
} from '@/features/scenario/entities';
import {
  ensureObjectiveSpec,
  objectiveTermEnabled,
  removeObjectiveTerm,
  setObjectiveTermEnabled,
  updateObjectiveTerm,
} from '@/features/scenario/objectives';
import { useGraphStore } from '@/stores/graphStore';
import {
  getActiveStage,
  getScenario,
  getTopPaneState,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type { StudioScenarioEntity, StudioScenarioEntityRegistry } from '@/types/workspace';
import { PropertiesPanel } from '@/components/panels/PropertiesPanel';
import { Trash2 } from 'lucide-react';
import type { ParamValue } from '@/types/graph';

const GRAPH_ENTITY_KINDS = new Set(['graph_node', 'graph_edge', 'probe']);

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

function entityKindTone(kind: StudioScenarioEntity['kind']) {
  switch (kind) {
    case 'graph_node':
    case 'graph_port':
    case 'graph_edge':
      return 'bg-sky-50 text-sky-700 border-sky-100';
    case 'task_object':
      return 'bg-emerald-50 text-emerald-700 border-emerald-100';
    case 'mechanics_object':
      return 'bg-amber-50 text-amber-700 border-amber-100';
    case 'objective_term':
      return 'bg-violet-50 text-violet-700 border-violet-100';
    case 'probe':
      return 'bg-rose-50 text-rose-700 border-rose-100';
    default:
      return 'bg-slate-50 text-slate-600 border-slate-100';
  }
}

function EntityHeader({ entity }: { entity: StudioScenarioEntity }) {
  return (
    <div className="border-b border-slate-100 px-6 py-4">
      <div className="flex items-center gap-2">
        <span
          className={clsx(
            'rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em]',
            entityKindTone(entity.kind)
          )}
        >
          {entityKindLabel(entity.kind)}
        </span>
      </div>
      <div className="mt-2 break-words text-sm font-semibold text-slate-800">{entity.label}</div>
      {entity.summary && (
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
                {related ? `${entityKindLabel(related.kind)}: ${related.label}` : item.entity_id}
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
  const selectablePorts = Object.values(registry.entities)
    .filter((candidate) => candidate.kind === 'graph_port' && candidate.selector)
    .sort((a, b) => a.label.localeCompare(b.label));
  if (!term) {
    return <div className="text-sm text-slate-400">Objective term is no longer available.</div>;
  }

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
      <label className="block space-y-1 text-xs text-slate-500">
        <span>Source</span>
        <select
          value={selectorToEntityId(term.source_selector) ?? ''}
          onChange={(event) => {
            const source = registry.entities[event.target.value];
            updateTerm({
              source_selector: source?.selector ?? null,
            });
          }}
          className="w-full rounded border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-800"
        >
          <option value="">None</option>
          {selectablePorts.map((port) => (
            <option key={port.id} value={port.id}>
              {port.label}
            </option>
          ))}
        </select>
      </label>
      <div className="space-y-2 text-xs text-slate-600">
        <div className="grid grid-cols-[6rem_minmax(0,1fr)] gap-3">
          <div className="font-medium text-slate-500">Type</div>
          <div className="break-words">{formatValue(term.type_id)}</div>
        </div>
        <div className="grid grid-cols-[6rem_minmax(0,1fr)] gap-3">
          <div className="font-medium text-slate-500">Penalty</div>
          <div className="break-words">{formatValue(term.penalty)}</div>
        </div>
      </div>
    </section>
  );
}

function PortInspector({ entity }: { entity: StudioScenarioEntity }) {
  return (
    <section className="space-y-3">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Port Selector</div>
      <div className="space-y-2 text-xs text-slate-600">
        <div className="grid grid-cols-[5rem_minmax(0,1fr)] gap-3">
          <div className="font-medium text-slate-500">Node</div>
          <div className="break-words">{formatValue(entity.metadata.node_id)}</div>
        </div>
        <div className="grid grid-cols-[5rem_minmax(0,1fr)] gap-3">
          <div className="font-medium text-slate-500">Port</div>
          <div className="break-words">{formatValue(entity.metadata.port)}</div>
        </div>
        <div className="grid grid-cols-[5rem_minmax(0,1fr)] gap-3">
          <div className="font-medium text-slate-500">Direction</div>
          <div className="break-words">{formatValue(entity.metadata.direction)}</div>
        </div>
      </div>
    </section>
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
      {entity.kind === 'graph_port' && <PortInspector entity={entity} />}
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
      <EntityHeader entity={entity} />
      <EntityBody entity={entity} registry={registry} />
    </div>
  );
}
