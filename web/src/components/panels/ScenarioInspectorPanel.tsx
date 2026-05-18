import { useMemo } from 'react';
import clsx from 'clsx';
import { buildScenarioEntityRegistry, entityKindLabel, getScenarioEntity } from '@/features/scenario/entities';
import { useGraphStore } from '@/stores/graphStore';
import {
  getActiveStage,
  getScenario,
  getTopPaneState,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type { StudioScenarioEntity, StudioScenarioEntityRegistry } from '@/types/workspace';
import { PropertiesPanel } from '@/components/panels/PropertiesPanel';

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
  const task = entity.metadata.task_spec;
  const params =
    task && typeof task === 'object' && 'params' in task && task.params && typeof task.params === 'object'
      ? Object.entries(task.params as Record<string, unknown>)
      : [];
  return (
    <section className="space-y-3">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Task Parameters</div>
      {params.length === 0 ? (
        <div className="text-sm text-slate-400">No task parameters recorded.</div>
      ) : (
        <div className="space-y-2">
          {params.map(([key, value]) => (
            <div key={key} className="grid grid-cols-[7rem_minmax(0,1fr)] gap-3 text-xs">
              <div className="truncate font-medium text-slate-500">{key}</div>
              <div className="break-words text-slate-700">{formatValue(value)}</div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function MechanicsInspector({ entity }: { entity: StudioScenarioEntity }) {
  const nodeId = typeof entity.metadata.node_id === 'string' ? entity.metadata.node_id : null;
  const componentType =
    typeof entity.metadata.component_type === 'string' ? entity.metadata.component_type : null;
  return (
    <section className="space-y-3">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Mechanics Binding</div>
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
    </section>
  );
}

function ObjectiveInspector({ entity }: { entity: StudioScenarioEntity }) {
  const term = entity.metadata.term;
  const record = term && typeof term === 'object' ? (term as Record<string, unknown>) : {};
  return (
    <section className="space-y-3">
      <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Objective Term</div>
      <div className="space-y-2 text-xs text-slate-600">
        {['role', 'type_id', 'operator', 'penalty', 'weight', 'units'].map((key) => (
          <div key={key} className="grid grid-cols-[6rem_minmax(0,1fr)] gap-3">
            <div className="font-medium capitalize text-slate-500">{key.replace('_', ' ')}</div>
            <div className="break-words">{formatValue(record[key])}</div>
          </div>
        ))}
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
      {entity.kind === 'objective_term' && <ObjectiveInspector entity={entity} />}
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
