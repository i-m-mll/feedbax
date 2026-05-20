import { useMemo } from 'react';
import { PlugZap, Settings2 } from 'lucide-react';
import { createDefaultTaskBindingSpec, ensureTaskBindingSpec } from '@/features/scenario/taskBindings';
import { useGraphStore } from '@/stores/graphStore';
import {
  getTopPaneState,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import type { ParamValue } from '@/types/graph';
import type { TaskSpec } from '@/types/training';

const TASK_CATALOG: TaskSpec[] = [
  {
    type: 'SimpleReaches',
    params: {
      n_steps: 200,
      workspace: [
        [-1.0, -1.0],
        [1.0, 1.0],
      ],
      eval_n_directions: 7,
      eval_reach_length: 0.5,
      eval_grid_n: 1,
    },
  },
  {
    type: 'DelayedReaches',
    params: {
      n_steps: 140,
      train_endpoint_mode: 'center_out',
      p_catch_trial: 0.5,
      eval_n_directions: 8,
      eval_reach_length: 0.5,
    },
  },
  {
    type: 'Stabilization',
    params: {
      n_steps: 200,
      eval_n_directions: 8,
    },
  },
];

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
  if (typeof currentValue === 'boolean') return rawValue === 'true';
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

function ParamEditor({
  name,
  value,
  onChange,
}: {
  name: string;
  value: unknown;
  onChange: (value: ParamValue) => void;
}) {
  const structured = Array.isArray(value) || (value !== null && typeof value === 'object');
  if (typeof value === 'boolean') {
    return (
      <label className="flex items-center justify-between gap-3 text-xs">
        <span className="min-w-0 truncate font-medium text-slate-500">{name}</span>
        <input
          type="checkbox"
          checked={value}
          onChange={(event) => onChange(event.target.checked)}
          className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
        />
      </label>
    );
  }
  return (
    <label className="grid gap-1 text-xs">
      <span className="truncate font-medium text-slate-500">{name}</span>
      {structured ? (
        <textarea
          value={formatValue(value)}
          onChange={(event) => onChange(coerceParamValue(event.target.value, value))}
          className="min-h-14 rounded border border-slate-200 px-2 py-1.5 font-mono text-[11px] text-slate-700"
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

export function TaskScenarioPanel() {
  const graph = useGraphStore((state) => state.graph);
  const markDirty = useGraphStore((state) => state.markDirty);
  const workspace = useWorkspaceStore((state) => state.workspace);
  const updateTaskSpec = useWorkspaceStore((state) => state.updateActiveScenarioTaskSpec);
  const updateTaskBindingSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioTaskBindingSpec
  );
  const topPane = getTopPaneState(workspace);
  const scenario = getTrainingScenario(workspace);
  const task = scenario?.task_spec ?? TASK_CATALOG[0];
  const taskBindingSpec = useMemo(
    () => ensureTaskBindingSpec(scenario?.task_binding_spec ?? createDefaultTaskBindingSpec(graph), graph),
    [graph, scenario?.task_binding_spec]
  );

  if (topPane.active_projection !== 'graph') return null;

  const params = Object.entries(task.params ?? {});
  const bindableOutputs = taskBindingSpec.exposed_outputs.filter((output) => output.bindable);
  const protocolOutputs = taskBindingSpec.exposed_outputs.filter((output) => !output.bindable);
  const boundTarget = (nodeId: string, port: string) => `${nodeId}.${port}`;
  const updateParam = (key: string, value: ParamValue) => {
    updateTaskSpec({
      ...task,
      params: {
        ...task.params,
        [key]: value,
      },
    });
    markDirty();
  };
  const changeTaskType = (taskType: string) => {
    const next = TASK_CATALOG.find((candidate) => candidate.type === taskType);
    if (!next) return;
    updateTaskSpec(next);
    updateTaskBindingSpec(createDefaultTaskBindingSpec(graph));
    markDirty();
  };

  return (
    <aside className="relative z-20 flex w-72 shrink-0 flex-col overflow-visible border-r border-slate-100 bg-white/95">
      <div className="border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.24em] text-emerald-600">
          <Settings2 className="h-3.5 w-3.5" />
          Task
        </div>
        <select
          value={task.type}
          onChange={(event) => changeTaskType(event.target.value)}
          className="mt-2 h-9 w-full rounded border border-slate-200 bg-white px-2 text-sm font-medium text-slate-800"
        >
          {TASK_CATALOG.map((candidate) => (
            <option key={candidate.type} value={candidate.type}>
              {candidate.type}
            </option>
          ))}
        </select>
      </div>
      <section className="shrink-0 border-b border-slate-100 px-4 py-2.5">
        <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
          <PlugZap className="h-3 w-3" />
          Outputs
        </div>
        <div className="mt-2 space-y-1">
          {bindableOutputs.map((output) => (
            <div
              key={output.id}
              className="relative flex h-7 items-center justify-end pr-3 text-xs"
            >
              <span className="min-w-0 truncate text-right font-medium text-slate-700">
                {output.label}
              </span>
              <span
                data-task-output-port-id={output.id}
                className="absolute right-[-21px] top-1/2 z-30 h-2.5 w-2.5 -translate-y-1/2 rounded-full border border-white bg-emerald-500 shadow-soft"
                title={`${output.label} task output`}
              />
            </div>
          ))}
        </div>
      </section>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4">
        <section className="space-y-2">
          <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
            Parameters
          </div>
          {params.length > 0 ? (
            params.map(([key, value]) => (
              <ParamEditor
                key={key}
                name={key}
                value={value}
                onChange={(nextValue) => updateParam(key, nextValue)}
              />
            ))
          ) : (
            <div className="text-xs text-slate-400">None recorded</div>
          )}
        </section>
        {protocolOutputs.length > 0 && (
          <section className="space-y-2">
            <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
              Protocol
            </div>
            <div className="space-y-1">
              {protocolOutputs.map((output) => (
                <div
                  key={output.id}
                  className="flex h-7 items-center justify-between gap-3 rounded border border-slate-100 px-2 text-xs"
                >
                  <span className="truncate font-medium text-slate-600">{output.label}</span>
                  <span className="shrink-0 text-[10px] text-slate-400">{output.kind}</span>
                </div>
              ))}
            </div>
          </section>
        )}
        <section className="space-y-2">
          <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
            Bindings
          </div>
          <div className="space-y-1.5">
            {taskBindingSpec.bindings.map((binding) => (
              <div
                key={binding.id}
                className="rounded-md border border-emerald-100 bg-emerald-50/50 px-2.5 py-2 text-xs"
              >
                <div className="font-medium text-slate-700">{binding.source_output_id}</div>
                <div className="mt-0.5 truncate text-[11px] text-slate-500">
                  {boundTarget(binding.target_node_id, binding.target_port)}
                </div>
              </div>
            ))}
            {taskBindingSpec.bindings.length === 0 && (
              <div className="text-xs text-slate-400">None recorded</div>
            )}
          </div>
        </section>
      </div>
    </aside>
  );
}
