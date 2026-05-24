import {
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { Settings2, SlidersHorizontal } from 'lucide-react';
import {
  createDefaultTaskBindingSpec,
  ensureTaskBindingSpec,
  scopedTaskBindingSpec,
} from '@/features/scenario/taskBindings';
import {
  applyDelayedReachTimelineEdit,
  delayedReachTaskWithTimeline,
  delayedReachTimelineFromTask,
  isDelayedReachTimelineParam,
  toggleDelayedReachSignalEpoch,
  updateTaskTimelineSignalValueSpec,
  updateDelayedReachEpochRange,
} from '@/features/scenario/taskTimeline';
import {
  VALUE_SPEC_DISTRIBUTIONS,
  VALUE_SPEC_FUNCTION_TEMPLATES,
  VALUE_SPEC_MODE_OPTIONS,
  VALUE_SPEC_SCOPE_OPTIONS,
  setValueSpecDistributionFamily,
  setValueSpecFunction,
  setValueSpecMode,
  setValueSpecScope,
  valueSpecAllowedModes,
  valueSpecAllowedScopes,
  valueSpecChipLabel,
} from '@/features/scenario/valueSpecs';
import { useGraphStore } from '@/stores/graphStore';
import {
  getTopPaneState,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { useLayoutStore } from '@/stores/layoutStore';
import type { ParamValue } from '@/types/graph';
import type { TaskSpec } from '@/types/training';
import type { StudioTaskTimelineSpec, StudioValueSpec } from '@/types/workspace';

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
      epoch_len_ranges: [
        [0, 1],
        [10, 30],
      ],
      target_on_epochs: [1, 2],
      hold_epochs: [0, 1],
      move_epochs: [2],
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

function formatValueSpec(valueSpec: StudioValueSpec | null | undefined): string {
  return valueSpecChipLabel(valueSpec);
}

function ValueSpecPreview({ valueSpec }: { valueSpec: StudioValueSpec }) {
  const mode = valueSpec.mode;
  if (mode === 'distribution') {
    return (
      <div className="grid grid-cols-8 gap-1">
        {Array.from({ length: 8 }, (_, index) => (
          <span
            key={index}
            className="block rounded-sm bg-emerald-100"
            style={{ height: `${8 + ((index * 7) % 18)}px` }}
          />
        ))}
      </div>
    );
  }
  if (mode === 'function' || mode === 'schedule') {
    return (
      <div className="flex h-9 items-end gap-1">
        {[0.15, 0.2, 0.35, 0.55, 0.75, 0.9, 0.9, 0.9].map((height, index) => (
          <span
            key={index}
            className="block flex-1 rounded-sm bg-emerald-500/70"
            style={{ height: `${height * 100}%` }}
          />
        ))}
      </div>
    );
  }
  return (
    <div className="h-9 rounded-sm bg-slate-100 px-2 py-2 font-mono text-[11px] text-slate-500">
      {formatValueSpec(valueSpec)}
    </div>
  );
}

function ValueSpecInlineEditor({
  signal,
  onChange,
  onClose,
}: {
  signal: StudioTaskTimelineSpec['signals'][number];
  onChange: (valueSpec: StudioValueSpec) => void;
  onClose: () => void;
}) {
  const valueSpec = signal.value_spec;
  if (!valueSpec) return null;
  const modes = valueSpecAllowedModes(signal);
  const scopes = valueSpecAllowedScopes(signal);
  return (
    <div className="fixed left-4 top-28 z-50 w-[min(26rem,calc(100vw-2rem))] rounded border border-slate-200 bg-white/95 px-3 py-2 shadow-lg backdrop-blur">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 truncate text-xs font-semibold text-slate-700">
          {signal.label}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded px-2 py-1 text-[11px] font-medium text-slate-500 hover:bg-white"
        >
          Done
        </button>
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {VALUE_SPEC_MODE_OPTIONS.filter((option) => modes.includes(option.value)).map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(setValueSpecMode(valueSpec, option.value))}
            className={
              valueSpec.mode === option.value
                ? 'rounded border border-emerald-300 bg-emerald-50 px-2 py-1 text-[11px] font-medium text-emerald-700'
                : 'rounded border border-slate-200 bg-white px-2 py-1 text-[11px] font-medium text-slate-500 hover:border-emerald-200'
            }
          >
            {option.label}
          </button>
        ))}
      </div>
      <div className="mt-2 grid grid-cols-[minmax(0,1fr)_7.5rem] gap-3">
        <div className="min-w-0 space-y-2">
          {valueSpec.mode === 'function' && (
            <label className="grid gap-1 text-xs text-slate-500">
              <span>Function</span>
              <select
                value={valueSpec.function_id ?? VALUE_SPEC_FUNCTION_TEMPLATES[0].id}
                onChange={(event) => onChange(setValueSpecFunction(valueSpec, event.target.value))}
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs text-slate-700"
              >
                {VALUE_SPEC_FUNCTION_TEMPLATES.map((template) => (
                  <option key={template.id} value={template.id}>
                    {template.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          {valueSpec.mode === 'distribution' && (
            <label className="grid gap-1 text-xs text-slate-500">
              <span>Distribution</span>
              <select
                value={String(valueSpec.distribution?.family ?? 'uniform')}
                onChange={(event) =>
                  onChange(setValueSpecDistributionFamily(valueSpec, event.target.value))
                }
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs text-slate-700"
              >
                {VALUE_SPEC_DISTRIBUTIONS.map((family) => (
                  <option key={family} value={family}>
                    {family}
                  </option>
                ))}
              </select>
            </label>
          )}
          {valueSpec.mode === 'schedule' && (
            <label className="grid gap-1 text-xs text-slate-500">
              <span>Domain</span>
              <select
                value={String(valueSpec.schedule?.domain ?? 'epoch')}
                onChange={(event) =>
                  onChange({
                    ...valueSpec,
                    schedule: { ...(valueSpec.schedule ?? {}), domain: event.target.value },
                  })
                }
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs text-slate-700"
              >
                <option value="epoch">epoch</option>
                <option value="time">time</option>
                <option value="trial">trial</option>
              </select>
            </label>
          )}
          {valueSpec.mode === 'expression' && (
            <label className="grid gap-1 text-xs text-slate-500">
              <span>Expression</span>
              <input
                value={valueSpec.expression ?? ''}
                onChange={(event) => onChange({ ...valueSpec, expression: event.target.value })}
                className="h-8 rounded border border-slate-200 bg-white px-2 font-mono text-xs text-slate-700"
              />
            </label>
          )}
          <label className="grid gap-1 text-xs text-slate-500">
            <span>Scope</span>
            <select
              value={valueSpec.sampling_scope ?? scopes[0] ?? 'trial'}
              onChange={(event) => onChange(setValueSpecScope(valueSpec, event.target.value))}
              className="h-8 rounded border border-slate-200 bg-white px-2 text-xs text-slate-700"
            >
              {VALUE_SPEC_SCOPE_OPTIONS.filter((option) => scopes.includes(option.value)).map(
                (option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                )
              )}
            </select>
          </label>
        </div>
        <div className="rounded border border-slate-200 bg-white p-2">
          <ValueSpecPreview valueSpec={valueSpec} />
          <div className="mt-1 truncate text-[10px] text-slate-400">
            {formatValueSpec(valueSpec)}
          </div>
        </div>
      </div>
    </div>
  );
}

function DelayedReachTimelineEditor({
  timeline,
  onChange,
}: {
  timeline: StudioTaskTimelineSpec;
  onChange: (timeline: StudioTaskTimelineSpec) => void;
}) {
  const [editingSignalId, setEditingSignalId] = useState<string | null>(null);
  const editableEpochs = timeline.epochs.slice(0, -1);
  const signalRows = timeline.signals.filter((signal) => signal.kind === 'signal');
  const editingSignal = timeline.signals.find((signal) => signal.id === editingSignalId);
  const timelineGridColumns = `7rem 5.75rem repeat(${timeline.epochs.length}, minmax(4.875rem, 1fr))`;
  const timelineMinWidth = `${12.75 + timeline.epochs.length * 4.875}rem`;
  const updateVisibleRange = (
    epoch: StudioTaskTimelineSpec['epochs'][number],
    key: 'min' | 'max',
    nextValue: number
  ) => {
    const current = epoch.length.value as { min?: unknown; max?: unknown } | null;
    const currentMin = Number(current?.min ?? 0);
    const currentMaxExclusive = Number(current?.max ?? currentMin + 1);
    const currentMaxInclusive = Math.max(currentMin, currentMaxExclusive - 1);
    const nextMin = Math.max(
      0,
      Math.round(key === 'min' ? nextValue : currentMin)
    );
    const nextMaxInclusive = Math.max(
      nextMin,
      Math.round(key === 'max' ? nextValue : currentMaxInclusive)
    );
    const withMin = updateDelayedReachEpochRange(timeline, epoch.id, 'min', nextMin);
    return updateDelayedReachEpochRange(withMin, epoch.id, 'max', nextMaxInclusive + 1);
  };
  return (
    <section className="space-y-2">
      <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
        Timeline
      </div>
      {editingSignal && (
        <ValueSpecInlineEditor
          signal={editingSignal}
          onChange={(valueSpec) =>
            onChange(updateTaskTimelineSignalValueSpec(timeline, editingSignal.id, valueSpec))
          }
          onClose={() => setEditingSignalId(null)}
        />
      )}
      <div className="overflow-hidden rounded border border-slate-100">
        <div className="local-x-scrollbar overflow-x-scroll pb-2">
          <div className="w-full" style={{ minWidth: timelineMinWidth }}>
            <div
              className="grid bg-slate-50/80 text-[10px] font-medium uppercase tracking-[0.16em] text-slate-400"
              style={{ gridTemplateColumns: timelineGridColumns }}
            >
              <div className="px-3 py-1.5">Signal</div>
              <div className="border-l border-slate-100 px-2 py-1.5 text-center">Value</div>
              {timeline.epochs.map((epoch) => (
                <div key={epoch.id} className="border-l border-slate-100 px-2 py-1.5 text-center">
                  {epoch.index}
                </div>
              ))}
            </div>
            <div
              className="grid items-center border-t border-slate-100 text-xs"
              style={{ gridTemplateColumns: timelineGridColumns }}
            >
              <div className="px-3 py-1.5 text-[10px] font-medium uppercase tracking-[0.14em] text-slate-400">
                Length
              </div>
              <div className="border-l border-slate-100 px-2 py-1.5 text-center text-[10px] leading-3 text-slate-400">
                min-max
                <br />
                steps/trial
              </div>
              {timeline.epochs.map((epoch) => {
                const value = epoch.length.value as { min?: unknown; max?: unknown } | null;
                const inferred = Boolean(epoch.length.metadata.inferred_from_remaining_steps);
                const storedMin = Number(value?.min ?? 0);
                const storedMaxExclusive = Number(value?.max ?? 0);
                const visibleMin = Number.isFinite(storedMin) ? storedMin : 0;
                const visibleMax = Math.max(
                  visibleMin,
                  Number.isFinite(storedMaxExclusive) ? storedMaxExclusive - 1 : visibleMin
                );
                return (
                  <div key={epoch.id} className="border-l border-slate-100 px-1.5 py-1.5">
                    {inferred ? (
                      <div className="whitespace-nowrap text-center text-[10px] text-slate-400">
                        remaining
                      </div>
                    ) : (
                      <div className="grid gap-1">
                        <div className="grid grid-cols-[1.25rem_2.45rem] justify-center gap-1">
                          <span className="self-center text-right text-[9px] font-medium uppercase tracking-[0.08em] text-slate-300">
                            min
                          </span>
                          <input
                            type="number"
                            min={0}
                            value={visibleMin}
                            onChange={(event) =>
                              onChange(updateVisibleRange(epoch, 'min', Number(event.target.value)))
                            }
                            className="h-6 rounded border border-slate-200 px-1 text-center text-[11px] text-slate-700"
                            aria-label={`${epoch.label} min length`}
                          />
                        </div>
                        <div className="grid grid-cols-[1.25rem_2.45rem] justify-center gap-1">
                          <span className="self-center text-right text-[9px] font-medium uppercase tracking-[0.08em] text-slate-300">
                            max
                          </span>
                          <input
                            type="number"
                            min={0}
                            value={visibleMax}
                            onChange={(event) =>
                              onChange(updateVisibleRange(epoch, 'max', Number(event.target.value)))
                            }
                            className="h-6 rounded border border-slate-200 px-1 text-center text-[11px] text-slate-700"
                            aria-label={`${epoch.label} max length`}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            {signalRows.map((signal) => (
              <div
                key={signal.id}
                className="grid items-center border-t border-slate-100 text-xs"
                style={{ gridTemplateColumns: timelineGridColumns }}
              >
                <div className="truncate px-2.5 py-1.5 font-medium text-slate-600">
                  {signal.label}
                </div>
                <div className="border-l border-slate-100 px-1.5 py-1.5">
                  <button
                    type="button"
                    onClick={() =>
                      setEditingSignalId((current) => (current === signal.id ? null : signal.id))
                    }
                    className="flex h-6 w-full min-w-0 items-center justify-center gap-1 rounded border border-slate-200 bg-white px-1.5 text-[10px] font-medium text-slate-600 hover:border-emerald-200 hover:text-slate-800"
                    title={`Edit ${signal.label} value spec`}
                  >
                    <SlidersHorizontal className="h-3 w-3 shrink-0" />
                    <span className="min-w-0 truncate">{formatValueSpec(signal.value_spec)}</span>
                  </button>
                </div>
                {timeline.epochs.map((epoch) => (
                  <label
                    key={epoch.id}
                    className="flex h-8 items-center justify-center border-l border-slate-100"
                    title={`${signal.label} during ${epoch.label}`}
                  >
                    <input
                      type="checkbox"
                      checked={signal.epoch_ids.includes(epoch.id)}
                      onChange={(event) =>
                        onChange(
                          toggleDelayedReachSignalEpoch(
                            timeline,
                            signal.id,
                            epoch.id,
                            event.target.checked
                          )
                        )
                      }
                      className="h-3.5 w-3.5 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                    />
                  </label>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>
      {editableEpochs.length > 0 && (
        <div className="text-[10px] text-slate-400">
          Length ranges are sampled per trial; final epoch uses remaining steps.
        </div>
      )}
    </section>
  );
}

export function TaskScenarioPanel() {
  const { taskSidebarWidth, setTaskSidebarWidth } = useLayoutStore();
  const asideRef = useRef<HTMLElement | null>(null);
  const taskDataSectionRef = useRef<HTMLElement | null>(null);
  const [taskDataResizeGap, setTaskDataResizeGap] = useState({ top: 0, bottom: 0 });
  const graph = useGraphStore((state) => state.graph);
  const graphStack = useGraphStore((state) => state.graphStack);
  const markDirty = useGraphStore((state) => state.markDirty);
  const workspace = useWorkspaceStore((state) => state.workspace);
  const updateTaskSpec = useWorkspaceStore((state) => state.updateActiveScenarioTaskSpec);
  const updateTaskBindingSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioTaskBindingSpec
  );
  const updateScenarioDraft = useWorkspaceStore((state) => state.updateScenarioDraft);
  const topPane = getTopPaneState(workspace);
  const scenario = getTrainingScenario(workspace);
  const task = scenario?.task_spec ?? TASK_CATALOG[0];
  const currentGraphPath = useMemo(
    () => graphStack.map((layer) => layer.childNodeId).filter((item): item is string => Boolean(item)),
    [graphStack]
  );
  const rootGraph = graphStack.length > 0 ? graphStack[0].graph : graph;
  const allTaskBindingSpec = useMemo(
    () =>
      ensureTaskBindingSpec(
        scenario?.task_binding_spec ?? createDefaultTaskBindingSpec(rootGraph, task),
        rootGraph,
        task
      ),
    [rootGraph, scenario?.task_binding_spec, task]
  );
  const taskBindingSpec = useMemo(
    () => scopedTaskBindingSpec(allTaskBindingSpec, currentGraphPath),
    [allTaskBindingSpec, currentGraphPath]
  );
  const timeline = useMemo(() => delayedReachTimelineFromTask(task), [task]);
  const params = Object.entries(task.params ?? {}).filter(
    ([key]) => !(timeline && isDelayedReachTimelineParam(key))
  );
  const bindableData = taskBindingSpec.exposed_data.filter((data) => data.bindable);
  const protocolData = taskBindingSpec.exposed_data.filter((data) => !data.bindable);
  const boundTarget = (nodeId: string, port: string) => `${nodeId}.${port}`;
  useLayoutEffect(() => {
    if (topPane.active_projection !== 'task') return;
    const aside = asideRef.current;
    const taskDataSection = taskDataSectionRef.current;
    if (!aside || !taskDataSection) return;

    const updateGap = () => {
      const asideRect = aside.getBoundingClientRect();
      const sectionRect = taskDataSection.getBoundingClientRect();
      setTaskDataResizeGap({
        top: Math.max(0, sectionRect.top - asideRect.top),
        bottom: Math.max(0, sectionRect.bottom - asideRect.top),
      });
    };

    updateGap();
    const observer = new ResizeObserver(updateGap);
    observer.observe(aside);
    observer.observe(taskDataSection);
    window.addEventListener('resize', updateGap);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updateGap);
    };
  }, [bindableData.length, taskSidebarWidth, topPane.active_projection]);

  if (topPane.active_projection !== 'task') return null;

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
    updateTaskBindingSpec(createDefaultTaskBindingSpec(rootGraph, next));
    markDirty();
  };
  const updateTimeline = (nextTimeline: StudioTaskTimelineSpec) => {
    const edited = applyDelayedReachTimelineEdit(task, allTaskBindingSpec, nextTimeline);
    if (scenario?.id) {
      updateScenarioDraft(
        scenario.id,
        {
          task_spec: edited.task_spec,
          task_binding_spec: ensureTaskBindingSpec(
            edited.task_binding_spec,
            rootGraph,
            edited.task_spec
          ),
        },
        'task_timeline_updated'
      );
    } else {
      updateTaskSpec(delayedReachTaskWithTimeline(task, nextTimeline));
    }
    markDirty();
  };
  const startResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const startX = event.clientX;
    const startWidth = taskSidebarWidth;
    const onMove = (moveEvent: PointerEvent) => {
      setTaskSidebarWidth(startWidth + (moveEvent.clientX - startX));
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };
  return (
    <aside
      ref={asideRef}
      style={{ width: taskSidebarWidth }}
      className="relative z-20 flex shrink-0 flex-col overflow-visible border-r border-slate-100 bg-white/95"
    >
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
      <section
        ref={taskDataSectionRef}
        className="shrink-0 border-b border-slate-100 bg-white py-3 pl-4 pr-0"
      >
        <div className="ml-auto w-fit min-w-[15rem] max-w-[calc(100%-1.25rem)] overflow-visible rounded-xl border-2 border-slate-200 bg-white/90 shadow-soft backdrop-blur">
          <div className="flex items-center justify-between gap-3 rounded-t-xl border-b border-slate-100 bg-slate-50/70 px-3 py-2">
            <div className="min-w-0 truncate text-sm font-medium text-slate-800">Task Data</div>
            <div className="shrink-0 text-[11px] text-slate-500">Task</div>
          </div>
          <div className="space-y-1 px-3 py-2 text-xs text-slate-600">
          {bindableData.map((data) => (
            <div
              key={data.id}
              className="relative flex h-7 items-center justify-end overflow-visible pr-3"
            >
              <span className="min-w-0 truncate text-right text-slate-600">
                {data.label}
              </span>
              <span
                data-task-data-port-id={data.id}
                className="pointer-events-none absolute right-[-20px] top-1/2 z-30 h-2.5 w-2.5 -translate-y-1/2 rounded-full border border-white bg-emerald-500 shadow-soft"
                title={`${data.label} Task Data`}
              />
            </div>
          ))}
          </div>
        </div>
      </section>
      <div className="min-h-0 flex-1 space-y-4 overflow-x-hidden overflow-y-auto bg-white px-4 py-4">
        {timeline && <DelayedReachTimelineEditor timeline={timeline} onChange={updateTimeline} />}
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
        {protocolData.length > 0 && (
          <section className="space-y-2">
            <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
              Protocol
            </div>
            <div className="space-y-1">
              {protocolData.map((data) => (
                <div
                  key={data.id}
                  className="flex h-7 items-center justify-between gap-3 rounded border border-slate-100 px-2 text-xs"
                >
                  <span className="truncate font-medium text-slate-600">{data.label}</span>
                  <span className="shrink-0 text-[10px] text-slate-400">{data.kind}</span>
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
                <div className="font-medium text-slate-700">{binding.source_data_id}</div>
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
      <div className="pointer-events-none absolute right-0 top-0 bottom-0 z-10 w-1">
        <div
          className="pointer-events-auto absolute right-0 top-0 w-1 cursor-col-resize touch-none hover:bg-brand-300/50 active:bg-brand-400/50"
          style={{ height: taskDataResizeGap.top }}
          aria-label="Resize task sidebar"
          role="separator"
          onPointerDown={startResize}
        />
        <div
          className="pointer-events-auto absolute right-0 bottom-0 w-1 cursor-col-resize touch-none hover:bg-brand-300/50 active:bg-brand-400/50"
          style={{ top: taskDataResizeGap.bottom }}
          aria-label="Resize task sidebar"
          role="separator"
          onPointerDown={startResize}
        />
      </div>
    </aside>
  );
}
