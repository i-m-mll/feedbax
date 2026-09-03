import {
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { Settings2 } from 'lucide-react';
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
  signalEpochValueSpec,
  updateTaskTimelineSignalEpochValueSpec,
} from '@/features/scenario/taskTimeline';
import {
  ValueSpecField,
  descriptorForTaskSignal,
  humanizeLabel,
  isStudioValueSpec,
  type ValueSpecFieldDescriptor,
} from '@/components/values/ValueSpecField';
import { useGraphStore } from '@/stores/graphStore';
import {
  getTopPaneState,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { useLayoutStore } from '@/stores/layoutStore';
import type { ParamValue } from '@/types/graph';
import type { TaskSpec } from '@/types/training';
import type { StudioTaskTimelineSpec } from '@/types/workspace';

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

function ParamEditor({
  name,
  value,
  onChange,
}: {
  name: string;
  value: unknown;
  onChange: (value: ParamValue) => void;
}) {
  const descriptor = descriptorForTaskParam(name);
  return (
    <div className="grid gap-1 text-xs">
      <span className="truncate font-medium text-slate-500">{humanizeLabel(name)}</span>
      <ValueSpecField
        descriptor={descriptor}
        value={value}
        onChange={(nextValue) => onChange(nextValue as ParamValue)}
      />
    </div>
  );
}

function isStaticShapeParam(name: string) {
  return /(^|_)(size|sizes|shape|count|dim|dims|n|num)(_|\b)/i.test(name);
}

export function descriptorForTaskParam(name: string): ValueSpecFieldDescriptor {
  const staticShape = isStaticShapeParam(name);
  return {
    id: `task_param:${name}`,
    label: humanizeLabel(name),
    ownerKind: 'task_param',
    semanticKind: staticShape ? 'static_shape' : 'static_leaf',
    allowedModes: ['constant', 'expression', 'distribution'],
    allowedScopes: staticShape ? ['fixed', 'run'] : ['fixed', 'run', 'sweep'],
    defaultScope: 'run',
    loweringTarget: staticShape ? 'run_manifest' : 'sweep_axis',
  };
}

function epochLengthDescriptor(
  epoch: StudioTaskTimelineSpec['epochs'][number]
): ValueSpecFieldDescriptor {
  return {
    id: `epoch_length:${epoch.id}`,
    label: `${epoch.label} length`,
    ownerKind: 'epoch_length',
    semanticKind: 'epoch_length',
    valueSchema:
      (epoch.metadata.value_schema as Record<string, unknown> | undefined) ??
      (epoch.length.metadata.value_schema as Record<string, unknown> | undefined) ??
      null,
    allowedModes: ['constant', 'distribution', 'expression'],
    allowedScopes: ['run', 'sweep', 'trial', 'epoch'],
    defaultScope: 'trial',
    loweringTarget: 'timeline_mask',
  };
}

function signalEpochDescriptor(
  signal: StudioTaskTimelineSpec['signals'][number],
  epoch: StudioTaskTimelineSpec['epochs'][number]
): ValueSpecFieldDescriptor {
  const base = descriptorForTaskSignal(signal);
  return {
    ...base,
    id: `${base.id}:epoch:${epoch.id}`,
    label: humanizeLabel(`${signal.label} in ${epoch.label}`),
    defaultScope: 'epoch',
    loweringTarget: 'timeline_mask',
  };
}

function updateEpochLengthValueSpec(
  timeline: StudioTaskTimelineSpec,
  epochId: string,
  nextValue: unknown
): StudioTaskTimelineSpec {
  if (!isStudioValueSpec(nextValue)) return timeline;
  return {
    ...timeline,
    epochs: timeline.epochs.map((epoch) =>
      epoch.id === epochId ? { ...epoch, length: nextValue } : epoch
    ),
  };
}

function DelayedReachTimelineEditor({
  timeline,
  onChange,
}: {
  timeline: StudioTaskTimelineSpec;
  onChange: (timeline: StudioTaskTimelineSpec) => void;
}) {
  const editableEpochs = timeline.epochs.slice(0, -1);
  const signalRows = timeline.signals.filter((signal) => signal.kind === 'signal');
  const timelineGridColumns = `7rem repeat(${timeline.epochs.length}, minmax(4.875rem, 1fr))`;
  const timelineMinWidth = `${7 + timeline.epochs.length * 4.875}rem`;
  return (
    <section className="space-y-2">
      <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
        Timeline
      </div>
      <div className="overflow-hidden rounded border border-slate-100">
        <div className="local-x-scrollbar overflow-x-scroll pb-2">
          <div className="w-full" style={{ minWidth: timelineMinWidth }}>
            <div
              className="grid bg-slate-50/80 text-[10px] font-medium uppercase tracking-[0.16em] text-slate-400"
              style={{ gridTemplateColumns: timelineGridColumns }}
            >
              <div className="px-3 py-1.5">Signal</div>
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
              {timeline.epochs.map((epoch) => {
                const inferred = Boolean(epoch.length.metadata.inferred_from_remaining_steps);
                return (
                  <div key={epoch.id} className="border-l border-slate-100 px-1.5 py-1.5">
                    {inferred ? (
                      <div className="whitespace-nowrap text-center text-[10px] text-slate-400">
                        remaining
                      </div>
                    ) : (
                      <ValueSpecField
                        descriptor={epochLengthDescriptor(epoch)}
                        value={epoch.length}
                        forceValueSpec
                        compact
                        onChange={(nextValue) =>
                          onChange(updateEpochLengthValueSpec(timeline, epoch.id, nextValue))
                        }
                      />
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
                {timeline.epochs.map((epoch) => (
                  <div
                    key={epoch.id}
                    className="flex h-8 items-center justify-center border-l border-slate-100 px-1.5"
                    title={`${signal.label} during ${epoch.label}`}
                  >
                    <ValueSpecField
                      descriptor={signalEpochDescriptor(signal, epoch)}
                      value={signalEpochValueSpec(timeline, signal.id, epoch.id)}
                      forceValueSpec
                      compact
                      onChange={(nextValue) => {
                        if (!isStudioValueSpec(nextValue)) return;
                        onChange(
                          updateTaskTimelineSignalEpochValueSpec(
                            timeline,
                            signal.id,
                            epoch.id,
                            nextValue
                          )
                        );
                      }}
                    />
                  </div>
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
