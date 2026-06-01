import type { ParamValue } from '@/types/graph';
import type { TaskSpec } from '@/types/training';
import type {
  StudioTaskTimelineSignalSpec,
  StudioTaskTimelineSpec,
  StudioValueSpec,
  StudioTaskBindingSpec,
  StudioTaskTimelineSegmentSpec,
  TaskDataSchema,
  ValueSchema,
} from '@/types/workspace';

export const TASK_TIMELINE_SCHEMA_VERSION = 'feedbax.studio.task_timeline.v1';
export const VALUE_SCHEMA_VERSION = 'feedbax.studio.value.v1';

const DELAYED_REACH_EPOCH_LABELS = ['hold', 'target_on', 'movement'];
const TIMELINE_PARAM_KEYS = new Set([
  'epoch_len_ranges',
  'target_on_epochs',
  'hold_epochs',
  'move_epochs',
]);

export function isDelayedReachTimelineParam(key: string): boolean {
  return TIMELINE_PARAM_KEYS.has(key);
}

const EPOCH_LENGTH_VALUE_SCHEMA: ValueSchema = {
  id: 'value:task_timeline:epoch_length',
  label: 'Epoch length',
  kind: 'epoch_length',
  dtype: 'int32',
  shape: ['range', 2],
  rank: 2,
  units: 'steps',
  frame: 'task_time',
  origin: 'inferred_static',
  metadata: { temporal_support: 'epoch_window' },
};

export interface DelayedReachTimelinePreviewEpoch {
  id: string;
  label: string;
  start_min: number;
  start_max: number;
  end_min: number;
  end_max: number;
  inferred: boolean;
}

export interface DelayedReachTimelinePreviewSignal {
  id: string;
  label: string;
  kind: string;
  active_epoch_ids: string[];
  active_ranges: Array<Pick<DelayedReachTimelinePreviewEpoch, 'start_min' | 'end_max'>>;
}

export interface DelayedReachTimelinePreview {
  n_steps: number | null;
  epochs: DelayedReachTimelinePreviewEpoch[];
  signals: DelayedReachTimelinePreviewSignal[];
}

function timelineValueSchema(
  id: string,
  label: string,
  kind: string,
  path: string,
  value: Pick<ValueSchema, 'dtype' | 'shape' | 'units' | 'frame'>,
  metadata: Record<string, unknown> = {}
): ValueSchema {
  return {
    id: `value:task_timeline:${id}`,
    label,
    kind,
    dtype: value.dtype ?? null,
    shape: value.shape ?? null,
    units: value.units ?? null,
    frame: value.frame ?? null,
    origin: 'inferred_static',
    metadata: { ...metadata, task_data_path: path },
  };
}

function timelineTaskDataSchema(
  id: string,
  label: string,
  kind: StudioTaskTimelineSignalSpec['kind'],
  path: string,
  valueSchema: ValueSchema
): TaskDataSchema {
  const role = kind === 'signal' ? 'model_input' : kind;
  return {
    id: `task_data:${id}`,
    label,
    kind,
    role,
    path,
    bindable: kind === 'signal',
    value_schema: valueSchema,
    origin: 'inferred_static',
    metadata: {
      source: 'task_timeline',
      value_schema_id: valueSchema.id,
      task_data_role: role,
      task_data_surface: kind === 'signal' ? 'graph_input' : 'protocol',
    },
  };
}

function signalValueSchema(
  id: string,
  label: string,
  kind: StudioTaskTimelineSignalSpec['kind'],
  path: string
): ValueSchema {
  if (id === 'target_position') {
    return timelineValueSchema(
      id,
      label,
      'task_signal',
      path,
      { dtype: 'float32', shape: ['time', 4], units: null, frame: 'cartesian_effector' },
      {
        temporal_support: 'trajectory',
        component_fields: ['pos', 'vel'],
        component_shapes: { pos: [2], vel: [2] },
      }
    );
  }
  if (id === 'hold' || id === 'target_on') {
    return timelineValueSchema(
      id,
      label,
      'task_signal',
      path,
      { dtype: 'float32', shape: ['time', 1], units: null, frame: 'task_time' },
      { temporal_support: 'epoch_masked_signal' }
    );
  }
  if (kind === 'target') {
    return timelineValueSchema(
      id,
      label,
      'task_target',
      path,
      { dtype: 'float32', shape: ['time', 2], units: null, frame: 'cartesian_effector' },
      {
        temporal_support: 'epoch_materialized_trajectory',
        storage: 'compact_task_params',
        compact_representation: 'delayed_reach_task_params_v1',
        materializes_to: { dtype: 'float32', shape: ['time', 2] },
      }
    );
  }
  return timelineValueSchema(
    id,
    label,
    'task_signal',
    path,
    { dtype: 'bool', shape: ['time'], units: null, frame: 'task_time' },
    { temporal_support: 'epoch_masked_signal' }
  );
}

function constantValue(
  value: unknown,
  metadata: Record<string, unknown> = {},
  valueSchema: ValueSchema | null = null
): StudioValueSpec {
  return {
    schema_version: VALUE_SCHEMA_VERSION,
    mode: 'constant',
    value,
    dtype: valueSchema?.dtype ?? null,
    shape: valueSchema?.shape ?? null,
    units: valueSchema?.units ?? null,
    frame: valueSchema?.frame ?? null,
    metadata: {
      ...metadata,
      value_schema: valueSchema,
      value_schema_id: valueSchema?.id ?? null,
    },
  };
}

function distributionValue(
  family: string,
  parameters: Record<string, unknown>,
  samplingScope: string,
  metadata: Record<string, unknown> = {},
  valueSchema: ValueSchema | null = null
): StudioValueSpec {
  return {
    schema_version: VALUE_SCHEMA_VERSION,
    mode: 'distribution',
    distribution: { family, parameters },
    sampling_scope: samplingScope,
    dtype: valueSchema?.dtype ?? null,
    shape: valueSchema?.shape ?? null,
    units: valueSchema?.units ?? null,
    frame: valueSchema?.frame ?? null,
    metadata: {
      ...metadata,
      value_schema: valueSchema,
      value_schema_id: valueSchema?.id ?? null,
    },
  };
}

function functionValue(
  functionId: string,
  parameters: Record<string, unknown>,
  metadata: Record<string, unknown> = {},
  valueSchema: ValueSchema | null = null
): StudioValueSpec {
  return {
    schema_version: VALUE_SCHEMA_VERSION,
    mode: 'function',
    function_id: functionId,
    parameters,
    dtype: valueSchema?.dtype ?? null,
    shape: valueSchema?.shape ?? null,
    units: valueSchema?.units ?? null,
    frame: valueSchema?.frame ?? null,
    metadata: {
      ...metadata,
      value_schema: valueSchema,
      value_schema_id: valueSchema?.id ?? null,
    },
  };
}

function cloneValueSpecWithValue(valueSpec: StudioValueSpec | null | undefined, value: unknown) {
  const metadata = valueSpec?.metadata ?? {};
  return constantValue(value, metadata, (metadata.value_schema as ValueSchema | null) ?? null);
}

function inactiveValueForSignal(valueSpec: StudioValueSpec | null | undefined): unknown {
  const value = valueSpec?.value;
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    if ('inactive' in record) return record.inactive;
  }
  if (Array.isArray(value)) return value.map(() => 0);
  if (typeof value === 'boolean') return false;
  return 0;
}

function activeValueForSignal(valueSpec: StudioValueSpec | null | undefined): unknown {
  const value = valueSpec?.value;
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    if ('active' in record) return record.active;
  }
  if (value !== undefined && value !== null) return value;
  return 1;
}

function activeEpochValueSpec(valueSpec: StudioValueSpec | null | undefined): StudioValueSpec | null {
  if (!valueSpec) return null;
  if (valueSpec.mode !== 'constant') return valueSpec;
  return cloneValueSpecWithValue(valueSpec, activeValueForSignal(valueSpec));
}

function inactiveEpochValueSpec(valueSpec: StudioValueSpec | null | undefined): StudioValueSpec | null {
  if (!valueSpec) return null;
  return cloneValueSpecWithValue(valueSpec, inactiveValueForSignal(valueSpec));
}

function isActiveEpochValueSpec(valueSpec: StudioValueSpec | null | undefined): boolean {
  if (!valueSpec) return false;
  if (valueSpec.mode !== 'constant') return true;
  const value = valueSpec.value;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  if (Array.isArray(value)) return value.some((item) => Number(item) !== 0);
  if (value && typeof value === 'object') return true;
  return Boolean(value);
}

function asIndexSet(value: unknown): Set<number> {
  if (!Array.isArray(value)) return new Set();
  return new Set(
    value
      .map((item) => Number(item))
      .filter((item) => Number.isInteger(item) && item >= 0)
  );
}

function asRanges(value: unknown): Array<[number, number]> {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!Array.isArray(item) || item.length < 2) return [];
    const min = Number(item[0]);
    const max = Number(item[1]);
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
    return [[Math.max(0, Math.round(min)), Math.max(0, Math.round(max))] as [number, number]];
  });
}

function signal(
  id: string,
  label: string,
  kind: StudioTaskTimelineSignalSpec['kind'],
  path: string,
  epochSet: Set<number>,
  task: TaskSpec
): StudioTaskTimelineSignalSpec {
  const valueSchema = signalValueSchema(id, label, kind, path);
  const valueSpec = delayedReachTaskDataValueSpec(id, task, valueSchema);
  const epochIds = [...epochSet].sort((a, b) => a - b).map((index) => `epoch:${index}`);
  return {
    id,
    label,
    kind,
    task_data_id: id,
    path,
    epoch_ids: epochIds,
    value_spec: valueSpec,
    epoch_value_specs: Object.fromEntries(
      Array.from({ length: DELAYED_REACH_EPOCH_LABELS.length }, (_, index) => {
        const epochId = `epoch:${index}`;
        return [
          epochId,
          epochIds.includes(epochId)
            ? activeEpochValueSpec(valueSpec)
            : inactiveEpochValueSpec(valueSpec),
        ];
      })
    ),
    value_schema: valueSchema,
    task_data_schema: timelineTaskDataSchema(id, label, kind, path, valueSchema),
    metadata: {
      value_schema: valueSchema,
      value_schema_id: valueSchema.id,
      task_data_schema_id: `task_data:${id}`,
      temporal_support: valueSchema.metadata.temporal_support,
      value_spec_modes:
        id === 'target_position' || id === 'movement_target'
          ? ['constant', 'function', 'distribution', 'schedule', 'expression']
          : ['constant', 'function', 'distribution', 'schedule', 'expression'],
      value_spec_scopes:
        id === 'target_position' || id === 'movement_target'
          ? ['run', 'sweep', 'trial', 'epoch', 'timestep']
          : ['run', 'sweep', 'trial', 'epoch', 'timestep'],
    },
  };
}

function existingTimelineFromTask(task: TaskSpec): StudioTaskTimelineSpec | null {
  const timeline = task.timeline;
  if (!timeline || typeof timeline !== 'object' || Array.isArray(timeline)) return null;
  const record = timeline as unknown as Partial<StudioTaskTimelineSpec>;
  if (!Array.isArray(record.epochs) || !Array.isArray(record.signals)) return null;
  return {
    schema_version: record.schema_version ?? TASK_TIMELINE_SCHEMA_VERSION,
    epochs: record.epochs,
    signals: record.signals.map((signalSpec) => {
      if (signalSpec.epoch_value_specs) return signalSpec;
      return {
        ...signalSpec,
        epoch_value_specs: Object.fromEntries(
          record.epochs.map((epoch) => {
            return [
              epoch.id,
              signalSpec.epoch_ids.includes(epoch.id)
                ? activeEpochValueSpec(signalSpec.value_spec)
                : inactiveEpochValueSpec(signalSpec.value_spec),
            ];
          })
        ),
      };
    }),
    segments: record.segments,
    metadata: {
      ...(record.metadata ?? {}),
      n_steps: task.params?.n_steps ?? record.metadata?.n_steps ?? null,
    },
  };
}

function delayedReachSegments(epochCount: number): StudioTaskTimelineSegmentSpec[] {
  const base = DELAYED_REACH_EPOCH_LABELS.slice(0, epochCount).map((label, index) => ({
    id: label,
    label,
    epoch_ids: [`epoch:${index}`],
    metadata: { source: 'delayed_reach_epoch' },
  }));
  if (epochCount >= 2) {
    base.push({
      id: 'cue_window',
      label: 'cue window',
      epoch_ids: Array.from({ length: Math.min(epochCount, 2) }, (_, index) => `epoch:${index}`),
      metadata: { source: 'delayed_reach_group' },
    });
  }
  return base;
}

export function delayedReachTaskDataValueSpec(
  taskDataId: string,
  task?: TaskSpec | null,
  valueSchemaOverride: ValueSchema | null = null
): StudioValueSpec | null {
  const params = task?.params ?? {};
  const endpointParameters = {
    endpoint_mode: params.train_endpoint_mode ?? 'workspace',
    workspace: params.workspace ?? null,
    eval_reach_length: params.eval_reach_length ?? null,
    p_catch_trial: params.p_catch_trial ?? null,
  };
  if (taskDataId === 'target_position') {
    const schema =
      valueSchemaOverride ??
      signalValueSchema('target_position', 'Target position', 'signal', 'inputs.effector_target');
    return functionValue(
      'delayed_reach_target_position',
      endpointParameters,
      {
        compact_representation: 'delayed_reach_task_params_v1',
        task_data_id: taskDataId,
      },
      schema
    );
  }
  if (taskDataId === 'movement_target') {
    const schema =
      valueSchemaOverride ??
      signalValueSchema('movement_target', 'Movement target', 'target', 'targets.effector');
    return functionValue(
      'delayed_reach_movement_target',
      endpointParameters,
      {
        compact_representation: 'delayed_reach_task_params_v1',
        task_data_id: taskDataId,
      },
      schema
    );
  }
  if (taskDataId === 'hold') {
    const schema = valueSchemaOverride ?? signalValueSchema('hold', 'Hold/go cue', 'signal', 'inputs.hold');
    return constantValue(
      { active: 1, inactive: 0 },
      { task_data_id: taskDataId, cue_polarity: 'hold_is_1_go_is_0' },
      schema
    );
  }
  if (taskDataId === 'target_on') {
    const schema =
      valueSchemaOverride ??
      signalValueSchema('target_on', 'Target shown', 'signal', 'inputs.target_on');
    return constantValue(
      { active: 1, inactive: 0 },
      { task_data_id: taskDataId },
      schema
    );
  }
  return null;
}

export function delayedReachTimelineFromTask(task: TaskSpec): StudioTaskTimelineSpec | null {
  if (task.type !== 'DelayedReaches') return null;
  const existingTimeline = existingTimelineFromTask(task);
  if (existingTimeline) {
    return {
      ...existingTimeline,
      segments: existingTimeline.segments ?? delayedReachSegments(existingTimeline.epochs.length),
    };
  }
  const params = task.params ?? {};
  const ranges = asRanges(params.epoch_len_ranges);
  const epochCount = Math.max(DELAYED_REACH_EPOCH_LABELS.length, ranges.length + 1);
  const epochs = Array.from({ length: epochCount }, (_, index) => {
    const range = ranges[index];
    const isInferred = range === undefined;
    return {
      id: `epoch:${index}`,
      label: DELAYED_REACH_EPOCH_LABELS[index] ?? `epoch ${index + 1}`,
      index,
      length: isInferred ? constantValue(
        null,
        {
          scope: 'trial',
          inferred_from_remaining_steps: isInferred,
          temporal_window: { mode: 'epoch', epoch_id: `epoch:${index}` },
        },
        EPOCH_LENGTH_VALUE_SCHEMA
      ) : distributionValue(
        'uniform',
        { min: range[0], max: range[1] },
        'trial',
        {
          scope: 'trial',
          inferred_from_remaining_steps: isInferred,
          range_upper_bound: 'exclusive',
          temporal_window: { mode: 'epoch', epoch_id: `epoch:${index}` },
        },
        EPOCH_LENGTH_VALUE_SCHEMA
      ),
      metadata: {
        value_schema: EPOCH_LENGTH_VALUE_SCHEMA,
        temporal_window: { mode: 'epoch', epoch_id: `epoch:${index}` },
      },
    };
  });
  return {
    schema_version: TASK_TIMELINE_SCHEMA_VERSION,
    epochs,
    signals: [
      signal(
        'target_position',
        'Target position',
        'signal',
        'inputs.effector_target',
        asIndexSet(params.target_on_epochs),
        task
      ),
      signal(
        'hold',
        'Hold/go cue',
        'signal',
        'inputs.hold',
        asIndexSet(params.hold_epochs),
        task
      ),
      signal(
        'target_on',
        'Target shown',
        'signal',
        'inputs.target_on',
        asIndexSet(params.target_on_epochs),
        task
      ),
      signal(
        'movement_target',
        'Movement target',
        'target',
        'targets.effector',
        asIndexSet(params.move_epochs),
        task
      ),
    ],
    segments: delayedReachSegments(epochCount),
    metadata: {
      task_type: task.type,
      n_steps: params.n_steps ?? null,
      representation: 'delayed_reach_task_params_v1',
      storage: 'compact_task_params',
      materializes_targets: true,
    },
  };
}

function signalEpochIndexes(
  timeline: StudioTaskTimelineSpec,
  signalId: string,
  fallbackIds: string[] = []
): number[] {
  const signalSpec = timeline.signals.find((item) => item.id === signalId)
    ?? timeline.signals.find((item) => fallbackIds.includes(item.id));
  if (!signalSpec) return [];
  return signalSpec.epoch_ids
    .map((id) => Number(id.replace(/^epoch:/, '')))
    .filter((index) => Number.isInteger(index) && index >= 0)
    .sort((a, b) => a - b);
}

function rangeFromValue(value: StudioValueSpec): [number, number] | null {
  if (value.mode === 'distribution') {
    const distribution = value.distribution;
    if (distribution && typeof distribution === 'object' && !Array.isArray(distribution)) {
      const record = distribution as Record<string, unknown>;
      const parameters = record.parameters as Record<string, unknown> | undefined;
      if (record.family === 'uniform' && parameters) {
        const min = Number(parameters.min);
        const max = Number(parameters.max);
        if (Number.isFinite(min) && Number.isFinite(max)) {
          return [Math.max(0, Math.round(min)), Math.max(0, Math.round(max))];
        }
      }
    }
  }
  const raw = value.value;
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    const record = raw as Record<string, unknown>;
    const min = Number(record.min);
    const max = Number(record.max);
    if (Number.isFinite(min) && Number.isFinite(max)) {
      return [Math.max(0, Math.round(min)), Math.max(0, Math.round(max))];
    }
  }
  return null;
}

export function delayedReachTaskWithTimeline(
  task: TaskSpec,
  timeline: StudioTaskTimelineSpec
): TaskSpec {
  const epochLenRanges = timeline.epochs
    .slice(0, -1)
    .map((epoch) => rangeFromValue(epoch.length) ?? [0, 0]);
  return {
    ...task,
    params: {
      ...task.params,
      epoch_len_ranges: epochLenRanges as ParamValue,
      target_on_epochs: signalEpochIndexes(
        timeline,
        'target_on',
        ['target_position']
      ) as ParamValue,
      hold_epochs: signalEpochIndexes(timeline, 'hold') as ParamValue,
      move_epochs: signalEpochIndexes(timeline, 'movement_target', ['move']) as ParamValue,
    },
    timeline: timeline as unknown as Record<string, ParamValue>,
  };
}

export function updateDelayedReachEpochRange(
  timeline: StudioTaskTimelineSpec,
  epochId: string,
  key: 'min' | 'max',
  value: number
): StudioTaskTimelineSpec {
  return {
    ...timeline,
    epochs: timeline.epochs.map((epoch) => {
      if (epoch.id !== epochId) return epoch;
      const current = rangeFromValue(epoch.length) ?? [0, 0];
      const next = key === 'min' ? [value, current[1]] : [current[0], value];
      const min = Math.max(0, Math.round(Math.min(next[0], next[1])));
      const max = Math.max(min, Math.round(Math.max(next[0], next[1])));
      return {
        ...epoch,
        length: distributionValue(
          'uniform',
          { min, max },
          'trial',
          { ...epoch.length.metadata, range_upper_bound: 'exclusive' },
          EPOCH_LENGTH_VALUE_SCHEMA
        ),
      };
    }),
  };
}

export function toggleDelayedReachSignalEpoch(
  timeline: StudioTaskTimelineSpec,
  signalId: string,
  epochId: string,
  enabled: boolean
): StudioTaskTimelineSpec {
  const linkedSignalIds =
    signalId === 'target_on' || signalId === 'target_position'
      ? new Set(['target_on', 'target_position'])
      : new Set([signalId]);
  return {
    ...timeline,
    signals: timeline.signals.map((item) => {
      if (!linkedSignalIds.has(item.id)) return item;
      const epochIds = new Set(item.epoch_ids);
      if (enabled) {
        epochIds.add(epochId);
      } else {
        epochIds.delete(epochId);
      }
      return {
        ...item,
        epoch_ids: [...epochIds].sort(),
      };
    }),
  };
}

export function signalEpochValueSpec(
  signalSpec: StudioTaskTimelineSignalSpec,
  epochId: string
): StudioValueSpec | null {
  if (signalSpec.epoch_value_specs?.[epochId]) return signalSpec.epoch_value_specs[epochId] ?? null;
  return signalSpec.epoch_ids.includes(epochId)
    ? activeEpochValueSpec(signalSpec.value_spec)
    : inactiveEpochValueSpec(signalSpec.value_spec);
}

export function updateTaskTimelineSignalEpochValueSpec(
  timeline: StudioTaskTimelineSpec,
  signalId: string,
  epochId: string,
  valueSpec: StudioValueSpec
): StudioTaskTimelineSpec {
  const linkedSignalIds =
    signalId === 'target_on' || signalId === 'target_position'
      ? new Set(['target_on', 'target_position'])
      : new Set([signalId]);
  return {
    ...timeline,
    signals: timeline.signals.map((item) => {
      if (!linkedSignalIds.has(item.id)) return item;
      const epochValueSpecs = {
        ...(item.epoch_value_specs ?? {}),
        [epochId]: valueSpec,
      };
      const epochIds = Object.entries(epochValueSpecs)
        .filter(([, candidate]) => isActiveEpochValueSpec(candidate))
        .map(([id]) => id)
        .sort();
      return {
        ...item,
        epoch_ids: epochIds,
        epoch_value_specs: epochValueSpecs,
        metadata: {
          ...item.metadata,
          epoch_value_specs_updated_from: 'task_timeline_epoch_value_editor',
        },
      };
    }),
  };
}

export function updateTaskTimelineSignalValueSpec(
  timeline: StudioTaskTimelineSpec,
  signalId: string,
  valueSpec: StudioValueSpec
): StudioTaskTimelineSpec {
  return {
    ...timeline,
    signals: timeline.signals.map((item) => {
      if (item.id !== signalId) return item;
      return {
        ...item,
        value_spec: valueSpec,
        metadata: {
          ...item.metadata,
          value_spec_updated_from: 'task_timeline_value_editor',
        },
      };
    }),
  };
}

export function delayedReachTimelinePreview(
  timeline: StudioTaskTimelineSpec
): DelayedReachTimelinePreview {
  const rawSteps = Number(timeline.metadata.n_steps);
  const nSteps = Number.isFinite(rawSteps) && rawSteps > 0 ? Math.round(rawSteps) : null;
  let cursorMin = 0;
  let cursorMax = 0;
  const epochs = timeline.epochs.map((epoch) => {
    const range = rangeFromValue(epoch.length);
    const inferred = range === null || Boolean(epoch.length.metadata.inferred_from_remaining_steps);
    const endMin = inferred && nSteps !== null ? nSteps : cursorMin + (range?.[0] ?? 0);
    const endMax = inferred && nSteps !== null ? nSteps : cursorMax + (range?.[1] ?? 0);
    const preview = {
      id: epoch.id,
      label: epoch.label,
      start_min: cursorMin,
      start_max: cursorMax,
      end_min: endMin,
      end_max: endMax,
      inferred,
    };
    cursorMin = preview.end_min;
    cursorMax = preview.end_max;
    return preview;
  });
  return {
    n_steps: nSteps,
    epochs,
    signals: timeline.signals.map((signalSpec) => {
      const activeEpochs = new Set(signalSpec.epoch_ids);
      return {
        id: signalSpec.id,
        label: signalSpec.label,
        kind: signalSpec.kind,
        active_epoch_ids: signalSpec.epoch_ids,
        active_ranges: epochs
          .filter((epoch) => activeEpochs.has(epoch.id))
          .map((epoch) => ({ start_min: epoch.start_min, end_max: epoch.end_max })),
      };
    }),
  };
}

export function applyDelayedReachTimelineEdit(
  task: TaskSpec,
  taskBindingSpec: StudioTaskBindingSpec | null | undefined,
  timeline: StudioTaskTimelineSpec
): {
  task_spec: TaskSpec;
  task_binding_spec: StudioTaskBindingSpec | null | undefined;
} {
  const taskSpec = delayedReachTaskWithTimeline(task, timeline);
  if (!taskBindingSpec) {
    return { task_spec: taskSpec, task_binding_spec: taskBindingSpec };
  }
  const valueSpecByDataId = new Map(
    timeline.signals.flatMap((signalSpec) => {
      const taskDataId = signalSpec.task_data_id ?? signalSpec.id;
      return signalSpec.value_spec ? [[taskDataId, signalSpec.value_spec] as const] : [];
    })
  );
  return {
    task_spec: taskSpec,
    task_binding_spec: {
      ...taskBindingSpec,
      exposed_data: taskBindingSpec.exposed_data.map((data) => {
        const valueSpec = valueSpecByDataId.get(data.id);
        if (!valueSpec) return data;
        return {
          ...data,
          value_spec: valueSpec,
          metadata: {
            ...data.metadata,
            task_timeline_signal_id: data.id,
          },
        };
      }),
      metadata: {
        ...taskBindingSpec.metadata,
        updated_from: 'task_timeline_editor',
      },
    },
  };
}
