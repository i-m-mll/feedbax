import type { ParamValue } from '@/types/graph';
import type { TaskSpec } from '@/types/training';
import type {
  StudioTaskTimelineSignalSpec,
  StudioTaskTimelineSpec,
  StudioValueSpec,
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
  epochSet: Set<number>
): StudioTaskTimelineSignalSpec {
  const valueSchema = signalValueSchema(id, label, kind, path);
  return {
    id,
    label,
    kind,
    path,
    epoch_ids: [...epochSet].sort((a, b) => a - b).map((index) => `epoch:${index}`),
    value_schema: valueSchema,
    task_data_schema: timelineTaskDataSchema(id, label, kind, path, valueSchema),
    metadata: {
      value_schema: valueSchema,
      task_data_schema_id: `task_data:${id}`,
      temporal_support: valueSchema.metadata.temporal_support,
    },
  };
}

export function delayedReachTimelineFromTask(task: TaskSpec): StudioTaskTimelineSpec | null {
  if (task.type !== 'DelayedReaches') return null;
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
      length: constantValue(
        isInferred ? null : { min: range[0], max: range[1] },
        {
          scope: 'trial',
          inferred_from_remaining_steps: isInferred,
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
      signal('target_on', 'Target shown', 'signal', 'inputs.target_on', asIndexSet(params.target_on_epochs)),
      signal('hold', 'Hold cue', 'signal', 'inputs.hold', asIndexSet(params.hold_epochs)),
      signal('move', 'Move target', 'target', 'targets.effector', asIndexSet(params.move_epochs)),
    ],
    metadata: {
      task_type: task.type,
      n_steps: params.n_steps ?? null,
      representation: 'delayed_reach_task_params_v1',
      storage: 'compact_task_params',
      materializes_targets: true,
    },
  };
}

function signalEpochIndexes(timeline: StudioTaskTimelineSpec, signalId: string): number[] {
  const signalSpec = timeline.signals.find((item) => item.id === signalId);
  if (!signalSpec) return [];
  return signalSpec.epoch_ids
    .map((id) => Number(id.replace(/^epoch:/, '')))
    .filter((index) => Number.isInteger(index) && index >= 0)
    .sort((a, b) => a - b);
}

function rangeFromValue(value: StudioValueSpec): [number, number] | null {
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
      target_on_epochs: signalEpochIndexes(timeline, 'target_on') as ParamValue,
      hold_epochs: signalEpochIndexes(timeline, 'hold') as ParamValue,
      move_epochs: signalEpochIndexes(timeline, 'move') as ParamValue,
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
        length: constantValue({ min, max }, epoch.length.metadata),
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
  return {
    ...timeline,
    signals: timeline.signals.map((item) => {
      if (item.id !== signalId) return item;
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
