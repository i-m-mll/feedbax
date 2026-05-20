import type { ParamValue } from '@/types/graph';
import type { TaskSpec } from '@/types/training';
import type {
  StudioTaskTimelineSignalSpec,
  StudioTaskTimelineSpec,
  StudioValueSpec,
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

function constantValue(value: unknown, metadata: Record<string, unknown> = {}): StudioValueSpec {
  return {
    schema_version: VALUE_SCHEMA_VERSION,
    mode: 'constant',
    value,
    metadata,
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
  return {
    id,
    label,
    kind,
    path,
    epoch_ids: [...epochSet].sort((a, b) => a - b).map((index) => `epoch:${index}`),
    metadata: {},
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
      length: constantValue(isInferred ? null : { min: range[0], max: range[1] }, {
        scope: 'trial',
        inferred_from_remaining_steps: isInferred,
      }),
      metadata: {},
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
      representation: 'delayed_reach_epochs_v1',
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
