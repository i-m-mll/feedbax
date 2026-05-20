import { describe, expect, it } from 'vitest';
import {
  delayedReachTaskWithTimeline,
  delayedReachTimelineFromTask,
  toggleDelayedReachSignalEpoch,
  updateDelayedReachEpochRange,
} from './taskTimeline';
import type { TaskSpec } from '@/types/training';

const task: TaskSpec = {
  type: 'DelayedReaches',
  params: {
    n_steps: 140,
    epoch_len_ranges: [
      [0, 1],
      [10, 30],
    ],
    target_on_epochs: [1, 2],
    hold_epochs: [0, 1],
    move_epochs: [2],
  },
};

describe('delayed reach task timeline helpers', () => {
  it('projects delayed reach arrays into epoch and signal specs', () => {
    const timeline = delayedReachTimelineFromTask(task)!;

    expect(timeline.epochs.map((epoch) => epoch.label)).toEqual([
      'hold',
      'target_on',
      'movement',
    ]);
    expect(timeline.epochs[0].length.value).toEqual({ min: 0, max: 1 });
    expect(timeline.epochs[0].length).toMatchObject({
      dtype: 'int32',
      units: 'steps',
      metadata: {
        value_schema_id: 'value:task_timeline:epoch_length',
        temporal_window: { mode: 'epoch', epoch_id: 'epoch:0' },
      },
    });
    expect(timeline.epochs[2].length.metadata.inferred_from_remaining_steps).toBe(true);
    expect(timeline.signals.find((signal) => signal.id === 'target_on')?.epoch_ids).toEqual([
      'epoch:1',
      'epoch:2',
    ]);
    expect(timeline.signals.find((signal) => signal.id === 'move')).toMatchObject({
      value_schema: {
        kind: 'task_target',
        dtype: 'float32',
        shape: ['time', 2],
        frame: 'cartesian_effector',
        metadata: {
          storage: 'compact_task_params',
          compact_representation: 'delayed_reach_task_params_v1',
          materializes_to: { dtype: 'float32', shape: ['time', 2] },
        },
      },
      task_data_schema: {
        id: 'task_data:move',
        path: 'targets.effector',
        value_schema: {
          id: 'value:task_timeline:move',
        },
      },
    });
    expect(timeline.metadata).toMatchObject({
      representation: 'delayed_reach_task_params_v1',
      storage: 'compact_task_params',
      materializes_targets: true,
    });
  });

  it('writes timeline edits back into backend-compatible delayed reach params', () => {
    const timeline = delayedReachTimelineFromTask(task)!;
    const editedRange = updateDelayedReachEpochRange(timeline, 'epoch:1', 'min', 12);
    const editedSignals = toggleDelayedReachSignalEpoch(
      editedRange,
      'hold',
      'epoch:1',
      false
    );

    const editedTask = delayedReachTaskWithTimeline(task, editedSignals);

    expect(editedTask.params.epoch_len_ranges).toEqual([
      [0, 1],
      [12, 30],
    ]);
    expect(editedTask.params.hold_epochs).toEqual([0]);
    expect(editedTask.timeline?.schema_version).toBe('feedbax.studio.task_timeline.v1');
  });
});
