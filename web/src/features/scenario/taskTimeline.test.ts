import { describe, expect, it } from 'vitest';
import {
  applyDelayedReachTimelineEdit,
  delayedReachTaskWithTimeline,
  delayedReachTimelinePreview,
  delayedReachTimelineFromTask,
  toggleDelayedReachSignalEpoch,
  updateDelayedReachEpochRange,
} from './taskTimeline';
import type { TaskSpec } from '@/types/training';
import type { StudioTaskBindingSpec } from '@/types/workspace';

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
    expect(timeline.signals.map((signal) => signal.id)).toEqual([
      'target_position',
      'hold',
      'target_on',
      'movement_target',
    ]);
    expect(timeline.signals.find((signal) => signal.id === 'target_position')).toMatchObject({
      task_data_id: 'target_position',
      epoch_ids: ['epoch:1', 'epoch:2'],
      value_spec: {
        mode: 'function',
        function_id: 'delayed_reach_target_position',
      },
    });
    expect(timeline.signals.find((signal) => signal.id === 'movement_target')).toMatchObject({
      task_data_id: 'movement_target',
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
      value_spec: {
        mode: 'function',
        function_id: 'delayed_reach_movement_target',
      },
      task_data_schema: {
        id: 'task_data:movement_target',
        path: 'targets.effector',
        value_schema: {
          id: 'value:task_timeline:movement_target',
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
    expect(editedRange.epochs[1].length.metadata.value_schema_id).toBe(
      'value:task_timeline:epoch_length'
    );
  });

  it('keeps target position and target-on rows linked through target_on_epochs', () => {
    const timeline = delayedReachTimelineFromTask(task)!;

    const editedSignals = toggleDelayedReachSignalEpoch(
      timeline,
      'target_position',
      'epoch:0',
      true
    );
    const editedTask = delayedReachTaskWithTimeline(task, editedSignals);

    expect(
      editedSignals.signals.find((signal) => signal.id === 'target_position')?.epoch_ids
    ).toEqual(['epoch:0', 'epoch:1', 'epoch:2']);
    expect(editedSignals.signals.find((signal) => signal.id === 'target_on')?.epoch_ids).toEqual([
      'epoch:0',
      'epoch:1',
      'epoch:2',
    ]);
    expect(editedTask.params.target_on_epochs).toEqual([0, 1, 2]);
  });

  it('previews sampled epoch ranges and remaining final epoch', () => {
    const timeline = delayedReachTimelineFromTask(task)!;

    const preview = delayedReachTimelinePreview(timeline);

    expect(preview.epochs).toMatchObject([
      { id: 'epoch:0', start_min: 0, start_max: 0, end_min: 0, end_max: 1 },
      { id: 'epoch:1', start_min: 0, start_max: 1, end_min: 10, end_max: 31 },
      { id: 'epoch:2', start_min: 10, start_max: 31, end_min: 140, end_max: 140 },
    ]);
    expect(preview.signals.find((signal) => signal.id === 'movement_target')).toMatchObject({
      active_epoch_ids: ['epoch:2'],
      active_ranges: [{ start_min: 10, end_max: 140 }],
    });
  });

  it('applies timeline edits to task params and Task Data value specs', () => {
    const timeline = delayedReachTimelineFromTask(task)!;
    const taskBindingSpec: StudioTaskBindingSpec = {
      schema_version: 'feedbax.studio.task_bindings.v2',
      exposed_data: [
        {
          id: 'target_position',
          label: 'Target position',
          kind: 'signal',
          role: 'model_input',
          path: 'inputs.effector_target',
          bindable: true,
          metadata: {},
        },
      ],
      bindings: [],
      metadata: {},
    };

    const edited = applyDelayedReachTimelineEdit(task, taskBindingSpec, timeline);

    expect(edited.task_spec.params.move_epochs).toEqual([2]);
    expect(edited.task_binding_spec?.exposed_data[0].value_spec).toMatchObject({
      mode: 'function',
      function_id: 'delayed_reach_target_position',
    });
    expect(edited.task_binding_spec?.metadata.updated_from).toBe('task_timeline_editor');
  });
});
