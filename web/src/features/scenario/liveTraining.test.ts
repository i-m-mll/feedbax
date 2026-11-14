import { describe, expect, it } from 'vitest';
import {
  LIVE_TRAINING_TRAJECTORY_SCHEMA_ID,
  LIVE_TRAINING_TRAJECTORY_SCHEMA_VERSION,
  liveTrainingEffectorTrack,
  liveTrainingTargetTrack,
  normalizeTrainingTrajectoryPayload,
} from '@/features/scenario/liveTraining';

describe('live training trajectory normalization', () => {
  it('uses selector-keyed replay tracks as the live frame source', () => {
    const frame = normalizeTrainingTrajectoryPayload(
      {
        schema_id: LIVE_TRAINING_TRAJECTORY_SCHEMA_ID,
        schema_version: LIVE_TRAINING_TRAJECTORY_SCHEMA_VERSION,
        fidelity: 'lower_fidelity_live_snapshot',
        time: { length: 2, units: 'step', values: [4, 5] },
        tracks: {
          'graph_output:effector': {
            anchor_id: 'effector',
            selector: {
              namespace: 'graph_output',
              compact: 'graph_output:effector',
              target_id: 'effector',
              role: 'observed',
            },
            samples: [[0.1, 0.2], [0.3, 0.4]],
            dim: 2,
            dtype: 'float32',
            frame: 'world',
          },
          'task_data:target': {
            anchor_id: 'target',
            selector: {
              namespace: 'task_data',
              compact: 'task_data:target',
              target_id: 'target',
              role: 'target',
            },
            samples: [[0.5, 0.6], [0.7, 0.8]],
            dim: 2,
            dtype: 'float32',
            frame: 'world',
          },
        },
      }
    );

    expect(liveTrainingEffectorTrack(frame)?.selector.compact).toBe('graph_output:effector');
    expect(liveTrainingTargetTrack(frame)?.selector.compact).toBe('task_data:target');
  });

  it('rejects unsupported explicit live trajectory schema versions', () => {
    expect(() =>
      normalizeTrainingTrajectoryPayload(
        {
          schema_version: 'feedbax.event.studio.training_trajectory.v0' as never,
        }
      )
    ).toThrow("Unsupported training trajectory schema_version");
  });
});
