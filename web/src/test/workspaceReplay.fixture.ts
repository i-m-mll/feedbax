import type { WorkspaceReplayProduct } from '@/generated/studioContracts';

export const WORKSPACE_REPLAY_FIXTURE: WorkspaceReplayProduct = {
  schema_id: 'feedbax.manifest.studio.workspace_replay',
  schema_version: 'feedbax.manifest.studio.workspace_replay.v1',
  product_kind: 'workspace_replay',
  source_mode: 'resolved_scene',
  trials: [
    {
      identity: {
        index: 0,
        stable_id: 'fixture-validation-0',
        source: 'stable_id',
        label: 'Validation fixture trial',
        metadata: { condition: 'validation' },
      },
      time: {
        length: 61,
        units: 's',
        dt: 0.02,
        values: Array.from({ length: 61 }, (_, index) => Number((index * 0.02).toFixed(2))),
      },
      tracks: [
        {
          anchor_id: 'effector',
          selector: {
            namespace: 'graph_output',
            compact: 'graph_output:effector',
            target_id: 'effector',
            role: 'observed',
          },
          samples: Array.from({ length: 61 }, (_, index) => {
            const progress = index / 60;
            return [
              -0.36 + progress * 0.72,
              -0.18 + Math.sin(progress * Math.PI) * 0.28,
            ];
          }),
          dim: 2,
          dtype: 'float32',
          units: 'm',
          frame: 'world',
          label: 'Effector',
        },
      ],
      trial_spec: {
        summary: { condition: 'validation', note: 'typed browser fixture' },
        timeline: {
          epochs: [
            { id: 'prep', label: 'Prep', start: 0, end: 0.28 },
            { id: 'movement', label: 'Movement', start: 0.28, end: 1.2 },
          ],
          events: [{ id: 'go_cue', label: 'Go cue', time: 0.28 }],
        },
        schema_id: 'feedbax.fixture.workspace_replay_trial',
        schema_version: 'v1',
      },
      manifest_refs: {
        checkpoint: {
          kind: 'checkpoint',
          id: 'fixture:checkpoint:browser',
          role: 'validation_checkpoint',
          provider: 'fixture',
          metadata: { checkpoint_step: 8000 },
        },
        spec_snapshot: {
          kind: 'trial_spec_snapshot',
          id: 'fixture:trial-spec:browser',
          role: 'spec_snapshot',
          provider: 'fixture',
          metadata: {},
        },
        seed: 7,
      },
      warnings: [
        {
          code: 'missing_manifest_refs',
          message: 'Using typed browser fixture because no workspace replay product is attached.',
        },
      ],
      metadata: { run_ref: 'fixture-run', checkpoint_step: 8000 },
    },
    {
      identity: {
        index: 1,
        stable_id: 'fixture-validation-1',
        source: 'stable_id',
        label: 'Validation fixture alternate',
        metadata: { condition: 'holdout' },
      },
      time: {
        length: 61,
        units: 's',
        dt: 0.02,
        values: Array.from({ length: 61 }, (_, index) => Number((index * 0.02).toFixed(2))),
      },
      tracks: [
        {
          anchor_id: 'effector',
          selector: {
            namespace: 'graph_output',
            compact: 'graph_output:effector',
            target_id: 'effector',
            role: 'observed',
          },
          samples: Array.from({ length: 61 }, (_, index) => {
            const progress = index / 60;
            return [
              -0.32 + progress * 0.64,
              -0.22 + Math.sin(progress * Math.PI) * 0.2 + 0.05,
            ];
          }),
          dim: 2,
          dtype: 'float32',
          units: 'm',
          frame: 'world',
          label: 'Effector',
        },
      ],
      trial_spec: {
        summary: { condition: 'holdout', note: 'static comparison fixture' },
        timeline: {
          epochs: [
            { id: 'prep', label: 'Prep', start: 0, end: 0.28 },
            { id: 'movement', label: 'Movement', start: 0.28, end: 1.2 },
          ],
          events: [{ id: 'go_cue', label: 'Go cue', time: 0.28 }],
        },
        schema_id: 'feedbax.fixture.workspace_replay_trial',
        schema_version: 'v1',
      },
      manifest_refs: { seed: 8 },
      warnings: [],
      metadata: { run_ref: 'fixture-run', checkpoint_step: 8000 },
    },
  ],
  warnings: [
    {
      code: 'missing_manifest_refs',
      message: 'No attached workspace replay product was found; Studio is showing typed fixture data.',
    },
  ],
  metadata: { source: 'studio_fixture' },
};
