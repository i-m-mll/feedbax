import { describe, expect, it } from 'vitest';
import type { WorkspaceReplayProduct } from '@/generated/studioContracts';
import {
  objectiveLossWindowBands,
  resolveWorkspaceReplayComparison,
  resolveWorkspaceReplayModel,
  resolveWorkspaceReplaySources,
  selectWorkspaceReplayTrial,
  workspaceReplayDuration,
  workspaceReplayEventTicks,
  workspaceReplayFrameIndex,
  workspaceReplayFrameTimes,
  workspaceReplayProvenance,
  workspaceReplaySampleAt,
  workspaceReplayTimelineBands,
  workspaceReplayTrialRef,
} from '@/features/scenario/workspaceReplay';
import type { StudioObjectiveSpec, StudioStageSpec, StudioWorkspaceSpec } from '@/types/workspace';

const embeddedProduct: WorkspaceReplayProduct = {
  schema_id: 'feedbax.manifest.studio.workspace_replay',
  schema_version: 'feedbax.manifest.studio.workspace_replay.v1',
  product_kind: 'workspace_replay',
  source_mode: 'resolved_scene',
  trials: [
    {
      identity: { index: 2, stable_id: 'stable-a', label: 'Stable A' },
      time: { length: 3, dt: 0.5, units: 's' },
      tracks: [
        {
          anchor_id: 'effector',
          selector: { namespace: 'graph_output', compact: 'graph_output:effector' },
          samples: [[0, 0], [1, 1], [2, 0]],
          dim: 2,
        },
      ],
      trial_spec: {
        timeline: {
          epochs: [{ id: 'reach', length: 1 }],
          events: [{ id: 'go_cue', time: 0.5 }],
        },
      },
      manifest_refs: {
        checkpoint: {
          kind: 'checkpoint',
          id: 'artifact:ckpt',
          provider: 'test',
          metadata: {},
        },
      },
      metadata: { run_ref: 'run:test' },
    },
  ],
};

const workspace: StudioWorkspaceSpec = {
  id: 'workspace:test',
  schema_version: 'feedbax.studio.workspace.v1',
  label: 'Workspace',
  active_stage_id: 'stage:eval',
  stages: [],
  scenarios: {},
  collections: [],
  manifest_refs: [],
  artifact_refs: [],
  validation: { valid: null, checked_at: null, errors: [], warnings: [], metadata: {} },
  ui_state: {},
  metadata: {},
};

const stage: StudioStageSpec = {
  id: 'stage:eval',
  kind: 'eval',
  label: 'Eval',
  status: 'completed',
  input_collections: [],
  output_collections: [],
  manifest_refs: [],
  artifact_refs: [],
  execution_spec: null,
  selection_spec: {},
  validation: { valid: null, checked_at: null, errors: [], warnings: [], metadata: {} },
  ui_state: {},
  metadata: { workspace_replay_product: embeddedProduct },
};

const candidateProduct: WorkspaceReplayProduct = {
  ...embeddedProduct,
  trials: [
    {
      ...embeddedProduct.trials![0],
      tracks: [
        {
          ...embeddedProduct.trials![0].tracks![0],
          samples: [[0, 0], [1, 1.5], [2, 0.5]],
        },
      ],
      manifest_refs: {
        checkpoint: {
          kind: 'checkpoint',
          id: 'artifact:ckpt-candidate',
          provider: 'test',
          metadata: {},
        },
      },
      metadata: { run_ref: 'run:candidate' },
    },
  ],
};

describe('workspace replay helpers', () => {
  it('prefers embedded replay products and falls back to a typed fixture', () => {
    expect(resolveWorkspaceReplayModel(workspace, stage).source).toBe('embedded');

    const artifactStage = {
      ...stage,
      metadata: {},
      artifact_refs: [
        {
          kind: 'workspace_replay',
          id: 'artifact:replay',
          provider: 'test',
          metadata: { product: embeddedProduct },
        },
      ],
    };
    expect(resolveWorkspaceReplayModel(workspace, artifactStage).source).toBe('embedded');

    const fallback = resolveWorkspaceReplayModel(workspace, { ...stage, metadata: {} });
    expect(fallback.source).toBe('fixture');
    expect(fallback.product.trials?.length).toBeGreaterThan(0);
  });

  it('resolves ref-addressable replay products for comparison selection', () => {
    const comparisonStage: StudioStageSpec = {
      ...stage,
      metadata: {},
      artifact_refs: [
        {
          kind: 'workspace_replay',
          id: 'artifact:baseline',
          provider: 'test',
          metadata: { label: 'Baseline run', product: embeddedProduct },
        },
        {
          kind: 'workspace_replay',
          id: 'artifact:candidate',
          provider: 'test',
          metadata: { label: 'Candidate run', product: candidateProduct },
        },
      ],
    };

    const sources = resolveWorkspaceReplaySources(workspace, comparisonStage);
    expect(sources.map((source) => source.ref)).toEqual([
      'artifact:baseline',
      'artifact:candidate',
    ]);

    const comparison = resolveWorkspaceReplayComparison(
      sources,
      { baseline_ref: 'artifact:baseline', candidate_ref: 'artifact:candidate' },
      'trial:stable-a'
    );

    expect(comparison.members.map((member) => [member.role, member.label])).toEqual([
      ['baseline', 'Baseline run'],
      ['candidate', 'Candidate run'],
    ]);
    expect(workspaceReplaySampleAt(comparison.members[0].track, 1)).toEqual([1, 1]);
    expect(workspaceReplaySampleAt(comparison.members[1].track, 1)).toEqual([1, 1.5]);
    expect(comparison.primaryTrial?.identity.stable_id).toBe('stable-a');
  });

  it('selects trials and honors replay dt when stepping frames', () => {
    const trial = selectWorkspaceReplayTrial(embeddedProduct, 'trial:stable-a')!;
    expect(workspaceReplayTrialRef(trial)).toBe('trial:stable-a');
    expect(workspaceReplayFrameTimes(trial)).toEqual([0, 0.5, 1]);
    expect(workspaceReplayDuration(trial)).toBe(1);
    expect(workspaceReplayFrameIndex(trial, 0.7)).toBe(1);
    expect(workspaceReplaySampleAt(trial.tracks?.[0], 2)).toEqual([2, 0]);
  });

  it('extracts timeline bands, event ticks, loss windows, and provenance', () => {
    const trial = embeddedProduct.trials![0];
    const objectiveSpec: StudioObjectiveSpec = {
      schema_version: 'feedbax.studio.objective.v1',
      terms: [
        {
          id: 'endpoint',
          type_id: 'target_state',
          label: 'Endpoint',
          role: 'loss',
          temporal_selector: { mode: 'range', start: 0.25, end: 0.75 },
          weight: 1,
          metadata: {},
        },
      ],
      legacy_loss_spec: null,
      metadata: {},
    };

    expect(workspaceReplayTimelineBands(trial)).toMatchObject([
      { id: 'reach', label: 'reach', start: 0, end: 1, kind: 'epoch' },
    ]);
    expect(workspaceReplayEventTicks(trial)).toMatchObject([
      { id: 'go_cue', label: 'go_cue', time: 0.5 },
    ]);
    expect(objectiveLossWindowBands(objectiveSpec, 1)).toMatchObject([
      { id: 'loss:endpoint:range', start: 0.25, end: 0.75, kind: 'loss_window' },
    ]);
    expect(workspaceReplayProvenance(trial)).toContain('Eval playback');
    expect(workspaceReplayProvenance(trial)).toContain('run:test');
  });
});
