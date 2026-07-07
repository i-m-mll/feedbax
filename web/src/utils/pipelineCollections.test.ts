import { describe, expect, it } from 'vitest';
import { createRlrmpModelGraph } from '@/data/rlrmp-model-graph';
import {
  createRlrmpMovementRampAnalysis,
  seedRlrmpMovementRampWorkspace,
} from '@/data/rlrmp-run-example';
import { buildWorkspaceSnapshot } from '@/stores/workspaceStore';
import { defaultTaskSpec, defaultTrainingSpec } from '@/stores/trainingStore';
import {
  bestTrainingRun,
  buildBillableSpecLock,
  buildLineageProjection,
  buildQueueProjection,
  evaluationProtocolLabel,
  evaluationRunSummaries,
  selectedIds,
  trainingInputSummaries,
  trainingRunSummaries,
} from '@/utils/pipelineCollections';

function seededWorkspace() {
  const { graph, uiState } = createRlrmpModelGraph('RLRMP movement-ramp runs');
  const analysisSnapshot = createRlrmpMovementRampAnalysis();
  return seedRlrmpMovementRampWorkspace(
    buildWorkspaceSnapshot({
      workspace: null,
      graph,
      uiState,
      trainingSpec: defaultTrainingSpec,
      taskSpec: defaultTaskSpec,
      analysisSnapshot,
      projectName: 'RLRMP movement-ramp runs',
    })
  );
}

describe('pipeline collection summaries', () => {
  it('summarizes training runs by user-facing variant and metrics', () => {
    const workspace = seededWorkspace();
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train');
    const rows = trainingRunSummaries(trainStage);

    expect(rows).toHaveLength(7);
    expect(rows[0]).toMatchObject({
      label: 'Power 6 ramp, duration 80',
      finalValidationLoss: 0.1021,
      metrics: expect.objectContaining({
        final_validation_loss: 0.1021,
        peak_velocity_m_per_s: 1.157,
      }),
      rampDurationSteps: 80,
      sourceIssue: 'b399efc',
      jobId: null,
      axisCoordinates: {},
    });
    expect(bestTrainingRun(rows)?.id).toBe(
      'rlrmp:b399efc:movement_ramp__power6_dur80'
    );
  });

  it('summarizes evaluation input selection without exposing collection ids', () => {
    const workspace = seededWorkspace();
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval');

    expect(trainingInputSummaries(evalStage)).toHaveLength(7);
    expect(selectedIds(evalStage, 'training_run_ids')).toEqual([
      'rlrmp:b399efc:movement_ramp__power6_dur80',
    ]);
  });

  it('keeps manifest job ids for progress binding', () => {
    const rows = trainingRunSummaries({
      output_collections: [{
        item_refs: [{
          kind: 'TrainingRun',
          id: 'feedbax-training-run:pending',
          role: 'training_run',
          uri: '/tmp/pending.json',
          metadata: {
            name: 'Pending train',
            status: 'pending',
            job_id: 'studio-train-123',
            planned: true,
          },
        }],
      }],
    } as any);

    expect(rows[0]).toMatchObject({
      id: 'feedbax-training-run:pending',
      jobId: 'studio-train-123',
      planned: true,
    });
  });

  it('summarizes completed evaluation protocol details', () => {
    const workspace = seededWorkspace();
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval');
    const [summary] = evaluationRunSummaries(evalStage);

    expect(summary.label).toBe('8-direction validation, SISU 0.5, zero perturbation');
    expect(evaluationProtocolLabel(summary)).toBe(
      '8-direction center-out - SISU 0.5 - no perturbation'
    );
  });

  it('projects manifest ParentRefs as run-set grouped lineage edges', () => {
    const workspace = seededWorkspace();
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train');
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval');
    const analysisStage = workspace.stages.find((stage) => stage.kind === 'analysis');
    if (!trainStage || !evalStage || !analysisStage) throw new Error('seed missing stages');
    const trainRef = {
      ...trainStage.output_collections[0].item_refs[0],
      metadata: {
        ...trainStage.output_collections[0].item_refs[0].metadata,
        run_set_id: 'movement-ramp',
      },
    };
    const evalRef = {
      ...evalStage.output_collections[0].item_refs[0],
      kind: 'EvaluationRunManifest',
      metadata: {
        job_id: 'studio-pipeline',
        status: 'stale',
        staleness_reason: 'training manifest was superseded',
        parent_refs: [
          {
            kind: 'TrainingRunManifest',
            id: trainRef.id,
            role: 'training_run',
            metadata: { status: 'completed' },
          },
        ],
      },
    };
    const analysisRef = {
      ...analysisStage.output_collections[0].item_refs[0],
      kind: 'AnalysisRunManifest',
      metadata: {
        job_id: 'studio-pipeline',
        status: 'skipped',
        skip_reason: 'optional output disabled',
        inputs: [{ kind: 'EvaluationRunManifest', id: evalRef.id, role: 'evaluation_run' }],
      },
    };
    const projection = buildLineageProjection({
      ...workspace,
      collections: [
        {
          ...trainStage.output_collections[0],
          item_refs: [trainRef],
        },
        {
          ...evalStage.output_collections[0],
          item_refs: [evalRef],
        },
        {
          ...analysisStage.output_collections[0],
          item_refs: [analysisRef],
        },
      ],
      stages: workspace.stages.map((stage) => {
        if (stage.id === trainStage.id) {
          return {
            ...stage,
            output_collections: [{ ...stage.output_collections[0], item_refs: [trainRef] }],
            manifest_refs: [trainRef],
          };
        }
        if (stage.id === evalStage.id) {
          return {
            ...stage,
            output_collections: [{ ...stage.output_collections[0], item_refs: [evalRef] }],
            manifest_refs: [evalRef],
          };
        }
        if (stage.id === analysisStage.id) {
          return {
            ...stage,
            input_collections: [{ ...stage.input_collections[0], item_refs: [evalRef] }],
            output_collections: [{ ...stage.output_collections[0], item_refs: [analysisRef] }],
            manifest_refs: [analysisRef],
          };
        }
        return stage;
      }),
      manifest_refs: [trainRef, evalRef, analysisRef],
    });

    expect(projection.groups.find((group) => group.id === 'run-set:movement-ramp')).toMatchObject({
      label: 'Run set movement-ramp',
      nodeIds: [trainRef.id],
    });
    expect(projection.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          parentId: trainRef.id,
          childId: evalRef.id,
          role: 'training_run',
          status: 'completed',
        }),
        expect.objectContaining({
          parentId: evalRef.id,
          childId: analysisRef.id,
          role: 'evaluation_run',
        }),
      ])
    );
    expect(projection.nodes.find((node) => node.id === evalRef.id)).toMatchObject({
      status: 'stale',
      statusReason: 'training manifest was superseded',
      focusStageId: evalStage.id,
      focusCollectionId: evalStage.output_collections[0].id,
    });
    expect(projection.nodes.find((node) => node.id === analysisRef.id)).toMatchObject({
      status: 'skipped',
      statusReason: 'optional output disabled',
    });
  });

  it('projects pending manifests across stages with target assignment and events', () => {
    const workspace = seededWorkspace();
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train');
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval');
    if (!trainStage || !evalStage) throw new Error('seed missing stages');
    const pendingTrain = {
      kind: 'TrainingRunManifest',
      id: 'train:pending-runpod',
      role: 'training_run',
      provider: 'feedbax',
      metadata: {
        name: 'RunPod train',
        status: 'pending',
        axis_coordinates: { learning_rate: 0.001, seed: 1 },
        execution_target: 'runpod',
        queue_order: 2,
        estimated_duration_minutes: 40,
        estimated_cost_usd: 1.5,
      },
    };
    const pendingEval = {
      kind: 'EvaluationRunManifest',
      id: 'eval:pending-local',
      role: 'evaluation_run',
      provider: 'feedbax',
      metadata: {
        name: 'Local eval',
        status: 'pending',
        axis_coordinates: { learning_rate: 0.003, seed: 2 },
        queue_order: 1,
      },
    };
    const superseded = {
      kind: 'TrainingRunManifest',
      id: 'train:superseded',
      role: 'training_run',
      provider: 'feedbax',
      metadata: {
        name: 'Superseded run',
        status: 'pending',
        superseded_by: 'train:new',
      },
    };
    const projection = buildQueueProjection({
      ...workspace,
      stages: workspace.stages.map((stage) => {
        if (stage.id === trainStage.id) {
          return {
            ...stage,
            execution_spec: { protocol: { compute_target: 'runpod' } },
            manifest_refs: [pendingTrain, superseded],
          };
        }
        if (stage.id === evalStage.id) {
          return {
            ...stage,
            execution_spec: { protocol: { compute_target: 'local' } },
            manifest_refs: [pendingEval],
          };
        }
        return stage;
      }),
      manifest_refs: [pendingTrain, pendingEval, superseded],
    });

    expect(projection.items.map((item) => item.manifestId)).toEqual([
      'eval:pending-local',
      'train:pending-runpod',
    ]);
    expect(projection.items[0]).toMatchObject({
      target: 'local',
      billable: false,
      canLaunch: true,
    });
    expect(projection.items[1]).toMatchObject({
      target: 'runpod',
      billable: true,
      canLaunch: true,
      estimatedCostUsd: 1.5,
    });
    expect(projection.events).toEqual([
      expect.objectContaining({
        manifestId: 'train:superseded',
        reason: 'Superseded by train:new',
      }),
    ]);
  });

  it('requires spec-lock only for billable queue targets', () => {
    const runpodItem = {
      manifestId: 'runpod-a',
      target: 'runpod',
      targetLabel: 'RunPod',
      billable: true,
      runCount: 1,
      axisCoordinates: { learning_rate: 0.001 },
      estimatedDurationMinutes: 20,
      estimatedCostUsd: 0.8,
    } as any;
    const localItem = {
      manifestId: 'local-a',
      target: 'local',
      targetLabel: 'Local worker',
      billable: false,
      runCount: 1,
      axisCoordinates: { learning_rate: 0.003 },
      estimatedDurationMinutes: null,
      estimatedCostUsd: null,
    } as any;

    expect(buildBillableSpecLock([localItem])).toMatchObject({
      required: false,
      confirmationToken: null,
    });
    expect(buildBillableSpecLock([runpodItem, localItem])).toMatchObject({
      required: true,
      target: 'runpod',
      runCount: 1,
      estimatedCostUsd: 0.8,
      confirmationToken: 'confirm-runpod-queue-launch',
    });
  });

  it('requires target-specific spec locks for mixed GCP and RunPod billable queues', () => {
    const runpodItem = {
      manifestId: 'runpod-a',
      target: 'runpod',
      targetLabel: 'RunPod',
      billable: true,
      runCount: 2,
      axisCoordinates: { learning_rate: 0.001, seed: 1 },
      estimatedDurationMinutes: 45,
      estimatedCostUsd: 1.6,
    } as any;
    const gcpItem = {
      manifestId: 'gcp-a',
      target: 'gcp',
      targetLabel: 'GCP',
      billable: true,
      runCount: 3,
      axisCoordinates: { learning_rate: 0.003, seed: 2 },
      estimatedDurationMinutes: 30,
      estimatedCostUsd: 4.5,
    } as any;

    expect(buildBillableSpecLock([runpodItem, gcpItem])).toMatchObject({
      required: true,
      target: null,
      targetLabel: 'Choose billable target',
      runCount: 5,
      confirmationToken: null,
      targetOptions: [
        { target: 'gcp', targetLabel: 'GCP', itemCount: 1, runCount: 3 },
        { target: 'runpod', targetLabel: 'RunPod', itemCount: 1, runCount: 2 },
      ],
    });

    expect(buildBillableSpecLock([runpodItem, gcpItem], 'runpod')).toMatchObject({
      required: true,
      target: 'runpod',
      runCount: 2,
      estimatedDurationMinutes: 45,
      estimatedCostUsd: 1.6,
      confirmationToken: 'confirm-runpod-queue-launch',
      variedAxes: [],
    });
    expect(buildBillableSpecLock([runpodItem, gcpItem], 'gcp')).toMatchObject({
      required: true,
      target: 'gcp',
      runCount: 3,
      estimatedDurationMinutes: 30,
      estimatedCostUsd: 4.5,
      confirmationToken: 'confirm-gcp-queue-launch',
      variedAxes: [],
    });
  });
});
