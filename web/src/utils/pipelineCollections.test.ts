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
  buildLineageProjection,
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
});
