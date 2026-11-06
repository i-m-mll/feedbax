import { describe, expect, it } from 'vitest';
import { TRAINING_RUN_ID, pipelineWorkspaceFixture } from '@/test/pipelineWorkspaceFixture';
import {
  bestTrainingRun,
  buildBillableSpecLock,
  buildLineageProjection,
  buildQueueProjection,
  currentDraftSpecHashesForScenario,
  evaluationProtocolLabel,
  evaluationRunSummaries,
  compareSpecHashes,
  selectedIds,
  trainingInputSummaries,
  trainingRunSummaries,
} from '@/utils/pipelineCollections';
import { studioDraftHashes } from '@/utils/studioDraftHash';

describe('pipeline collection summaries', () => {
  it('summarizes training runs by user-facing variant and metrics', () => {
    const workspace = pipelineWorkspaceFixture();
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train');
    const rows = trainingRunSummaries(trainStage);

    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({
      label: 'Tuned run',
      finalValidationLoss: 0.2,
      metrics: expect.objectContaining({
        final_validation_loss: 0.2,
      }),
      jobId: null,
      axisCoordinates: { learning_rate: 0.002 },
    });
    expect(bestTrainingRun(rows)?.id).toBe(
      TRAINING_RUN_ID
    );
  });

  it('summarizes evaluation input selection without exposing collection ids', () => {
    const workspace = pipelineWorkspaceFixture();
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval');

    expect(trainingInputSummaries(evalStage)).toHaveLength(2);
    expect(selectedIds(evalStage, 'training_run_ids')).toEqual([
      TRAINING_RUN_ID,
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

  it('marks pending training rows stale when draft spec hashes changed', () => {
    const currentTrainingSpec = { n_batches: 50 };
    const rows = trainingRunSummaries(
      {
        output_collections: [{
          item_refs: [{
            kind: 'TrainingRun',
            id: 'feedbax-training-run:pending',
            role: 'training_run',
            provider: 'manifest',
            uri: '/tmp/pending.json',
            metadata: {
              name: 'Pending train',
              status: 'pending',
              planned: true,
              spec_hashes: studioDraftHashes({ training_spec: { n_batches: 25 } }),
            },
          }],
        }],
      } as any,
      {
        currentSpecHashes: currentDraftSpecHashesForScenario(
          { training_spec: currentTrainingSpec },
          { schema_id: 'feedbax.spec.graph', schema_version: '5' }
        ),
      }
    );

    expect(rows[0]).toMatchObject({
      status: 'stale',
      stale: true,
      staleReason: 'draft changed',
      statusReason: 'draft changed',
    });
    expect(rows[0].specHashComparisons).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: 'training_spec',
          status: 'changed',
        }),
      ])
    );
  });

  it('checks native matrix launch against the staging draft instead of the varied row', () => {
    const stagedDraft = studioDraftHashes({ training_spec: { n_batches: 10 } });
    const metadata = {
      status: 'pending',
      planned: true,
      native_launch_supported: true,
      execution_target: 'local',
      staging_generation: 'generation-a',
      staging_draft_spec_hashes: stagedDraft,
      spec_hashes: studioDraftHashes({ training_spec: { n_batches: 2 } }),
    };
    const summarize = (patch: Partial<typeof metadata> = {}, currentSpecHashes = stagedDraft) =>
      trainingRunSummaries({
        output_collections: [{ item_refs: [{
          kind: 'TrainingRunManifest',
          id: 'run:matrix-row',
          role: 'training_run',
          metadata: { ...metadata, ...patch },
        }] }],
      } as any, { currentSpecHashes })[0];

    expect(summarize().nativeLaunchGeneration).toBe('generation-a');
    expect(summarize({}, studioDraftHashes({ training_spec: { n_batches: 12 } })))
      .toMatchObject({ status: 'stale', nativeLaunchGeneration: undefined });
    expect(summarize({ execution_target: 'runpod' }).nativeLaunchGeneration).toBeUndefined();
    expect(summarize({ native_launch_supported: false }).nativeLaunchGeneration).toBeUndefined();
    expect(summarize({ status: 'claimed' }).nativeLaunchGeneration).toBeUndefined();
    expect(summarize({ staging_generation: '' }).nativeLaunchGeneration).toBeUndefined();
  });

  it('marks legacy hashes rehash-required and agrees after current JSON round trips', () => {
    const authored = { integral: 1, negative_zero: -0, nested: ['café'] };
    const snapshot = studioDraftHashes({ training_spec: authored });
    const current = studioDraftHashes({
      training_spec: JSON.parse(JSON.stringify(authored)),
    });
    expect(compareSpecHashes(JSON.parse(JSON.stringify(snapshot)), current)).toEqual([
      expect.objectContaining({ key: 'training_spec', status: 'unchanged' }),
    ]);

    const currentRows = trainingRunSummaries(
      {
        output_collections: [{
          item_refs: [{
            kind: 'TrainingRun',
            id: 'feedbax-training-run:current',
            role: 'training_run',
            metadata: {
              name: 'Current pending train',
              status: 'pending',
              planned: true,
              spec_hashes: JSON.parse(JSON.stringify(snapshot)),
            },
          }],
        }],
      } as any,
      { currentSpecHashes: current }
    );
    expect(currentRows[0]).toMatchObject({ status: 'pending', stale: false });

    const rows = trainingRunSummaries(
      {
        output_collections: [{
          item_refs: [{
            kind: 'TrainingRun',
            id: 'feedbax-training-run:legacy',
            role: 'training_run',
            metadata: {
              name: 'Legacy pending train',
              status: 'pending',
              planned: true,
              spec_hashes: JSON.parse(JSON.stringify({
                training_spec: 'fnv1a:12345678',
              })),
            },
          }],
        }],
      } as any,
      { currentSpecHashes: current }
    );
    expect(rows[0]).toMatchObject({
      status: 'stale',
      stale: true,
      staleReason: 'draft hash requires rehash',
      specHashComparisons: [
        expect.objectContaining({ key: 'training_spec', status: 'rehash-required' }),
      ],
    });
  });

  it('filters superseded training rows without hiding self-linked replacements', () => {
    const stage = {
      output_collections: [{
        item_refs: [
          {
            kind: 'TrainingRun',
            id: 'run:old',
            role: 'training_run',
            provider: 'manifest',
            metadata: {
              name: 'Old run',
              status: 'completed',
              superseded_by: 'run:new',
            },
          },
          {
            kind: 'TrainingRun',
            id: 'run:new',
            role: 'training_run',
            provider: 'manifest',
            metadata: {
              name: 'New run',
              status: 'pending',
              superseded_by: 'run:new',
              supersedes: 'run:old',
            },
          },
        ],
      }],
    } as any;

    const visibleRows = trainingRunSummaries(stage);
    expect(visibleRows.map((row) => row.id)).toEqual(['run:new']);
    expect(visibleRows[0]).toMatchObject({
      supersededBy: null,
      supersedes: 'run:old',
    });
    expect(trainingRunSummaries(stage, { includeSuperseded: true }).map((row) => row.id).sort())
      .toEqual(['run:new', 'run:old']);
  });

  it('summarizes completed evaluation protocol details', () => {
    const workspace = pipelineWorkspaceFixture();
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval');
    const [summary] = evaluationRunSummaries(evalStage);

    expect(summary.label).toBe('Held-out validation');
    expect(evaluationProtocolLabel(summary)).toBe(
      'Held-out targets - no perturbation'
    );
  });

  it('prefers the authored condition label and preserves zero in the display', () => {
    const [summary] = evaluationRunSummaries({
      output_collections: [{
        item_refs: [{
          kind: 'EvaluationRunManifest',
          id: 'eval:zero-gain',
          role: 'evaluation_run',
          metadata: {
            condition_label: 'Feedback gain=0',
            condition_axis_coordinates: { gain: 0 },
            eval_protocol: { targets: 'Held-out targets' },
          },
        }],
      }],
    } as any);

    expect(evaluationProtocolLabel(summary)).toBe('Feedback gain=0');
  });

  it('marks downstream evaluation rows stale when an upstream parent was superseded', () => {
    const rows = evaluationRunSummaries({
      output_collections: [{
        item_refs: [{
          kind: 'EvaluationRunManifest',
          id: 'eval:old-parent',
          role: 'evaluation_run',
          provider: 'manifest',
          metadata: {
            name: 'Validation',
            status: 'pending',
            parent_refs: [
              {
                kind: 'TrainingRunManifest',
                id: 'train:old',
                role: 'training_run',
                metadata: { superseded_by: 'train:new' },
              },
            ],
          },
        }],
      }],
    } as any);

    expect(rows[0]).toMatchObject({
      status: 'stale',
      staleReason: 'upstream superseded',
      statusReason: 'upstream superseded',
    });
  });

  it('does not mark downstream evaluation stale for self-superseded parents', () => {
    const rows = evaluationRunSummaries({
      output_collections: [{
        item_refs: [{
          kind: 'EvaluationRunManifest',
          id: 'eval:current-parent',
          role: 'evaluation_run',
          provider: 'manifest',
          metadata: {
            name: 'Validation',
            status: 'pending',
            parent_refs: [
              {
                kind: 'TrainingRunManifest',
                id: 'train:current',
                role: 'training_run',
                metadata: { superseded_by: 'train:current' },
              },
            ],
          },
        }],
      }],
    } as any);

    expect(rows[0]).toMatchObject({
      status: 'pending',
      stale: false,
      staleReason: null,
    });
  });

  it('projects manifest ParentRefs as run-set grouped lineage edges', () => {
    const workspace = pipelineWorkspaceFixture();
    const trainStage = workspace.stages.find((stage) => stage.kind === 'train');
    const evalStage = workspace.stages.find((stage) => stage.kind === 'eval');
    const analysisStage = workspace.stages.find((stage) => stage.kind === 'analysis');
    const reportStage = workspace.stages.find((stage) => stage.kind === 'report');
    if (!trainStage || !evalStage || !analysisStage || !reportStage) {
      throw new Error('fixture missing pipeline stages');
    }
    const trainRef = {
      ...trainStage.output_collections[0].item_refs[0],
      metadata: {
        ...trainStage.output_collections[0].item_refs[0].metadata,
        run_set_id: 'validation',
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
        if (stage.id === reportStage.id) {
          return {
            ...stage,
            input_collections: [{ ...stage.input_collections[0], item_refs: [analysisRef] }],
          };
        }
        return stage;
      }),
      manifest_refs: [trainRef, evalRef, analysisRef],
    });

    expect(projection.groups.find((group) => group.id === 'run-set:validation')).toMatchObject({
      label: 'Run set validation',
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
    const workspace = pipelineWorkspaceFixture();
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
            metadata: { backend_realization: { execution_target: 'runpod' } },
            manifest_refs: [pendingTrain, superseded],
          };
        }
        if (stage.id === evalStage.id) {
          return {
            ...stage,
            metadata: { backend_realization: { execution_target: 'local' } },
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
