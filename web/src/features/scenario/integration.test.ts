import { describe, expect, it } from 'vitest';
import { ANALYSIS_PRODUCT_ID, EVALUATION_RUN_ID, TRAINING_RUN_ID, pipelineWorkspaceFixture } from '@/test/pipelineWorkspaceFixture';
import {
  artifactOverlaysForWorkspace,
  scenarioMetricSpecs,
  stageProductReferences,
} from '@/features/scenario/integration';

describe('scenario integration derivation', () => {
  it('derives metric specs from task defaults, analysis pages, and imported manifests', () => {
    const workspace = pipelineWorkspaceFixture();
    const metrics = scenarioMetricSpecs(workspace);

    expect(metrics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: 'target_reach_error',
          source: 'task_default',
          selector: expect.stringContaining('task:'),
          valueSchema: expect.objectContaining({
            id: 'value:task_data:targets',
            kind: 'task_data',
          }),
          metadata: expect.objectContaining({
            task_data_schema: expect.objectContaining({
              id: 'task_data:targets',
            }),
            temporal_aggregation: 'final',
          }),
        }),
        expect.objectContaining({
          id: 'peak_velocity_m_per_s',
          source: 'analysis',
          sourceId: expect.stringContaining('summary_metrics'),
          valueSchema: expect.objectContaining({
            kind: 'metric',
            units: 'm/s',
          }),
        }),
        expect.objectContaining({
          id: 'final_validation_loss',
          source: 'manifest',
          sourceId: TRAINING_RUN_ID,
          metadata: expect.objectContaining({
            value_schema: expect.objectContaining({
              id: 'value:metric:manifest:final_validation_loss',
            }),
          }),
        }),
      ])
    );
  });

  it('does not treat ordinary analysis figures as workspace overlays', () => {
    const workspace = pipelineWorkspaceFixture();
    const overlays = artifactOverlaysForWorkspace(workspace);

    expect(overlays).toEqual([]);
  });

  it('exposes stage-owned analysis and report references', () => {
    const workspace = pipelineWorkspaceFixture();
    const analysisStage = workspace.stages.find((stage) => stage.kind === 'analysis');
    const reportStage = workspace.stages.find((stage) => stage.kind === 'report');

    expect(stageProductReferences(workspace, analysisStage?.id)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'analysis_page',
          label: 'Validation summary',
          manifestIds: [EVALUATION_RUN_ID],
        }),
      ])
    );
    expect(stageProductReferences(workspace, reportStage?.id)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'report_section',
          collectionId: 'collection:analysis-products',
          manifestIds: [ANALYSIS_PRODUCT_ID],
        }),
      ])
    );
  });
});
