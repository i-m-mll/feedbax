import { createBlankGraph } from '@/stores/graphStore';
import { defaultTaskSpec, defaultTrainingSpec } from '@/stores/trainingStore';
import { buildWorkspaceSnapshot } from '@/stores/workspaceStore';
import type { AnalysisSnapshot } from '@/types/analysis';
import type { StudioManifestRef, StudioStageKind } from '@/types/workspace';

export const TRAINING_RUN_ID = 'train:best';
export const EVALUATION_RUN_ID = 'eval:validation';
export const ANALYSIS_PRODUCT_ID = 'analysis:metrics';

const analysisSnapshot: AnalysisSnapshot = {
  pages: [{
    id: 'analysis:summary',
    name: 'Validation summary',
    graphSpec: {
      dataSourceId: '__data_source__',
      nodes: {
        summary_metrics: {
          id: 'summary_metrics',
          type: 'Violins',
          label: 'Validation metrics',
          category: 'Statistical Plots',
          inputPorts: ['input'],
          outputPorts: ['figure'],
          params: { metrics: ['peak_velocity_m_per_s'] },
          role: 'analysis',
        },
      },
      wires: [],
    },
    evalParams: {},
    viewport: { x: 0, y: 0, zoom: 1 },
    evalRunId: EVALUATION_RUN_ID,
    expandedFieldPaths: [],
  }],
  activePageId: 'analysis:summary',
};

function trainingRun(id: string, name: string, loss: number): StudioManifestRef {
  return {
    kind: 'TrainingRunManifest',
    id,
    role: 'training_run',
    provider: 'manifest',
    metadata: {
      name,
      status: 'completed',
      final_validation_loss: loss,
      axis_coordinates: { learning_rate: loss / 100 },
    },
  };
}

export function pipelineWorkspaceFixture() {
  const workspace = buildWorkspaceSnapshot({
    workspace: null,
    graph: createBlankGraph(),
    uiState: null,
    taskSpec: defaultTaskSpec,
    trainingSpec: defaultTrainingSpec,
    analysisSnapshot,
  });
  const outputs: Partial<Record<StudioStageKind, StudioManifestRef[]>> = {
    train: [
      trainingRun('train:baseline', 'Baseline run', 0.8),
      trainingRun(TRAINING_RUN_ID, 'Tuned run', 0.2),
    ],
    eval: [{
      kind: 'EvaluationRunManifest',
      id: EVALUATION_RUN_ID,
      role: 'evaluation_run',
      provider: 'manifest',
      metadata: {
        name: 'Held-out validation',
        status: 'completed',
        eval_protocol: { targets: 'Held-out targets', perturbation: 'none' },
      },
    }],
    analysis: [{
      kind: 'AnalysisProduct',
      id: ANALYSIS_PRODUCT_ID,
      role: 'analysis_product',
      provider: 'manifest',
      metadata: { name: 'Validation metrics', status: 'completed' },
    }],
  };

  const stages = workspace.stages.map((stage) => ({
    ...stage,
    selection_spec: stage.kind === 'eval'
      ? { training_run_ids: [TRAINING_RUN_ID] }
      : stage.selection_spec,
    output_collections: stage.output_collections.map((collection) => ({
      ...collection,
      item_refs: outputs[stage.kind] ?? [],
    })),
  }));
  const collections = stages.flatMap((stage) => stage.output_collections);
  const report = stages.find((stage) => stage.kind === 'report')!;
  const reportScenario = workspace.scenarios[report.scenario_id!];
  return {
    ...workspace,
    stages: stages.map((stage) => ({
      ...stage,
      input_collections: collections.filter((collection) =>
        stage.input_collections.some((input) => input.id === collection.id)
      ),
    })),
    collections,
    scenarios: {
      ...workspace.scenarios,
      [reportScenario.id]: {
        ...reportScenario,
        report_spec: {
          sections: [{
            id: 'validation',
            title: 'Validation results',
            collection_id: 'collection:analysis-products',
            manifest_ids: [ANALYSIS_PRODUCT_ID],
          }],
        },
      },
    },
  };
}
