import type { AnalysisSnapshot } from '@/types/analysis';
import type {
  StudioArtifactRef,
  StudioCollectionRef,
  StudioManifestRef,
  StudioWorkspaceSpec,
} from '@/types/workspace';

const TRAIN_STAGE_ID = 'stage:train';
const EVAL_STAGE_ID = 'stage:eval';
const ANALYSIS_STAGE_ID = 'stage:analysis';
const REPORT_STAGE_ID = 'stage:report';

const TRAINING_COLLECTION_ID = 'collection:training-runs';
const EVAL_INPUT_COLLECTION_ID = 'collection:b399efc-training-runs-for-eval';
const EVALUATION_COLLECTION_ID = 'collection:evaluation-runs';
const ANALYSIS_INPUT_COLLECTION_ID = 'collection:b399efc-evaluation-runs-for-analysis';
const ANALYSIS_OUTPUT_COLLECTION_ID = 'collection:b399efc-analysis-products';
const REPORT_COLLECTION_ID = 'collection:b399efc-reports';

const EVALUATION_RUN_ID = 'rlrmp:b399efc:eval:centerout-sisu0.5-zero-perturbation';
const WINNER_RUN_ID = 'rlrmp:b399efc:movement_ramp__power6_dur80';
const ANALYSIS_PRODUCT_ID = 'rlrmp:b399efc:analysis:summary-products';
const REPORT_MANIFEST_ID = 'rlrmp:b399efc:report:movement-ramp-summary';

interface MovementRampRunSeed {
  variant: string;
  label: string;
  rampShape: string;
  rampDurationSteps: number;
  nnOutputPreGo: number;
  finalValLoss: number;
  withinCellVelRmse: number;
  peakVelocityMean: number;
  peakVelocitySd: number;
  holdDriftMeanMm: number;
  holdDriftSdMm: number;
}

const MOVEMENT_RAMP_RUNS: MovementRampRunSeed[] = [
  {
    variant: 'movement_ramp__linear',
    label: 'Linear ramp',
    rampShape: 'linear',
    rampDurationSteps: 60,
    nnOutputPreGo: 1,
    finalValLoss: 0.7225,
    withinCellVelRmse: 0.0867,
    peakVelocityMean: 1.938,
    peakVelocitySd: 0.128,
    holdDriftMeanMm: 0.02,
    holdDriftSdMm: 0.01,
  },
  {
    variant: 'movement_ramp__cosine',
    label: 'Cosine ramp',
    rampShape: 'cosine',
    rampDurationSteps: 60,
    nnOutputPreGo: 1,
    finalValLoss: 0.5127,
    withinCellVelRmse: 0.0974,
    peakVelocityMean: 1.983,
    peakVelocitySd: 0.12,
    holdDriftMeanMm: 0.02,
    holdDriftSdMm: 0.02,
  },
  {
    variant: 'movement_ramp__power2',
    label: 'Power 2 ramp',
    rampShape: 'power',
    rampDurationSteps: 60,
    nnOutputPreGo: 1,
    finalValLoss: 0.3619,
    withinCellVelRmse: 0.091,
    peakVelocityMean: 1.716,
    peakVelocitySd: 0.128,
    holdDriftMeanMm: 0.01,
    holdDriftSdMm: 0.01,
  },
  {
    variant: 'movement_ramp__power4',
    label: 'Power 4 ramp',
    rampShape: 'power',
    rampDurationSteps: 60,
    nnOutputPreGo: 1,
    finalValLoss: 0.1811,
    withinCellVelRmse: 0.099,
    peakVelocityMean: 1.506,
    peakVelocitySd: 0.129,
    holdDriftMeanMm: 0.02,
    holdDriftSdMm: 0.02,
  },
  {
    variant: 'movement_ramp__power6',
    label: 'Power 6 ramp',
    rampShape: 'power',
    rampDurationSteps: 60,
    nnOutputPreGo: 1,
    finalValLoss: 0.1509,
    withinCellVelRmse: 0.1287,
    peakVelocityMean: 1.397,
    peakVelocitySd: 0.087,
    holdDriftMeanMm: 0.02,
    holdDriftSdMm: 0.02,
  },
  {
    variant: 'movement_ramp__power6_prego5',
    label: 'Power 6 ramp, pre-go 5',
    rampShape: 'power',
    rampDurationSteps: 60,
    nnOutputPreGo: 5,
    finalValLoss: 0.2383,
    withinCellVelRmse: 0.1724,
    peakVelocityMean: 1.205,
    peakVelocitySd: 0.177,
    holdDriftMeanMm: 0.03,
    holdDriftSdMm: 0.01,
  },
  {
    variant: 'movement_ramp__power6_dur80',
    label: 'Power 6 ramp, duration 80',
    rampShape: 'power',
    rampDurationSteps: 80,
    nnOutputPreGo: 1,
    finalValLoss: 0.1021,
    withinCellVelRmse: 0.0969,
    peakVelocityMean: 1.157,
    peakVelocitySd: 0.09,
    holdDriftMeanMm: 0.02,
    holdDriftSdMm: 0.01,
  },
];

function emptyValidation() {
  return {
    valid: null,
    checked_at: null,
    errors: [],
    warnings: [],
    metadata: {},
  };
}

function trainingRunRef(run: MovementRampRunSeed): StudioManifestRef {
  return {
    kind: 'TrainingRun',
    id: `rlrmp:b399efc:${run.variant}`,
    role: 'training_run',
    provider: 'rlrmp',
    uri: `results/b399efc/runs/${run.variant}/run.json`,
    metadata: {
      name: run.label,
      status: 'completed',
      source_issue: 'b399efc',
      run_variant: run.variant,
      n_replicates: 5,
      n_warmup_batches: 12000,
      batch_size: 250,
      hidden_type: 'gru',
      ramp_shape: run.rampShape,
      ramp_duration_steps: run.rampDurationSteps,
      nn_output_pre_go: run.nnOutputPreGo,
      final_validation_loss: run.finalValLoss,
      within_cell_velocity_rmse_m_per_s: run.withinCellVelRmse,
      peak_velocity_m_per_s: {
        mean: run.peakVelocityMean,
        sd: run.peakVelocitySd,
      },
      hold_drift_mm: {
        mean: run.holdDriftMeanMm,
        sd: run.holdDriftSdMm,
      },
    },
  };
}

function collection(
  id: string,
  kind: string,
  label: string,
  sourceStageId: string,
  itemRefs: StudioManifestRef[],
  metadata: Record<string, unknown> = {}
): StudioCollectionRef {
  return {
    id,
    kind,
    label,
    source_stage_id: sourceStageId,
    item_refs: itemRefs,
    filters: {},
    facets: {},
    metadata,
  };
}

function evaluationRunRef(): StudioManifestRef {
  return {
    kind: 'EvaluationRun',
    id: EVALUATION_RUN_ID,
    role: 'evaluation_run',
    provider: 'rlrmp',
    uri: 'results/b399efc/notes/matrix_results.md',
    metadata: {
      name: '8-direction validation, SISU 0.5, zero perturbation',
      status: 'completed',
      source_issue: 'b399efc',
      training_run_ids: MOVEMENT_RAMP_RUNS.map((run) => `rlrmp:b399efc:${run.variant}`),
      selected_training_run_id: WINNER_RUN_ID,
      eval_protocol: {
        targets: '8-direction center-out',
        sisu: 0.5,
        perturbation: 'none',
      },
    },
  };
}

function analysisProductRef(): StudioManifestRef {
  return {
    kind: 'AnalysisProduct',
    id: ANALYSIS_PRODUCT_ID,
    role: 'analysis_product',
    provider: 'rlrmp',
    uri: 'results/b399efc/figures/summary-products.json',
    metadata: {
      name: 'Movement-ramp summary analysis products',
      status: 'completed',
      source_issue: 'b399efc',
      page_id: 'analysis:b399efc:summary',
      eval_run_ids: [EVALUATION_RUN_ID],
      figure_topics: [
        'forward_velocity_profiles',
        'hold_drift_profiles',
        'peak_velocity_distributions',
        'summary_metrics',
      ],
      metric_ids: ['peak_velocity_m_per_s', 'hold_drift_mm', 'within_cell_velocity_rmse_m_per_s'],
    },
  };
}

function reportManifestRef(): StudioManifestRef {
  return {
    kind: 'ReportManifest',
    id: REPORT_MANIFEST_ID,
    role: 'report',
    provider: 'rlrmp',
    uri: 'results/b399efc/report/manifest.json',
    metadata: {
      name: 'Movement-ramp summary report',
      status: 'completed',
      source_issue: 'b399efc',
      analysis_product_ids: [ANALYSIS_PRODUCT_ID],
      source_stage_id: ANALYSIS_STAGE_ID,
    },
  };
}

function artifactRefs(): StudioArtifactRef[] {
  return [
    {
      kind: 'CheckpointBundle',
      id: 'rlrmp:b399efc:checkpoints',
      role: 'model_checkpoints',
      provider: 'rlrmp',
      uri: '_artifacts/b399efc/runs/<variant>/warmup_model.eqx',
      media_type: 'application/octet-stream',
      metadata: {
        note: 'One warmup_model.eqx per movement-ramp variant.',
      },
    },
    {
      kind: 'FigureBundle',
      id: 'rlrmp:b399efc:figures',
      role: 'summary_figures',
      provider: 'rlrmp',
      uri: '_artifacts/b399efc/figures/<topic>/figure.html',
      media_type: 'text/html',
      metadata: {
        topics: [
          'forward_velocity_profiles',
          'hold_drift_profiles',
          'peak_velocity_distributions',
          'summary_metrics',
          'training_loss',
          'training_loss_per_term',
        ],
      },
    },
  ];
}

export function createRlrmpMovementRampAnalysis(): AnalysisSnapshot {
  return {
    pages: [
      {
        id: 'analysis:b399efc:summary',
        name: 'b399efc summary',
        graphSpec: {
          dataSourceId: '__data_source__',
          nodes: {
            training_loss: {
              id: 'training_loss',
              type: 'Profiles',
              label: 'Training loss by ramp variant',
              category: 'Trajectory Plots',
              inputPorts: ['data'],
              outputPorts: ['figure'],
              params: {
                source_figure: 'results/b399efc/figures/training_loss',
                color_by: 'run_variant',
              },
              role: 'analysis',
            },
            forward_velocity: {
              id: 'forward_velocity',
              type: 'Profiles',
              label: 'Forward velocity profiles',
              category: 'Trajectory Plots',
              inputPorts: ['data'],
              outputPorts: ['figure'],
              params: {
                source_figure: 'results/b399efc/figures/forward_velocity_profiles',
                align_epoch: 'go_cue',
              },
              role: 'analysis',
            },
            summary_metrics: {
              id: 'summary_metrics',
              type: 'Violins',
              label: 'Peak velocity and hold drift',
              category: 'Statistical Plots',
              inputPorts: ['input'],
              outputPorts: ['figure'],
              params: {
                source_figure: 'results/b399efc/figures/summary_metrics',
                metrics: ['peak_velocity_m_per_s', 'hold_drift_mm', 'within_cell_velocity_rmse_m_per_s'],
              },
              role: 'analysis',
            },
          },
          wires: [],
        },
        evalParams: {
          source_issue: 'b399efc',
          training_run_ids: MOVEMENT_RAMP_RUNS.map((run) => `rlrmp:b399efc:${run.variant}`),
          selected_training_run_id: WINNER_RUN_ID,
          sisu_values: [0.5],
          perturbation_type: 'none',
          task_variants: {
            validation: '8-direction center-out',
          },
        },
        viewport: { x: 0, y: 0, zoom: 1 },
        evalRunId: EVALUATION_RUN_ID,
        expandedFieldPaths: ['states', 'states.mechanics', 'states.net'],
      },
    ],
    activePageId: 'analysis:b399efc:summary',
  };
}

export function seedRlrmpMovementRampWorkspace(
  baseWorkspace: StudioWorkspaceSpec
): StudioWorkspaceSpec {
  const trainingRefs = MOVEMENT_RAMP_RUNS.map(trainingRunRef);
  const evalRef = evaluationRunRef();
  const analysisProduct = analysisProductRef();
  const reportManifest = reportManifestRef();
  const artifacts = artifactRefs();
  const trainingCollection = collection(
    TRAINING_COLLECTION_ID,
    'training_runs',
    'Completed movement-ramp runs',
    TRAIN_STAGE_ID,
    trainingRefs,
    { source_issue: 'b399efc', seeded_example: true }
  );
  const evalInputCollection = collection(
    EVAL_INPUT_COLLECTION_ID,
    'training_runs',
    'Training runs ready for evaluation',
    TRAIN_STAGE_ID,
    trainingRefs,
    {
      source_collection_id: TRAINING_COLLECTION_ID,
      selected_training_run_id: WINNER_RUN_ID,
      seeded_example: true,
    }
  );
  const evaluationCollection = collection(
    EVALUATION_COLLECTION_ID,
    'evaluation_runs',
    'Completed validation runs',
    EVAL_STAGE_ID,
    [evalRef],
    { selected_training_run_id: WINNER_RUN_ID, seeded_example: true }
  );
  const analysisInputCollection = collection(
    ANALYSIS_INPUT_COLLECTION_ID,
    'evaluation_runs',
    'Evaluation runs ready for analysis',
    EVAL_STAGE_ID,
    [evalRef],
    {
      source_collection_id: EVALUATION_COLLECTION_ID,
      selected_eval_run_id: EVALUATION_RUN_ID,
      seeded_example: true,
    }
  );
  const analysisOutputCollection = collection(
    ANALYSIS_OUTPUT_COLLECTION_ID,
    'analysis_products',
    'Movement-ramp analysis products',
    ANALYSIS_STAGE_ID,
    [analysisProduct],
    {
      source_collection_id: ANALYSIS_INPUT_COLLECTION_ID,
      selected_eval_run_id: EVALUATION_RUN_ID,
      seeded_example: true,
    }
  );
  const reportCollection = collection(
    REPORT_COLLECTION_ID,
    'reports',
    'Movement-ramp report products',
    REPORT_STAGE_ID,
    [reportManifest],
    {
      source_collection_id: ANALYSIS_OUTPUT_COLLECTION_ID,
      analysis_product_ids: [ANALYSIS_PRODUCT_ID],
      seeded_example: true,
    }
  );

  return {
    ...baseWorkspace,
    label: 'RLRMP movement-ramp run review',
    active_stage_id: EVAL_STAGE_ID,
    collections: [
      trainingCollection,
      evalInputCollection,
      evaluationCollection,
      analysisInputCollection,
      analysisOutputCollection,
      reportCollection,
    ],
    manifest_refs: [...trainingRefs, evalRef, analysisProduct, reportManifest],
    artifact_refs: artifacts,
    stages: baseWorkspace.stages.map((stage) => {
      if (stage.kind === 'train') {
        return {
          ...stage,
          label: 'Train',
          status: 'completed' as const,
          output_collections: [trainingCollection],
          manifest_refs: trainingRefs,
          artifact_refs: artifacts,
          selection_spec: {
            source_issue: 'b399efc',
            winning_training_run_id: WINNER_RUN_ID,
            training_run_ids: trainingRefs.map((ref) => ref.id),
          },
          metadata: {
            ...stage.metadata,
            source_issue: 'b399efc',
            seeded_example: true,
            summary: 'Seven completed movement-ramp training cells from rlrmp.',
          },
        };
      }
      if (stage.kind === 'eval') {
        return {
          ...stage,
          label: 'Evaluate',
          status: 'completed' as const,
          input_collections: [evalInputCollection],
          output_collections: [evaluationCollection],
          manifest_refs: [evalRef],
          selection_spec: {
            source_collection_id: TRAINING_COLLECTION_ID,
            training_run_ids: [WINNER_RUN_ID],
            candidate_training_run_ids: trainingRefs.map((ref) => ref.id),
            evaluation_run_ids: [EVALUATION_RUN_ID],
          },
          metadata: {
            ...stage.metadata,
            source_issue: 'b399efc',
            seeded_example: true,
            summary: 'Seeded validation over the b399efc matrix results.',
          },
        };
      }
      if (stage.kind === 'analysis') {
        return {
          ...stage,
          label: 'Analyze',
          status: 'completed' as const,
          input_collections: [analysisInputCollection],
          output_collections: [analysisOutputCollection],
          manifest_refs: [analysisProduct],
          artifact_refs: artifacts,
          selection_spec: {
            source_collection_id: EVALUATION_COLLECTION_ID,
            eval_run_ids: [EVALUATION_RUN_ID],
            input_collection_ids: [ANALYSIS_INPUT_COLLECTION_ID],
            output_collection_ids: [ANALYSIS_OUTPUT_COLLECTION_ID],
          },
          metadata: {
            ...stage.metadata,
            source_issue: 'b399efc',
            seeded_example: true,
            summary: 'Seeded analysis products for the b399efc validation run.',
          },
        };
      }
      if (stage.kind === 'report') {
        return {
          ...stage,
          label: 'Report',
          status: 'completed' as const,
          input_collections: [analysisOutputCollection],
          output_collections: [reportCollection],
          manifest_refs: [reportManifest],
          selection_spec: {
            source_collection_id: ANALYSIS_OUTPUT_COLLECTION_ID,
            analysis_product_ids: [ANALYSIS_PRODUCT_ID],
            report_ids: [REPORT_MANIFEST_ID],
          },
          metadata: {
            ...stage.metadata,
            source_issue: 'b399efc',
            seeded_example: true,
            summary: 'Seeded report consuming the movement-ramp analysis products.',
          },
        };
      }
      return stage;
    }),
    scenarios: Object.fromEntries(
      Object.entries(baseWorkspace.scenarios).map(([id, scenario]) => {
        if (scenario.stage_id === TRAIN_STAGE_ID) {
          return [
            id,
            {
              ...scenario,
              label: 'Movement-ramp training matrix',
              metadata: {
                ...scenario.metadata,
                source_issue: 'b399efc',
                seeded_example: true,
              },
            },
          ];
        }
        if (scenario.stage_id === EVAL_STAGE_ID) {
          return [
            id,
            {
              ...scenario,
              label: '8-direction validation at SISU 0.5',
              metadata: {
                ...scenario.metadata,
                source_issue: 'b399efc',
                seeded_example: true,
              },
            },
          ];
        }
        if (scenario.stage_id === ANALYSIS_STAGE_ID) {
          return [
            id,
            {
              ...scenario,
              label: 'Movement-ramp summary analysis',
              metadata: {
                ...scenario.metadata,
                source_issue: 'b399efc',
                seeded_example: true,
              },
            },
          ];
        }
        if (scenario.stage_id === REPORT_STAGE_ID) {
          return [
            id,
            {
              ...scenario,
              label: 'Movement-ramp summary report',
              report_spec: {
                schema_version: 'feedbax.studio.report.v1',
                consumes: [
                  {
                    stage_id: ANALYSIS_STAGE_ID,
                    collection_id: ANALYSIS_OUTPUT_COLLECTION_ID,
                    manifest_ids: [ANALYSIS_PRODUCT_ID],
                  },
                ],
                sections: [
                  {
                    id: 'section:movement-ramp-summary',
                    title: 'Movement-ramp summary',
                    role: 'results_summary',
                    source_stage_id: ANALYSIS_STAGE_ID,
                    collection_id: ANALYSIS_OUTPUT_COLLECTION_ID,
                    manifest_ids: [ANALYSIS_PRODUCT_ID],
                    artifact_ids: ['rlrmp:b399efc:figures'],
                  },
                ],
                metadata: {
                  source_issue: 'b399efc',
                  seeded_example: true,
                },
              },
              metadata: {
                ...scenario.metadata,
                source_issue: 'b399efc',
                seeded_example: true,
              },
            },
          ];
        }
        return [id, scenario];
      })
    ),
    validation: emptyValidation(),
    metadata: {
      ...baseWorkspace.metadata,
      source_issue: 'b399efc',
      seeded_example: true,
      source_project: 'rlrmp',
      source_run_spec_commit: '69c91eb',
      summary:
        'Seeded from rlrmp b399efc: seven completed movement-ramp GRU training cells.',
    },
  };
}

export const RLRMP_MOVEMENT_RAMP_TEMPLATE = {
  id: 'rlrmp-movement-ramp-runs',
  name: 'RLRMP movement-ramp training runs',
  description: 'Seven completed training runs seeded into the Train and Evaluate tables.',
  pageNames: ['b399efc summary'],
} as const;
