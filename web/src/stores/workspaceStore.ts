import { create } from 'zustand';
import type { AnalysisSnapshot } from '@/types/analysis';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { TaskSpec, TrainingSpec } from '@/types/training';
import type {
  AnalysisPageWire,
  StudioCollectionRef,
  StudioPipelineMaterializationResult,
  StudioScenarioSpec,
  StudioStageKind,
  StudioStageSpec,
  StudioTrainingLocalRunResult,
  StudioTrainingExecutionPreparation,
  StudioValidationState,
  StudioWorkspaceSpec,
} from '@/types/workspace';

const WORKSPACE_SCHEMA_VERSION = 'feedbax.studio.workspace.v1';
const SCENARIO_SCHEMA_VERSION = 'feedbax.studio.scenario.v1';

const DEFAULT_STAGE_IDS = {
  train: 'stage:train',
  eval: 'stage:eval',
  analysis: 'stage:analysis',
  report: 'stage:report',
} as const;

const DEFAULT_SCENARIO_IDS = {
  train: 'scenario:train',
  eval: 'scenario:eval',
  analysis: 'scenario:analysis',
  report: 'scenario:report',
} as const;

function emptyValidation(): StudioValidationState {
  return {
    valid: null,
    checked_at: null,
    errors: [],
    warnings: [],
    metadata: {},
  };
}

function generateId(prefix: string): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `${prefix}:${crypto.randomUUID()}`;
  }
  return `${prefix}:${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
}

function collection(
  id: string,
  kind: string,
  label: string,
  sourceStageId: string
): StudioCollectionRef {
  return {
    id,
    kind,
    label,
    source_stage_id: sourceStageId,
    item_refs: [],
    filters: {},
    facets: {},
    metadata: {},
  };
}

function defaultStage(
  kind: StudioStageKind,
  label: string,
  scenarioId: string,
  inputCollections: StudioCollectionRef[] = [],
  outputCollections: StudioCollectionRef[] = []
): StudioStageSpec {
  return {
    id: DEFAULT_STAGE_IDS[kind as keyof typeof DEFAULT_STAGE_IDS] ?? generateId('stage'),
    kind,
    label,
    status: 'draft',
    scenario_id: scenarioId,
    input_collections: inputCollections,
    output_collections: outputCollections,
    manifest_refs: [],
    artifact_refs: [],
    execution_spec: null,
    selection_spec: {},
    validation: emptyValidation(),
    ui_state: {},
    metadata: {},
  };
}

function defaultScenario(
  id: string,
  label: string,
  stageId: string,
  overrides: Partial<StudioScenarioSpec> = {}
): StudioScenarioSpec {
  return {
    id,
    schema_version: SCENARIO_SCHEMA_VERSION,
    label,
    stage_id: stageId,
    parent_scenario_id: null,
    graph: null,
    graph_ui_state: null,
    training_spec: null,
    task_spec: null,
    objective_spec: null,
    probe_specs: [],
    temporal_spec: null,
    biomechanics_spec: null,
    analysis_spec: null,
    report_spec: null,
    validation: emptyValidation(),
    ui_state: {},
    metadata: {},
    ...overrides,
  };
}

function analysisPagesFromSnapshot(snapshot: AnalysisSnapshot | null): AnalysisPageWire[] {
  if (!snapshot || snapshot.pages.length === 0) return [];
  return snapshot.pages.map((page) => ({
    id: page.id,
    name: page.name,
    graph_spec: page.graphSpec as unknown as Record<string, unknown>,
    eval_params: page.evalParams,
    viewport: page.viewport,
    eval_run_id: page.evalRunId,
    expanded_field_paths: page.expandedFieldPaths ?? [],
  }));
}

function mergeAnalysisSpec(
  existing: Record<string, unknown> | null | undefined,
  pages: AnalysisPageWire[],
  activePageId: string | null
): Record<string, unknown> | null {
  if (pages.length === 0 && !existing) return null;
  return {
    ...(existing ?? {}),
    pages,
    active_page_id: activePageId,
  };
}

function ensureDefaultStages(workspace: StudioWorkspaceSpec): StudioWorkspaceSpec {
  const existingByKind = new Map(workspace.stages.map((stage) => [stage.kind, stage]));
  const trainingRuns = collection(
    'collection:training-runs',
    'training_runs',
    'Training runs',
    DEFAULT_STAGE_IDS.train
  );
  const evaluationRuns = collection(
    'collection:evaluation-runs',
    'evaluation_runs',
    'Evaluation runs',
    DEFAULT_STAGE_IDS.eval
  );
  const analysisProducts = collection(
    'collection:analysis-products',
    'analysis_products',
    'Analysis products',
    DEFAULT_STAGE_IDS.analysis
  );
  const reports = collection(
    'collection:reports',
    'reports',
    'Reports',
    DEFAULT_STAGE_IDS.report
  );

  const defaults = [
    defaultStage('train', 'Train', DEFAULT_SCENARIO_IDS.train, [], [trainingRuns]),
    defaultStage('eval', 'Evaluate', DEFAULT_SCENARIO_IDS.eval, [trainingRuns], [evaluationRuns]),
    defaultStage(
      'analysis',
      'Analyze',
      DEFAULT_SCENARIO_IDS.analysis,
      [evaluationRuns],
      [analysisProducts]
    ),
    defaultStage('report', 'Report', DEFAULT_SCENARIO_IDS.report, [analysisProducts], [reports]),
  ];
  const mergedDefaults = defaults.filter((stage) => !existingByKind.has(stage.kind));

  return {
    ...workspace,
    stages: [...workspace.stages, ...mergedDefaults],
  };
}

export function buildWorkspaceSnapshot({
  workspace,
  graph,
  uiState,
  trainingSpec,
  taskSpec,
  analysisSnapshot,
  projectName,
}: {
  workspace: StudioWorkspaceSpec | null;
  graph: GraphSpec;
  uiState: GraphUIState | null;
  trainingSpec: TrainingSpec;
  taskSpec: TaskSpec;
  analysisSnapshot: AnalysisSnapshot | null;
  projectName?: string;
}): StudioWorkspaceSpec {
  const base: StudioWorkspaceSpec =
    workspace ??
    {
      id: generateId('studio-workspace'),
      schema_version: WORKSPACE_SCHEMA_VERSION,
      label: projectName ?? graph.metadata?.name ?? 'Studio workspace',
      active_stage_id: DEFAULT_STAGE_IDS.train,
      stages: [],
      scenarios: {},
      collections: [],
      manifest_refs: [],
      artifact_refs: [],
      validation: emptyValidation(),
      ui_state: {},
      metadata: { source: 'frontend_workspace_store' },
    };

  const withStages = ensureDefaultStages(base);
  const trainStage =
    withStages.stages.find((stage) => stage.kind === 'train') ?? withStages.stages[0];
  const evalStage = withStages.stages.find((stage) => stage.kind === 'eval');
  const analysisStage = withStages.stages.find((stage) => stage.kind === 'analysis');
  const reportStage = withStages.stages.find((stage) => stage.kind === 'report');

  const trainScenarioId = trainStage?.scenario_id ?? DEFAULT_SCENARIO_IDS.train;
  const evalScenarioId = evalStage?.scenario_id ?? DEFAULT_SCENARIO_IDS.eval;
  const analysisScenarioId = analysisStage?.scenario_id ?? DEFAULT_SCENARIO_IDS.analysis;
  const reportScenarioId = reportStage?.scenario_id ?? DEFAULT_SCENARIO_IDS.report;
  const analysisPages = analysisPagesFromSnapshot(analysisSnapshot);

  const scenarios = { ...withStages.scenarios };
  const existingTrain = scenarios[trainScenarioId];
  scenarios[trainScenarioId] = {
    ...defaultScenario(trainScenarioId, existingTrain?.label ?? 'Training scenario', trainStage.id),
    ...existingTrain,
    graph,
    graph_ui_state: uiState,
    training_spec: trainingSpec,
    task_spec: taskSpec,
    metadata: {
      ...(existingTrain?.metadata ?? {}),
      updated_from: 'active_studio_state',
    },
  };

  if (evalStage && !scenarios[evalScenarioId]) {
    scenarios[evalScenarioId] = defaultScenario(
      evalScenarioId,
      'Evaluation scenario',
      evalStage.id,
      {
        parent_scenario_id: trainScenarioId,
        metadata: { inheritance: 'training_default' },
      }
    );
  }
  if (analysisStage) {
    const existingAnalysis = scenarios[analysisScenarioId];
    scenarios[analysisScenarioId] = {
      ...defaultScenario(analysisScenarioId, 'Analysis scenario', analysisStage.id),
      ...existingAnalysis,
      analysis_spec: mergeAnalysisSpec(
        existingAnalysis?.analysis_spec,
        analysisPages,
        analysisSnapshot?.activePageId ?? null
      ),
    };
  }
  if (reportStage && !scenarios[reportScenarioId]) {
    scenarios[reportScenarioId] = defaultScenario(
      reportScenarioId,
      'Report scenario',
      reportStage.id
    );
  }

  return {
    ...withStages,
    label: withStages.label || projectName || graph.metadata?.name || 'Studio workspace',
    active_stage_id: withStages.active_stage_id ?? trainStage?.id ?? null,
    scenarios,
  };
}

interface WorkspaceStoreState {
  workspace: StudioWorkspaceSpec | null;
  lastTrainingExecutionPreparation: StudioTrainingExecutionPreparation | null;
  lastTrainingLocalRunResult: StudioTrainingLocalRunResult | null;
  lastPipelineMaterializationResult: StudioPipelineMaterializationResult | null;
  setWorkspace: (workspace: StudioWorkspaceSpec | null) => void;
  setTrainingExecutionPreparation: (
    preparation: StudioTrainingExecutionPreparation | null
  ) => void;
  setTrainingLocalRunResult: (result: StudioTrainingLocalRunResult | null) => void;
  setPipelineMaterializationResult: (
    result: StudioPipelineMaterializationResult | null
  ) => void;
  updateActiveScenarioTrainingSpec: (trainingSpec: TrainingSpec) => void;
  updateActiveScenarioTaskSpec: (taskSpec: TaskSpec) => void;
}

function activeTrainScenario(workspace: StudioWorkspaceSpec | null): string | null {
  if (!workspace) return null;
  const activeStage = workspace.stages.find((stage) => stage.id === workspace.active_stage_id);
  const trainStage = workspace.stages.find((stage) => stage.kind === 'train');
  return trainStage?.scenario_id ?? activeStage?.scenario_id ?? null;
}

export const useWorkspaceStore = create<WorkspaceStoreState>((set) => ({
  workspace: null,
  lastTrainingExecutionPreparation: null,
  lastTrainingLocalRunResult: null,
  lastPipelineMaterializationResult: null,

  setWorkspace: (workspace) => set({ workspace }),

  setTrainingExecutionPreparation: (preparation) =>
    set((state) => ({
      lastTrainingExecutionPreparation: preparation,
      workspace: preparation?.workspace ?? state.workspace,
    })),

  setTrainingLocalRunResult: (result) =>
    set((state) => ({
      lastTrainingLocalRunResult: result,
      workspace: result?.workspace ?? state.workspace,
    })),

  setPipelineMaterializationResult: (result) =>
    set((state) => ({
      lastPipelineMaterializationResult: result,
      workspace: result?.workspace ?? state.workspace,
    })),

  updateActiveScenarioTrainingSpec: (trainingSpec) =>
    set((state) => {
      const scenarioId = activeTrainScenario(state.workspace);
      if (!state.workspace || !scenarioId) return {};
      const scenario = state.workspace.scenarios[scenarioId];
      if (!scenario) return {};
      return {
        workspace: {
          ...state.workspace,
          scenarios: {
            ...state.workspace.scenarios,
            [scenarioId]: {
              ...scenario,
              training_spec: trainingSpec,
            },
          },
        },
      };
    }),

  updateActiveScenarioTaskSpec: (taskSpec) =>
    set((state) => {
      const scenarioId = activeTrainScenario(state.workspace);
      if (!state.workspace || !scenarioId) return {};
      const scenario = state.workspace.scenarios[scenarioId];
      if (!scenario) return {};
      return {
        workspace: {
          ...state.workspace,
          scenarios: {
            ...state.workspace.scenarios,
            [scenarioId]: {
              ...scenario,
              task_spec: taskSpec,
            },
          },
        },
      };
    }),
}));
