import { create } from 'zustand';
import {
  lossSpecFromObjectiveSpec,
  selectorWithSubpath,
} from '@/features/scenario/objectives';
import {
  createDefaultTaskBindingSpec,
  ensureTaskBindingSpec,
  retargetTaskBindingsForNodeRename,
} from '@/features/scenario/taskBindings';
import {
  normalizeGraphForStudioAuthoring,
  normalizeTaskBindingSpecForStudioAuthoring,
} from '@/features/graph/normalization';
import type { AnalysisSnapshot } from '@/types/analysis';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { LossTermSpec, TaskSpec, TrainingSpec } from '@/types/training';
import type {
  AnalysisPageWire,
  StudioCollectionRef,
  StudioObjectiveSpec,
  StudioPipelineMaterializationResult,
  StudioScenarioSpec,
  StudioSelectorRef,
  StudioTaskBindingSpec,
  StudioTopPaneProjection,
  StudioTopPaneState,
  StudioStageKind,
  StudioStageSpec,
  StudioTrainingLocalRunResult,
  StudioTrainingExecutionPreparation,
  StudioValidationState,
  StudioWorkspaceSpec,
} from '@/types/workspace';

const WORKSPACE_SCHEMA_VERSION = 'feedbax.studio.workspace.v1';
const SCENARIO_SCHEMA_VERSION = 'feedbax.studio.scenario.v1';
const OBJECTIVE_SCHEMA_VERSION = 'feedbax.studio.objective.v1';

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

const DEFAULT_TOP_PANE_STATE: StudioTopPaneState = {
  active_projection: 'model',
  selected_entity_id: null,
  hovered_entity_id: null,
  pinned_inspector_entity_id: null,
  metadata: {},
};

const LEGACY_PROBE_SELECTOR_MAP: Record<
  string,
  { nodeId: string; port: string; subpath: string }
> = {
  effector_pos: { nodeId: 'mechanics', port: 'effector', subpath: 'position' },
  effector_vel: { nodeId: 'mechanics', port: 'effector', subpath: 'velocity' },
  network_hidden: { nodeId: 'network', port: 'hidden', subpath: 'hidden' },
};

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

function nowIso(): string {
  return new Date().toISOString();
}

function markDraftMetadata(
  metadata: Record<string, unknown>,
  reason: string
): Record<string, unknown> {
  const currentVersion =
    typeof metadata.draft_version === 'number' ? metadata.draft_version : 0;
  return {
    ...metadata,
    dirty: true,
    draft_version: currentVersion + 1,
    updated_at: nowIso(),
    updated_reason: reason,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function normalizeTopPaneState(value: unknown): StudioTopPaneState {
  const record = isRecord(value) ? value : {};
  const activeProjection =
    record.active_projection === 'task' ||
    record.active_projection === 'workspace' ||
    record.active_projection === 'objectives'
      ? record.active_projection
      : 'model';
  return {
    active_projection: activeProjection,
    selected_entity_id:
      typeof record.selected_entity_id === 'string' ? record.selected_entity_id : null,
    hovered_entity_id:
      typeof record.hovered_entity_id === 'string' ? record.hovered_entity_id : null,
    pinned_inspector_entity_id:
      typeof record.pinned_inspector_entity_id === 'string'
        ? record.pinned_inspector_entity_id
        : null,
    metadata: isRecord(record.metadata) ? record.metadata : {},
  };
}

function updateTopPaneState(
  workspace: StudioWorkspaceSpec,
  patch: Partial<StudioTopPaneState>,
  reason: string,
  markDirty = true
): StudioWorkspaceSpec {
  const topPane = {
    ...normalizeTopPaneState(workspace.ui_state.top_pane),
    ...patch,
    metadata: {
      ...normalizeTopPaneState(workspace.ui_state.top_pane).metadata,
      ...(patch.metadata ?? {}),
    },
  };
  return {
    ...workspace,
    ui_state: {
      ...workspace.ui_state,
      top_pane: topPane,
    },
    metadata: markDirty ? markDraftMetadata(workspace.metadata, reason) : workspace.metadata,
  };
}

function selectorRefFromString(selector: string | undefined): StudioSelectorRef | null {
  if (!selector) return null;
  if (selector.startsWith('probe:')) {
    const probeId = selector.slice('probe:'.length);
    const mappedProbe = LEGACY_PROBE_SELECTOR_MAP[probeId];
    if (mappedProbe) {
      const baseSelector: StudioSelectorRef = {
        namespace: 'graph_port',
        compact: `port:${mappedProbe.nodeId}.${mappedProbe.port}`,
        target_id: mappedProbe.nodeId,
        path: mappedProbe.port,
        role: 'observed',
        metadata: {
          source: 'legacy_loss_selector',
          legacy_selector: selector,
          direction: 'output',
        },
      };
      return selectorWithSubpath(baseSelector, mappedProbe.subpath);
    }
    return {
      namespace: 'probe',
      compact: selector,
      target_id: probeId,
      path: null,
      metadata: { source: 'legacy_loss_selector' },
    };
  }
  if (selector.startsWith('port:')) {
    const portRef = selector.slice('port:'.length);
    const [nodeId, ...portParts] = portRef.split('.');
    return {
      namespace: 'graph_port',
      compact: selector,
      target_id: nodeId || null,
      path: portParts.join('.') || null,
      metadata: { source: 'legacy_loss_selector' },
    };
  }
  if (selector.startsWith('path:')) {
    return {
      namespace: 'state_path',
      compact: selector,
      target_id: null,
      path: selector.slice('path:'.length),
      metadata: { source: 'legacy_loss_selector' },
    };
  }
  return {
    namespace: 'custom',
    compact: selector,
    target_id: null,
    path: selector,
    metadata: { source: 'legacy_loss_selector' },
  };
}

export function objectiveSpecFromLossSpec(loss: LossTermSpec): StudioObjectiveSpec {
  const terms: StudioObjectiveSpec['terms'] = [];

  const visit = (term: LossTermSpec, path: string[]) => {
    const children = term.children ?? {};
    const childEntries = Object.entries(children);
    if (childEntries.length > 0) {
      childEntries.forEach(([key, child]) => visit(child, [...path, key]));
      return;
    }

    const stablePath = path.length > 0 ? path.join('.') : 'root';
    terms.push({
      id: `objective:${stablePath}`,
      type_id: term.type,
      label: term.label,
      role: term.type.toLowerCase().includes('regular') ? 'regularizer' : 'loss',
      source_selector: selectorRefFromString(term.selector),
      target_selector: null,
      operator: 'minimize',
      penalty: term.norm ?? null,
      temporal_selector: term.time_agg ?? null,
      weight: term.weight,
      metadata: {
        legacy_loss_path: path,
        legacy_loss_type: term.type,
      },
    });
  };

  visit(loss, []);
  return {
    schema_version: OBJECTIVE_SCHEMA_VERSION,
    terms,
    legacy_loss_spec: loss,
    metadata: {
      lowered_from: 'training_spec.loss',
    },
  };
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
    task_binding_spec: null,
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

function normalizeScenarioTaskBindingSpec(
  scenario: StudioScenarioSpec
): StudioScenarioSpec {
  if (!scenario.graph) return scenario;
  const graph = normalizeGraphForStudioAuthoring(
    scenario.graph,
    scenario.task_binding_spec
  );
  const normalizedExistingTaskBindingSpec = normalizeTaskBindingSpecForStudioAuthoring(
    scenario.task_binding_spec,
    graph
  );
  const taskBindingSpec = ensureTaskBindingSpec(
    normalizedExistingTaskBindingSpec,
    graph,
    scenario.task_spec
  );
  if (graph === scenario.graph && taskBindingSpec === scenario.task_binding_spec) {
    return scenario;
  }
  return {
    ...scenario,
    graph,
    task_binding_spec: taskBindingSpec,
  };
}

function normalizeWorkspaceTaskBindingSpecs(
  workspace: StudioWorkspaceSpec | null
): StudioWorkspaceSpec | null {
  if (!workspace) return workspace;
  let changed = false;
  const scenarios = Object.fromEntries(
    Object.entries(workspace.scenarios).map(([scenarioId, scenario]) => {
      const normalized = normalizeScenarioTaskBindingSpec(scenario);
      if (normalized !== scenario) changed = true;
      return [scenarioId, normalized];
    })
  );
  return changed ? { ...workspace, scenarios } : workspace;
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
  const scenarioTrainingSpec = existingTrain?.training_spec ?? trainingSpec;
  const scenarioTaskSpec = existingTrain?.task_spec ?? taskSpec;
  const scenarioTaskBindingSpec = ensureTaskBindingSpec(
    existingTrain?.task_binding_spec ?? createDefaultTaskBindingSpec(graph, scenarioTaskSpec),
    graph,
    scenarioTaskSpec
  );
  scenarios[trainScenarioId] = {
    ...defaultScenario(trainScenarioId, existingTrain?.label ?? 'Training scenario', trainStage.id),
    ...existingTrain,
    graph,
    graph_ui_state: uiState,
    training_spec: scenarioTrainingSpec,
    task_spec: scenarioTaskSpec,
    task_binding_spec: scenarioTaskBindingSpec,
    objective_spec:
      existingTrain?.objective_spec ?? objectiveSpecFromLossSpec(scenarioTrainingSpec.loss),
    metadata: {
      ...(existingTrain?.metadata ?? {}),
      draft_owner: 'studio_workspace',
      updated_from: existingTrain ? 'workspace_draft' : 'legacy_active_studio_state',
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

  return normalizeWorkspaceTaskBindingSpecs({
    ...withStages,
    label: withStages.label || projectName || graph.metadata?.name || 'Studio workspace',
    active_stage_id: withStages.active_stage_id ?? trainStage?.id ?? null,
    scenarios,
  })!;
}

interface WorkspaceStoreState {
  workspace: StudioWorkspaceSpec | null;
  lastTrainingExecutionPreparation: StudioTrainingExecutionPreparation | null;
  lastTrainingLocalRunResult: StudioTrainingLocalRunResult | null;
  lastPipelineMaterializationResult: StudioPipelineMaterializationResult | null;
  setWorkspace: (workspace: StudioWorkspaceSpec | null) => void;
  setActiveStage: (stageId: string | null) => void;
  setActiveStageByKind: (kind: StudioStageKind) => void;
  updateStageDraft: (stageId: string, patch: Partial<StudioStageSpec>, reason?: string) => void;
  updateScenarioDraft: (
    scenarioId: string,
    patch: Partial<StudioScenarioSpec>,
    reason?: string
  ) => void;
  setTrainingExecutionPreparation: (
    preparation: StudioTrainingExecutionPreparation | null
  ) => void;
  setTrainingLocalRunResult: (result: StudioTrainingLocalRunResult | null) => void;
  setPipelineMaterializationResult: (
    result: StudioPipelineMaterializationResult | null
  ) => void;
  updateActiveScenarioTrainingSpec: (trainingSpec: TrainingSpec) => void;
  updateActiveScenarioTaskSpec: (taskSpec: TaskSpec) => void;
  updateActiveScenarioTaskBindingSpec: (taskBindingSpec: StudioTaskBindingSpec) => void;
  retargetActiveScenarioTaskBindingsForNodeRename: (
    previousNodeId: string,
    nextNodeId: string
  ) => void;
  updateActiveScenarioObjectiveSpec: (objectiveSpec: StudioObjectiveSpec) => void;
  setTopPaneProjection: (projection: StudioTopPaneProjection) => void;
  selectTopPaneEntity: (entityId: string | null, reason?: string) => void;
  hoverTopPaneEntity: (entityId: string | null) => void;
  updateStageCollections: (
    stageId: string,
    collections: {
      input_collections?: StudioCollectionRef[];
      output_collections?: StudioCollectionRef[];
    },
    reason?: string
  ) => void;
}

export function getStageByKind(
  workspace: StudioWorkspaceSpec | null,
  kind: StudioStageKind
): StudioStageSpec | null {
  return workspace?.stages.find((stage) => stage.kind === kind) ?? null;
}

export function getActiveStage(workspace: StudioWorkspaceSpec | null): StudioStageSpec | null {
  if (!workspace) return null;
  return (
    workspace.stages.find((stage) => stage.id === workspace.active_stage_id) ??
    workspace.stages[0] ??
    null
  );
}

export function getScenario(
  workspace: StudioWorkspaceSpec | null,
  scenarioId: string | null | undefined
): StudioScenarioSpec | null {
  if (!workspace || !scenarioId) return null;
  return workspace.scenarios[scenarioId] ?? null;
}

export function getActiveScenario(workspace: StudioWorkspaceSpec | null): StudioScenarioSpec | null {
  return getScenario(workspace, getActiveStage(workspace)?.scenario_id);
}

export function getTrainingScenario(
  workspace: StudioWorkspaceSpec | null
): StudioScenarioSpec | null {
  const trainStage = getStageByKind(workspace, 'train');
  return getScenario(workspace, trainStage?.scenario_id);
}

function activeTrainScenario(workspace: StudioWorkspaceSpec | null): string | null {
  if (!workspace) return null;
  const trainStage = getStageByKind(workspace, 'train');
  const activeStage = getActiveStage(workspace);
  return trainStage?.scenario_id ?? activeStage?.scenario_id ?? null;
}

function activeScenarioId(workspace: StudioWorkspaceSpec | null): string | null {
  if (!workspace) return null;
  return getActiveStage(workspace)?.scenario_id ?? activeTrainScenario(workspace);
}

export function getTopPaneState(
  workspace: StudioWorkspaceSpec | null | undefined
): StudioTopPaneState {
  return normalizeTopPaneState(workspace?.ui_state.top_pane);
}

export const useWorkspaceStore = create<WorkspaceStoreState>((set) => ({
  workspace: null,
  lastTrainingExecutionPreparation: null,
  lastTrainingLocalRunResult: null,
  lastPipelineMaterializationResult: null,

  setWorkspace: (workspace) => set({ workspace: normalizeWorkspaceTaskBindingSpecs(workspace) }),

  setActiveStage: (stageId) =>
    set((state) => {
      if (!state.workspace) return {};
      const stageExists = state.workspace.stages.some((stage) => stage.id === stageId);
      const activeStageId = stageId && stageExists ? stageId : null;
      return {
        workspace: {
          ...state.workspace,
          active_stage_id: activeStageId,
          ui_state: {
            ...state.workspace.ui_state,
            active_stage_id: activeStageId,
          },
          metadata: markDraftMetadata(state.workspace.metadata, 'active_stage_changed'),
        },
      };
    }),

  setActiveStageByKind: (kind) =>
    set((state) => {
      const stage = getStageByKind(state.workspace, kind);
      if (!state.workspace || !stage) return {};
      return {
        workspace: {
          ...state.workspace,
          active_stage_id: stage.id,
          ui_state: {
            ...state.workspace.ui_state,
            active_stage_id: stage.id,
          },
          metadata: markDraftMetadata(state.workspace.metadata, 'active_stage_changed'),
        },
      };
    }),

  updateStageDraft: (stageId, patch, reason = 'stage_draft_updated') =>
    set((state) => {
      if (!state.workspace) return {};
      let changed = false;
      const stages = state.workspace.stages.map((stage) => {
        if (stage.id !== stageId) return stage;
        changed = true;
        return {
          ...stage,
          ...patch,
          id: stage.id,
          metadata: markDraftMetadata(
            {
              ...stage.metadata,
              ...(patch.metadata ?? {}),
            },
            reason
          ),
        };
      });
      if (!changed) return {};
      return {
        workspace: {
          ...state.workspace,
          stages,
          metadata: markDraftMetadata(state.workspace.metadata, reason),
        },
      };
    }),

  updateScenarioDraft: (scenarioId, patch, reason = 'scenario_draft_updated') =>
    set((state) => {
      if (!state.workspace) return {};
      const scenario = state.workspace.scenarios[scenarioId];
      if (!scenario) return {};
      return {
        workspace: {
          ...state.workspace,
          scenarios: {
            ...state.workspace.scenarios,
            [scenarioId]: {
              ...scenario,
              ...patch,
              id: scenario.id,
              metadata: markDraftMetadata(
                {
                  ...scenario.metadata,
                  ...(patch.metadata ?? {}),
                },
                reason
              ),
            },
          },
          metadata: markDraftMetadata(state.workspace.metadata, reason),
        },
      };
    }),

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
              objective_spec: objectiveSpecFromLossSpec(trainingSpec.loss),
              metadata: markDraftMetadata(
                {
                  ...scenario.metadata,
                  draft_owner: 'studio_workspace',
                },
                'training_spec_updated'
              ),
            },
          },
          metadata: markDraftMetadata(state.workspace.metadata, 'training_spec_updated'),
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
              metadata: markDraftMetadata(
                {
                  ...scenario.metadata,
                  draft_owner: 'studio_workspace',
                },
                'task_spec_updated'
              ),
            },
          },
          metadata: markDraftMetadata(state.workspace.metadata, 'task_spec_updated'),
        },
      };
    }),

  updateActiveScenarioTaskBindingSpec: (taskBindingSpec) =>
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
              task_binding_spec: taskBindingSpec,
              metadata: markDraftMetadata(
                {
                  ...scenario.metadata,
                  draft_owner: 'studio_workspace',
                },
                'task_binding_spec_updated'
              ),
            },
          },
          metadata: markDraftMetadata(state.workspace.metadata, 'task_binding_spec_updated'),
        },
      };
    }),

  retargetActiveScenarioTaskBindingsForNodeRename: (previousNodeId, nextNodeId) =>
    set((state) => {
      const scenarioId = activeTrainScenario(state.workspace);
      if (!state.workspace || !scenarioId) return {};
      const scenario = state.workspace.scenarios[scenarioId];
      if (!scenario?.task_binding_spec) return {};
      const taskBindingSpec = retargetTaskBindingsForNodeRename(
        scenario.task_binding_spec,
        previousNodeId,
        nextNodeId
      );
      if (taskBindingSpec === scenario.task_binding_spec) return {};
      return {
        workspace: {
          ...state.workspace,
          scenarios: {
            ...state.workspace.scenarios,
            [scenarioId]: {
              ...scenario,
              task_binding_spec: taskBindingSpec,
              metadata: markDraftMetadata(
                {
                  ...scenario.metadata,
                  draft_owner: 'studio_workspace',
                },
                'task_binding_target_renamed'
              ),
            },
          },
          metadata: markDraftMetadata(state.workspace.metadata, 'task_binding_target_renamed'),
        },
      };
    }),

  updateActiveScenarioObjectiveSpec: (objectiveSpec) =>
    set((state) => {
      const scenarioId = activeScenarioId(state.workspace);
      if (!state.workspace || !scenarioId) return {};
      const scenario = state.workspace.scenarios[scenarioId];
      if (!scenario) return {};
      const trainingSpec = scenario.training_spec
        ? {
            ...scenario.training_spec,
            loss: lossSpecFromObjectiveSpec(objectiveSpec),
          }
        : scenario.training_spec;
      return {
        workspace: {
          ...state.workspace,
          scenarios: {
            ...state.workspace.scenarios,
            [scenarioId]: {
              ...scenario,
              training_spec: trainingSpec,
              objective_spec: objectiveSpec,
              metadata: markDraftMetadata(
                {
                  ...scenario.metadata,
                  draft_owner: 'studio_workspace',
                },
                'objective_spec_updated'
              ),
            },
          },
          metadata: markDraftMetadata(state.workspace.metadata, 'objective_spec_updated'),
        },
      };
    }),

  setTopPaneProjection: (projection) =>
    set((state) => {
      if (!state.workspace) return {};
      return {
        workspace: updateTopPaneState(
          state.workspace,
          { active_projection: projection },
          'top_pane_projection_changed'
        ),
      };
    }),

  selectTopPaneEntity: (entityId, reason = 'top_pane_selection_changed') =>
    set((state) => {
      if (!state.workspace) return {};
      return {
        workspace: updateTopPaneState(
          state.workspace,
          { selected_entity_id: entityId, hovered_entity_id: null },
          reason
        ),
      };
    }),

  hoverTopPaneEntity: (entityId) =>
    set((state) => {
      if (!state.workspace) return {};
      return {
        workspace: updateTopPaneState(
          state.workspace,
          { hovered_entity_id: entityId },
          'top_pane_hover_changed',
          false
        ),
      };
    }),

  updateStageCollections: (stageId, collections, reason = 'stage_collections_updated') =>
    set((state) => {
      if (!state.workspace) return {};
      let changed = false;
      const stages = state.workspace.stages.map((stage) => {
        if (stage.id !== stageId) return stage;
        changed = true;
        return {
          ...stage,
          input_collections: collections.input_collections ?? stage.input_collections,
          output_collections: collections.output_collections ?? stage.output_collections,
          metadata: markDraftMetadata(stage.metadata, reason),
        };
      });
      if (!changed) return {};
      return {
        workspace: {
          ...state.workspace,
          stages,
          metadata: markDraftMetadata(state.workspace.metadata, reason),
        },
      };
    }),
}));
