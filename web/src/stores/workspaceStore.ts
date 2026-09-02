import { create } from 'zustand';
import {
  frozenSnapshotProjectionFromWorkspace,
  useSelectionContextStore,
} from '@/stores/selectionContextStore';
import {
  lossSpecFromObjectiveSpec,
  selectorWithSubpath,
} from '@/features/scenario/objectives';
import {
  createDefaultTaskBindingSpec,
  ensureTaskBindingSpec,
  retargetTaskBindingsForNodeRename,
  retargetTaskBindingsForNodePortRename,
} from '@/features/scenario/taskBindings';
import { WORKSPACE_VIEW_STATE_SCHEMA_VERSION } from '@/types/workspace';
import type { AnalysisSnapshot } from '@/types/analysis';
import { isCausalGraphSpec, type GraphSpec, type GraphUIState } from '@/types/graph';
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
  StudioTrainingExecutionPreparation,
  StudioValidationState,
  StudioWorkspaceSpec,
  WorkspaceViewMode,
  WorkspaceViewState,
} from '@/types/workspace';
import type { WorkspaceDocument } from '@/generated/studioContracts';
import { useGraphStore } from '@/stores/graphStore';

export function buildWorkspaceDocumentSnapshot(
  document: WorkspaceDocument | null,
  graphUiState: GraphUIState,
  analysisSnapshot: AnalysisSnapshot | null,
  workspace: StudioWorkspaceSpec | null,
): WorkspaceDocument {
  if (!document) {
    throw new Error('WorkspaceDocument is required to save an existing semantic graph.');
  }
  return {
    ...document,
    graph_ui_state: graphUiState,
    workspace_ui_state: workspace?.ui_state ?? {},
    stage_ui_state: Object.fromEntries(
      (workspace?.stages ?? []).map((stage) => [stage.id, stage.ui_state])
    ),
    scenario_ui_state: Object.fromEntries(
      Object.entries(workspace?.scenarios ?? {}).map(([id, scenario]) => [id, scenario.ui_state])
    ),
    analysis_pages: analysisPagesFromSnapshot(
      analysisSnapshot
    ) as unknown as WorkspaceDocument['analysis_pages'],
    active_analysis_page_id: analysisSnapshot?.activePageId ?? null,
  };
}

export function buildNewWorkspaceDocumentSnapshot(
  graphUiState: GraphUIState,
  analysisSnapshot: AnalysisSnapshot | null,
  workspace: StudioWorkspaceSpec | null,
): WorkspaceDocument {
  return buildWorkspaceDocumentSnapshot(
    {
      schema_id: 'feedbax.workspace_document',
      schema_version: '1',
      semantic_root: {
        semantic_document_sha256: '0'.repeat(64),
        authored_path: '/graph',
      },
      semantic_anchors: {},
    },
    graphUiState,
    analysisSnapshot,
    workspace,
  );
}

export function hydrateWorkspacePresentation(
  workspace: StudioWorkspaceSpec | null,
  document: WorkspaceDocument,
): StudioWorkspaceSpec | null {
  if (!workspace) return null;
  return {
    ...workspace,
    ui_state: document.workspace_ui_state ?? {},
    stages: workspace.stages.map((stage) => ({
      ...stage,
      ui_state: document.stage_ui_state?.[stage.id] ?? {},
    })),
    scenarios: Object.fromEntries(
      Object.entries(workspace.scenarios).map(([id, scenario]) => [
        id,
        { ...scenario, ui_state: document.scenario_ui_state?.[id] ?? {} },
      ])
    ),
  };
}

const WORKSPACE_SCHEMA_VERSION = 'feedbax.spec.studio.workspace.v2';
const WORKSPACE_SCHEMA_ID = 'feedbax.spec.studio.workspace';
const STAGE_SCHEMA_ID = 'feedbax.spec.studio.stage';
const STAGE_SCHEMA_VERSION = 'feedbax.spec.studio.stage.v2';
const SCENARIO_SCHEMA_VERSION = 'feedbax.spec.studio.scenario.v3';
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

const WORKSPACE_VIEW_STATE_KEY = 'workspace_view_state';

const DEFAULT_WORKSPACE_VIEW_STATE: WorkspaceViewState = {
  schema_version: WORKSPACE_VIEW_STATE_SCHEMA_VERSION,
  camera: { zoom: 1, pan: { x: 0, y: 0 } },
  selected_artifact_ref: null,
  selected_trial_ref: null,
  overlay_visibility: {
    mechanics: true,
    task: true,
    objectives: true,
    observables: true,
    artifacts: true,
    trials: true,
    comparisons: true,
  },
  playback: { position: 0, speed: 1 },
  comparison_selection: { baseline_ref: null, candidate_ref: null },
};

type WorkspaceViewStatePatch = Partial<
  Omit<WorkspaceViewState, 'camera' | 'playback' | 'comparison_selection'>
> & {
  camera?: Partial<Omit<WorkspaceViewState['camera'], 'pan'>> & {
    pan?: Partial<WorkspaceViewState['camera']['pan']>;
  };
  playback?: Partial<WorkspaceViewState['playback']>;
  comparison_selection?: Partial<WorkspaceViewState['comparison_selection']>;
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
    record.active_projection === 'observables' ||
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

function numberOrDefault(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function stringRefOrNull(value: unknown, availableRefs: Set<string> | null): string | null {
  if (typeof value !== 'string' || value.length === 0) return null;
  if (availableRefs && !availableRefs.has(value)) return null;
  return value;
}

function normalizeWorkspaceViewState(
  value: unknown,
  availableRefs: Set<string> | null = null
): WorkspaceViewState {
  const record = isRecord(value) ? value : {};
  if (
    record.schema_version !== WORKSPACE_VIEW_STATE_SCHEMA_VERSION &&
    record.schema_version !== undefined
  ) {
    return DEFAULT_WORKSPACE_VIEW_STATE;
  }
  const camera = isRecord(record.camera) ? record.camera : {};
  const pan = isRecord(camera.pan) ? camera.pan : {};
  const playback = isRecord(record.playback) ? record.playback : {};
  const comparison = isRecord(record.comparison_selection) ? record.comparison_selection : {};
  const overlayVisibility = isRecord(record.overlay_visibility)
    ? record.overlay_visibility
    : {};
  const normalizedOverlayVisibility = Object.entries(overlayVisibility).reduce<
    Record<string, boolean>
  >((result, [key, visible]) => {
    if (typeof visible === 'boolean') result[key] = visible;
    return result;
  }, {});
  return {
    schema_version: WORKSPACE_VIEW_STATE_SCHEMA_VERSION,
    camera: {
      zoom: Math.max(0.35, Math.min(8, numberOrDefault(camera.zoom, 1))),
      pan: {
        x: numberOrDefault(pan.x, 0),
        y: numberOrDefault(pan.y, 0),
      },
    },
    selected_artifact_ref: stringRefOrNull(record.selected_artifact_ref, availableRefs),
    selected_trial_ref: stringRefOrNull(record.selected_trial_ref, null),
    overlay_visibility: {
      ...DEFAULT_WORKSPACE_VIEW_STATE.overlay_visibility,
      ...normalizedOverlayVisibility,
    },
    playback: {
      position: Math.max(0, numberOrDefault(playback.position, 0)),
      speed: Math.max(0.1, Math.min(16, numberOrDefault(playback.speed, 1))),
    },
    comparison_selection: {
      baseline_ref: stringRefOrNull(comparison.baseline_ref, availableRefs),
      candidate_ref: stringRefOrNull(comparison.candidate_ref, availableRefs),
    },
  };
}

function collectRefIdsFromRef(value: unknown, refs: Set<string>) {
  if (!isRecord(value)) return;
  for (const key of ['id', 'uri']) {
    const ref = value[key];
    if (typeof ref === 'string' && ref.length > 0) refs.add(ref);
  }
}

function availableWorkspaceViewRefs(
  workspace: StudioWorkspaceSpec | null | undefined,
  stage: StudioStageSpec | null | undefined
): Set<string> {
  const refs = new Set<string>();
  for (const ref of workspace?.manifest_refs ?? []) collectRefIdsFromRef(ref, refs);
  for (const ref of workspace?.artifact_refs ?? []) collectRefIdsFromRef(ref, refs);
  for (const collection of workspace?.collections ?? []) {
    refs.add(collection.id);
    for (const ref of collection.item_refs) collectRefIdsFromRef(ref, refs);
  }
  if (stage) {
    for (const ref of stage.manifest_refs) collectRefIdsFromRef(ref, refs);
    for (const ref of stage.artifact_refs ?? []) collectRefIdsFromRef(ref, refs);
    for (const collection of [...stage.input_collections, ...stage.output_collections]) {
      refs.add(collection.id);
      for (const ref of collection.item_refs) collectRefIdsFromRef(ref, refs);
    }
  }
  return refs;
}

function mergeWorkspaceViewStatePatch(
  current: WorkspaceViewState,
  patch: WorkspaceViewStatePatch,
  availableRefs: Set<string> | null
): WorkspaceViewState {
  return normalizeWorkspaceViewState(
    {
      ...current,
      ...patch,
      camera: {
        ...current.camera,
        ...(patch.camera ?? {}),
        pan: {
          ...current.camera.pan,
          ...(patch.camera?.pan ?? {}),
        },
      },
      overlay_visibility: {
        ...current.overlay_visibility,
        ...(patch.overlay_visibility ?? {}),
      },
      playback: {
        ...current.playback,
        ...(patch.playback ?? {}),
      },
      comparison_selection: {
        ...current.comparison_selection,
        ...(patch.comparison_selection ?? {}),
      },
    },
    availableRefs
  );
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
    schema_id: STAGE_SCHEMA_ID,
    schema_version: STAGE_SCHEMA_VERSION,
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

function stageForScenario(
  workspace: StudioWorkspaceSpec,
  scenario: StudioScenarioSpec
): StudioStageSpec | null {
  return (
    workspace.stages.find((stage) => stage.scenario_id === scenario.id) ??
    workspace.stages.find((stage) => stage.id === scenario.stage_id) ??
    null
  );
}

function normalizeWorkspaceViewStates(
  workspace: StudioWorkspaceSpec | null
): StudioWorkspaceSpec | null {
  if (!workspace) return workspace;
  let changed = false;
  const activeStage = getActiveStage(workspace);
  const workspaceRefs = availableWorkspaceViewRefs(workspace, activeStage);
  const workspaceViewState = normalizeWorkspaceViewState(
    workspace.ui_state[WORKSPACE_VIEW_STATE_KEY],
    workspaceRefs
  );
  const uiState = {
    ...workspace.ui_state,
    [WORKSPACE_VIEW_STATE_KEY]: workspaceViewState,
  };
  if (workspace.ui_state[WORKSPACE_VIEW_STATE_KEY] !== workspaceViewState) changed = true;

  const scenarios = Object.fromEntries(
    Object.entries(workspace.scenarios).map(([scenarioId, scenario]) => {
      const stage = stageForScenario(workspace, scenario);
      const refs = availableWorkspaceViewRefs(workspace, stage);
      const viewState = normalizeWorkspaceViewState(
        scenario.ui_state[WORKSPACE_VIEW_STATE_KEY],
        refs
      );
      if (scenario.ui_state[WORKSPACE_VIEW_STATE_KEY] === viewState) {
        return [scenarioId, scenario];
      }
      changed = true;
      return [
        scenarioId,
        {
          ...scenario,
          ui_state: {
            ...scenario.ui_state,
            [WORKSPACE_VIEW_STATE_KEY]: viewState,
          },
        },
      ];
    })
  );

  return changed ? { ...workspace, ui_state: uiState, scenarios } : workspace;
}

function normalizeWorkspaceForStudioState(
  workspace: StudioWorkspaceSpec | null
): StudioWorkspaceSpec | null {
  return normalizeWorkspaceViewStates(workspace);
}

function analysisPagesFromSnapshot(snapshot: AnalysisSnapshot | null): AnalysisPageWire[] {
  if (!snapshot || snapshot.pages.length === 0) return [];
  return snapshot.pages.map((page) => ({
    id: page.id,
    name: page.name,
    graph_spec: page.graphSpec as unknown as Record<string, unknown>,
    input_requirements: page.inputRequirements ?? [],
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

function assertGraphUiStateConsistency(
  graph: GraphSpec,
  uiState: GraphUIState | null,
  path = 'graph'
): void {
  if (!uiState) return;

  for (const nodeId of Object.keys(uiState.node_states ?? {})) {
    if (!graph.nodes[nodeId]) {
      throw new Error(
        `Cannot build workspace snapshot: ${path} UI state references missing node "${nodeId}".`
      );
    }
  }

  for (const [nodeId, subgraphUiState] of Object.entries(uiState.subgraph_states ?? {})) {
    const subgraph = graph.subgraphs?.[nodeId];
    if (!subgraph) {
      throw new Error(
        `Cannot build workspace snapshot: ${path} UI state references missing subgraph "${nodeId}".`
      );
    }
    if (!isCausalGraphSpec(subgraph)) continue;
    assertGraphUiStateConsistency(subgraph, subgraphUiState, `${path}.subgraphs.${nodeId}`);
  }
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
  graphStackPath,
}: {
  workspace: StudioWorkspaceSpec | null;
  graph: GraphSpec;
  uiState: GraphUIState | null;
  trainingSpec: TrainingSpec;
  taskSpec: TaskSpec;
  analysisSnapshot: AnalysisSnapshot | null;
  projectName?: string;
  graphStackPath?: string[] | null;
}): StudioWorkspaceSpec {
  assertGraphUiStateConsistency(graph, uiState);

  const base: StudioWorkspaceSpec =
    workspace ??
    {
      id: generateId('studio-workspace'),
      schema_id: WORKSPACE_SCHEMA_ID,
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

  return normalizeWorkspaceForStudioState({
    ...withStages,
    label: withStages.label || projectName || graph.metadata?.name || 'Studio workspace',
    active_stage_id: withStages.active_stage_id ?? trainStage?.id ?? null,
    ui_state: {
      ...withStages.ui_state,
      graph_stack_path:
        graphStackPath === undefined
          ? withStages.ui_state.graph_stack_path ?? []
          : graphStackPath ?? [],
    },
    scenarios,
  })!;
}

interface WorkspaceStoreState {
  workspace: StudioWorkspaceSpec | null;
  workspaceDocument: WorkspaceDocument | null;
  lastTrainingExecutionPreparation: StudioTrainingExecutionPreparation | null;
  lastPipelineMaterializationResult: StudioPipelineMaterializationResult | null;
  setWorkspace: (workspace: StudioWorkspaceSpec | null) => void;
  restoreWorkspace: (workspace: StudioWorkspaceSpec | null) => void;
  setWorkspaceDocument: (document: WorkspaceDocument | null) => void;
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
  setPipelineMaterializationResult: (
    result: StudioPipelineMaterializationResult | null
  ) => void;
  updateActiveScenarioTrainingSpec: (trainingSpec: TrainingSpec) => void;
  updateActiveScenarioTaskSpec: (taskSpec: TaskSpec) => void;
  updateActiveScenarioTaskBindingSpec: (taskBindingSpec: StudioTaskBindingSpec) => void;
  retargetActiveScenarioTaskBindingsForNodeRename: (
    previousNodeId: string,
    nextNodeId: string,
    graphPath?: string[] | null
  ) => void;
  retargetActiveScenarioTaskBindingsForNodePortRename: (
    nodeId: string,
    previousPort: string,
    nextPort: string,
    graphPath?: string[] | null
  ) => void;
  updateActiveScenarioObjectiveSpec: (objectiveSpec: StudioObjectiveSpec) => void;
  updateActiveWorkspaceViewState: (
    patch: WorkspaceViewStatePatch,
    reason?: string
  ) => void;
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

export function getProjectedScenario(
  workspace: StudioWorkspaceSpec | null,
  stage: StudioStageSpec | null | undefined
): StudioScenarioSpec | null {
  const scenario = getScenario(workspace, stage?.scenario_id);
  if (!workspace || !scenario) return scenario;
  const parent = getScenario(workspace, scenario.parent_scenario_id);
  if (!parent) return scenario;
  const inheritTrainingObjectives = stage?.kind === 'train';
  return {
    ...parent,
    ...scenario,
    id: scenario.id,
    label: scenario.label,
    stage_id: scenario.stage_id ?? stage?.id ?? null,
    parent_scenario_id: scenario.parent_scenario_id,
    task_spec: scenario.task_spec ?? parent.task_spec,
    task_binding_spec: scenario.task_binding_spec ?? parent.task_binding_spec,
    probe_specs:
      (scenario.probe_specs ?? []).length > 0 ? scenario.probe_specs : parent.probe_specs,
    biomechanics_spec: scenario.biomechanics_spec ?? parent.biomechanics_spec,
    training_spec: inheritTrainingObjectives
      ? scenario.training_spec ?? parent.training_spec
      : scenario.training_spec ?? null,
    objective_spec: inheritTrainingObjectives
      ? scenario.objective_spec ?? parent.objective_spec
      : scenario.objective_spec ?? null,
    ui_state: {
      ...parent.ui_state,
      ...scenario.ui_state,
    },
    metadata: {
      ...parent.metadata,
      ...scenario.metadata,
      projected_from_parent_scenario_id: parent.id,
      projected_for_stage_kind: stage?.kind ?? null,
    },
  };
}

export function getActiveScenario(workspace: StudioWorkspaceSpec | null): StudioScenarioSpec | null {
  return getProjectedScenario(workspace, getActiveStage(workspace));
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

export function getWorkspaceViewState(
  workspace: StudioWorkspaceSpec | null | undefined,
  stage: StudioStageSpec | null | undefined = getActiveStage(workspace ?? null)
): WorkspaceViewState {
  if (!workspace) return DEFAULT_WORKSPACE_VIEW_STATE;
  const scenario = getScenario(workspace, stage?.scenario_id);
  const availableRefs = availableWorkspaceViewRefs(workspace, stage);
  return normalizeWorkspaceViewState(
    scenario?.ui_state[WORKSPACE_VIEW_STATE_KEY] ??
      workspace.ui_state[WORKSPACE_VIEW_STATE_KEY],
    availableRefs
  );
}

export function getWorkspaceViewMode(
  workspace: StudioWorkspaceSpec | null | undefined,
  stage: StudioStageSpec | null | undefined = getActiveStage(workspace ?? null)
): WorkspaceViewMode {
  if (!workspace || !stage) return 'authoring';
  const viewState = getWorkspaceViewState(workspace, stage);
  if (stage.kind === 'compare' || viewState.comparison_selection.baseline_ref) {
    return 'comparison';
  }
  if (viewState.selected_trial_ref || viewState.playback.position > 0) return 'playback';
  if (
    viewState.selected_artifact_ref ||
    stage.manifest_refs.length > 0 ||
    (stage.artifact_refs?.length ?? 0) > 0
  ) {
    return 'artifact';
  }
  return 'authoring';
}

export const useWorkspaceStore = create<WorkspaceStoreState>((baseSet) => {
  const set: typeof baseSet = (partial, replace) => {
    let changedPersistedWorkspace = false;
    baseSet((state) => {
      const patch = typeof partial === 'function' ? partial(state) : partial;
      changedPersistedWorkspace = Boolean(
        patch &&
        typeof patch === 'object' &&
        patch !== state &&
        'workspace' in patch &&
        patch.workspace !== state.workspace
      );
      return patch;
    }, replace as false);
    if (changedPersistedWorkspace) useGraphStore.getState().markDirty();
  };
  const replaceWorkspace = (
    workspace: StudioWorkspaceSpec | null,
    setter: typeof baseSet,
  ) => {
    const normalizedWorkspace = normalizeWorkspaceForStudioState(workspace);
    useSelectionContextStore
      .getState()
      .setFrozenSnapshot(frozenSnapshotProjectionFromWorkspace(normalizedWorkspace));
    setter({ workspace: normalizedWorkspace });
  };

  return {
  workspace: null,
  workspaceDocument: null,
  lastTrainingExecutionPreparation: null,
  lastPipelineMaterializationResult: null,

  setWorkspace: (workspace) => replaceWorkspace(workspace, set),
  restoreWorkspace: (workspace) => replaceWorkspace(workspace, baseSet),
  setWorkspaceDocument: (workspaceDocument) => baseSet({ workspaceDocument }),

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

  retargetActiveScenarioTaskBindingsForNodeRename: (previousNodeId, nextNodeId, graphPath) =>
    set((state) => {
      const scenarioId = activeTrainScenario(state.workspace);
      if (!state.workspace || !scenarioId) return {};
      const scenario = state.workspace.scenarios[scenarioId];
      if (!scenario?.task_binding_spec) return {};
      const taskBindingSpec = retargetTaskBindingsForNodeRename(
        scenario.task_binding_spec,
        previousNodeId,
        nextNodeId,
        graphPath
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

  retargetActiveScenarioTaskBindingsForNodePortRename: (nodeId, previousPort, nextPort, graphPath) =>
    set((state) => {
      const scenarioId = activeTrainScenario(state.workspace);
      if (!state.workspace || !scenarioId) return {};
      const scenario = state.workspace.scenarios[scenarioId];
      if (!scenario?.task_binding_spec) return {};
      const taskBindingSpec = retargetTaskBindingsForNodePortRename(
        scenario.task_binding_spec,
        nodeId,
        previousPort,
        nextPort,
        graphPath
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
                'task_binding_target_port_renamed'
              ),
            },
          },
          metadata: markDraftMetadata(state.workspace.metadata, 'task_binding_target_port_renamed'),
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

  updateActiveWorkspaceViewState: (patch, reason = 'workspace_view_state_updated') =>
    set((state) => {
      if (!state.workspace) return {};
      const activeStage = getActiveStage(state.workspace);
      const scenario = getScenario(state.workspace, activeStage?.scenario_id);
      const refs = availableWorkspaceViewRefs(state.workspace, activeStage);
      const current = getWorkspaceViewState(state.workspace, activeStage);
      const nextViewState = mergeWorkspaceViewStatePatch(current, patch, refs);
      if (scenario) {
        return {
          workspace: {
            ...state.workspace,
            scenarios: {
              ...state.workspace.scenarios,
              [scenario.id]: {
                ...scenario,
                ui_state: {
                  ...scenario.ui_state,
                  [WORKSPACE_VIEW_STATE_KEY]: nextViewState,
                },
                metadata: markDraftMetadata(scenario.metadata, reason),
              },
            },
            metadata: markDraftMetadata(state.workspace.metadata, reason),
          },
        };
      }
      return {
        workspace: {
          ...state.workspace,
          ui_state: {
            ...state.workspace.ui_state,
            [WORKSPACE_VIEW_STATE_KEY]: nextViewState,
          },
          metadata: markDraftMetadata(state.workspace.metadata, reason),
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
  };
});
