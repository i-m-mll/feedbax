import { create } from 'zustand';
import type { TrainingRun, EvalRun, LegacyCheckpointInfo } from '@/types/runs';
import { fetchTrainingRuns, fetchEvalRuns } from '@/api/runAPI';
import { apiErrorMessage } from '@/api/request';
import { withStoreActionFeedback } from '@/stores/storeActions';
import {
  type SelectionContext,
  useSelectionContextStore,
} from '@/stores/selectionContextStore';
import { getStageByKind, useWorkspaceStore } from '@/stores/workspaceStore';
import type { StudioCollectionRef, StudioManifestRef, StudioWorkspaceSpec } from '@/types/workspace';

const TRAINING_COLLECTION_ID = 'collection:training-runs';
const SELECTED_TRAINING_COLLECTION_ID = 'collection:selected-training-runs';
const EVALUATION_COLLECTION_ID = 'collection:evaluation-runs';
const SELECTED_EVALUATION_COLLECTION_ID = 'collection:selected-evaluation-runs';

function manifestRefForTrainingRun(run: TrainingRun): StudioManifestRef {
  return {
    kind: run.uri ? 'TrainingRunManifest' : 'TrainingRun',
    id: run.id,
    role: 'training_run',
    provider: 'feedbax',
    uri: run.uri ?? null,
    metadata: {
      name: run.name,
      status: run.status,
      created_at: run.createdAt,
      hyperparams: run.hyperparams,
      metrics: run.metrics ?? {},
      stage_id: run.stageId ?? null,
      scenario_id: run.scenarioId ?? null,
      planned: run.planned ?? false,
      checkpoint_available: run.checkpointAvailable ?? false,
      source_issue: run.sourceIssue ?? null,
      provenance_id: run.provenanceId ?? run.id,
      superseded_by: run.supersededBy ?? null,
      legacy_checkpoint: run.legacyCheckpoint ?? null,
      legacy_run_record: !run.uri,
      ...run.hyperparams,
      ...(run.metrics ?? {}),
    },
  };
}

function manifestRefForEvalRun(run: EvalRun): StudioManifestRef {
  return {
    kind: run.uri ? 'EvaluationRunManifest' : 'EvaluationRun',
    id: run.id,
    role: 'evaluation_run',
    provider: 'feedbax',
    uri: run.uri ?? null,
    metadata: {
      name: run.name,
      status: run.status,
      created_at: run.createdAt,
      training_run_id: run.trainingRunId,
      training_run_ids: run.trainingRunIds ?? [run.trainingRunId],
      description: run.description ?? null,
      legacy_run_record: !run.uri,
    },
  };
}

function collectionFromRefs(
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

function selectionIds(stageSelection: Record<string, unknown>, key: string): string[] {
  const value = stageSelection[key];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function manifestRefsFromCollections(collections: StudioCollectionRef[]): StudioManifestRef[] {
  const refs = new Map<string, StudioManifestRef>();
  for (const collection of collections) {
    for (const ref of collection.item_refs) {
      refs.set(ref.id, ref);
    }
  }
  return Array.from(refs.values());
}

function runStatus(value: unknown): TrainingRun['status'] {
  return value === 'pending' ||
    value === 'running' ||
    value === 'completed' ||
    value === 'failed' ||
    value === 'cancelled' ||
    value === 'stopped'
    ? value
    : 'completed';
}

function evalStatus(value: unknown): EvalRun['status'] {
  return value === 'pending' ||
    value === 'running' ||
    value === 'completed' ||
    value === 'failed' ||
    value === 'cancelled'
    ? value
    : 'completed';
}

function displayNameFromRef(ref: StudioManifestRef): string {
  return typeof ref.metadata.name === 'string' ? ref.metadata.name : ref.id;
}

function createdAtFromRef(ref: StudioManifestRef): string {
  return typeof ref.metadata.created_at === 'string' ? ref.metadata.created_at : '';
}

function hyperparamsFromRef(ref: StudioManifestRef): Record<string, string | number> {
  const hyperparams = ref.metadata.hyperparams;
  if (hyperparams && typeof hyperparams === 'object' && !Array.isArray(hyperparams)) {
    return Object.fromEntries(
      Object.entries(hyperparams as Record<string, unknown>).filter(
        (entry): entry is [string, string | number] =>
          typeof entry[1] === 'string' || typeof entry[1] === 'number'
      )
    );
  }

  const keys = [
    'hidden_type',
    'n_replicates',
    'n_warmup_batches',
    'batch_size',
    'ramp_shape',
    'ramp_duration_steps',
    'nn_output_pre_go',
    'final_validation_loss',
  ];
  return Object.fromEntries(
    keys.flatMap((key) => {
      const value = ref.metadata[key];
      return typeof value === 'string' || typeof value === 'number' ? [[key, value]] : [];
    })
  );
}

function metricsFromRef(ref: StudioManifestRef): Record<string, unknown> {
  const metrics = ref.metadata.metrics;
  return metrics && typeof metrics === 'object' && !Array.isArray(metrics)
    ? { ...(metrics as Record<string, unknown>) }
    : {};
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function optionalBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined;
}

function legacyCheckpointFromMetadata(value: unknown): LegacyCheckpointInfo | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return undefined;
  }
  const payload = value as Record<string, unknown>;
  if (
    typeof payload.layout_id !== 'string' ||
    typeof payload.layout_name !== 'string' ||
    typeof payload.message !== 'string'
  ) {
    return undefined;
  }
  return {
    layout_id: payload.layout_id,
    layout_name: payload.layout_name,
    message: payload.message,
    docs: optionalString(payload.docs),
    adoption_entrypoint: optionalString(payload.adoption_entrypoint),
  };
}

function trainingRunFromManifestRef(ref: StudioManifestRef): TrainingRun {
  return {
    id: ref.id,
    name: displayNameFromRef(ref),
    createdAt: createdAtFromRef(ref),
    status: runStatus(ref.metadata.status),
    hyperparams: hyperparamsFromRef(ref),
    metrics: metricsFromRef(ref),
    uri: ref.uri ?? undefined,
    stageId: optionalString(ref.metadata.stage_id),
    scenarioId: optionalString(ref.metadata.scenario_id),
    planned: optionalBoolean(ref.metadata.planned),
    checkpointAvailable: optionalBoolean(ref.metadata.checkpoint_available),
    sourceIssue: optionalString(ref.metadata.source_issue),
    provenanceId: optionalString(ref.metadata.provenance_id) ?? ref.id,
    supersededBy: optionalString(ref.metadata.superseded_by),
    legacyCheckpoint: legacyCheckpointFromMetadata(ref.metadata.legacy_checkpoint),
  };
}

function evalRunFromManifestRef(ref: StudioManifestRef): EvalRun {
  const trainingRunId =
    typeof ref.metadata.training_run_id === 'string'
      ? ref.metadata.training_run_id
      : Array.isArray(ref.metadata.training_run_ids) &&
          typeof ref.metadata.training_run_ids[0] === 'string'
        ? ref.metadata.training_run_ids[0]
        : '';
  return {
    id: ref.id,
    trainingRunId,
    name: displayNameFromRef(ref),
    createdAt: createdAtFromRef(ref),
    status: evalStatus(ref.metadata.status),
    description:
      typeof ref.metadata.description === 'string'
        ? ref.metadata.description
        : typeof ref.metadata.name === 'string'
          ? ref.metadata.name
          : undefined,
    trainingRunIds: Array.isArray(ref.metadata.training_run_ids)
      ? ref.metadata.training_run_ids.filter((item): item is string => typeof item === 'string')
      : trainingRunId
        ? [trainingRunId]
        : [],
    uri: ref.uri ?? undefined,
  };
}

export function selectedTrainingRunIdFromWorkspace(): string | null {
  const workspace = useWorkspaceStore.getState().workspace;
  const evalStage = getStageByKind(workspace, 'eval');
  return selectionIds(evalStage?.selection_spec ?? {}, 'training_run_ids')[0] ?? null;
}

export function selectedEvalRunIdFromWorkspace(): string | null {
  const workspace = useWorkspaceStore.getState().workspace;
  const analysisStage = getStageByKind(workspace, 'analysis');
  return selectionIds(analysisStage?.selection_spec ?? {}, 'eval_run_ids')[0] ?? null;
}

function writeTrainingRunsToWorkspace(runs: TrainingRun[]) {
  const workspaceStore = useWorkspaceStore.getState();
  const workspace = workspaceStore.workspace;
  const trainStage = getStageByKind(workspace, 'train');
  if (!workspace || !trainStage) return;
  workspaceStore.updateStageCollections(
    trainStage.id,
    {
      output_collections: [
        collectionFromRefs(
          TRAINING_COLLECTION_ID,
          'training_runs',
          'Training runs',
          trainStage.id,
          runs.map(manifestRefForTrainingRun),
          { populated_from: 'run_store' }
        ),
      ],
    },
    'training_collection_indexed'
  );
}

function writeSelectedTrainingRunToWorkspace(run: TrainingRun | null) {
  const workspaceStore = useWorkspaceStore.getState();
  const workspace = workspaceStore.workspace;
  const trainStage = getStageByKind(workspace, 'train');
  const evalStage = getStageByKind(workspace, 'eval');
  if (!workspace || !trainStage || !evalStage) return;
  const itemRefs = run ? [manifestRefForTrainingRun(run)] : [];
  workspaceStore.updateStageCollections(
    evalStage.id,
    {
      input_collections: [
        collectionFromRefs(
          SELECTED_TRAINING_COLLECTION_ID,
          'training_runs',
          'Selected training runs',
          trainStage.id,
          itemRefs,
          { selected_for_stage_id: evalStage.id }
        ),
      ],
    },
    'eval_input_collection_selected'
  );
  workspaceStore.updateStageDraft(
    evalStage.id,
    {
      selection_spec: {
        ...evalStage.selection_spec,
        source_collection_id: TRAINING_COLLECTION_ID,
        training_run_ids: run ? [run.id] : [],
      },
    },
    'eval_input_collection_selected'
  );
  useSelectionContextStore.getState().setContext({
    stage: evalStage.id,
    collection: SELECTED_TRAINING_COLLECTION_ID,
    selectedIds: run ? [run.id] : [],
    focusedId: run?.id ?? null,
  });
}

function writeEvalRunsToWorkspace(runs: EvalRun[]) {
  const workspaceStore = useWorkspaceStore.getState();
  const workspace = workspaceStore.workspace;
  const evalStage = getStageByKind(workspace, 'eval');
  if (!workspace || !evalStage) return;
  workspaceStore.updateStageCollections(
    evalStage.id,
    {
      output_collections: [
        collectionFromRefs(
          EVALUATION_COLLECTION_ID,
          'evaluation_runs',
          'Evaluation runs',
          evalStage.id,
          runs.map(manifestRefForEvalRun),
          { populated_from: 'run_store' }
        ),
      ],
    },
    'evaluation_collection_indexed'
  );
}

function writeSelectedEvalRunToWorkspace(run: EvalRun | null) {
  const workspaceStore = useWorkspaceStore.getState();
  const workspace = workspaceStore.workspace;
  const evalStage = getStageByKind(workspace, 'eval');
  const analysisStage = getStageByKind(workspace, 'analysis');
  if (!workspace || !evalStage || !analysisStage) return;
  const itemRefs = run ? [manifestRefForEvalRun(run)] : [];
  const inputCollections = [
    collectionFromRefs(
      SELECTED_EVALUATION_COLLECTION_ID,
      'evaluation_runs',
      'Selected evaluation runs',
      evalStage.id,
      itemRefs,
      { selected_for_stage_id: analysisStage.id }
    ),
  ];
  workspaceStore.updateStageCollections(
    analysisStage.id,
    { input_collections: inputCollections },
    'analysis_input_collection_selected'
  );
  workspaceStore.updateStageDraft(
    analysisStage.id,
    {
      selection_spec: {
        ...analysisStage.selection_spec,
        source_collection_id: EVALUATION_COLLECTION_ID,
        eval_run_ids: run ? [run.id] : [],
        input_collection_ids: inputCollections.map((collection) => collection.id),
      },
    },
    'analysis_input_collection_selected'
  );
  useSelectionContextStore.getState().setContext({
    stage: analysisStage.id,
    collection: SELECTED_EVALUATION_COLLECTION_ID,
    selectedIds: run ? [run.id] : [],
    focusedId: run?.id ?? null,
  });
}

function primarySelectedId(context: SelectionContext): string | null {
  return context.focusedId ?? context.selectedIds[0] ?? null;
}

interface RunStoreState {
  /** All known training runs. */
  trainingRuns: TrainingRun[];
  /** Eval runs for the currently selected training run. */
  evalRuns: EvalRun[];
  /** Currently selected training run ID (null = none selected). */
  selectedTrainingRunId: string | null;
  /** Currently selected evaluation run ID (null = none selected). */
  selectedEvalRunId: string | null;
  /** Whether runs are being loaded. */
  loading: boolean;
  /** Most recent training-run load failure, if any. */
  trainingError: string | null;
  /** Most recent evaluation-run load failure, if any. */
  evalError: string | null;

  // Actions
  loadTrainingRuns: () => Promise<void>;
  selectTrainingRun: (id: string | null) => Promise<void>;
  selectEvalRun: (id: string | null) => void;
  addTrainingRun: (run: TrainingRun) => void;
  addEvalRun: (run: EvalRun) => void;
  updateEvalRunStatus: (id: string, status: EvalRun['status']) => void;
  hydrateFromWorkspace: (workspace: StudioWorkspaceSpec | null | undefined) => void;
}

let evaluationLoadEpoch = 0;

export const useRunStore = create<RunStoreState>((set, get) => ({
  trainingRuns: [],
  evalRuns: [],
  selectedTrainingRunId: null,
  selectedEvalRunId: null,
  loading: false,
  trainingError: null,
  evalError: null,

  loadTrainingRuns: async () => {
    set({ loading: true, trainingError: null });
    const runs = await withStoreActionFeedback(
      () => fetchTrainingRuns(),
      {
        errorToast: (error) => apiErrorMessage(error, 'Could not load training runs'),
        toastId: 'training-runs-load-error',
        onError: (error) =>
          set({
            loading: false,
            trainingError: apiErrorMessage(error, 'Could not load training runs'),
          }),
      },
    );
    if (!runs) return;
    set({ trainingRuns: runs, loading: false, trainingError: null });
    writeTrainingRunsToWorkspace(runs);
    const workspaceSelected = selectedTrainingRunIdFromWorkspace();
    if (runs.length > 0 && get().selectedTrainingRunId === null) {
      const selectedId =
        workspaceSelected && runs.some((run) => run.id === workspaceSelected)
          ? workspaceSelected
          : runs[0].id;
      await get().selectTrainingRun(selectedId);
    }
  },

  selectTrainingRun: async (id) => {
    const requestEpoch = ++evaluationLoadEpoch;
    set({ selectedTrainingRunId: id, selectedEvalRunId: null, evalRuns: [], evalError: null });
    const selected = get().trainingRuns.find((run) => run.id === id) ?? null;
    writeSelectedTrainingRunToWorkspace(selected);
    if (id === null) return;
    const isCurrentEvaluationLoad = () =>
      evaluationLoadEpoch === requestEpoch && get().selectedTrainingRunId === id;
    const evals = await withStoreActionFeedback(
      () => fetchEvalRuns(id),
      {
        errorToast: (error) => apiErrorMessage(error, 'Could not load evaluation runs'),
        toastId: 'eval-runs-load-error',
        onError: (error) => {
          if (isCurrentEvaluationLoad()) {
            set({ evalError: apiErrorMessage(error, 'Could not load evaluation runs') });
          }
        },
      },
    );
    if (!evals || !isCurrentEvaluationLoad()) return;
    set({ evalRuns: evals, evalError: null });
    writeEvalRunsToWorkspace(evals);
    const workspaceSelected = selectedEvalRunIdFromWorkspace();
    if (evals.length > 0) {
      const selectedId =
        workspaceSelected && evals.some((run) => run.id === workspaceSelected)
          ? workspaceSelected
          : evals[0].id;
      get().selectEvalRun(selectedId);
    }
  },

  selectEvalRun: (id) => {
    set({ selectedEvalRunId: id });
    const selected = get().evalRuns.find((run) => run.id === id) ?? null;
    writeSelectedEvalRunToWorkspace(selected);
  },

  addTrainingRun: (run) => {
    set((state) => ({
      trainingRuns: [run, ...state.trainingRuns],
    }));
    writeTrainingRunsToWorkspace(get().trainingRuns);
  },

  addEvalRun: (run) => {
    set((state) => ({
      evalRuns: [run, ...state.evalRuns],
      selectedEvalRunId: run.id,
    }));
    writeEvalRunsToWorkspace(get().evalRuns);
    writeSelectedEvalRunToWorkspace(run);
  },

  updateEvalRunStatus: (id, status) => {
    set((state) => ({
      evalRuns: state.evalRuns.map((r) =>
        r.id === id ? { ...r, status } : r,
      ),
    }));
  },

  hydrateFromWorkspace: (workspace) => {
    const trainStage = getStageByKind(workspace ?? null, 'train');
    const evalStage = getStageByKind(workspace ?? null, 'eval');
    const analysisStage = getStageByKind(workspace ?? null, 'analysis');
    const trainingRefs = trainStage
      ? manifestRefsFromCollections([
          ...trainStage.output_collections,
          ...(evalStage?.input_collections ?? []),
        ]).filter((ref) => ref.role === 'training_run' || ref.kind === 'TrainingRun')
      : [];
    const evalRefs = evalStage
      ? manifestRefsFromCollections([
          ...evalStage.output_collections,
          ...(analysisStage?.input_collections ?? []),
        ]).filter((ref) => ref.role === 'evaluation_run' || ref.kind === 'EvaluationRun')
      : [];
    const selectedTrainingRunId =
      selectionIds(evalStage?.selection_spec ?? {}, 'training_run_ids')[0] ??
      trainingRefs[0]?.id ??
      null;
    const selectedEvalRunId =
      selectionIds(analysisStage?.selection_spec ?? {}, 'eval_run_ids')[0] ??
      selectionIds(evalStage?.selection_spec ?? {}, 'evaluation_run_ids')[0] ??
      evalRefs[0]?.id ??
      null;

    set({
      trainingRuns: trainingRefs.map(trainingRunFromManifestRef),
      evalRuns: evalRefs.map(evalRunFromManifestRef),
      selectedTrainingRunId,
      selectedEvalRunId,
      loading: false,
      trainingError: null,
      evalError: null,
    });
    const selectionStore = useSelectionContextStore.getState();
    if (selectedEvalRunId && analysisStage) {
      selectionStore.setContext({
        stage: analysisStage.id,
        collection: SELECTED_EVALUATION_COLLECTION_ID,
        selectedIds: [selectedEvalRunId],
        focusedId: selectedEvalRunId,
      });
    } else if (selectedTrainingRunId && evalStage) {
      selectionStore.setContext({
        stage: evalStage.id,
        collection: SELECTED_TRAINING_COLLECTION_ID,
        selectedIds: [selectedTrainingRunId],
        focusedId: selectedTrainingRunId,
      });
    } else {
      selectionStore.setContext({
        stage: null,
        collection: null,
        selectedIds: [],
        focusedId: null,
      });
    }
  },
}));

useSelectionContextStore.subscribe((state) => {
  const selectedId = primarySelectedId(state.context);
  if (state.context.collection === null && selectedId === null) {
    useRunStore.setState({
      selectedTrainingRunId: null,
      selectedEvalRunId: null,
    });
    return;
  }
  if (state.context.collection === SELECTED_TRAINING_COLLECTION_ID) {
    useRunStore.setState({
      selectedTrainingRunId: selectedId,
      selectedEvalRunId: null,
    });
    return;
  }
  if (state.context.collection === SELECTED_EVALUATION_COLLECTION_ID) {
    useRunStore.setState({ selectedEvalRunId: selectedId });
  }
});
