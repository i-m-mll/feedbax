import { create } from 'zustand';
import type { TrainingRun, EvalRun } from '@/types/runs';
import { fetchTrainingRuns, fetchEvalRuns } from '@/api/runAPI';
import { getStageByKind, useWorkspaceStore } from '@/stores/workspaceStore';
import type { StudioCollectionRef, StudioManifestRef } from '@/types/workspace';

const TRAINING_COLLECTION_ID = 'collection:training-runs';
const SELECTED_TRAINING_COLLECTION_ID = 'collection:selected-training-runs';
const EVALUATION_COLLECTION_ID = 'collection:evaluation-runs';
const SELECTED_EVALUATION_COLLECTION_ID = 'collection:selected-evaluation-runs';

function manifestRefForTrainingRun(run: TrainingRun): StudioManifestRef {
  return {
    kind: 'TrainingRun',
    id: run.id,
    role: 'training_run',
    provider: 'feedbax',
    uri: null,
    metadata: {
      name: run.name,
      status: run.status,
      created_at: run.createdAt,
      hyperparams: run.hyperparams,
      legacy_run_record: true,
    },
  };
}

function manifestRefForEvalRun(run: EvalRun): StudioManifestRef {
  return {
    kind: 'EvaluationRun',
    id: run.id,
    role: 'evaluation_run',
    provider: 'feedbax',
    uri: null,
    metadata: {
      name: run.name,
      status: run.status,
      created_at: run.createdAt,
      training_run_id: run.trainingRunId,
      description: run.description ?? null,
      legacy_run_record: true,
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

  // Actions
  loadTrainingRuns: () => Promise<void>;
  selectTrainingRun: (id: string | null) => Promise<void>;
  selectEvalRun: (id: string | null) => void;
  addTrainingRun: (run: TrainingRun) => void;
  addEvalRun: (run: EvalRun) => void;
  updateEvalRunStatus: (id: string, status: EvalRun['status']) => void;
}

export const useRunStore = create<RunStoreState>((set, get) => ({
  trainingRuns: [],
  evalRuns: [],
  selectedTrainingRunId: null,
  selectedEvalRunId: null,
  loading: false,

  loadTrainingRuns: async () => {
    set({ loading: true });
    try {
      const runs = await fetchTrainingRuns();
      set({ trainingRuns: runs, loading: false });
      writeTrainingRunsToWorkspace(runs);
      const workspaceSelected = selectedTrainingRunIdFromWorkspace();
      if (runs.length > 0 && get().selectedTrainingRunId === null) {
        const selectedId =
          workspaceSelected && runs.some((run) => run.id === workspaceSelected)
            ? workspaceSelected
            : runs[0].id;
        await get().selectTrainingRun(selectedId);
      }
    } catch {
      set({ loading: false });
    }
  },

  selectTrainingRun: async (id) => {
    set({ selectedTrainingRunId: id, selectedEvalRunId: null, evalRuns: [] });
    const selected = get().trainingRuns.find((run) => run.id === id) ?? null;
    writeSelectedTrainingRunToWorkspace(selected);
    if (id === null) return;
    try {
      const evals = await fetchEvalRuns(id);
      set({ evalRuns: evals });
      writeEvalRunsToWorkspace(evals);
      const workspaceSelected = selectedEvalRunIdFromWorkspace();
      if (evals.length > 0) {
        const selectedId =
          workspaceSelected && evals.some((run) => run.id === workspaceSelected)
            ? workspaceSelected
            : evals[0].id;
        get().selectEvalRun(selectedId);
      }
    } catch {
      // eval fetch failed silently
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
}));
