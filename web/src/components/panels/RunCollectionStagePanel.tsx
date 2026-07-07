import { useCallback, useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import {
  Activity,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Circle,
  Cpu,
  Download,
  Eye,
  GitBranch,
  Info,
  Layers3,
  Pencil,
  Pin,
  PinOff,
  Plus,
  PlayCircle,
  RotateCcw,
  Server,
  SlidersHorizontal,
  Trash2,
  X,
  XCircle,
} from 'lucide-react';
import { getScenario, getStageByKind, useWorkspaceStore } from '@/stores/workspaceStore';
import { useGraphStore } from '@/stores/graphStore';
import { useSelectionContextStore } from '@/stores/selectionContextStore';
import { useTrainingStore } from '@/stores/trainingStore';
import type { FrozenSnapshotProjection } from '@/stores/selectionContextStore';
import {
  prepareStudioTrainingExecution,
  previewStudioEvaluationMatrix,
  runStudioEvaluationLocalExecution,
  stageStudioEvaluationMatrix,
} from '@/api/client';
import {
  cancelTrainingRun,
  deleteTrainingRun,
  fetchEvalRunManifest,
  fetchTrainingRunManifest,
  supersedeTrainingRun,
} from '@/api/runAPI';
import {
  bestTrainingRun,
  buildLineageProjection,
  compareSpecHashes,
  currentDraftSpecHashesForScenario,
  evaluationProtocolLabel,
  evaluationRunSummaries,
  formatMetric,
  runParameterSummary,
  selectedIds,
  stableHash,
  trainingRunMetricValue,
  trainingInputSummaries,
  trainingRunSummaries,
  type EvaluationRunSummary,
  type LineageProjection,
  type LineageProjectionEdge,
  type LineageProjectionNode,
  type SpecHashStatus,
  type TrainingRunSummary,
} from '@/utils/pipelineCollections';
import {
  stageExecutionSpecWithProtocolPatch,
  stageExecutionTarget,
  trainingProtocolSnapshot,
  trainingSpecWithProtocolPatch,
  type ExecutionTargetChoice,
  type TrainingProtocolSnapshot,
} from '@/utils/stageProtocol';
import {
  scenarioMetricSpecs,
  type ScenarioMetricSpec,
} from '@/features/scenario/integration';
import {
  runMetricColumns,
  type MetricColumnSpec,
} from '@/utils/runMetricColumns';
import {
  bulkEditGhostRows,
  formatAxisValue,
  ghostRowsForMatrix,
  initialMatrixSpec,
  matrixSpecFromGhostRows,
  matrixSpecToValuesInput,
  parseAxisValuesInput,
  runCountExpression,
  trainAxisColumns,
  validateAxisPath,
  workspaceWithMatrixSelection,
  selectionSpecWithoutMatrix,
  type BulkEditVerb,
  type TrainAxisColumn,
  type TrainMatrixAxisDraft,
  type TrainMatrixGhostRow,
  type TrainMatrixMode,
  type TrainMatrixSpec,
} from '@/utils/trainMatrix';
import {
  progressBindingsForRuns,
  sortTrainingRows,
  stageWithTrainingRunLifecyclePatch,
  trainingRunGroupId,
  UNGROUPED_RUN_GROUP_ID,
} from '@/utils/trainRunTable';
import type { TrainingProgress } from '@/types/training';
import type { TrainingRun } from '@/types/runs';
import type {
  EvalCheckpointPolicyMode,
  EvalReprocessMode,
  StudioEvaluationMatrixPreview,
  StudioTopPaneState,
  StudioWorkspaceSpec,
} from '@/types/workspace';

type RunView = 'all' | 'selected' | 'best';
type SortKey = string;
type SortDirection = 'asc' | 'desc';

interface SortState {
  key: SortKey;
  direction: SortDirection;
}


export function TrainCollectionPanel() {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const setActiveStage = useWorkspaceStore((state) => state.setActiveStage);
  const setWorkspace = useWorkspaceStore((state) => state.setWorkspace);
  const updateStageDraft = useWorkspaceStore((state) => state.updateStageDraft);
  const setTrainingExecutionPreparation = useWorkspaceStore(
    (state) => state.setTrainingExecutionPreparation
  );
  const lastTrainingExecutionPreparation = useWorkspaceStore(
    (state) => state.lastTrainingExecutionPreparation
  );
  const updateActiveScenarioTrainingSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioTrainingSpec
  );
  const markDirty = useGraphStore((state) => state.markDirty);
  const trainingProgress = useTrainingStore((state) => state.progress);
  const trainingJobId = useTrainingStore((state) => state.jobId);
  const lossHistory = useTrainingStore((state) => state.lossHistory);
  const selectionContext = useSelectionContextStore((state) => state.context);
  const previewRunId = useSelectionContextStore((state) => state.previewId);
  const syncMode = useSelectionContextStore((state) => state.syncMode);
  const frozenSnapshot = useSelectionContextStore((state) => state.frozenSnapshot);
  const setSelectionContext = useSelectionContextStore((state) => state.setContext);
  const syncCollection = useSelectionContextStore((state) => state.syncCollection);
  const toggleSelectionId = useSelectionContextStore((state) => state.toggleSelectedId);
  const setSelectedIds = useSelectionContextStore((state) => state.setSelectedIds);
  const focusSelectionId = useSelectionContextStore((state) => state.focusId);
  const previewSelectionId = useSelectionContextStore((state) => state.previewFocus);
  const setSyncMode = useSelectionContextStore((state) => state.setSyncMode);
  const setFrozenSnapshot = useSelectionContextStore((state) => state.setFrozenSnapshot);
  const [view, setView] = useState<RunView>('all');
  const [sort, setSort] = useState<SortState>({ key: 'final_validation_loss', direction: 'asc' });
  const [collapsedSets, setCollapsedSets] = useState<Set<string>>(() => new Set());
  const [matrix, setMatrix] = useState<TrainMatrixSpec>(() => initialMatrixSpec(null, null));
  const [editingAxisId, setEditingAxisId] = useState<string | null>(null);
  const [axisDraft, setAxisDraft] = useState({
    label: 'Learning rate',
    path: 'training_spec.optimizer.params.learning_rate',
    values: '0.001, 0.0003',
  });
  const [bulkVerb, setBulkVerb] = useState<BulkEditVerb>('keep');
  const [bulkAxisId, setBulkAxisId] = useState<string>('');
  const [bulkValues, setBulkValues] = useState('0.001, 0.0003');
  const [stageState, setStageState] = useState({ busy: false, error: null as string | null });
  const [runActionState, setRunActionState] = useState({ busy: false, error: null as string | null });
  const [showHistory, setShowHistory] = useState(false);

  const trainStage = getStageByKind(workspace, 'train');
  const trainScenario = getScenario(workspace, trainStage?.scenario_id);
  const evalStage = getStageByKind(workspace, 'eval');
  const protocol = trainingProtocolSnapshot(trainStage, trainScenario);
  const metrics = useMemo(() => scenarioMetricSpecs(workspace), [workspace]);
  const lineage = useMemo(() => buildLineageProjection(workspace), [workspace]);
  const currentSpecHashes = useMemo(
    () => currentDraftSpecHashesForScenario(trainScenario),
    [trainScenario]
  );
  const rows = useMemo(
    () =>
      trainingRunSummaries(trainStage, {
        currentSpecHashes,
        includeSuperseded: showHistory,
      }),
    [currentSpecHashes, showHistory, trainStage]
  );
  const trainingCollectionId =
    trainStage?.output_collections.find((collection) => collection.item_refs.length > 0)?.id ??
    'collection:training-runs';
  const contextMatchesTrain =
    selectionContext.stage === trainStage?.id && selectionContext.collection === trainingCollectionId;
  const selectedRunIds = useMemo(
    () => new Set(contextMatchesTrain ? selectionContext.selectedIds : []),
    [contextMatchesTrain, selectionContext.selectedIds]
  );
  const focusedRunId = contextMatchesTrain ? selectionContext.focusedId : null;
  const effectiveFocusedRunId =
    syncMode === 'linked' && previewRunId !== null ? previewRunId : focusedRunId;
  const ghostRows = useMemo(() => ghostRowsForMatrix(matrix), [matrix]);
  const axisColumns = useMemo(() => trainAxisColumns(rows, matrix.axes), [matrix.axes, rows]);
  const metricColumns = useMemo(() => runMetricColumns(metrics, rows), [metrics, rows]);
  const bestRow = useMemo(() => bestTrainingRun(rows), [rows]);
  const selectedRows = useMemo(
    () => rows.filter((row) => selectedRunIds.has(row.id)),
    [rows, selectedRunIds]
  );
  const focusedRun = useMemo(
    () => rows.find((row) => row.id === effectiveFocusedRunId) ?? selectedRows[0] ?? rows[0] ?? null,
    [effectiveFocusedRunId, rows, selectedRows]
  );
  const bulkAxis = useMemo(
    () => axisColumns.find((axis) => axis.id === bulkAxisId) ?? axisColumns[0] ?? null,
    [axisColumns, bulkAxisId]
  );
  const bulkGhostRows = useMemo(
    () =>
      bulkEditGhostRows({
        rows: selectedRows.filter((row) => row.status === 'pending'),
        axis: bulkAxis,
        verb: bulkVerb,
        values: parseAxisValuesInput(bulkValues),
      }),
    [bulkAxis, bulkValues, bulkVerb, selectedRows]
  );
  const bulkMatrixPlan = useMemo(
    () =>
      matrixSpecFromGhostRows({
        name: `Bulk ${bulkVerb} ${bulkAxis?.label ?? 'axis'}`,
        rows: bulkGhostRows,
        axes: axisColumns,
      }),
    [axisColumns, bulkAxis?.label, bulkGhostRows, bulkVerb]
  );
  const visibleRows = useMemo(() => {
    const base =
      view === 'selected'
        ? selectedRows
        : view === 'best' && bestRow
          ? [bestRow]
          : rows;
    return sortTrainingRows(base, sort);
  }, [bestRow, rows, selectedRows, sort, view]);
  const pendingSelectionCount = selectedRows.filter((row) => row.status === 'pending').length;
  const matrixPathError = useMemo(
    () => validateAxisPath(axisDraft.path, trainScenario),
    [axisDraft.path, trainScenario]
  );
  const progressBindings = useMemo(
    () =>
      progressBindingsForRuns(
        rows,
        trainingProgress,
        trainingJobId ?? lastTrainingExecutionPreparation?.plan.job_id ?? null
      ),
    [lastTrainingExecutionPreparation?.plan.job_id, rows, trainingJobId, trainingProgress]
  );

  useEffect(() => {
    if (!trainStage) return;
    const next = initialMatrixSpec(trainStage, trainScenario);
    setMatrix((current) => (current.axes.length === 0 ? next : current));
  }, [trainScenario, trainStage]);

  useEffect(() => {
    syncCollection(trainStage?.id ?? null, trainingCollectionId, rows.map((row) => row.id));
  }, [rows, syncCollection, trainStage?.id, trainingCollectionId]);

  const toggleRow = useCallback((id: string) => {
    toggleSelectionId(id);
  }, [toggleSelectionId]);

  const selectAll = useCallback(() => {
    setSelectedIds(rows.map((row) => row.id));
  }, [rows, setSelectedIds]);

  const selectBest = useCallback(() => {
    setSelectedIds(bestRow ? [bestRow.id] : []);
  }, [bestRow, setSelectedIds]);

  const clearSelection = useCallback(() => {
    setSelectedIds([]);
  }, [setSelectedIds]);

  const toggleSetCollapsed = useCallback((runSetId: string) => {
    setCollapsedSets((current) => {
      const next = new Set(current);
      if (next.has(runSetId)) next.delete(runSetId);
      else next.add(runSetId);
      return next;
    });
  }, []);

  const useForEvaluation = useCallback(() => {
    if (!evalStage || selectedRunIds.size === 0) return;
    const sourceCollectionId =
      trainStage?.output_collections.find((collection) => collection.item_refs.length > 0)?.id ??
      evalStage.selection_spec.source_collection_id;
    updateStageDraft(
      evalStage.id,
      {
        selection_spec: {
          ...evalStage.selection_spec,
          source_collection_id: sourceCollectionId,
          candidate_training_run_ids: rows.map((row) => row.id),
          training_run_ids: Array.from(selectedRunIds),
        },
      },
      'evaluation_selection_transferred_from_train'
    );
    setActiveStage(evalStage.id);
  }, [evalStage, rows, selectedRunIds, setActiveStage, trainStage, updateStageDraft]);

  const commitAxisDraft = useCallback(() => {
    if (matrixPathError) return;
    const values = parseAxisValuesInput(axisDraft.values);
    if (values.length === 0) return;
    setMatrix((current) => {
      const nextAxis: TrainMatrixAxisDraft = {
        id:
          editingAxisId ??
          axisDraft.path
            .replace(/^(training_spec|task_spec|task_binding_spec)\./, '')
            .replace(/[^a-zA-Z0-9]+/g, '_')
            .replace(/^_+|_+$/g, '')
            .toLowerCase(),
        label: axisDraft.label.trim() || axisDraft.path,
        path: axisDraft.path.trim(),
        values,
        source: 'manual',
      };
      return {
        ...current,
        manualCoordinates: undefined,
        axes: editingAxisId
          ? current.axes.map((axis) => (axis.id === editingAxisId ? { ...nextAxis, id: axis.id } : axis))
          : [...current.axes, nextAxis],
      };
    });
    setEditingAxisId(null);
  }, [axisDraft.label, axisDraft.path, axisDraft.values, editingAxisId, matrixPathError]);

  const editAxis = useCallback((axis: TrainMatrixAxisDraft) => {
    setEditingAxisId(axis.id);
    setAxisDraft({
      label: axis.label,
      path: axis.path,
      values: matrixSpecToValuesInput(axis),
    });
  }, []);

  const removeAxis = useCallback((axisId: string) => {
    setMatrix((current) => ({
      ...current,
      manualCoordinates: undefined,
      axes: current.axes.filter((axis) => axis.id !== axisId),
    }));
    setEditingAxisId((current) => (current === axisId ? null : current));
  }, []);

  const stageWorkspace = useCallback(async (
    nextWorkspace: typeof workspace,
    stageId: string,
    metadata: Record<string, unknown>
  ) => {
    if (!nextWorkspace) return false;
    setStageState({ busy: true, error: null });
    try {
      setWorkspace(nextWorkspace);
      const preparation = await prepareStudioTrainingExecution({
        workspace: nextWorkspace,
        stage_id: stageId,
        backend: 'local',
        issues: ['3a6d02e'],
        metadata,
      });
      setTrainingExecutionPreparation(preparation);
      markDirty();
      setStageState({ busy: false, error: null });
      return true;
    } catch (error) {
      setStageState({
        busy: false,
        error: error instanceof Error ? error.message : 'Failed to stage training runs.',
      });
      return false;
    }
  }, [markDirty, setTrainingExecutionPreparation, setWorkspace]);

  const stageMatrixSpec = useCallback(async (
    matrixToStage: TrainMatrixSpec,
    metadata: Record<string, unknown>
  ) => {
    if (!workspace || !trainStage || matrixToStage.axes.length === 0) return;
    const previewRows = ghostRowsForMatrix(matrixToStage);
    if (previewRows.length === 0) return;
    const nextWorkspace = workspaceWithMatrixSelection(workspace, trainStage.id, matrixToStage);
    setMatrix(matrixToStage);
    return stageWorkspace(nextWorkspace, trainStage.id, metadata);
  }, [stageWorkspace, trainStage, workspace]);

  const stageMatrix = useCallback(async () => {
    if (matrix.axes.length === 0 || ghostRows.length === 0) return;
    await stageMatrixSpec(matrix, {
      source: 'train_collection_panel',
      matrix_preview_count: ghostRows.length,
    });
  }, [ghostRows.length, matrix, stageMatrixSpec]);

  const applyBulkEdit = useCallback(async () => {
    if (!bulkMatrixPlan.matrix) {
      setStageState({
        busy: false,
        error: bulkMatrixPlan.error ?? 'Bulk edit preview cannot be staged.',
      });
      return;
    }
    const staged = await stageMatrixSpec(bulkMatrixPlan.matrix, {
      source: 'train_collection_bulk_edit',
      bulk_verb: bulkVerb,
      selected_pending_count: pendingSelectionCount,
      matrix_preview_count: bulkGhostRows.length,
    });
    if (staged) setSelectedIds([]);
  }, [
    bulkGhostRows.length,
    bulkMatrixPlan.error,
    bulkMatrixPlan.matrix,
    bulkVerb,
    pendingSelectionCount,
    setSelectedIds,
    stageMatrixSpec,
  ]);

  const viewSnapshot = useCallback(async (run: TrainingRunSummary) => {
    focusSelectionId(run.id);
    setStageState({ busy: true, error: null });
    try {
      const manifest = await fetchTrainingRunManifest(run.id);
      const projection = frozenSnapshotProjectionForRun(run, manifest);
      setFrozenSnapshot(projection);
      if (workspace) {
        setWorkspace(workspaceWithTopPaneProvenance(workspace, projection));
      }
      setStageState({ busy: false, error: null });
    } catch (error) {
      setStageState({
        busy: false,
        error: error instanceof Error ? error.message : 'Failed to load run snapshot.',
      });
    }
  }, [focusSelectionId, setFrozenSnapshot, setWorkspace, workspace]);

  const backToDraft = useCallback(() => {
    setFrozenSnapshot(null);
    if (workspace) setWorkspace(workspaceWithTopPaneProvenance(workspace, null));
  }, [setFrozenSnapshot, setWorkspace, workspace]);

  const restageRun = useCallback(async (run: TrainingRunSummary) => {
    if (!workspace || !trainStage) return;
    focusSelectionId(run.id);
    setFrozenSnapshot(null);
    setStageState({ busy: true, error: null });
    try {
      const manifest = await fetchTrainingRunManifest(run.id);
      const nextWorkspace = workspaceWithTopPaneProvenance(
        workspaceWithTrainingSnapshot(
          workspace,
          trainStage.id,
          trainStage.scenario_id,
          manifest,
          run.axisCoordinates
        ),
        null
      );
      await stageWorkspace(nextWorkspace, trainStage.id, {
        source: 'train_collection_snapshot_restage',
        restaged_from_run_id: run.id,
        restaged_from_manifest_id: manifest.id,
        matrix_preview_count: 1,
      });
    } catch (error) {
      setStageState({
        busy: false,
        error: error instanceof Error ? error.message : 'Failed to restage run snapshot.',
      });
    }
  }, [focusSelectionId, setFrozenSnapshot, stageWorkspace, trainStage, workspace]);

  const promoteToDraft = useCallback(async (run: TrainingRunSummary) => {
    if (!workspace || !trainStage) return;
    focusSelectionId(run.id);
    setStageState({ busy: true, error: null });
    try {
      const manifest = await fetchTrainingRunManifest(run.id);
      const nextWorkspace = workspaceWithTopPaneProvenance(
        workspaceWithTrainingSnapshot(
          workspace,
          trainStage.id,
          trainStage.scenario_id,
          manifest,
          run.axisCoordinates
        ),
        null
      );
      setWorkspace(nextWorkspace);
      setFrozenSnapshot(null);
      markDirty();
      setStageState({ busy: false, error: null });
    } catch (error) {
      setStageState({
        busy: false,
        error: error instanceof Error ? error.message : 'Failed to promote run snapshot.',
      });
    }
  }, [focusSelectionId, markDirty, setFrozenSnapshot, setWorkspace, trainStage, workspace]);

  const patchLifecycleRun = useCallback(
    (action: 'cancel' | 'delete' | 'supersede', run: TrainingRun) => {
      if (!workspace || !trainStage) return;
      const patchAction = action === 'delete' ? 'remove' : 'update';
      const nextWorkspace = {
        ...workspace,
        stages: workspace.stages.map((stage) =>
          stage.id === trainStage.id
            ? stageWithTrainingRunLifecyclePatch(stage, patchAction, run)
            : stage
        ),
      };
      setWorkspace(nextWorkspace);
      if (action === 'delete') {
        const selected = useSelectionContextStore
          .getState()
          .context.selectedIds.filter((id) => id !== run.id);
        setSelectedIds(selected);
        if (useSelectionContextStore.getState().context.focusedId === run.id) focusSelectionId(null);
        if (useSelectionContextStore.getState().frozenSnapshot?.runId === run.id) {
          setFrozenSnapshot(null);
        }
      }
      markDirty();
    },
    [focusSelectionId, markDirty, setFrozenSnapshot, setSelectedIds, setWorkspace, trainStage, workspace]
  );

  const runLifecycleAction = useCallback(
    async (action: 'cancel' | 'delete' | 'supersede', run: TrainingRunSummary) => {
      setRunActionState({ busy: true, error: null });
      try {
        const updated =
          action === 'cancel'
            ? await cancelTrainingRun(run.id)
            : action === 'delete'
              ? await deleteTrainingRun(run.id)
              : await supersedeTrainingRun(run.id, { reason: 'Superseded from Train tab' });
        patchLifecycleRun(action, updated);
        setRunActionState({ busy: false, error: null });
      } catch (error) {
        setRunActionState({
          busy: false,
          error: error instanceof Error ? error.message : `Failed to ${action} run.`,
        });
      }
    },
    [patchLifecycleRun]
  );

  const setTarget = useCallback(
    (target: ExecutionTargetChoice) => {
      if (!trainStage) return;
      updateStageDraft(
        trainStage.id,
        {
          execution_spec: stageExecutionSpecWithProtocolPatch(trainStage, {
            compute_target: target,
          }),
        },
        'training_compute_target_changed'
      );
    },
    [trainStage, updateStageDraft]
  );

  const updateProtocol = useCallback(
    (patch: Partial<TrainingProtocolSnapshot>) => {
      if (!trainScenario?.training_spec) return;
      updateActiveScenarioTrainingSpec(
        trainingSpecWithProtocolPatch(trainScenario.training_spec, patch)
      );
      markDirty();
    },
    [markDirty, trainScenario, updateActiveScenarioTrainingSpec]
  );

  const focusLineageNode = useCallback(
    (node: LineageProjectionNode) => {
      if (!node.focusStageId || !node.focusCollectionId) return;
      setSelectionContext({
        stage: node.focusStageId,
        collection: node.focusCollectionId,
        selectedIds: [node.id],
        focusedId: node.id,
      });
      setActiveStage(node.focusStageId);
    },
    [setActiveStage, setSelectionContext]
  );

  return (
    <div className="relative h-full overflow-hidden bg-slate-50/40">
      <div className="h-full overflow-y-auto">
        <div className="mx-auto grid w-full max-w-7xl min-w-0 gap-5 px-6 py-5 text-sm text-slate-600 xl:grid-cols-[minmax(0,1fr)_22rem]">
          <div className="space-y-5">
            <MatrixBuilderStrip
              matrix={matrix}
              axisDraft={axisDraft}
              editingAxisId={editingAxisId}
              pathError={matrixPathError}
              ghostRows={ghostRows}
              busy={stageState.busy}
              onMatrixNameChange={(name) => setMatrix((current) => ({ ...current, name }))}
              onModeChange={(mode) => setMatrix((current) => ({ ...current, mode, manualCoordinates: undefined }))}
              onAxisDraftChange={setAxisDraft}
              onCommitAxis={commitAxisDraft}
              onEditAxis={editAxis}
              onRemoveAxis={removeAxis}
              onCancelEdit={() => setEditingAxisId(null)}
              onStage={stageMatrix}
            />

            {stageState.error && (
              <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                {stageState.error}
              </div>
            )}

            <RunTable
              title="Training runs"
              rows={visibleRows}
              allRows={rows}
              ghostRows={ghostRows}
              axisColumns={axisColumns}
              metricColumns={metricColumns}
              selectedIds={selectedRunIds}
              focusedId={focusedRunId}
              previewId={previewRunId}
              syncMode={syncMode}
              collapsedSets={collapsedSets}
              view={view}
              showHistory={showHistory}
              sort={sort}
              bestRunId={bestRow?.id ?? null}
              progressByRunId={progressBindings.byRunId}
              progressByGroupId={progressBindings.byGroupId}
              onViewChange={setView}
              onToggleHistory={() => setShowHistory((current) => !current)}
              onSortChange={setSort}
              onToggle={toggleRow}
              onSelectAll={selectAll}
              onSelectBest={selectBest}
              onClear={clearSelection}
              onPreview={(id) => previewSelectionId(id)}
              onCommitFocus={(run) => focusSelectionId(run.id)}
              onSyncModeChange={setSyncMode}
              onToggleSet={toggleSetCollapsed}
              onOpenDetails={(run) => focusSelectionId(run.id)}
              onViewSnapshot={viewSnapshot}
              onRestageRun={restageRun}
              onLifecycleAction={runLifecycleAction}
            />

            <LineageProjectionPanel
              projection={lineage}
              focusedId={selectionContext.focusedId}
              previewId={previewRunId}
              syncMode={syncMode}
              onPreview={previewSelectionId}
              onFocusNode={focusLineageNode}
            />

            <BulkEditPanel
              selectedCount={selectedRunIds.size}
              pendingSelectionCount={pendingSelectionCount}
              axes={axisColumns}
              axisId={bulkAxis?.id ?? ''}
              verb={bulkVerb}
              values={bulkValues}
              ghostRows={bulkGhostRows}
              applyError={bulkGhostRows.length > 0 ? bulkMatrixPlan.error : null}
              busy={stageState.busy}
              onAxisChange={setBulkAxisId}
              onVerbChange={setBulkVerb}
              onValuesChange={setBulkValues}
              onApply={applyBulkEdit}
            />

            {runActionState.error && (
              <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                {runActionState.error}
              </div>
            )}
          </div>

          <div className="space-y-3">
            <ExecutionTarget value={protocol.computeTarget} onChange={setTarget} />
            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="font-semibold text-slate-800">Protocol</div>
              <div className="mt-3">
                <TrainingProtocolEditor protocol={protocol} onProtocolChange={updateProtocol} />
              </div>
            </section>
            <RunDetailPane
              run={focusedRun}
              frozenSnapshot={frozenSnapshot}
              currentSpecHashes={currentSpecHashes}
              lossHistory={lossHistory}
              onViewSnapshot={viewSnapshot}
              onBackToDraft={backToDraft}
              onPromoteToDraft={promoteToDraft}
              onRestage={restageRun}
              onDownloadCheckpoint={(run) => {
                if (run.uri) window.open(run.uri, '_blank', 'noopener,noreferrer');
              }}
            />
            {selectedRunIds.size > 0 && (
              <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <div className="mb-3 text-xs font-semibold text-slate-700">
                  {selectedRunIds.size} run{selectedRunIds.size === 1 ? '' : 's'} selected
                </div>
                <button
                  type="button"
                  disabled={!evalStage}
                  onClick={useForEvaluation}
                  className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-brand-500 px-3 py-2 text-xs font-semibold text-white shadow-sm hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Use selection in Evaluate
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </section>
            )}
            <MetricTracePanel metrics={metrics} />
          </div>
        </div>
      </div>
    </div>
  );
}

export function EvaluateCollectionPanel() {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const setWorkspace = useWorkspaceStore((state) => state.setWorkspace);
  const setActiveStage = useWorkspaceStore((state) => state.setActiveStage);
  const updateStageDraft = useWorkspaceStore((state) => state.updateStageDraft);
  const markDirty = useGraphStore((state) => state.markDirty);
  const selectionContext = useSelectionContextStore((state) => state.context);
  const previewRunId = useSelectionContextStore((state) => state.previewId);
  const syncMode = useSelectionContextStore((state) => state.syncMode);
  const setSelectionContext = useSelectionContextStore((state) => state.setContext);
  const setSelectedIds = useSelectionContextStore((state) => state.setSelectedIds);
  const focusSelectionId = useSelectionContextStore((state) => state.focusId);
  const previewSelectionId = useSelectionContextStore((state) => state.previewFocus);
  const setSyncMode = useSelectionContextStore((state) => state.setSyncMode);
  const setFrozenSnapshot = useSelectionContextStore((state) => state.setFrozenSnapshot);
  const [view, setView] = useState<RunView>('all');
  const [sort, setSort] = useState<SortState>({ key: 'final_validation_loss', direction: 'asc' });
  const [detailsRun, setDetailsRun] = useState<TrainingRunSummary | null>(null);
  const [evalActionState, setEvalActionState] = useState({
    busyRunId: null as string | null,
    error: null as string | null,
  });
  const [evalPreview, setEvalPreview] = useState<StudioEvaluationMatrixPreview | null>(null);
  const [evalMatrixBusy, setEvalMatrixBusy] = useState<'preview' | 'stage' | 'run' | null>(null);
  const [checkpointPolicyMode, setCheckpointPolicyMode] =
    useState<EvalCheckpointPolicyMode>('last');
  const [checkpointMetric, setCheckpointMetric] = useState('final_validation_loss');
  const [checkpointEveryK, setCheckpointEveryK] = useState(5);
  const [reprocessMode, setReprocessMode] = useState<EvalReprocessMode>('missing');
  const [showHistory, setShowHistory] = useState(false);

  const trainStage = getStageByKind(workspace, 'train');
  const evalStage = getStageByKind(workspace, 'eval');
  const analysisStage = getStageByKind(workspace, 'analysis');
  const metrics = useMemo(() => scenarioMetricSpecs(workspace), [workspace]);
  const lineage = useMemo(() => buildLineageProjection(workspace), [workspace]);
  const rows = useMemo(
    () => trainingInputSummaries(evalStage, { includeSuperseded: showHistory }),
    [evalStage, showHistory]
  );
  const metricColumns = useMemo(() => runMetricColumns(metrics, rows), [metrics, rows]);
  const bestRow = useMemo(() => bestTrainingRun(rows), [rows]);
  const evaluationRows = useMemo(
    () => evaluationRunSummaries(evalStage, { includeSuperseded: showHistory }),
    [evalStage, showHistory]
  );
  const evalSelectionCollectionId =
    evalStage?.input_collections.find((collection) => collection.item_refs.length > 0)?.id ??
    'collection:selected-training-runs';
  const evaluationCollectionId =
    evalStage?.output_collections.find((collection) => collection.item_refs.length > 0)?.id ??
    'collection:evaluation-runs';
  const contextMatchesEvalInput =
    selectionContext.stage === evalStage?.id &&
    selectionContext.collection === evalSelectionCollectionId;
  const contextMatchesEvalOutput =
    selectionContext.stage === evalStage?.id &&
    selectionContext.collection === evaluationCollectionId;
  const selectionContextSelectedKey = selectionContext.selectedIds.join('\0');
  const workspaceSelectedIds = useMemo(
    () => selectedIds(evalStage, 'training_run_ids'),
    [evalStage]
  );
  const selectedIdsForEval = useMemo(
    () => new Set(contextMatchesEvalInput ? selectionContext.selectedIds : workspaceSelectedIds),
    [contextMatchesEvalInput, selectionContext.selectedIds, workspaceSelectedIds]
  );
  const focusedRunId = contextMatchesEvalInput ? selectionContext.focusedId : null;
  const focusedEvalRunId = contextMatchesEvalOutput ? selectionContext.focusedId : null;
  const selectedRows = useMemo(
    () => rows.filter((row) => selectedIdsForEval.has(row.id)),
    [rows, selectedIdsForEval]
  );
  const visibleRows = useMemo(() => {
    const base =
      view === 'selected'
        ? selectedRows
        : view === 'best' && bestRow
          ? [bestRow]
          : rows;
    return sortTrainingRows(base, sort);
  }, [bestRow, rows, selectedRows, sort, view]);
  const evalPayload = useMemo(() => {
    const checkpointPolicy = {
      mode: checkpointPolicyMode,
      ...(checkpointPolicyMode === 'best-by-metric'
        ? {
            metric: checkpointMetric.trim() || 'final_validation_loss',
            objective: 'minimize' as const,
          }
        : {}),
      ...(checkpointPolicyMode === 'every-k'
        ? { every_k: Math.max(1, Math.floor(checkpointEveryK)) }
        : {}),
      params: {},
    };
    return {
      workspace: workspace as StudioWorkspaceSpec,
      stage_id: evalStage?.id ?? null,
      training_run_ids: Array.from(selectedIdsForEval),
      eval_params: {
        targets: '8-direction center-out',
        sisu: 0.5,
        perturbation: 'none',
      },
      checkpoint_policy: checkpointPolicy,
      reprocess: reprocessMode,
    };
  }, [
    checkpointEveryK,
    checkpointMetric,
    checkpointPolicyMode,
    evalStage?.id,
    reprocessMode,
    selectedIdsForEval,
    workspace,
  ]);

  useEffect(() => {
    if (!evalStage) return;
    const available = new Set(rows.map((row) => row.id));
    const availableEvaluationRuns = new Set(evaluationRows.map((row) => row.id));
    if (
      contextMatchesEvalOutput &&
      ((selectionContext.focusedId && availableEvaluationRuns.has(selectionContext.focusedId)) ||
        selectionContext.selectedIds.some((id) => availableEvaluationRuns.has(id)))
    ) {
      return;
    }
    const selected = workspaceSelectedIds.filter((id) => available.has(id));
    setSelectionContext({
      stage: evalStage.id,
      collection: evalSelectionCollectionId,
      selectedIds: selected,
      focusedId:
        contextMatchesEvalInput &&
        selectionContext.focusedId &&
        available.has(selectionContext.focusedId)
          ? selectionContext.focusedId
          : selected[0] ?? null,
    });
  }, [
    contextMatchesEvalInput,
    contextMatchesEvalOutput,
    evalSelectionCollectionId,
    evalStage,
    evaluationRows,
    rows,
    selectionContext.focusedId,
    selectionContextSelectedKey,
    setSelectionContext,
    workspaceSelectedIds,
  ]);

  const writeSelection = useCallback(
    (ids: string[]) => {
      if (!evalStage) return;
      setSelectedIds(ids);
      const sourceCollectionId =
        trainStage?.output_collections.find((collection) => collection.item_refs.length > 0)?.id ??
        evalStage.selection_spec.source_collection_id;
      updateStageDraft(
        evalStage.id,
        {
          selection_spec: {
            ...evalStage.selection_spec,
            source_collection_id: sourceCollectionId,
            candidate_training_run_ids: rows.map((row) => row.id),
            training_run_ids: ids,
          },
        },
        'evaluation_selection_changed'
      );
    },
    [evalStage, rows, setSelectedIds, trainStage, updateStageDraft]
  );

  const toggleRow = useCallback(
    (id: string) => {
      const next = new Set(selectedIdsForEval);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      writeSelection(Array.from(next));
    },
    [selectedIdsForEval, writeSelection]
  );

  const openAnalyze = useCallback(() => {
    if (analysisStage) setActiveStage(analysisStage.id);
  }, [analysisStage, setActiveStage]);

  const focusEvaluationResult = useCallback(
    (row: EvaluationRunSummary) => {
      if (!evalStage) return;
      setSelectionContext({
        stage: evalStage.id,
        collection: evaluationCollectionId,
        selectedIds: [row.id],
        focusedId: row.id,
      });
    },
    [evalStage, evaluationCollectionId, setSelectionContext]
  );

  const focusLineageNode = useCallback(
    (node: LineageProjectionNode) => {
      if (!node.focusStageId || !node.focusCollectionId) return;
      setSelectionContext({
        stage: node.focusStageId,
        collection: node.focusCollectionId,
        selectedIds: [node.id],
        focusedId: node.id,
      });
      setActiveStage(node.focusStageId);
    },
    [setActiveStage, setSelectionContext]
  );

  const viewEvalSnapshot = useCallback(async (row: EvaluationRunSummary) => {
    if (!evalStage || !workspace || !row.uri) return;
    setSelectionContext({
      stage: evalStage.id,
      collection: evaluationCollectionId,
      selectedIds: [row.id],
      focusedId: row.id,
    });
    setEvalActionState({ busyRunId: row.id, error: null });
    try {
      const manifest = await fetchEvalRunManifest(row.id);
      const projection = frozenSnapshotProjectionForEvaluationRun(row, manifest);
      setFrozenSnapshot(projection);
      setWorkspace(workspaceWithTopPaneProvenance(workspace, projection));
      setEvalActionState({ busyRunId: null, error: null });
    } catch (error) {
      setEvalActionState({
        busyRunId: null,
        error: error instanceof Error ? error.message : 'Failed to load evaluation snapshot.',
      });
    }
  }, [
    evalStage,
    evaluationCollectionId,
    setFrozenSnapshot,
    setSelectionContext,
    setWorkspace,
    workspace,
  ]);

  const setTarget = useCallback(
    (target: ExecutionTargetChoice) => {
      if (!evalStage) return;
      updateStageDraft(
        evalStage.id,
        {
          execution_spec: stageExecutionSpecWithProtocolPatch(evalStage, {
            compute_target: target,
          }),
        },
        'evaluation_compute_target_changed'
      );
    },
    [evalStage, updateStageDraft]
  );

  const previewEvalMatrix = useCallback(async () => {
    if (!workspace || !evalStage || selectedIdsForEval.size === 0) return;
    setEvalMatrixBusy('preview');
    setEvalActionState({ busyRunId: null, error: null });
    try {
      const preview = await previewStudioEvaluationMatrix(evalPayload);
      setEvalPreview(preview);
    } catch (error) {
      setEvalActionState({
        busyRunId: null,
        error: error instanceof Error ? error.message : 'Failed to preview evaluation matrix.',
      });
    } finally {
      setEvalMatrixBusy(null);
    }
  }, [evalPayload, evalStage, selectedIdsForEval.size, workspace]);

  const stageEvalMatrix = useCallback(async () => {
    if (!workspace || !evalStage || selectedIdsForEval.size === 0) return;
    setEvalMatrixBusy('stage');
    setEvalActionState({ busyRunId: null, error: null });
    try {
      const result = await stageStudioEvaluationMatrix(evalPayload);
      setEvalPreview(result.preview);
      setWorkspace(result.workspace);
      markDirty();
    } catch (error) {
      setEvalActionState({
        busyRunId: null,
        error: error instanceof Error ? error.message : 'Failed to stage evaluation manifests.',
      });
    } finally {
      setEvalMatrixBusy(null);
    }
  }, [evalPayload, evalStage, markDirty, selectedIdsForEval.size, setWorkspace, workspace]);

  const runEvalMatrix = useCallback(async () => {
    if (!workspace || !evalStage || selectedIdsForEval.size === 0) return;
    setEvalMatrixBusy('run');
    setEvalActionState({ busyRunId: null, error: null });
    try {
      const result = await runStudioEvaluationLocalExecution(evalPayload);
      setEvalPreview(result.preview);
      setWorkspace(result.workspace);
      markDirty();
      if (result.errors.length > 0) {
        setEvalActionState({
          busyRunId: null,
          error: result.errors.join('; '),
        });
      }
    } catch (error) {
      setEvalActionState({
        busyRunId: null,
        error: error instanceof Error ? error.message : 'Failed to run evaluation matrix.',
      });
    } finally {
      setEvalMatrixBusy(null);
    }
  }, [evalPayload, evalStage, markDirty, selectedIdsForEval.size, setWorkspace, workspace]);

  return (
    <div className="relative h-full overflow-hidden bg-slate-50/40">
      <div className="h-full overflow-y-auto">
        <div className="mx-auto grid w-full max-w-7xl min-w-0 gap-5 px-6 py-5 text-sm text-slate-600 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="space-y-5">
          <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="text-base font-semibold text-slate-800">Validation set</div>
            <div className="mt-1 text-xs text-slate-500">
              Select training runs, then run the validation conditions on the right.
            </div>
          </section>

          <RunTable
            title="Run set"
            rows={visibleRows}
            allRows={rows}
            metricColumns={metricColumns}
            selectedIds={selectedIdsForEval}
            focusedId={focusedRunId}
            previewId={previewRunId}
            syncMode={syncMode}
            view={view}
            showHistory={showHistory}
            sort={sort}
            bestRunId={bestRow?.id ?? null}
            onViewChange={setView}
            onToggleHistory={() => setShowHistory((current) => !current)}
            onSortChange={setSort}
            onToggle={toggleRow}
            onSelectAll={() => writeSelection(rows.map((row) => row.id))}
            onSelectBest={() => writeSelection(bestRow ? [bestRow.id] : [])}
            onClear={() => writeSelection([])}
            onPreview={(id) => previewSelectionId(id)}
            onCommitFocus={(run) => focusSelectionId(run.id)}
            onSyncModeChange={setSyncMode}
            onOpenDetails={(run) => {
              focusSelectionId(run.id);
              setDetailsRun(run);
            }}
          />
          <LineageProjectionPanel
            projection={lineage}
            focusedId={selectionContext.focusedId}
            previewId={previewRunId}
            syncMode={syncMode}
            onPreview={previewSelectionId}
            onFocusNode={focusLineageNode}
          />
        </div>

          <div className="space-y-3">
            <ExecutionTarget value={stageExecutionTarget(evalStage)} onChange={setTarget} />
            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="h-4 w-4 text-slate-400" />
                <div className="font-semibold text-slate-800">Validation conditions</div>
              </div>
              <div className="mt-3 space-y-2">
                <ProtocolRow label="Targets" value="8-direction center-out" />
                <ProtocolRow label="SISU" value="0.5" />
                <ProtocolRow label="Perturbation" value="None" />
              </div>
              <div className="mt-4 grid gap-3">
                <label className="grid gap-1 text-xs font-semibold text-slate-600">
                  Checkpoint
                  <select
                    value={checkpointPolicyMode}
                    onChange={(event) =>
                      setCheckpointPolicyMode(event.target.value as EvalCheckpointPolicyMode)
                    }
                    className="rounded-md border border-slate-200 bg-white px-2 py-1.5 text-xs font-medium text-slate-700"
                  >
                    <option value="last">Last</option>
                    <option value="best-by-metric">Best by metric</option>
                    <option value="every-k">Every k</option>
                  </select>
                </label>
                {checkpointPolicyMode === 'best-by-metric' && (
                  <input
                    value={checkpointMetric}
                    onChange={(event) => setCheckpointMetric(event.target.value)}
                    className="rounded-md border border-slate-200 px-2 py-1.5 text-xs text-slate-700"
                    aria-label="Checkpoint metric"
                  />
                )}
                {checkpointPolicyMode === 'every-k' && (
                  <input
                    type="number"
                    min={1}
                    value={checkpointEveryK}
                    onChange={(event) => setCheckpointEveryK(Number(event.target.value) || 1)}
                    className="rounded-md border border-slate-200 px-2 py-1.5 text-xs text-slate-700"
                    aria-label="Checkpoint interval"
                  />
                )}
                <label className="grid gap-1 text-xs font-semibold text-slate-600">
                  Reprocess
                  <select
                    value={reprocessMode}
                    onChange={(event) => setReprocessMode(event.target.value as EvalReprocessMode)}
                    className="rounded-md border border-slate-200 bg-white px-2 py-1.5 text-xs font-medium text-slate-700"
                  >
                    <option value="missing">Missing</option>
                    <option value="missing_failed">Missing + failed</option>
                    <option value="stale">Stale</option>
                    <option value="all">All</option>
                  </select>
                </label>
              </div>
              {evalPreview && (
                <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
                  <div className="font-semibold text-slate-800">{evalPreview.summary}</div>
                  <div className="mt-1">
                    {evalPreview.pending_count} pending, {evalPreview.failed_count} failed,{' '}
                    {evalPreview.launch_count} eligible to launch.
                  </div>
                </div>
              )}
              <div className="mt-4 grid grid-cols-3 gap-2">
                <button
                  type="button"
                  disabled={evalMatrixBusy !== null || selectedIdsForEval.size === 0}
                  onClick={previewEvalMatrix}
                  className="inline-flex items-center justify-center gap-2 rounded-md border border-slate-200 bg-white px-2 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Eye className="h-3.5 w-3.5" />
                  {evalMatrixBusy === 'preview' ? 'Previewing' : 'Preview'}
                </button>
                <button
                  type="button"
                  disabled={evalMatrixBusy !== null || selectedIdsForEval.size === 0}
                  onClick={stageEvalMatrix}
                  className="inline-flex items-center justify-center gap-2 rounded-md border border-emerald-200 bg-emerald-50 px-2 py-2 text-xs font-semibold text-emerald-700 hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Layers3 className="h-3.5 w-3.5" />
                  {evalMatrixBusy === 'stage' ? 'Staging' : 'Stage'}
                </button>
                <button
                  type="button"
                  disabled={evalMatrixBusy !== null || selectedIdsForEval.size === 0}
                  onClick={runEvalMatrix}
                  className="inline-flex items-center justify-center gap-2 rounded-md bg-brand-500 px-2 py-2 text-xs font-semibold text-white hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <PlayCircle className="h-3.5 w-3.5" />
                  {evalMatrixBusy === 'run' ? 'Running' : 'Run selected'}
                </button>
              </div>
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-slate-400" />
                <div className="font-semibold text-slate-800">Results</div>
              </div>
              <div className="mt-3 space-y-3">
                {evalActionState.error && (
                  <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                    {evalActionState.error}
                  </div>
                )}
                {evaluationRows.length === 0 ? (
                  <div className="rounded-md border border-dashed border-slate-200 p-3 text-xs text-slate-500">
                    No validation results yet.
                  </div>
                ) : (
                  evaluationRows.map((row) => (
                    <EvaluationResult
                      key={row.id}
                      row={row}
                      selectedRun={rows.find(
                        (candidate) => candidate.id === row.selectedTrainingRunId
                      )}
                      focused={focusedEvalRunId === row.id}
                      previewed={previewRunId === row.id}
                      onPreview={() => previewSelectionId(row.id)}
                      onPreviewEnd={() => previewSelectionId(null)}
                      onCommitFocus={() => focusEvaluationResult(row)}
                      onOpenAnalyze={openAnalyze}
                      onViewSnapshot={() => viewEvalSnapshot(row)}
                      snapshotBusy={evalActionState.busyRunId === row.id}
                    />
                  ))
                )}
              </div>
            </section>

            <MetricTracePanel metrics={metrics} />
          </div>
        </div>
      </div>
      {detailsRun && <RunDetailOverlay run={detailsRun} onClose={() => setDetailsRun(null)} />}
    </div>
  );
}

function workspaceWithTrainingSnapshot(
  workspace: StudioWorkspaceSpec,
  stageId: string,
  scenarioId: string | null | undefined,
  manifest: Record<string, unknown>,
  axisCoordinates: Record<string, unknown>
): StudioWorkspaceSpec {
  const graphSpec = specPayloadInlineValue(manifest, 'graph_spec');
  const trainingSpec = specPayloadInlineValue(manifest, 'training_spec');
  const taskSpec = specPayloadInlineValue(manifest, 'task_spec');
  const taskBindingSpec = specPayloadInlineValue(manifest, 'task_binding_spec');
  if (!scenarioId) throw new Error('Run snapshot cannot be restaged without a scenario id.');
  if (!trainingSpec || !taskSpec) {
    throw new Error('Run snapshot is missing inline training or task specs.');
  }
  return {
    ...workspace,
    scenarios: Object.fromEntries(
      Object.entries(workspace.scenarios).map(([id, scenario]) => [
        id,
        id === scenarioId
          ? {
              ...scenario,
              graph: (graphSpec ?? scenario.graph) as typeof scenario.graph,
              training_spec: trainingSpec as unknown as typeof scenario.training_spec,
              task_spec: taskSpec as unknown as typeof scenario.task_spec,
              task_binding_spec:
                taskBindingSpec === undefined
                  ? scenario.task_binding_spec
                  : (taskBindingSpec as unknown as typeof scenario.task_binding_spec),
            }
          : scenario,
      ])
    ),
    stages: workspace.stages.map((stage) =>
      stage.id === stageId
        ? {
            ...stage,
            selection_spec: {
              ...selectionSpecWithoutMatrix(stage.selection_spec),
              axis_coordinates: axisCoordinates,
              restaged_from_manifest_id:
                typeof manifest.id === 'string' ? manifest.id : undefined,
            },
          }
        : stage
    ),
  };
}

function workspaceWithTopPaneProvenance(
  workspace: StudioWorkspaceSpec,
  projection: FrozenSnapshotProjection | null
): StudioWorkspaceSpec {
  const topPane = workspace.ui_state.top_pane as StudioTopPaneState | undefined;
  const metadata = { ...(topPane?.metadata ?? {}) };
  if (projection) {
    metadata.run_snapshot_provenance = {
      source: projection.source,
      run_id: projection.runId,
      run_label: projection.runLabel,
      run_status: projection.runStatus,
      manifest_id: projection.manifestId,
      manifest_hash: projection.manifestHash,
      spec_hashes: projection.specHashes,
      mode: 'frozen_snapshot',
      read_only: true,
    };
  } else {
    delete metadata.run_snapshot_provenance;
  }
  return {
    ...workspace,
    ui_state: {
      ...workspace.ui_state,
      top_pane: {
        active_projection: topPane?.active_projection ?? 'model',
        selected_entity_id: topPane?.selected_entity_id ?? null,
        hovered_entity_id: topPane?.hovered_entity_id ?? null,
        pinned_inspector_entity_id: topPane?.pinned_inspector_entity_id ?? null,
        metadata,
      },
    },
  };
}

function frozenSnapshotProjectionForRun(
  run: TrainingRunSummary,
  manifest: Record<string, unknown>
): FrozenSnapshotProjection {
  const snapshot = {
    graph_spec: specPayloadInlineValue(manifest, 'graph_spec') ?? null,
    training_spec: specPayloadInlineValue(manifest, 'training_spec') ?? null,
    task_spec: specPayloadInlineValue(manifest, 'task_spec') ?? null,
    task_binding_spec: specPayloadInlineValue(manifest, 'task_binding_spec') ?? null,
  };
  const manifestId = typeof manifest.id === 'string' ? manifest.id : run.id;
  const manifestHash =
    typeof manifest.manifest_hash === 'string'
      ? manifest.manifest_hash
      : typeof manifest.hash === 'string'
        ? manifest.hash
        : stableHash(manifest);
  return {
    source: 'training_run',
    runId: run.id,
    runLabel: run.label,
    runStatus: run.status,
    manifestId,
    manifestHash,
    specHashes: {
      graph_spec: snapshot.graph_spec ? stableHash(snapshot.graph_spec) : null,
      training_spec: snapshot.training_spec ? stableHash(snapshot.training_spec) : null,
      task_spec: snapshot.task_spec ? stableHash(snapshot.task_spec) : null,
      task_binding_spec: snapshot.task_binding_spec ? stableHash(snapshot.task_binding_spec) : null,
    },
    snapshot,
  };
}

function frozenSnapshotProjectionForEvaluationRun(
  run: EvaluationRunSummary,
  manifest: Record<string, unknown>
): FrozenSnapshotProjection {
  const snapshot = {
    evaluation_spec: specPayloadInlineValue(manifest, 'evaluation_spec') ?? null,
  };
  const manifestId = typeof manifest.id === 'string' ? manifest.id : run.id;
  const manifestHash =
    typeof manifest.manifest_hash === 'string'
      ? manifest.manifest_hash
      : typeof manifest.hash === 'string'
        ? manifest.hash
        : stableHash(manifest);
  return {
    source: 'evaluation_run',
    runId: run.id,
    runLabel: run.label,
    runStatus: run.status,
    manifestId,
    manifestHash,
    specHashes: {
      evaluation_spec: snapshot.evaluation_spec ? stableHash(snapshot.evaluation_spec) : null,
    },
    snapshot,
  };
}

function specHashStatusLabel(status: SpecHashStatus): string {
  switch (status) {
    case 'changed':
      return 'changed';
    case 'unchanged':
      return 'unchanged';
    case 'missing-current':
      return 'missing in draft';
    case 'missing-snapshot':
      return 'missing in snapshot';
  }
}

function specHashStatusClass(status: SpecHashStatus): string {
  switch (status) {
    case 'changed':
      return 'text-amber-700 ring-amber-200';
    case 'unchanged':
      return 'text-emerald-700 ring-emerald-200';
    case 'missing-current':
      return 'text-slate-600 ring-slate-200';
    case 'missing-snapshot':
      return 'text-red-700 ring-red-200';
  }
}

function specPayloadInlineValue(
  manifest: Record<string, unknown>,
  key: string
): Record<string, unknown> | null | undefined {
  const value = manifest[key];
  if (value === null) return null;
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const inline = (value as Record<string, unknown>).inline;
  return inline && typeof inline === 'object' && !Array.isArray(inline)
    ? (inline as Record<string, unknown>)
    : undefined;
}

function MatrixBuilderStrip({
  matrix,
  axisDraft,
  editingAxisId,
  pathError,
  ghostRows,
  busy,
  onMatrixNameChange,
  onModeChange,
  onAxisDraftChange,
  onCommitAxis,
  onEditAxis,
  onRemoveAxis,
  onCancelEdit,
  onStage,
}: {
  matrix: TrainMatrixSpec;
  axisDraft: { label: string; path: string; values: string };
  editingAxisId: string | null;
  pathError: string | null;
  ghostRows: TrainMatrixGhostRow[];
  busy: boolean;
  onMatrixNameChange: (name: string) => void;
  onModeChange: (mode: TrainMatrixMode) => void;
  onAxisDraftChange: (draft: { label: string; path: string; values: string }) => void;
  onCommitAxis: () => void;
  onEditAxis: (axis: TrainMatrixAxisDraft) => void;
  onRemoveAxis: (axisId: string) => void;
  onCancelEdit: () => void;
  onStage: () => void;
}) {
  const values = parseAxisValuesInput(axisDraft.values);
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 font-semibold text-slate-800">
            <Layers3 className="h-4 w-4 text-slate-400" />
            Matrix builder
          </div>
          <div className="mt-1 text-xs text-slate-500">
            {runCountExpression(matrix.axes, matrix.mode, matrix.manualCoordinates)}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={matrix.name}
            onChange={(event) => onMatrixNameChange(event.target.value)}
            className="h-8 w-48 rounded-md border border-slate-200 px-2.5 text-xs text-slate-700 focus:border-brand-300 focus:outline-none focus:ring-2 focus:ring-brand-100"
            aria-label="Matrix name"
          />
          <ModeToggle value={matrix.mode} onChange={onModeChange} />
          <button
            type="button"
            disabled={busy || ghostRows.length === 0}
            onClick={onStage}
            className="inline-flex h-8 items-center gap-2 rounded-md bg-brand-500 px-3 text-xs font-semibold text-white hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <PlayCircle className="h-3.5 w-3.5" />
            Stage {ghostRows.length} run{ghostRows.length === 1 ? '' : 's'}
          </button>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {matrix.axes.map((axis) => (
          <div
            key={axis.id}
            className="inline-flex max-w-full items-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs"
          >
            <span className="min-w-0">
              <span className="font-semibold text-slate-700">{axis.label}</span>
              <span className="ml-1 text-slate-400">{axis.values.length}</span>
              <span className="ml-2 text-slate-500">{axis.path}</span>
            </span>
            <button
              type="button"
              className="text-slate-400 hover:text-slate-700"
              onClick={() => onEditAxis(axis)}
              title="Edit axis"
            >
              <Pencil className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              className="text-slate-400 hover:text-red-600"
              onClick={() => onRemoveAxis(axis.id)}
              title="Remove axis"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
        {matrix.axes.length === 0 && (
          <div className="rounded-md border border-dashed border-slate-200 px-3 py-2 text-xs text-slate-400">
            Add one sweep axis to preview staged runs.
          </div>
        )}
      </div>

      <div className="mt-4 grid gap-3 border-t border-slate-100 pt-4 md:grid-cols-[minmax(8rem,0.7fr)_minmax(14rem,1.2fr)_minmax(12rem,1fr)_auto] md:items-end">
        <label className="block text-xs font-medium text-slate-600">
          <span>Label</span>
          <input
            value={axisDraft.label}
            onChange={(event) => onAxisDraftChange({ ...axisDraft, label: event.target.value })}
            className="mt-1 h-9 w-full rounded-md border border-slate-200 px-2.5 text-sm text-slate-700 focus:border-brand-300 focus:outline-none focus:ring-2 focus:ring-brand-100"
          />
        </label>
        <label className="block text-xs font-medium text-slate-600">
          <span>Path</span>
          <input
            value={axisDraft.path}
            onChange={(event) => onAxisDraftChange({ ...axisDraft, path: event.target.value })}
            className={clsx(
              'mt-1 h-9 w-full rounded-md border px-2.5 text-sm text-slate-700 focus:outline-none focus:ring-2',
              pathError
                ? 'border-red-200 focus:border-red-300 focus:ring-red-100'
                : 'border-slate-200 focus:border-brand-300 focus:ring-brand-100'
            )}
          />
          {pathError && <span className="mt-1 block text-[11px] text-red-600">{pathError}</span>}
        </label>
        <label className="block text-xs font-medium text-slate-600">
          <span>Values</span>
          <input
            value={axisDraft.values}
            onChange={(event) => onAxisDraftChange({ ...axisDraft, values: event.target.value })}
            className="mt-1 h-9 w-full rounded-md border border-slate-200 px-2.5 text-sm text-slate-700 focus:border-brand-300 focus:outline-none focus:ring-2 focus:ring-brand-100"
          />
          <span className="mt-1 block text-[11px] text-slate-400">
            {values.length} value{values.length === 1 ? '' : 's'}
          </span>
        </label>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onCommitAxis}
            disabled={Boolean(pathError) || values.length === 0}
            className="inline-flex h-9 items-center gap-2 rounded-md bg-slate-800 px-3 text-xs font-semibold text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Plus className="h-3.5 w-3.5" />
            {editingAxisId ? 'Update' : 'Add'}
          </button>
          {editingAxisId && (
            <button
              type="button"
              onClick={onCancelEdit}
              className="h-9 rounded-md border border-slate-200 px-3 text-xs font-semibold text-slate-600 hover:bg-slate-50"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

function ModeToggle({
  value,
  onChange,
}: {
  value: TrainMatrixMode;
  onChange: (value: TrainMatrixMode) => void;
}) {
  return (
    <div className="inline-flex h-8 rounded-md border border-slate-200 bg-slate-50 p-0.5">
      {(['cross', 'zip'] as const).map((mode) => (
        <button
          key={mode}
          type="button"
          onClick={() => onChange(mode)}
          className={clsx(
            'rounded px-2.5 text-xs font-semibold capitalize',
            value === mode ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
          )}
        >
          {mode}
        </button>
      ))}
    </div>
  );
}

function BulkEditPanel({
  selectedCount,
  pendingSelectionCount,
  axes,
  axisId,
  verb,
  values,
  ghostRows,
  applyError,
  busy,
  onAxisChange,
  onVerbChange,
  onValuesChange,
  onApply,
}: {
  selectedCount: number;
  pendingSelectionCount: number;
  axes: TrainAxisColumn[];
  axisId: string;
  verb: BulkEditVerb;
  values: string;
  ghostRows: TrainMatrixGhostRow[];
  applyError: string | null;
  busy: boolean;
  onAxisChange: (axisId: string) => void;
  onVerbChange: (verb: BulkEditVerb) => void;
  onValuesChange: (values: string) => void;
  onApply: () => void;
}) {
  const canApply = pendingSelectionCount > 0 && ghostRows.length > 0 && !applyError && !busy;
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="font-semibold text-slate-800">Bulk edit</div>
          <div className="mt-1 text-xs text-slate-500">
            {pendingSelectionCount} pending of {selectedCount} selected
          </div>
        </div>
        <div className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600">
          {ghostRows.length} preview
        </div>
      </div>
      <div className="mt-3 grid gap-3 md:grid-cols-[10rem_10rem_minmax(12rem,1fr)_auto] md:items-end">
        <label className="text-xs font-medium text-slate-600">
          <span>Verb</span>
          <select
            value={verb}
            onChange={(event) => onVerbChange(event.target.value as BulkEditVerb)}
            className="mt-1 h-9 w-full rounded-md border border-slate-200 px-2 text-sm text-slate-700"
          >
            <option value="keep">Keep</option>
            <option value="set">Set</option>
            <option value="distribute">Distribute</option>
            <option value="cross">Cross</option>
          </select>
        </label>
        <label className="text-xs font-medium text-slate-600">
          <span>Axis</span>
          <select
            value={axisId}
            onChange={(event) => onAxisChange(event.target.value)}
            className="mt-1 h-9 w-full rounded-md border border-slate-200 px-2 text-sm text-slate-700"
          >
            {axes.map((axis) => (
              <option key={axis.id} value={axis.id}>{axis.label}</option>
            ))}
          </select>
        </label>
        <label className="text-xs font-medium text-slate-600">
          <span>Values</span>
          <input
            value={values}
            onChange={(event) => onValuesChange(event.target.value)}
            className="mt-1 h-9 w-full rounded-md border border-slate-200 px-2.5 text-sm text-slate-700"
          />
        </label>
        <button
          type="button"
          disabled={!canApply}
          onClick={onApply}
          className="inline-flex h-9 items-center justify-center gap-2 rounded-md bg-slate-800 px-3 text-xs font-semibold text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Apply & restage
        </button>
      </div>
      {applyError && ghostRows.length > 0 && (
        <div className="mt-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {applyError}
        </div>
      )}
      <div className="mt-3 max-h-24 overflow-y-auto rounded-md border border-slate-100 bg-slate-50 p-2">
        {ghostRows.length === 0 ? (
          <div className="text-xs text-slate-400">
            Select pending rows and choose Set, Distribute, or Cross to preview edited rows.
          </div>
        ) : (
          <div className="grid gap-1 text-xs text-slate-600 sm:grid-cols-2">
            {ghostRows.slice(0, 12).map((row) => (
              <div key={row.id} className="truncate">{row.label}</div>
            ))}
            {ghostRows.length > 12 && <div className="text-slate-400">+{ghostRows.length - 12} more</div>}
          </div>
        )}
      </div>
    </section>
  );
}

function LineageProjectionPanel({
  projection,
  focusedId,
  previewId,
  syncMode,
  onPreview,
  onFocusNode,
}: {
  projection: LineageProjection;
  focusedId: string | null;
  previewId: string | null;
  syncMode: 'linked' | 'decoupled';
  onPreview: (id: string | null) => void;
  onFocusNode: (node: LineageProjectionNode) => void;
}) {
  const nodesById = useMemo(
    () => new Map(projection.nodes.map((node) => [node.id, node])),
    [projection.nodes]
  );
  const visibleGroups = projection.groups
    .map((group) => ({
      ...group,
      nodes: group.nodeIds
        .map((id) => nodesById.get(id))
        .filter((node): node is LineageProjectionNode => Boolean(node)),
    }))
    .filter((group) => group.nodes.length > 0);

  return (
    <section className="max-w-full overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          <GitBranch className="h-4 w-4 shrink-0 text-slate-400" />
          <div className="font-semibold text-slate-800">Lineage</div>
          <div className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600">
            {projection.nodes.length} nodes
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="rounded-full bg-slate-100 px-2 py-0.5">{projection.edges.length} ParentRefs</span>
          <span
            className={clsx(
              'inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium',
              syncMode === 'linked' ? 'bg-brand-50 text-brand-700' : 'bg-slate-100 text-slate-600'
            )}
          >
            {syncMode === 'linked' ? <Pin className="h-3 w-3" /> : <PinOff className="h-3 w-3" />}
            {syncMode === 'linked' ? 'Linked' : 'Decoupled'}
          </span>
        </div>
      </div>
      {projection.nodes.length === 0 ? (
        <EmptyCollection title="No lineage" detail="No manifests recorded in stage collections." />
      ) : (
        <div className="space-y-4 p-4">
          <div className="grid gap-3 xl:grid-cols-4">
            {visibleGroups.map((group) => (
              <div key={group.id} className="min-w-0 rounded-md border border-slate-100 bg-slate-50/60 p-3">
                <div className="mb-2 flex items-center justify-between gap-2 text-xs">
                  <span className="truncate font-semibold text-slate-700">{group.label}</span>
                  <span className="shrink-0 text-[11px] text-slate-400">{group.nodes.length}</span>
                </div>
                <div className="space-y-2">
                  {group.nodes.map((node) => (
                    <LineageNodeButton
                      key={node.id}
                      node={node}
                      focused={focusedId === node.id}
                      previewed={previewId === node.id}
                      onPreview={onPreview}
                      onFocusNode={onFocusNode}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          {projection.edges.length > 0 && (
            <div className="max-h-36 overflow-y-auto rounded-md border border-slate-100">
              {projection.edges.map((edge) => {
                const parent = nodesById.get(edge.parentId);
                const child = nodesById.get(edge.childId);
                return (
                  <div
                    key={edge.id}
                    className="grid gap-2 border-b border-slate-100 px-3 py-2 text-xs last:border-b-0 sm:grid-cols-[minmax(0,1fr)_12rem_minmax(0,1fr)]"
                  >
                    <span className="truncate font-medium text-slate-700" title={edge.parentId}>
                      {parent?.label ?? edge.parentId}
                    </span>
                    <LineageEdgeSummary edge={edge} />
                    <span className="truncate font-medium text-slate-700" title={edge.childId}>
                      {child?.label ?? edge.childId}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function LineageEdgeSummary({ edge }: { edge: LineageProjectionEdge }) {
  return (
    <div className="min-w-0 text-center">
      <div className="truncate text-[11px] font-semibold uppercase text-slate-400">
        {edge.role ?? 'parent'} -&gt;
      </div>
      {(edge.status || edge.reason) && (
        <div className="mt-1 flex min-w-0 flex-wrap justify-center gap-1">
          {edge.status && (
            <span
              className={clsx(
                'max-w-full truncate rounded-full px-2 py-0.5 text-[10px] font-semibold',
                lineageStatusClass(edge.status)
              )}
            >
              {lineageStatusLabel(edge.status)}
            </span>
          )}
          {edge.reason && (
            <span className="max-w-full truncate rounded bg-amber-50 px-1.5 py-0.5 text-[10px] text-amber-700">
              {edge.reason}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function LineageNodeButton({
  node,
  focused,
  previewed,
  onPreview,
  onFocusNode,
}: {
  node: LineageProjectionNode;
  focused: boolean;
  previewed: boolean;
  onPreview: (id: string | null) => void;
  onFocusNode: (node: LineageProjectionNode) => void;
}) {
  const clickable = Boolean(node.focusStageId && node.focusCollectionId);
  return (
    <button
      type="button"
      aria-disabled={!clickable}
      onMouseEnter={() => onPreview(node.id)}
      onMouseLeave={() => onPreview(null)}
      onClick={() => {
        if (clickable) onFocusNode(node);
      }}
      className={clsx(
        'w-full min-w-0 rounded-md border bg-white px-3 py-2 text-left text-xs transition-colors',
        clickable ? 'hover:border-brand-200 hover:bg-brand-50/40' : 'cursor-default',
        previewed && 'border-sky-200 bg-sky-50/70',
        focused ? 'border-brand-300 ring-1 ring-inset ring-brand-200' : 'border-slate-200'
      )}
      title={node.statusReason ?? node.id}
    >
      <div className="flex min-w-0 items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate font-semibold text-slate-800">{node.label}</div>
          <div className="mt-0.5 truncate text-[11px] text-slate-400">{node.id}</div>
        </div>
        <span className={clsx('shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold', lineageStatusClass(node.status))}>
          {lineageStatusLabel(node.status)}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
          {node.stageKind}
        </span>
        {node.role && (
          <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
            {node.role}
          </span>
        )}
        {node.statusReason && (
          <span className="truncate rounded bg-amber-50 px-1.5 py-0.5 text-[10px] text-amber-700">
            {node.statusReason}
          </span>
        )}
      </div>
    </button>
  );
}

function lineageStatusLabel(status: string): string {
  return status.replace(/_/g, ' ');
}

function lineageStatusClass(status: string): string {
  if (status === 'completed') return 'bg-emerald-50 text-emerald-700';
  if (status === 'running') return 'bg-brand-50 text-brand-700';
  if (status === 'failed' || status === 'cancelled') return 'bg-red-50 text-red-700';
  if (status === 'stale' || status === 'skipped' || status === 'not_applicable') {
    return 'bg-amber-50 text-amber-700';
  }
  if (status === 'pending' || status === 'planned') return 'bg-sky-50 text-sky-700';
  return 'bg-slate-100 text-slate-600';
}

function RunTable({
  title,
  rows,
  allRows,
  ghostRows = [],
  axisColumns = [],
  metricColumns,
  selectedIds,
  focusedId,
  previewId,
  syncMode = 'linked',
  collapsedSets = new Set<string>(),
  view,
  showHistory,
  sort,
  bestRunId,
  progressByRunId,
  progressByGroupId,
  onViewChange,
  onToggleHistory,
  onSortChange,
  onToggle,
  onSelectAll,
  onSelectBest,
  onClear,
  onPreview,
  onCommitFocus,
  onSyncModeChange,
  onToggleSet,
  onOpenDetails,
  onViewSnapshot,
  onRestageRun,
  onLifecycleAction,
}: {
  title: string;
  rows: TrainingRunSummary[];
  allRows: TrainingRunSummary[];
  ghostRows?: TrainMatrixGhostRow[];
  axisColumns?: TrainAxisColumn[];
  metricColumns: MetricColumnSpec[];
  selectedIds: Set<string>;
  focusedId?: string | null;
  previewId?: string | null;
  syncMode?: 'linked' | 'decoupled';
  collapsedSets?: Set<string>;
  view: RunView;
  showHistory: boolean;
  sort: SortState;
  bestRunId: string | null;
  progressByRunId?: Map<string, string>;
  progressByGroupId?: Map<string, string>;
  onViewChange: (view: RunView) => void;
  onToggleHistory: () => void;
  onSortChange: (sort: SortState) => void;
  onToggle: (id: string) => void;
  onSelectAll: () => void;
  onSelectBest: () => void;
  onClear: () => void;
  onPreview?: (id: string | null) => void;
  onCommitFocus?: (run: TrainingRunSummary) => void;
  onSyncModeChange?: (mode: 'linked' | 'decoupled') => void;
  onToggleSet?: (runSetId: string) => void;
  onOpenDetails: (run: TrainingRunSummary) => void;
  onViewSnapshot?: (run: TrainingRunSummary) => void;
  onRestageRun?: (run: TrainingRunSummary) => void;
  onLifecycleAction?: (
    action: 'cancel' | 'delete' | 'supersede',
    run: TrainingRunSummary
  ) => void;
}) {
  const axisTemplateColumns =
    axisColumns.length > 0 ? ` repeat(${axisColumns.length}, minmax(5.5rem,0.8fr))` : '';
  const metricTemplateColumns =
    metricColumns.length > 0
      ? ` repeat(${metricColumns.length}, minmax(4.75rem,0.8fr))`
      : '';
  const gridTemplateColumns = `1.5rem minmax(12rem,1.8fr) minmax(7rem,1fr)${axisTemplateColumns}${metricTemplateColumns} minmax(4.75rem,0.7fr) minmax(11rem,0.9fr)`;
  const groups = useMemo(() => groupTrainingRows(rows), [rows]);
  return (
    <section className="max-w-full overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="font-semibold text-slate-800">{title}</div>
          <div className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600">
            {selectedIds.size} selected
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <SegmentedFilter value={view} onChange={onViewChange} />
          <button
            type="button"
            className={clsx(
              'rounded-md border px-3 py-1.5 text-xs font-semibold',
              showHistory
                ? 'border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100'
                : 'border-slate-200 text-slate-600 hover:bg-slate-50'
            )}
            onClick={onToggleHistory}
          >
            {showHistory ? 'Hide history' : 'Show history'}
          </button>
          <button
            type="button"
            className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
            onClick={onSelectAll}
          >
            Select all
          </button>
          <button
            type="button"
            className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
            onClick={onSelectBest}
          >
            Select lowest loss
          </button>
          <button
            type="button"
            className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
            onClick={onClear}
          >
            Clear
          </button>
          {onSyncModeChange && (
            <button
              type="button"
              className={clsx(
                'inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-semibold',
                syncMode === 'linked'
                  ? 'border-brand-200 bg-brand-50 text-brand-700 hover:bg-brand-100'
                  : 'border-slate-200 text-slate-600 hover:bg-slate-50'
              )}
              onClick={() => onSyncModeChange(syncMode === 'linked' ? 'decoupled' : 'linked')}
              title={
                syncMode === 'linked'
                  ? 'Hover previews and row focus update the top pane.'
                  : 'Run table focus is decoupled from top-pane preview.'
              }
            >
              {syncMode === 'linked' ? (
                <Pin className="h-3.5 w-3.5" />
              ) : (
                <PinOff className="h-3.5 w-3.5" />
              )}
              {syncMode === 'linked' ? 'Linked' : 'Decoupled'}
            </button>
          )}
        </div>
      </div>
      {ghostRows.length > 0 && (
        <GhostRunPreview rows={ghostRows} axisColumns={axisColumns} gridTemplateColumns={gridTemplateColumns} />
      )}
      {rows.length === 0 ? (
        <EmptyCollection
          title={allRows.length === 0 ? 'No runs yet' : 'No rows in this view'}
          detail={
            allRows.length === 0
              ? 'Build a matrix above and stage pending runs.'
              : 'Change the filter or select runs from the full collection.'
          }
        />
      ) : (
        <div className="overflow-x-auto">
          <div className="min-w-full">
            <div
              className="grid items-start gap-2 border-b border-slate-100 px-4 py-2 text-[11px] font-semibold text-slate-500"
              style={{ gridTemplateColumns }}
            >
              <div />
              <div>Run</div>
              <SortHeader label="Progress" sortKey="progress" sort={sort} onChange={onSortChange} />
              {axisColumns.map((axis) => (
                <SortHeader
                  key={axis.id}
                  label={axis.label}
                  sortKey={`axis:${axis.id}`}
                  sort={sort}
                  onChange={onSortChange}
                />
              ))}
              {metricColumns.map((column) => (
                <SortHeader
                  key={column.id}
                  label={column.label}
                  units={column.units}
                  sortKey={column.id}
                  sort={sort}
                  onChange={onSortChange}
                />
              ))}
              <div>Checkpoint</div>
              <div>Actions</div>
            </div>
            <div className="divide-y divide-slate-100">
              {groups.map((group) => {
                const collapsed = collapsedSets.has(group.id);
                const groupProgress = progressByGroupId?.get(group.id) ?? null;
                const showHeader =
                  groups.length > 1 ||
                  group.id !== UNGROUPED_RUN_GROUP_ID ||
                  Boolean(groupProgress);
                return (
                  <div key={group.id}>
                    {showHeader && (
                      <RunSetHeader
                        group={group}
                        progressLabel={groupProgress}
                        collapsed={collapsed}
                        onToggle={() => onToggleSet?.(group.id)}
                      />
                    )}
                    {!collapsed &&
                      group.rows.map((row) => (
                        <TrainingRunRow
                          key={row.id}
                          row={row}
                          axisColumns={axisColumns}
                          metricColumns={metricColumns}
                          gridTemplateColumns={gridTemplateColumns}
                          selected={selectedIds.has(row.id)}
                          focused={focusedId === row.id}
                          previewed={previewId === row.id}
                          isBest={bestRunId === row.id}
                          progressLabel={progressByRunId?.get(row.id) ?? null}
                          onToggle={() => onToggle(row.id)}
                          onPreview={onPreview ? () => onPreview(row.id) : undefined}
                          onPreviewEnd={onPreview ? () => onPreview(null) : undefined}
                          onCommitFocus={onCommitFocus ? () => onCommitFocus(row) : undefined}
                          onOpenDetails={() => onOpenDetails(row)}
                          onViewSnapshot={onViewSnapshot ? () => onViewSnapshot(row) : undefined}
                          onRestageRun={onRestageRun ? () => onRestageRun(row) : undefined}
                          onLifecycleAction={onLifecycleAction}
                        />
                      ))}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

interface TrainingRunGroup {
  id: string;
  label: string;
  rows: TrainingRunSummary[];
  counts: Record<string, number>;
}

function groupTrainingRows(rows: TrainingRunSummary[]): TrainingRunGroup[] {
  const groups = new Map<string, TrainingRunGroup>();
  for (const row of rows) {
    const id = row.runSetId ?? 'ungrouped';
    const group = groups.get(id) ?? {
      id,
      label: row.runSetId ? `Run set ${row.runSetId}` : 'Individual runs',
      rows: [],
      counts: {},
    };
    group.rows.push(row);
    group.counts[row.status] = (group.counts[row.status] ?? 0) + 1;
    groups.set(id, group);
  }
  return Array.from(groups.values());
}

function RunSetHeader({
  group,
  progressLabel,
  collapsed,
  onToggle,
}: {
  group: TrainingRunGroup;
  progressLabel: string | null;
  collapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="flex w-full items-center justify-between gap-3 bg-slate-50 px-4 py-2 text-left text-xs hover:bg-slate-100"
    >
      <div className="flex min-w-0 items-center gap-2">
        {collapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
        <span className="truncate font-semibold text-slate-700">{group.label}</span>
        <span className="shrink-0 text-slate-400">{group.rows.length} runs</span>
        {progressLabel && (
          <span className="shrink-0 rounded-full bg-brand-50 px-2 py-0.5 text-[10px] font-semibold text-brand-700">
            job {progressLabel}
          </span>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-1">
        {Object.entries(group.counts).map(([status, count]) => (
          <span key={status} className={clsx(
            'rounded-full px-2 py-0.5 text-[10px] font-semibold',
            status === 'completed'
              ? 'bg-emerald-50 text-emerald-700'
              : status === 'failed'
                ? 'bg-red-50 text-red-700'
                : status === 'running'
                  ? 'bg-brand-50 text-brand-700'
                  : status === 'stale'
                    ? 'bg-amber-50 text-amber-700'
                    : 'bg-slate-100 text-slate-600'
          )}>
            {status} {count}
          </span>
        ))}
      </div>
    </button>
  );
}

function GhostRunPreview({
  rows,
  axisColumns,
  gridTemplateColumns,
}: {
  rows: TrainMatrixGhostRow[];
  axisColumns: TrainAxisColumn[];
  gridTemplateColumns: string;
}) {
  return (
    <div className="border-b border-slate-100 bg-sky-50/60">
      <div className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-sky-800">
        <Eye className="h-3.5 w-3.5" />
        Previewing {rows.length} unstaged run{rows.length === 1 ? '' : 's'}
      </div>
      <div className="max-h-44 overflow-y-auto">
        {rows.slice(0, 24).map((row) => (
          <div
            key={row.id}
            className="grid items-center gap-2 border-t border-sky-100 px-4 py-1.5 text-xs text-slate-600"
            style={{ gridTemplateColumns }}
          >
            <div />
            <div className="min-w-0 truncate font-medium text-slate-700">{row.label}</div>
            <div className="text-sky-700">Ghost</div>
            {axisColumns.map((axis) => (
              <MetricValue
                key={axis.id}
                value={
                  row.axisCoordinates[axis.id] === undefined
                    ? 'Not varied'
                    : formatAxisValue(row.axisCoordinates[axis.id])
                }
              />
            ))}
            <div className="text-slate-400">Pending</div>
            <div className="text-slate-400">Stage first</div>
          </div>
        ))}
        {rows.length > 24 && (
          <div className="border-t border-sky-100 px-4 py-2 text-xs text-sky-700">
            +{rows.length - 24} more preview rows
          </div>
        )}
      </div>
    </div>
  );
}

function TrainingRunRow({
  row,
  axisColumns,
  metricColumns,
  gridTemplateColumns,
  selected,
  focused,
  previewed,
  isBest,
  progressLabel,
  onToggle,
  onPreview,
  onPreviewEnd,
  onCommitFocus,
  onOpenDetails,
  onViewSnapshot,
  onRestageRun,
  onLifecycleAction,
}: {
  row: TrainingRunSummary;
  axisColumns: TrainAxisColumn[];
  metricColumns: MetricColumnSpec[];
  gridTemplateColumns: string;
  selected: boolean;
  focused: boolean;
  previewed: boolean;
  isBest: boolean;
  progressLabel: string | null;
  onToggle: () => void;
  onPreview?: () => void;
  onPreviewEnd?: () => void;
  onCommitFocus?: () => void;
  onOpenDetails: () => void;
  onViewSnapshot?: () => void;
  onRestageRun?: () => void;
  onLifecycleAction?: (
    action: 'cancel' | 'delete' | 'supersede',
    run: TrainingRunSummary
  ) => void;
}) {
  const progress = progressLabel ??
    (row.status === 'stale'
      ? `Stale${row.staleReason ? ` (${row.staleReason})` : ''}`
      : row.status === 'running'
      ? 'Running'
      : row.warmupBatches !== null
        ? `${row.warmupBatches.toLocaleString()}/${row.warmupBatches.toLocaleString()}`
        : row.status === 'pending'
          ? 'Pending'
          : 'Not recorded');
  const checkpoint = row.checkpointAvailable && row.warmupBatches !== null
    ? row.warmupBatches.toLocaleString()
    : row.checkpointAvailable
      ? 'Available'
      : 'None';
  const complete = row.status === 'completed';

  return (
    <div
      onMouseEnter={onPreview}
      onMouseLeave={onPreviewEnd}
      onClick={onCommitFocus}
      className={clsx(
        'grid items-center gap-2 px-4 py-2 text-xs transition-colors',
        selected && 'bg-brand-50/40',
        previewed && 'bg-sky-50/70',
        focused && 'ring-1 ring-inset ring-brand-200'
      )}
      style={{ gridTemplateColumns }}
    >
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          onToggle();
        }}
        className={clsx(
          'flex h-6 w-6 items-center justify-center rounded-md border transition-colors',
          selected
            ? 'border-brand-500 bg-brand-500 text-white'
            : 'border-slate-200 bg-white text-slate-400 hover:border-brand-200 hover:text-brand-500'
        )}
        aria-label={selected ? `Deselect ${row.label}` : `Select ${row.label}`}
      >
        {selected ? <Check className="h-3.5 w-3.5" /> : <Circle className="h-3.5 w-3.5" />}
      </button>
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="truncate font-semibold text-slate-800">{row.label}</span>
          {isBest && (
            <span className="shrink-0 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-emerald-700">
              lowest loss
            </span>
          )}
          {row.stale && (
            <span className="shrink-0 rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-700">
              stale
            </span>
          )}
          {row.supersededBy && (
            <span
              className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-600"
              title={`superseded by ${row.supersededBy}`}
            >
              superseded
            </span>
          )}
        </div>
        {row.statusReason && (
          <div className="mt-0.5 truncate text-[11px] text-amber-700" title={row.statusReason}>
            {row.statusReason}
          </div>
        )}
      </div>
      <div className="flex min-w-0 items-center gap-1.5 font-medium text-slate-700" title={progress}>
        {complete && <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-600" />}
        {row.status === 'running' && <Activity className="h-3.5 w-3.5 shrink-0 text-brand-500" />}
        {row.status === 'stale' && <RotateCcw className="h-3.5 w-3.5 shrink-0 text-amber-600" />}
        <span className="truncate">{progress}</span>
      </div>
      {axisColumns.map((axis) => (
        <MetricValue
          key={axis.id}
          value={
            row.axisCoordinates[axis.id] === undefined
              ? 'Not varied'
              : formatAxisValue(row.axisCoordinates[axis.id])
          }
        />
      ))}
      {metricColumns.map((column) => (
        <MetricValue
          key={column.id}
          value={formatMetric(trainingRunMetricValue(row, column.id), 3)}
        />
      ))}
      <div className="flex items-center gap-1 text-slate-600">
        {!row.checkpointAvailable && <XCircle className="h-3.5 w-3.5 shrink-0 text-red-500" />}
        <span className="truncate" title={checkpoint}>{checkpoint}</span>
      </div>
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            onOpenDetails();
          }}
          className="flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          aria-label={`Show details for ${row.label}`}
          title="Details"
        >
          <Info className="h-4 w-4" />
        </button>
        <button
          type="button"
          disabled
          className="flex h-8 w-8 cursor-not-allowed items-center justify-center rounded-md text-slate-300"
          aria-label={`Launch unavailable for ${row.label}`}
          title="Manifest launch requires queue execution support"
        >
          <PlayCircle className="h-4 w-4" />
        </button>
        {onViewSnapshot && (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onViewSnapshot();
            }}
            className="flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-sky-50 hover:text-sky-700"
            aria-label={`View snapshot for ${row.label}`}
            title="View snapshot"
          >
            <Eye className="h-4 w-4" />
          </button>
        )}
        {onRestageRun && (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onRestageRun();
            }}
            className="flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-700"
            aria-label={`Restage ${row.label}`}
            title="Restage run"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        )}
        {onLifecycleAction && row.status === 'pending' && (
          <>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                onLifecycleAction('cancel', row);
              }}
              className="flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-amber-50 hover:text-amber-700"
              title="Cancel pending run"
            >
              <X className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                onLifecycleAction('delete', row);
              }}
              className="flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-red-50 hover:text-red-700"
              title="Delete pending run"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </>
        )}
        {onLifecycleAction && row.status === 'completed' && (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onLifecycleAction('supersede', row);
            }}
            className="flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-700"
            title="Supersede run"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
}

function RunDetailPane({
  run,
  frozenSnapshot,
  currentSpecHashes,
  lossHistory,
  onViewSnapshot,
  onBackToDraft,
  onPromoteToDraft,
  onRestage,
  onDownloadCheckpoint,
}: {
  run: TrainingRunSummary | null;
  frozenSnapshot: FrozenSnapshotProjection | null;
  currentSpecHashes: Record<string, string | null>;
  lossHistory: TrainingProgress[];
  onViewSnapshot: (run: TrainingRunSummary) => void;
  onBackToDraft: () => void;
  onPromoteToDraft: (run: TrainingRunSummary) => void;
  onRestage: (run: TrainingRunSummary) => void;
  onDownloadCheckpoint: (run: TrainingRunSummary) => void;
}) {
  if (!run) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="font-semibold text-slate-800">Run detail</div>
        <div className="mt-3 rounded-md border border-dashed border-slate-200 p-3 text-xs text-slate-400">
          Select a run to inspect manifest details.
        </div>
      </section>
    );
  }
  const lossValues = lossHistory
    .map((point) => (typeof point.loss === 'number' ? point.loss : null))
    .filter((value): value is number => value !== null);
  const latestLoss = lossValues.at(-1) ?? run.finalValidationLoss;
  const snapshotOpen = frozenSnapshot?.runId === run.id;
  const specComparisons = snapshotOpen && frozenSnapshot
    ? compareSpecHashes(frozenSnapshot.specHashes, currentSpecHashes)
    : [];
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate font-semibold text-slate-800">{run.label}</div>
          <div className="mt-1 text-xs text-slate-500">{runParameterSummary(run)}</div>
        </div>
        <StatusDot status={run.status} />
      </div>
      <div className="mt-4 grid gap-2 text-xs">
        <DetailField label="Manifest" value={run.provenanceId} />
        <DetailField label="Job" value={run.jobId ?? 'Not recorded'} />
        <DetailField label="Set" value={run.runSetId ?? 'Not grouped'} />
        <DetailField label="Variant" value={run.variant ?? 'Not recorded'} />
        <DetailField label="Batch size" value={run.batchSize?.toLocaleString() ?? 'Not recorded'} />
        <DetailField label="Warmup batches" value={run.warmupBatches?.toLocaleString() ?? 'Not recorded'} />
        {Object.entries(run.axisCoordinates).map(([axisId, value]) => (
          <DetailField key={axisId} label={`Axis ${axisId}`} value={formatAxisValue(value)} />
        ))}
      </div>
      <div className="mt-4 rounded-md border border-slate-100 bg-slate-50 p-3">
        <div className="flex items-center justify-between gap-3 text-xs">
          <span className="font-semibold text-slate-700">Loss</span>
          <span className="text-slate-500">{formatMetric(latestLoss, 4)}</span>
        </div>
        <LossSparkline values={lossValues} />
      </div>
      {snapshotOpen && frozenSnapshot && (
        <div className="mt-4 rounded-md border border-sky-100 bg-sky-50 p-3 text-xs">
          <div className="flex items-center gap-2 font-semibold text-sky-800">
            <Eye className="h-3.5 w-3.5" />
            Frozen snapshot
          </div>
          <div className="mt-2 grid gap-1 text-slate-600">
            <DetailField label="Manifest id" value={frozenSnapshot.manifestId ?? run.id} />
            <DetailField label="URI" value={run.uri ?? 'Not recorded'} />
            <DetailField label="Status" value={frozenSnapshot.runStatus} />
          </div>
          {specComparisons.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1">
              {specComparisons.map((comparison) => (
                <span
                  key={comparison.key}
                  className={clsx(
                    'rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold ring-1',
                    specHashStatusClass(comparison.status)
                  )}
                  title={`snapshot=${comparison.snapshotHash ?? 'missing'} current=${comparison.currentHash ?? 'missing'}`}
                >
                  {comparison.label} {specHashStatusLabel(comparison.status)}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
      <div className="mt-4 grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => onViewSnapshot(run)}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-sky-200 bg-sky-50 px-3 py-2 text-xs font-semibold text-sky-700 hover:bg-sky-100"
        >
          <Eye className="h-3.5 w-3.5" />
          View snapshot
        </button>
        <button
          type="button"
          onClick={() => onPromoteToDraft(run)}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50"
        >
          <Pencil className="h-3.5 w-3.5" />
          Promote
        </button>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <button
          type="button"
          disabled={!snapshotOpen}
          onClick={onBackToDraft}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <ChevronRight className="h-3.5 w-3.5 rotate-180" />
          Back to draft
        </button>
        <button
          type="button"
          onClick={() => onRestage(run)}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Restage
        </button>
      </div>
      <button
        type="button"
        disabled={!run.checkpointAvailable || !run.uri}
        onClick={() => onDownloadCheckpoint(run)}
        className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <Download className="h-3.5 w-3.5" />
        Checkpoint
      </button>
    </section>
  );
}

function LossSparkline({ values }: { values: number[] }) {
  if (values.length < 2) {
    return <div className="mt-2 h-12 rounded bg-white text-center text-[11px] leading-[3rem] text-slate-400">No stream yet</div>;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = values.map((value, index) => {
    const x = (index / Math.max(1, values.length - 1)) * 100;
    const y = 36 - ((value - min) / span) * 32;
    return `${x},${y}`;
  });
  return (
    <svg viewBox="0 0 100 40" className="mt-2 h-12 w-full rounded bg-white" role="img" aria-label="Loss chart">
      <polyline fill="none" stroke="currentColor" strokeWidth="2" points={points.join(' ')} className="text-brand-500" />
    </svg>
  );
}

function RunDetailOverlay({
  run,
  onClose,
}: {
  run: TrainingRunSummary;
  onClose: () => void;
}) {
  return (
    <div className="absolute inset-4 z-20 flex items-start justify-center overflow-y-auto rounded-lg bg-white/80 p-6 backdrop-blur-sm">
      <section className="w-full max-w-2xl rounded-lg border border-slate-200 bg-white p-5 shadow-lift">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-base font-semibold text-slate-800">{run.label}</div>
            <div className="mt-1 text-xs text-slate-500">{runParameterSummary(run)}</div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
          >
            Close
          </button>
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          <DetailField label="Source issue" value={run.sourceIssue ?? 'Not recorded'} />
          <DetailField label="Manifest" value={run.provenanceId} />
          <DetailField label="Variant" value={run.variant ?? 'Not recorded'} />
          <DetailField label="Batch size" value={run.batchSize?.toLocaleString() ?? 'Not recorded'} />
          <DetailField label="Warmup batches" value={run.warmupBatches?.toLocaleString() ?? 'Not recorded'} />
          <DetailField label="Replicates" value={run.replicateCount?.toLocaleString() ?? 'Not recorded'} />
          {run.uri && <DetailField label="Path" value={run.uri} wide />}
        </div>
      </section>
    </div>
  );
}

function EvaluationResult({
  row,
  selectedRun,
  focused,
  previewed,
  onPreview,
  onPreviewEnd,
  onCommitFocus,
  onOpenAnalyze,
  onViewSnapshot,
  snapshotBusy,
}: {
  row: EvaluationRunSummary;
  selectedRun: TrainingRunSummary | undefined;
  focused: boolean;
  previewed: boolean;
  onPreview: () => void;
  onPreviewEnd: () => void;
  onCommitFocus: () => void;
  onOpenAnalyze: () => void;
  onViewSnapshot: () => void;
  snapshotBusy: boolean;
}) {
  const snapshotAvailable = Boolean(row.uri);
  return (
    <div
      onMouseEnter={onPreview}
      onMouseLeave={onPreviewEnd}
      onClick={onCommitFocus}
      className={clsx(
        'rounded-md border border-slate-200 p-3 transition-colors',
        previewed && 'bg-sky-50/70',
        focused && 'ring-1 ring-inset ring-brand-200'
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="font-semibold text-slate-800">{row.label}</div>
        <StatusDot status={row.status} />
      </div>
      <div className="mt-2 space-y-1 text-xs text-slate-500">
        <div>{evaluationProtocolLabel(row)}</div>
        <div>
          Run: <span className="font-medium text-slate-700">{selectedRun?.label ?? 'Selection snapshot'}</span>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <button
          type="button"
          disabled={!snapshotAvailable || snapshotBusy}
          onClick={(event) => {
            event.stopPropagation();
            onViewSnapshot();
          }}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-sky-200 bg-sky-50 px-3 py-2 text-xs font-semibold text-sky-700 hover:bg-sky-100 disabled:cursor-not-allowed disabled:opacity-50"
          title={
            snapshotAvailable
              ? 'View the evaluation manifest snapshot'
              : 'Legacy evaluation rows do not expose a durable manifest snapshot yet.'
          }
        >
          <Eye className="h-3.5 w-3.5" />
          {snapshotBusy ? 'Loading' : 'Snapshot'}
        </button>
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            onOpenAnalyze();
          }}
          className="inline-flex items-center justify-center gap-2 rounded-md bg-brand-500 px-3 py-2 text-xs font-semibold text-white hover:bg-brand-600"
        >
          Analyze
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}

function MetricTracePanel({ metrics }: { metrics: ScenarioMetricSpec[] }) {
  const visible = uniqueMetricTraces(metrics).slice(0, 5);
  return (
    <section className="max-w-3xl rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="font-semibold text-slate-800">Metric sources</div>
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        {visible.map((metric) => (
          <div key={`${metric.source}:${metric.id}`} className="min-w-0 rounded-md border border-slate-100 px-2.5 py-2 text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-medium text-slate-700">{metric.label}</span>
              <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
                {metric.source}
              </span>
            </div>
            {metric.summary && (
              <div className="mt-0.5 truncate text-[11px] text-slate-400">
                {metric.summary}
              </div>
            )}
          </div>
        ))}
        {visible.length === 0 && <div className="text-xs text-slate-400">No metrics derived</div>}
      </div>
    </section>
  );
}

function uniqueMetricTraces(metrics: ScenarioMetricSpec[]): ScenarioMetricSpec[] {
  const byKey = new Map<string, ScenarioMetricSpec>();
  for (const metric of metrics) {
    if (metric.source === 'task_default') continue;
    const key = `${metric.source}:${metric.id}`;
    const existing = byKey.get(key);
    if (!existing || sourceTraceRank(metric.source) < sourceTraceRank(existing.source)) {
      byKey.set(key, metric);
    }
  }
  return Array.from(byKey.values()).sort((a, b) => {
    const source = sourceTraceRank(a.source) - sourceTraceRank(b.source);
    return source || a.label.localeCompare(b.label);
  });
}

function sourceTraceRank(source: ScenarioMetricSpec['source']): number {
  return ['objective', 'analysis', 'manifest', 'task_default'].indexOf(source);
}

function ExecutionTarget({
  value,
  onChange,
}: {
  value: ExecutionTargetChoice;
  onChange: (value: ExecutionTargetChoice) => void;
}) {
  const targets: Array<{
    id: ExecutionTargetChoice;
    icon: typeof Cpu;
    title: string;
    detail: string;
  }> = [
    {
      id: 'local',
      icon: Cpu,
      title: 'This machine',
      detail: 'CPU now; accelerator when available',
    },
    {
      id: 'managed',
      icon: Server,
      title: 'Managed worker',
      detail: 'RunPod or lab worker once connected',
    },
    {
      id: 'manual',
      icon: Activity,
      title: 'Manual endpoint',
      detail: 'Advanced connection details',
    },
  ];
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="font-semibold text-slate-800">Run target</div>
      <div className="mt-3 space-y-2">
        {targets.map((target) => {
          const Icon = target.icon;
          const active = value === target.id;
          return (
            <button
              key={target.id}
              type="button"
              onClick={() => onChange(target.id)}
              aria-pressed={active}
              className={clsx(
                'relative flex w-full items-center gap-3 rounded-md border px-3 py-2 text-left transition-colors',
                active
                  ? 'border-brand-500 bg-brand-100 text-brand-900 shadow-[inset_0_0_0_1px_rgba(37,99,235,0.35)]'
                  : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50'
              )}
            >
              {active && <span className="absolute inset-y-2 left-0 w-1 rounded-r-full bg-brand-500" />}
              <span
                className={clsx(
                  'flex h-7 w-7 shrink-0 items-center justify-center rounded-full border',
                  active
                    ? 'border-brand-600 bg-brand-600 text-white'
                    : 'border-slate-200 bg-slate-50 text-slate-500'
                )}
              >
                {active ? <CheckCircle2 className="h-4 w-4" /> : <Icon className="h-4 w-4" />}
              </span>
              <span className="min-w-0">
                <span className="block text-xs font-semibold">{target.title}</span>
                <span className={clsx('block text-[11px]', active ? 'text-brand-700' : 'text-slate-500')}>
                  {target.detail}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function TrainingProtocolEditor({
  protocol,
  onProtocolChange,
}: {
  protocol: TrainingProtocolSnapshot;
  onProtocolChange: (patch: Partial<TrainingProtocolSnapshot>) => void;
}) {
  const updateNumber = (
    key: 'learningRate' | 'batchCount' | 'batchSize',
    rawValue: string
  ) => {
    const value = Number(rawValue);
    if (!Number.isFinite(value)) return;
    onProtocolChange({ [key]: value });
  };

  const updateInteger = (key: 'batchCount' | 'batchSize', rawValue: string) => {
    const value = Number(rawValue);
    if (!Number.isFinite(value)) return;
    onProtocolChange({ [key]: Math.max(0, Math.round(value)) });
  };

  const updateCheckpointInterval = (rawValue: string) => {
    if (rawValue.trim() === '') {
      onProtocolChange({ checkpointInterval: null });
      return;
    }
    const value = Number(rawValue);
    if (!Number.isFinite(value)) return;
    onProtocolChange({ checkpointInterval: Math.max(0, Math.round(value)) });
  };

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <NumberField
        label="Learning rate"
        min={0}
        step={0.0001}
        value={protocol.learningRate}
        onChange={(value) => updateNumber('learningRate', value)}
      />
      <NumberField
        label="Batches"
        min={0}
        step={1}
        value={protocol.batchCount}
        onChange={(value) => updateInteger('batchCount', value)}
      />
      <NumberField
        label="Batch size"
        min={0}
        step={1}
        value={protocol.batchSize}
        onChange={(value) => updateInteger('batchSize', value)}
      />
      <NumberField
        label="Checkpoint every"
        min={0}
        step={1}
        value={protocol.checkpointInterval ?? ''}
        onChange={updateCheckpointInterval}
      />
    </div>
  );
}

function NumberField({
  label,
  min,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  step: number;
  value: number | string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block text-xs font-medium text-slate-600">
      <span>{label}</span>
      <input
        type="number"
        min={min}
        step={step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-1 w-full rounded-md border border-slate-200 px-2.5 py-2 text-sm text-slate-700 shadow-sm focus:border-brand-300 focus:outline-none focus:ring-2 focus:ring-brand-100"
      />
    </label>
  );
}

function SortHeader({
  label,
  units,
  sortKey,
  sort,
  onChange,
}: {
  label: string;
  units?: string | null;
  sortKey: SortKey;
  sort: SortState;
  onChange: (sort: SortState) => void;
}) {
  const active = sort.key === sortKey;
  return (
    <button
      type="button"
      onClick={() =>
        onChange({
          key: sortKey,
          direction: active && sort.direction === 'asc' ? 'desc' : 'asc',
        })
      }
      className={clsx('text-left hover:text-slate-700', active && 'text-slate-700')}
      title={units ? `${label} (${units})` : label}
    >
      <span className="block truncate">{label}</span>
      {units && <span className="block truncate text-[9px] normal-case tracking-normal">{units}</span>}
      {active && <span className="ml-1">{sort.direction === 'asc' ? '↑' : '↓'}</span>}
    </button>
  );
}

function SegmentedFilter({
  value,
  onChange,
}: {
  value: RunView;
  onChange: (value: RunView) => void;
}) {
  const items: Array<{ id: RunView; label: string }> = [
    { id: 'all', label: 'All' },
    { id: 'selected', label: 'Selected' },
    { id: 'best', label: 'Lowest loss' },
  ];
  return (
    <div className="inline-flex rounded-md border border-slate-200 bg-slate-50 p-0.5">
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          onClick={() => onChange(item.id)}
          className={clsx(
            'rounded px-2.5 py-1 text-xs font-semibold',
            value === item.id
              ? 'bg-white text-slate-800 shadow-sm'
              : 'text-slate-500 hover:text-slate-700'
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

function MetricValue({
  value,
  done = false,
}: {
  value: string;
  done?: boolean;
}) {
  return (
    <div className={clsx('truncate font-medium text-slate-700', done && 'text-emerald-700')} title={value}>
      {value}
    </div>
  );
}

function StatusDot({ status }: { status: string }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 text-[11px] font-medium',
        status === 'completed'
          ? 'text-emerald-700'
          : status === 'failed'
            ? 'text-red-700'
            : 'text-slate-500'
      )}
    >
      <span
        className={clsx(
          'h-2 w-2 rounded-full',
          status === 'completed'
            ? 'bg-emerald-500'
            : status === 'failed'
              ? 'bg-red-500'
              : 'bg-slate-300'
        )}
      />
      {status}
    </span>
  );
}

function ProtocolRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-md bg-slate-50 px-3 py-2 text-xs">
      <span className="text-slate-500">{label}</span>
      <span className="font-medium text-slate-700">{value}</span>
    </div>
  );
}

function DetailField({
  label,
  value,
  wide = false,
}: {
  label: string;
  value: string;
  wide?: boolean;
}) {
  return (
    <div className={clsx('min-w-0 rounded-md bg-slate-50 p-3 text-xs', wide && 'sm:col-span-2')}>
      <div className="font-semibold uppercase tracking-[0.12em] text-slate-400">{label}</div>
      <div className="mt-1 break-words text-slate-700">{value}</div>
    </div>
  );
}

function EmptyCollection({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="flex min-h-32 flex-col items-center justify-center px-4 py-8 text-center">
      <div className="font-semibold text-slate-700">{title}</div>
      <div className="mt-1 max-w-sm text-xs leading-5 text-slate-500">{detail}</div>
    </div>
  );
}
