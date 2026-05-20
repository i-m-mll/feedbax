import { useCallback, useMemo, useState } from 'react';
import clsx from 'clsx';
import {
  Activity,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronRight,
  Circle,
  Cpu,
  Info,
  PlayCircle,
  Server,
  SlidersHorizontal,
  XCircle,
} from 'lucide-react';
import { getScenario, getStageByKind, useWorkspaceStore } from '@/stores/workspaceStore';
import { useGraphStore } from '@/stores/graphStore';
import {
  bestTrainingRun,
  evaluationProtocolLabel,
  evaluationRunSummaries,
  formatMetric,
  runParameterSummary,
  selectedIds,
  trainingRunMetricValue,
  trainingInputSummaries,
  trainingRunSummaries,
  type EvaluationRunSummary,
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
  const updateStageDraft = useWorkspaceStore((state) => state.updateStageDraft);
  const updateActiveScenarioTrainingSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioTrainingSpec
  );
  const markDirty = useGraphStore((state) => state.markDirty);
  const [view, setView] = useState<RunView>('all');
  const [sort, setSort] = useState<SortState>({ key: 'final_validation_loss', direction: 'asc' });
  const [selectedRunIds, setSelectedRunIds] = useState<Set<string>>(() => new Set());
  const [detailsRun, setDetailsRun] = useState<TrainingRunSummary | null>(null);

  const trainStage = getStageByKind(workspace, 'train');
  const trainScenario = getScenario(workspace, trainStage?.scenario_id);
  const evalStage = getStageByKind(workspace, 'eval');
  const protocol = trainingProtocolSnapshot(trainStage, trainScenario);
  const metrics = useMemo(() => scenarioMetricSpecs(workspace), [workspace]);
  const rows = useMemo(() => trainingRunSummaries(trainStage), [trainStage]);
  const metricColumns = useMemo(() => runMetricColumns(metrics, rows), [metrics, rows]);
  const bestRow = useMemo(() => bestTrainingRun(rows), [rows]);
  const selectedRows = useMemo(
    () => rows.filter((row) => selectedRunIds.has(row.id)),
    [rows, selectedRunIds]
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

  const toggleRow = useCallback((id: string) => {
    setSelectedRunIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedRunIds(new Set(rows.map((row) => row.id)));
  }, [rows]);

  const selectBest = useCallback(() => {
    setSelectedRunIds(new Set(bestRow ? [bestRow.id] : []));
  }, [bestRow]);

  const clearSelection = useCallback(() => {
    setSelectedRunIds(new Set());
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

  return (
    <div className="relative h-full overflow-hidden bg-slate-50/40">
      <div className="h-full overflow-y-auto">
        <div className="mx-auto flex w-full max-w-7xl min-w-0 flex-col gap-5 px-6 py-5 text-sm text-slate-600">
        <section className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_20rem] lg:items-start">
          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-base font-semibold text-slate-800">Movement-ramp batch</div>
                <div className="mt-1 text-xs text-slate-500">
                  {rows.length} imported run{rows.length === 1 ? '' : 's'}
                </div>
              </div>
              {selectedRunIds.size > 0 && (
                <button
                  type="button"
                  disabled={!evalStage}
                  onClick={useForEvaluation}
                  className="inline-flex items-center gap-2 rounded-md bg-brand-500 px-3 py-2 text-xs font-semibold text-white shadow-sm hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Use selection in Evaluate
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
            <div className="mt-4 border-t border-slate-100 pt-4">
              <TrainingProtocolEditor
                protocol={protocol}
                onProtocolChange={updateProtocol}
              />
            </div>
          </div>
          <div className="space-y-3">
            <ExecutionTarget value={protocol.computeTarget} onChange={setTarget} />
          </div>
        </section>

        <RunTable
          title="Run collection"
          rows={visibleRows}
          allRows={rows}
          metricColumns={metricColumns}
          selectedIds={selectedRunIds}
          view={view}
          sort={sort}
          bestRunId={bestRow?.id ?? null}
          onViewChange={setView}
          onSortChange={setSort}
          onToggle={toggleRow}
          onSelectAll={selectAll}
          onSelectBest={selectBest}
          onClear={clearSelection}
          onOpenDetails={setDetailsRun}
        />

        <MetricTracePanel metrics={metrics} />
        </div>
      </div>
      {detailsRun && <RunDetailOverlay run={detailsRun} onClose={() => setDetailsRun(null)} />}
    </div>
  );
}

export function EvaluateCollectionPanel() {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const setActiveStage = useWorkspaceStore((state) => state.setActiveStage);
  const updateStageDraft = useWorkspaceStore((state) => state.updateStageDraft);
  const [view, setView] = useState<RunView>('all');
  const [sort, setSort] = useState<SortState>({ key: 'final_validation_loss', direction: 'asc' });
  const [detailsRun, setDetailsRun] = useState<TrainingRunSummary | null>(null);

  const trainStage = getStageByKind(workspace, 'train');
  const evalStage = getStageByKind(workspace, 'eval');
  const analysisStage = getStageByKind(workspace, 'analysis');
  const metrics = useMemo(() => scenarioMetricSpecs(workspace), [workspace]);
  const rows = useMemo(() => trainingInputSummaries(evalStage), [evalStage]);
  const metricColumns = useMemo(() => runMetricColumns(metrics, rows), [metrics, rows]);
  const bestRow = useMemo(() => bestTrainingRun(rows), [rows]);
  const evaluationRows = useMemo(() => evaluationRunSummaries(evalStage), [evalStage]);
  const selectedIdsForEval = useMemo(
    () => new Set(selectedIds(evalStage, 'training_run_ids')),
    [evalStage]
  );
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

  const writeSelection = useCallback(
    (ids: string[]) => {
      if (!evalStage) return;
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
    [evalStage, rows, trainStage, updateStageDraft]
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
            view={view}
            sort={sort}
            bestRunId={bestRow?.id ?? null}
            onViewChange={setView}
            onSortChange={setSort}
            onToggle={toggleRow}
            onSelectAll={() => writeSelection(rows.map((row) => row.id))}
            onSelectBest={() => writeSelection(bestRow ? [bestRow.id] : [])}
            onClear={() => writeSelection([])}
            onOpenDetails={setDetailsRun}
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
              <button
                type="button"
                disabled
                className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-700 hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-50"
                title={
                  selectedIdsForEval.size === 0
                    ? 'Select at least one run first.'
                    : 'Backend execution wiring is pending; the selection is ready.'
                }
              >
                <PlayCircle className="h-3.5 w-3.5" />
                Run selected
              </button>
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-slate-400" />
                <div className="font-semibold text-slate-800">Results</div>
              </div>
              <div className="mt-3 space-y-3">
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
                      onOpenAnalyze={openAnalyze}
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

function RunTable({
  title,
  rows,
  allRows,
  metricColumns,
  selectedIds,
  view,
  sort,
  bestRunId,
  onViewChange,
  onSortChange,
  onToggle,
  onSelectAll,
  onSelectBest,
  onClear,
  onOpenDetails,
}: {
  title: string;
  rows: TrainingRunSummary[];
  allRows: TrainingRunSummary[];
  metricColumns: MetricColumnSpec[];
  selectedIds: Set<string>;
  view: RunView;
  sort: SortState;
  bestRunId: string | null;
  onViewChange: (view: RunView) => void;
  onSortChange: (sort: SortState) => void;
  onToggle: (id: string) => void;
  onSelectAll: () => void;
  onSelectBest: () => void;
  onClear: () => void;
  onOpenDetails: (run: TrainingRunSummary) => void;
}) {
  const metricTemplateColumns =
    metricColumns.length > 0
      ? ` repeat(${metricColumns.length}, minmax(4.75rem,0.8fr))`
      : '';
  const gridTemplateColumns = `1.5rem minmax(12rem,1.8fr) minmax(7rem,1fr)${metricTemplateColumns} minmax(4.75rem,0.7fr) 2rem`;
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
        </div>
      </div>
      {rows.length === 0 ? (
        <EmptyCollection
          title={allRows.length === 0 ? 'No runs yet' : 'No rows in this view'}
          detail={
            allRows.length === 0
              ? 'The seeded run table is in Examples > RLRMP movement-ramp training runs.'
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
              <div />
            </div>
            <div className="divide-y divide-slate-100">
              {rows.map((row) => (
                <TrainingRunRow
                  key={row.id}
                  row={row}
                  metricColumns={metricColumns}
                  gridTemplateColumns={gridTemplateColumns}
                  selected={selectedIds.has(row.id)}
                  isBest={bestRunId === row.id}
                  onToggle={() => onToggle(row.id)}
                  onOpenDetails={() => onOpenDetails(row)}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function TrainingRunRow({
  row,
  metricColumns,
  gridTemplateColumns,
  selected,
  isBest,
  onToggle,
  onOpenDetails,
}: {
  row: TrainingRunSummary;
  metricColumns: MetricColumnSpec[];
  gridTemplateColumns: string;
  selected: boolean;
  isBest: boolean;
  onToggle: () => void;
  onOpenDetails: () => void;
}) {
  const progress = row.warmupBatches !== null ? `${row.warmupBatches.toLocaleString()}/${row.warmupBatches.toLocaleString()}` : 'Not recorded';
  const checkpoint = row.checkpointAvailable && row.warmupBatches !== null
    ? row.warmupBatches.toLocaleString()
    : row.checkpointAvailable
      ? 'Available'
      : 'None';
  const complete = row.status === 'completed';

  return (
    <div
      className={clsx(
        'grid items-center gap-2 px-4 py-2 text-xs',
        selected && 'bg-brand-50/40'
      )}
      style={{ gridTemplateColumns }}
    >
      <button
        type="button"
        onClick={onToggle}
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
        </div>
      </div>
      <div className="flex min-w-0 items-center gap-1.5 font-medium text-slate-700" title={progress}>
        {complete && <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-600" />}
        <span className="truncate">{progress}</span>
      </div>
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
      <button
        type="button"
        onClick={onOpenDetails}
        className="flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-700"
        aria-label={`Show details for ${row.label}`}
      >
        <Info className="h-4 w-4" />
      </button>
    </div>
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
  onOpenAnalyze,
}: {
  row: EvaluationRunSummary;
  selectedRun: TrainingRunSummary | undefined;
  onOpenAnalyze: () => void;
}) {
  return (
    <div className="rounded-md border border-slate-200 p-3">
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
      <button
        type="button"
        onClick={onOpenAnalyze}
        className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-md bg-brand-500 px-3 py-2 text-xs font-semibold text-white hover:bg-brand-600"
      >
        Analyze result
        <ChevronRight className="h-3.5 w-3.5" />
      </button>
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

function sortTrainingRows(rows: TrainingRunSummary[], sort: SortState): TrainingRunSummary[] {
  return [...rows].sort((a, b) => {
    const direction = sort.direction === 'asc' ? 1 : -1;
    return compareMetric(metricValue(a, sort.key), metricValue(b, sort.key)) * direction;
  });
}

function metricValue(row: TrainingRunSummary, key: SortKey): number | null {
  if (key === 'progress') return row.status === 'completed' ? row.warmupBatches ?? 1 : 0;
  return trainingRunMetricValue(row, key);
}

function compareMetric(a: number | null, b: number | null): number {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return a - b;
}
