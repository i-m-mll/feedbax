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
  Filter,
  PlayCircle,
  Rows3,
  Server,
  SlidersHorizontal,
} from 'lucide-react';
import { getStageByKind, useWorkspaceStore } from '@/stores/workspaceStore';
import {
  bestTrainingRun,
  evaluationProtocolLabel,
  evaluationRunSummaries,
  formatMetric,
  runParameterSummary,
  selectedIds,
  trainingInputSummaries,
  trainingRunSummaries,
  type EvaluationRunSummary,
  type TrainingRunSummary,
} from '@/utils/pipelineCollections';

type RunView = 'all' | 'selected' | 'best';
type TargetChoice = 'local' | 'managed' | 'manual';

export function TrainCollectionPanel() {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const setActiveStage = useWorkspaceStore((state) => state.setActiveStage);
  const updateStageDraft = useWorkspaceStore((state) => state.updateStageDraft);
  const [view, setView] = useState<RunView>('all');
  const [target, setTarget] = useState<TargetChoice>('local');

  const trainStage = getStageByKind(workspace, 'train');
  const evalStage = getStageByKind(workspace, 'eval');
  const rows = useMemo(() => trainingRunSummaries(trainStage), [trainStage]);
  const bestRow = useMemo(() => bestTrainingRun(rows), [rows]);
  const selected = useMemo(
    () => new Set(selectedIds(evalStage, 'training_run_ids')),
    [evalStage]
  );
  const filteredRows = useMemo(() => {
    if (view === 'selected') return rows.filter((row) => selected.has(row.id));
    if (view === 'best') return bestRow ? [bestRow] : [];
    return rows;
  }, [bestRow, rows, selected, view]);

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
        'evaluation_selection_changed_from_train'
      );
    },
    [evalStage, rows, trainStage, updateStageDraft]
  );

  const toggleRow = useCallback(
    (id: string) => {
      const next = new Set(selected);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      writeSelection(Array.from(next));
    },
    [selected, writeSelection]
  );

  const openEvaluate = useCallback(() => {
    if (evalStage) setActiveStage(evalStage.id);
  }, [evalStage, setActiveStage]);

  const selectedCount = selected.size;
  const completedCount = rows.filter((row) => row.status === 'completed').length;

  return (
    <div className="h-full overflow-y-auto bg-slate-50/40">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 px-6 py-5 text-sm text-slate-600">
        <section className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="text-base font-semibold text-slate-800">
                  Movement-ramp batch
                </div>
                <div className="mt-1 max-w-2xl text-xs leading-5 text-slate-500">
                  Imported RLRMP runs are grouped as one batch. Select the rows that should feed
                  the next step; identifiers and file paths stay in provenance details.
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <MetricPill label="Completed" value={`${completedCount}/${rows.length}`} />
                <MetricPill
                  label="Best loss"
                  value={bestRow ? formatMetric(bestRow.finalValidationLoss, 4) : 'Not recorded'}
                />
                <MetricPill label="Selected" value={`${selectedCount}`} tone="brand" />
              </div>
            </div>
            <ReadinessStrip
              items={[
                { label: 'Model and task', state: 'ready', detail: 'Loaded in the top pane' },
                { label: 'Sweep', state: rows.length > 0 ? 'ready' : 'blocked', detail: `${rows.length} variants` },
                { label: 'Compute target', state: 'draft', detail: targetLabel(target) },
                {
                  label: 'Next step',
                  state: selectedCount > 0 ? 'ready' : 'blocked',
                  detail: selectedCount > 0 ? `${selectedCount} selected` : 'Select at least one run',
                },
              ]}
            />
          </div>
          <ExecutionTarget value={target} onChange={setTarget} />
        </section>

        <section className="rounded-lg border border-slate-200 bg-white shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-4 py-3">
            <div className="flex items-center gap-2">
              <Rows3 className="h-4 w-4 text-slate-400" />
              <div className="font-semibold text-slate-800">Run collection</div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <SegmentedFilter value={view} onChange={setView} />
              <button
                type="button"
                className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                onClick={() => writeSelection(rows.map((row) => row.id))}
              >
                Select all
              </button>
              <button
                type="button"
                className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                onClick={() => writeSelection(bestRow ? [bestRow.id] : [])}
              >
                Select best
              </button>
              <button
                type="button"
                className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                onClick={() => writeSelection([])}
              >
                Clear
              </button>
            </div>
          </div>
          {filteredRows.length === 0 ? (
            <EmptyCollection
              title="No rows in this view"
              detail="Change the filter or select runs from the full collection."
            />
          ) : (
            <div className="divide-y divide-slate-100">
              {filteredRows.map((row) => (
                <TrainingRunRow
                  key={row.id}
                  row={row}
                  selected={selected.has(row.id)}
                  isBest={bestRow?.id === row.id}
                  onToggle={() => toggleRow(row.id)}
                />
              ))}
            </div>
          )}
        </section>

        <section className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm">
          <div>
            <div className="font-semibold text-slate-800">Evaluation set</div>
            <div className="mt-0.5 text-xs text-slate-500">
              {selectedCount > 0
                ? `${selectedCount} run${selectedCount === 1 ? '' : 's'} ready to review in the next tab.`
                : 'Select at least one row to compose the next step.'}
            </div>
          </div>
          <button
            type="button"
            disabled={!evalStage || selectedCount === 0}
            onClick={openEvaluate}
            className="inline-flex items-center gap-2 rounded-md bg-brand-500 px-3 py-2 text-xs font-semibold text-white shadow-sm hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Open selected runs
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </section>
      </div>
    </div>
  );
}

export function EvaluateCollectionPanel() {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const setActiveStage = useWorkspaceStore((state) => state.setActiveStage);
  const updateStageDraft = useWorkspaceStore((state) => state.updateStageDraft);
  const [view, setView] = useState<RunView>('all');
  const [target, setTarget] = useState<TargetChoice>('local');

  const trainStage = getStageByKind(workspace, 'train');
  const evalStage = getStageByKind(workspace, 'eval');
  const analysisStage = getStageByKind(workspace, 'analysis');
  const rows = useMemo(() => trainingInputSummaries(evalStage), [evalStage]);
  const bestRow = useMemo(() => bestTrainingRun(rows), [rows]);
  const evaluationRows = useMemo(() => evaluationRunSummaries(evalStage), [evalStage]);
  const selected = useMemo(
    () => new Set(selectedIds(evalStage, 'training_run_ids')),
    [evalStage]
  );
  const filteredRows = useMemo(() => {
    if (view === 'selected') return rows.filter((row) => selected.has(row.id));
    if (view === 'best') return bestRow ? [bestRow] : [];
    return rows;
  }, [bestRow, rows, selected, view]);

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
      const next = new Set(selected);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      writeSelection(Array.from(next));
    },
    [selected, writeSelection]
  );

  const openAnalyze = useCallback(() => {
    if (analysisStage) setActiveStage(analysisStage.id);
  }, [analysisStage, setActiveStage]);

  const selectedRows = rows.filter((row) => selected.has(row.id));
  const selectedCount = selected.size;
  const selectedBestLoss = bestTrainingRun(selectedRows)?.finalValidationLoss ?? null;

  return (
    <div className="h-full overflow-y-auto bg-slate-50/40">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 px-6 py-5 text-sm text-slate-600">
        <section className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="text-base font-semibold text-slate-800">
                  Movement-ramp validation set
                </div>
                <div className="mt-1 max-w-2xl text-xs leading-5 text-slate-500">
                  Choose the training runs to validate, confirm the protocol, then send the
                  resulting evaluations to analysis. The run list is the control surface.
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <MetricPill label="Candidates" value={`${rows.length}`} />
                <MetricPill label="Selected" value={`${selectedCount}`} tone="brand" />
                <MetricPill label="Best selected loss" value={formatMetric(selectedBestLoss, 4)} />
              </div>
            </div>
            <ReadinessStrip
              items={[
                {
                  label: 'Run set',
                  state: selectedCount > 0 ? 'ready' : 'blocked',
                  detail: selectedCount > 0 ? `${selectedCount} selected` : 'Select runs',
                },
                { label: 'Protocol', state: 'ready', detail: '8-direction center-out' },
                { label: 'Compute target', state: 'draft', detail: targetLabel(target) },
                {
                  label: 'Results',
                  state: evaluationRows.length > 0 ? 'ready' : 'blocked',
                  detail:
                    evaluationRows.length > 0
                      ? `${evaluationRows.length} available`
                      : 'Run validation first',
                },
              ]}
            />
          </div>
          <ExecutionTarget value={target} onChange={setTarget} />
        </section>

        <section className="grid gap-5 xl:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.65fr)]">
          <div className="rounded-lg border border-slate-200 bg-white shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-4 py-3">
              <div className="flex items-center gap-2">
                <Filter className="h-4 w-4 text-slate-400" />
                <div className="font-semibold text-slate-800">Run set</div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <SegmentedFilter value={view} onChange={setView} />
                <button
                  type="button"
                  className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                  onClick={() => writeSelection(rows.map((row) => row.id))}
                >
                  Select all
                </button>
                <button
                  type="button"
                  className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                  onClick={() => writeSelection(bestRow ? [bestRow.id] : [])}
                >
                  Select best
                </button>
              </div>
            </div>
            {filteredRows.length === 0 ? (
              <EmptyCollection
                title="No rows in this view"
                detail="Use the full view to choose which runs should be evaluated."
              />
            ) : (
              <div className="divide-y divide-slate-100">
                {filteredRows.map((row) => (
                  <TrainingRunRow
                    key={row.id}
                    row={row}
                    selected={selected.has(row.id)}
                    isBest={bestRow?.id === row.id}
                    onToggle={() => toggleRow(row.id)}
                  />
                ))}
              </div>
            )}
          </div>

          <div className="flex flex-col gap-3">
            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="h-4 w-4 text-slate-400" />
                <div className="font-semibold text-slate-800">Protocol</div>
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
                  selectedCount === 0
                    ? 'Select at least one run first.'
                    : 'Backend execution wiring is pending; the selection is ready.'
                }
              >
                <PlayCircle className="h-3.5 w-3.5" />
                Run selected
              </button>
              <div className="mt-2 text-[11px] leading-4 text-slate-400">
                Execution wiring is pending; this slice makes the selection and readiness model visible.
              </div>
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
                      selectedRun={rows.find((candidate) => candidate.id === row.selectedTrainingRunId)}
                      onOpenAnalyze={openAnalyze}
                    />
                  ))
                )}
              </div>
            </section>
          </div>
        </section>
      </div>
    </div>
  );
}

function TrainingRunRow({
  row,
  selected,
  isBest,
  onToggle,
}: {
  row: TrainingRunSummary;
  selected: boolean;
  isBest: boolean;
  onToggle: () => void;
}) {
  return (
    <div className={clsx('px-4 py-3', selected && 'bg-brand-50/40')}>
      <div className="grid gap-3 lg:grid-cols-[2rem_minmax(12rem,1.35fr)_minmax(12rem,1fr)_8rem_8rem_8rem_7rem] lg:items-center">
        <button
          type="button"
          onClick={onToggle}
          className={clsx(
            'flex h-8 w-8 items-center justify-center rounded-md border transition-colors',
            selected
              ? 'border-brand-500 bg-brand-500 text-white'
              : 'border-slate-200 bg-white text-slate-400 hover:border-brand-200 hover:text-brand-500'
          )}
          aria-label={selected ? `Remove ${row.label} from set` : `Add ${row.label} to set`}
        >
          {selected ? <Check className="h-4 w-4" /> : <Circle className="h-4 w-4" />}
        </button>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold text-slate-800">{row.label}</span>
            {isBest && (
              <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-emerald-700">
                lowest loss
              </span>
            )}
            <StatusPill status={row.status} />
          </div>
          <div className="mt-1 text-xs text-slate-500">{runParameterSummary(row)}</div>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4 lg:contents">
          <MetricCell label="Val loss" value={formatMetric(row.finalValidationLoss, 4)} />
          <MetricCell label="Vel RMSE" value={formatMetric(row.velocityRmse, 3)} />
          <MetricCell
            label="Peak vel"
            value={`${formatMetric(row.peakVelocityMean, 3)} +/- ${formatMetric(row.peakVelocitySd, 2)}`}
          />
          <MetricCell
            label="Hold drift"
            value={`${formatMetric(row.holdDriftMeanMm, 2)} +/- ${formatMetric(row.holdDriftSdMm, 2)} mm`}
          />
        </div>
        <div className="text-xs">
          {row.checkpointAvailable ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-1 text-slate-600">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
              checkpoint
            </span>
          ) : (
            <span className="text-slate-400">No checkpoint</span>
          )}
        </div>
      </div>
      <details className="mt-2">
        <summary className="cursor-pointer text-[11px] font-medium text-slate-400 hover:text-slate-600">
          Provenance
        </summary>
        <div className="mt-2 grid gap-2 rounded-md bg-slate-50 p-3 text-[11px] text-slate-500 sm:grid-cols-2">
          <div className="min-w-0">
            <span className="font-medium text-slate-600">Source issue:</span>{' '}
            {row.sourceIssue ?? 'Not recorded'}
          </div>
          <div className="min-w-0 truncate" title={row.provenanceId}>
            <span className="font-medium text-slate-600">Manifest:</span> {row.provenanceId}
          </div>
          {row.uri && (
            <div className="min-w-0 truncate sm:col-span-2" title={row.uri}>
              <span className="font-medium text-slate-600">Path:</span> {row.uri}
            </div>
          )}
        </div>
      </details>
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
        <StatusPill status={row.status} />
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
      <details className="mt-2">
        <summary className="cursor-pointer text-[11px] font-medium text-slate-400 hover:text-slate-600">
          Provenance
        </summary>
        <div className="mt-2 space-y-1 rounded-md bg-slate-50 p-3 text-[11px] text-slate-500">
          <div className="truncate" title={row.provenanceId}>
            <span className="font-medium text-slate-600">Manifest:</span> {row.provenanceId}
          </div>
          {row.uri && (
            <div className="truncate" title={row.uri}>
              <span className="font-medium text-slate-600">Path:</span> {row.uri}
            </div>
          )}
        </div>
      </details>
    </div>
  );
}

function ExecutionTarget({
  value,
  onChange,
}: {
  value: TargetChoice;
  onChange: (value: TargetChoice) => void;
}) {
  const targets: Array<{
    id: TargetChoice;
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
              className={clsx(
                'flex w-full items-center gap-3 rounded-md border px-3 py-2 text-left transition-colors',
                active
                  ? 'border-brand-300 bg-brand-50 text-brand-800'
                  : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span className="min-w-0">
                <span className="block text-xs font-semibold">{target.title}</span>
                <span className="block text-[11px] text-slate-500">{target.detail}</span>
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function ReadinessStrip({
  items,
}: {
  items: Array<{ label: string; state: 'ready' | 'draft' | 'blocked'; detail: string }>;
}) {
  return (
    <div className="mt-4 grid gap-2 md:grid-cols-4">
      {items.map((item) => (
        <div key={item.label} className="rounded-md border border-slate-100 bg-slate-50 px-3 py-2">
          <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700">
            <span
              className={clsx(
                'h-2 w-2 rounded-full',
                item.state === 'ready' && 'bg-emerald-500',
                item.state === 'draft' && 'bg-amber-400',
                item.state === 'blocked' && 'bg-slate-300'
              )}
            />
            {item.label}
          </div>
          <div className="mt-1 truncate text-[11px] text-slate-500" title={item.detail}>
            {item.detail}
          </div>
        </div>
      ))}
    </div>
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
    { id: 'best', label: 'Best' },
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

function MetricPill({
  label,
  value,
  tone = 'slate',
}: {
  label: string;
  value: string;
  tone?: 'slate' | 'brand';
}) {
  return (
    <div
      className={clsx(
        'rounded-md border px-3 py-2',
        tone === 'brand' ? 'border-brand-100 bg-brand-50' : 'border-slate-100 bg-slate-50'
      )}
    >
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">
        {label}
      </div>
      <div className="mt-0.5 text-sm font-semibold text-slate-800">{value}</div>
    </div>
  );
}

function MetricCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">
        {label}
      </div>
      <div className="mt-0.5 truncate font-medium text-slate-700" title={value}>
        {value}
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  return (
    <span
      className={clsx(
        'rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em]',
        status === 'completed'
          ? 'bg-emerald-50 text-emerald-700'
          : status === 'failed'
            ? 'bg-red-50 text-red-700'
            : 'bg-slate-100 text-slate-500'
      )}
    >
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

function EmptyCollection({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="flex min-h-32 flex-col items-center justify-center px-4 py-8 text-center">
      <div className="font-semibold text-slate-700">{title}</div>
      <div className="mt-1 max-w-sm text-xs leading-5 text-slate-500">{detail}</div>
    </div>
  );
}

function targetLabel(value: TargetChoice): string {
  if (value === 'managed') return 'Managed worker';
  if (value === 'manual') return 'Manual endpoint';
  return 'This machine';
}
