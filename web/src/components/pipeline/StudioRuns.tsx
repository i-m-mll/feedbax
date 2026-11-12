import { useMemo, useState, type ReactNode } from "react";
import { StudioRegion, StudioRegionBoundary, type StudioRegionState } from "@/components/activity/StudioRegion";
import { StudioRunStatus } from "@/components/activity/StudioRunStatus";
import { studioAccessibleName, studioCountOf, studioName, studioRunStatus } from "@/components/activity/studioVocabulary";
import type { PlanRunView, PlanView } from "@/features/pipeline/planView";
import {
  extendsSelection,
  indexPlanLineage,
  selectedRuns,
  selectRun,
  selectStep,
  selectSweepGroup,
  type PipelineSelection,
  type RunSetsByStep,
} from "@/features/pipeline/selection";
import { StudioRunTable, StudioSelectionSummary, StudioSweepValues, type RunSelectionProps } from "./StudioRunTable";

const runCommandNames = ["run", "rerun", "stop", "forceStop", "resume", "accept", "newStep"] as const;
type RunCommandName = (typeof runCommandNames)[number];

export interface StudioRunCommand {
  readonly name: RunCommandName;
  readonly enabled: boolean;
  readonly perform: (selection: PipelineSelection) => void;
}

interface StudioRunsProps extends RunSelectionProps {
  plan: PlanView;
  region: StudioRegionState;
  definition?: ReactNode;
  sweep?: ReactNode;
  inputs?: ReactNode;
  commands?: readonly StudioRunCommand[];
}

function StudioRunLineage({
  plan,
  upstream,
  downstream,
  onChoose,
}: {
  plan: PlanView;
  upstream: RunSetsByStep;
  downstream: RunSetsByStep;
  onChoose: (stepId: string) => void;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {([["madeFrom", upstream], ["usedBy", downstream]] as const).map(([name, runs]) => (
        <section key={name} aria-label={studioName(name)} className="rounded-md border border-slate-200 p-2">
          <h3 className="font-medium">{studioName(name)}</h3>
          <ul>
            {plan.steps.map((step) => {
              const count = runs.get(step.id)?.size;
              if (!count) return null;
              const summary = studioCountOf(count, step.runs.length);
              return (
                <li key={step.id}>
                  <button
                    type="button" aria-label={studioAccessibleName("select", `${step.label} · ${summary}`)}
                    onClick={() => onChoose(step.id)} className="flex w-full justify-between gap-2 text-left hover:underline"
                  >
                    <span>{step.label}</span><span>{summary}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
}

function groupPivotRows(rows: readonly PlanRunView[], rowAxis: string, columnAxis: string) {
  const groups = new Map<string, Map<string, PlanRunView[]>>();
  const unplaced: PlanRunView[] = [];
  for (const run of rows) {
    const y = run.values[rowAxis];
    const x = run.values[columnAxis];
    if (y === undefined || x === undefined) {
      unplaced.push(run);
      continue;
    }
    let columns = groups.get(y);
    if (!columns) groups.set(y, columns = new Map());
    let cell = columns.get(x);
    if (!cell) columns.set(x, cell = []);
    cell.push(run);
  }
  return { groups, unplaced };
}

function StudioRunPivot({ step, state, onSelectionChange, rows, related }: RunSelectionProps & {
  rows: readonly PlanRunView[];
  related: RunSetsByStep;
}) {
  const [rowId, setRowId] = useState(step.sweeps[0]?.id);
  const [columnId, setColumnId] = useState(step.sweeps[1]?.id);
  const y = step.sweeps.find(({ id }) => id === rowId) ?? step.sweeps[0];
  const x = step.sweeps.find(({ id }) => id === columnId && id !== y?.id) ?? step.sweeps.find(({ id }) => id !== y?.id);
  const pivot = useMemo(() => y && x ? groupPivotRows(rows, y.id, x.id) : null, [rows, y, x]);
  const selected = new Set(selectedRuns(step, state.selection).map(({ id }) => id));
  if (!y || !x || !pivot) return null;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <label className="flex gap-2">
          {studioName("rows")}
          <select value={y.id} onChange={(event) => setRowId(event.currentTarget.value)}>
            {step.sweeps.map(({ id, label }) => <option key={id} value={id}>{label}</option>)}
          </select>
        </label>
        <label className="flex gap-2">
          {studioName("columns")}
          <select value={x.id} onChange={(event) => setColumnId(event.currentTarget.value)}>
            {step.sweeps.filter(({ id }) => id !== y.id).map(({ id, label }) => <option key={id} value={id}>{label}</option>)}
          </select>
        </label>
      </div>
      <div className="max-h-96 overflow-auto rounded border border-slate-200">
        <table aria-label={studioName("pivot")} className="w-full border-separate border-spacing-0 text-left text-sm">
          <thead><tr>
            <th scope="col" className="p-2">{y.label}</th>
            {x.values.map((value) => <th key={value.id} scope="col" className="p-2">{value.label}</th>)}
          </tr></thead>
          <tbody>
            {y.values.map((row) => (
              <tr key={row.id}>
                <th scope="row" className="border-t border-slate-200 p-2">{row.label}</th>
                {x.values.map((column) => {
                  const cell: readonly PlanRunView[] = pivot.groups.get(row.id)?.get(column.id) ?? [];
                  const count = new Map<PlanRunView["status"], number>();
                  for (const run of cell) count.set(run.status, (count.get(run.status) ?? 0) + 1);
                  const labels = [...count].map(([status, amount]) => cell.length > 1
                    ? `${amount} ${studioRunStatus(status)}`
                    : studioRunStatus(status));
                  return (
                    <td key={column.id} className="border-t border-slate-200 p-2 align-top">
                      {cell.length > 0 && (
                        <button
                          type="button"
                          aria-label={studioAccessibleName("select", [`${y.label} ${row.label}`, `${x.label} ${column.label}`, ...labels].join(" · "))}
                          aria-pressed={cell.every(({ id }) => selected.has(id))}
                          className="flex flex-wrap gap-2 rounded px-1 aria-pressed:bg-violet-100"
                          onClick={(event) => onSelectionChange(cell.length === 1
                            ? selectRun(state, { stepId: step.id, runId: cell[0].id }, extendsSelection(event))
                            : selectSweepGroup(step.id, [{ sweepId: y.id, valueIds: [row.id] }, { sweepId: x.id, valueIds: [column.id] }]))}
                        >
                          {[...count].map(([status, amount]) => (
                            <span className="flex items-center gap-1" key={status}>
                              {cell.length > 1 && <span>{amount}</span>}
                              <StudioRunStatus status={status} />
                            </span>
                          ))}
                        </button>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pivot.unplaced.length > 0 && (
        <StudioRunTable step={step} state={state} onSelectionChange={onSelectionChange} rows={pivot.unplaced} related={related} />
      )}
    </div>
  );
}

function RunsContent({ plan, step, state, onSelectionChange, definition, sweep, inputs, commands }: StudioRunsProps) {
  const [pivot, setPivot] = useState(false);
  const lineage = useMemo(() => indexPlanLineage(plan.edges), [plan.edges]);
  const selected = useMemo(() => selectedRuns(step, state.selection), [step, state.selection]);
  const roots = useMemo(() => selected.map(({ id }) => ({ stepId: step.id, runId: id })), [selected, step.id]);
  const upstream = useMemo(() => lineage.upstream(roots), [lineage, roots]);
  const downstream = useMemo(() => lineage.downstream(roots), [lineage, roots]);
  const related = useMemo(() => lineage.related(state.focus), [lineage, state.focus]);
  const canPivot = step.sweeps.length >= 2;
  const selection = state.selection;
  return (
    <div className="space-y-3 text-sm text-slate-900">
      {definition}
      {sweep}
      {inputs}
      <StudioSweepValues step={step} state={state} onSelectionChange={onSelectionChange} />
      <StudioSelectionSummary step={step} state={state} />
      <div className="flex flex-wrap items-center gap-2">
        {commands?.map(({ name, enabled, perform }) => (
          <button
            type="button" key={name} disabled={!enabled || !selection || !selected.length}
            className="rounded border border-slate-300 px-2 py-1 disabled:opacity-50"
            onClick={() => { if (selection) perform(selection); }}
          >
            {studioName(name)}
          </button>
        ))}
        {canPivot && (
          <button type="button" aria-pressed={pivot} onClick={() => setPivot(!pivot)} className="rounded border border-slate-300 px-2 py-1">
            {studioName("pivot")}
          </button>
        )}
      </div>
      {canPivot && pivot
        ? <StudioRunPivot step={step} state={state} onSelectionChange={onSelectionChange} rows={step.runs} related={related} />
        : <StudioRunTable step={step} state={state} onSelectionChange={onSelectionChange} rows={step.runs} related={related} />}
      {(upstream.size > 0 || downstream.size > 0) && (
        <StudioRunLineage plan={plan} upstream={upstream} downstream={downstream} onChoose={(id) => onSelectionChange(selectStep(state, id))} />
      )}
    </div>
  );
}

export function StudioRuns(props: StudioRunsProps) {
  return (
    <section aria-label={studioName("runs")}>
      <StudioRegionBoundary name="runs">
        <StudioRegion
          state={props.region}
          skeleton={<div className="h-28 animate-pulse rounded bg-slate-200" />}
          empty={<div className="h-12 rounded border border-dashed border-slate-200" />}
        >
          <RunsContent key={props.step.id} {...props} />
        </StudioRegion>
      </StudioRegionBoundary>
    </section>
  );
}
