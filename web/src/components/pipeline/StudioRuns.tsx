import { useMemo, useState, type ReactNode } from "react";
import { StudioRegion, StudioRegionBoundary, type StudioRegionState } from "@/components/activity/StudioRegion";
import { StudioRunStatus } from "@/components/activity/StudioRunStatus";
import { studioAccessibleName, studioCountOf, studioName, studioReason, studioRunStatus } from "@/components/activity/studioVocabulary";
import { commandAttributes, type StudioCommand } from "@/features/projectWindow/studioCommands";
import type { PlanRunView, PlanView } from "@/features/pipeline/planView";
import {
  selectionButton,
  indexPlanLineage,
  selectedRuns,
  selectRun,
  selectStep,
  selectSweepGroup,
  type RunSetsByStep,
} from "@/features/pipeline/selection";
import { StudioRunTable, StudioSelectionSummary, StudioSweepValues, type RunSelectionProps } from "./StudioRunTable";

interface StudioRunsProps extends RunSelectionProps {
  plan: PlanView;
  region: StudioRegionState;
  pivotActive?: boolean;
  upstreamDetail?: ReactNode;
}

function StudioRunLineage({
  plan,
  upstream,
  downstream,
  onChoose,
  single,
  upstreamDetail,
}: {
  plan: PlanView;
  upstream: RunSetsByStep;
  downstream: RunSetsByStep;
  onChoose: (stepId: string) => void;
  single: boolean;
  upstreamDetail?: ReactNode;
}) {
  return (
    <div className="space-y-2">
      {([["madeFrom", upstream], ["usedBy", downstream]] as const).map(([name, runs]) => (
        runs.size > 0 && <section key={name} aria-label={studioName(name)} className="border-t border-shell-divider pt-2">
          <h3 className="font-medium">{studioName(name)}</h3>
          <ul>
            {plan.steps.map((step) => {
              const members = runs.get(step.id);
              if (!members?.size) return null;
              const summary = studioCountOf(members.size, step.runs.length);
              return (
                <li key={step.id}>
                  <button
                    type="button" aria-label={studioAccessibleName("select", `${step.label} · ${summary}`)}
                    onClick={() => onChoose(step.id)} className="flex w-full justify-between gap-2 text-left hover:underline"
                  >
                    <span className="min-w-0">{step.label}</span><span className="shrink-0 tabular-nums text-shell-muted">{summary}</span>
                  </button>
                  {single && <ul className="ps-2 text-shell-muted">
                    {step.runs.filter(({ id }) => members.has(id)).map(({ id, label }) => <li key={id}>{label}</li>)}
                  </ul>}
                </li>
              );
            })}
          </ul>
          {name === "madeFrom" && single && upstreamDetail}
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
      <div className="max-h-96 overflow-auto rounded border border-shell-divider">
        <table aria-label={studioName("pivot")} className="w-full border-separate border-spacing-0 text-left text-sm">
          <thead><tr>
            <th scope="col" className="p-2">{y.label}</th>
            {x.values.map((value) => <th key={value.id} scope="col" className="p-2">{value.label}</th>)}
          </tr></thead>
          <tbody>
            {y.values.map((row) => (
              <tr key={row.id}>
                <th scope="row" className="border-t border-shell-divider p-2">{row.label}</th>
                {x.values.map((column) => {
                  const cell: readonly PlanRunView[] = pivot.groups.get(row.id)?.get(column.id) ?? [];
                  const count = new Map<PlanRunView["status"], number>();
                  for (const run of cell) count.set(run.status, (count.get(run.status) ?? 0) + 1);
                  const labels = [...count].map(([status, amount]) => cell.length > 1
                    ? `${amount} ${studioRunStatus(status)}`
                    : studioRunStatus(status));
                  return (
                    <td key={column.id} className="border-t border-shell-divider p-2 align-top">
                      {cell.length > 0 && (
                        <button
                          type="button"
                          aria-label={studioAccessibleName("select", [`${y.label} ${row.label}`, `${x.label} ${column.label}`, ...labels].join(" · "))}
                          aria-pressed={cell.every(({ id }) => selected.has(id))}
                          className="flex flex-wrap gap-2 rounded px-1 aria-pressed:bg-shell-selected"
                          {...selectionButton((extend) => onSelectionChange(cell.length === 1
                            ? selectRun(state, { stepId: step.id, runId: cell[0].id }, extend)
                            : selectSweepGroup(step.id, [{ sweepId: y.id, valueIds: [row.id] }, { sweepId: x.id, valueIds: [column.id] }])))}
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

export function StudioRunDetails({ step, state }: Pick<RunSelectionProps, "step" | "state">) {
  const selected = useMemo(() => selectedRuns(step, state.selection), [step, state.selection]);
  if (!selected.length) return null;
  const single = selected.length === 1;
  const counts = new Map<PlanRunView["status"], number>();
  const reasons = new Set<string>();
  for (const run of selected) {
    counts.set(run.status, (counts.get(run.status) ?? 0) + 1);
    if (run.reason) reasons.add(run.reason);
  }
  return <section className="space-y-2 text-sm" aria-label={studioName("inspector")}>
    {single && <h3 className="font-semibold">{selected[0].label}</h3>}
    <ul className="flex flex-wrap gap-x-3 gap-y-1">
      {[...counts].map(([status, count]) => <li key={status} className="inline-flex items-center gap-1">
        {!single && <span className="tabular-nums">{count}</span>}
        <StudioRunStatus status={status} />
      </li>)}
    </ul>
    {reasons.size > 0 && <ul className="space-y-1 text-xs text-shell-muted">
      {[...reasons].map((reason) => <li key={reason}>{studioReason("observed", reason)}</li>)}
    </ul>}
  </section>;
}

export function StudioRunControls({ commands, pivotCommand, pivotActive, reason }: {
  commands?: readonly StudioCommand[];
  pivotCommand?: StudioCommand;
  pivotActive?: boolean;
  reason?: string;
}) {
  return <div className="flex min-w-0 items-center gap-1 overflow-x-auto">
    {commands?.map((command) => (
      <button type="button" key={command.id} {...commandAttributes(command)} disabled={!command.enabled}
        aria-description={!command.enabled ? reason : undefined}
        className="shrink-0 rounded border border-shell-divider px-2 py-1 disabled:opacity-50" onClick={command.invoke}>
        {studioName(command.name)}
      </button>
    ))}
    {pivotCommand && (
      <button type="button" {...commandAttributes(pivotCommand)} aria-pressed={pivotActive}
        onClick={pivotCommand.invoke} className="shrink-0 rounded border border-shell-divider px-2 py-1">
        {studioName(pivotCommand.name)}
      </button>
    )}
  </div>;
}

function RunsContent({ plan, step, state, onSelectionChange, pivotActive, upstreamDetail }: StudioRunsProps) {
  const lineage = useMemo(() => indexPlanLineage(plan.edges), [plan.edges]);
  const related = useMemo(() => lineage.related(state.focus), [lineage, state.focus]);
  const selected = useMemo(() => selectedRuns(step, state.selection), [step, state.selection]);
  const roots = useMemo(() => selected.map(({ id }) => ({ stepId: step.id, runId: id })), [selected, step.id]);
  const upstream = useMemo(() => lineage.upstream(roots), [lineage, roots]);
  const downstream = useMemo(() => lineage.downstream(roots), [lineage, roots]);
  const selection = { step, state, onSelectionChange };
  return <div className="flex h-full min-h-0 flex-col">
    {step.sweeps.length > 0 && <div className="flex max-h-28 flex-wrap items-baseline justify-between gap-2 overflow-auto border-b border-shell-divider px-2 py-1 text-xs">
      <StudioSweepValues {...selection} />
      <StudioSelectionSummary step={step} state={state} />
    </div>}
    <div className="min-h-0 flex-1 overflow-auto">
      {pivotActive
        ? <StudioRunPivot {...selection} rows={step.runs} related={related} />
        : <StudioRunTable {...selection} rows={step.runs} related={related} fillSlot />}
    </div>
    {(upstream.size > 0 || downstream.size > 0) && <div className="max-h-36 shrink-0 overflow-auto px-2 pb-2 text-sm">
      <StudioRunLineage plan={plan} upstream={upstream} downstream={downstream} single={selected.length === 1}
        upstreamDetail={upstreamDetail} onChoose={(id) => onSelectionChange(selectStep(state, id))} />
    </div>}
  </div>;
}

export function StudioRuns(props: StudioRunsProps) {
  return (
    <section aria-label={studioName("runs")} className="h-full min-h-0">
      <StudioRegionBoundary name="runs">
        <StudioRegion
          state={props.region}
          skeleton={<div className="h-28 animate-pulse rounded bg-shell-divider" />}
          empty={<div className="h-12 rounded border border-dashed border-shell-divider" />}
        >
          <RunsContent key={props.step.id} {...props} />
        </StudioRegion>
      </StudioRegionBoundary>
    </section>
  );
}
