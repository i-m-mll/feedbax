import { useMemo, useState, type ReactNode } from "react";
import { StudioRegion, StudioRegionBoundary, useRegionPending, type StudioRegionState } from "@/components/activity/StudioRegion";
import { StudioRunStatus } from "@/components/activity/StudioRunStatus";
import {
  studioAccessibleName, studioCountOf, studioEnvironmentNotice, studioName, studioReason, studioRunStatus, studioValidityReason,
  type StudioResultState, type StudioValidityRows,
} from "@/components/activity/studioVocabulary";
import { StatusMark } from "@/components/ui/StatusMark";
import { commandAttributes, type StudioCommand, type StudioCommandId } from "@/features/projectWindow/studioCommands";
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
import { Button } from "@/components/ui/Button";
import { CommandBar, type BarControl, type BarEntry, type GlyphCommand } from "@/components/ui/CommandBar";
import { Select } from "@/components/ui/Select";
import { Table, TableCell, TableHeaderCell, TableRowHeader } from "@/components/ui/Table";
import { StudioRunTable, type RunSelectionProps } from "./StudioRunTable";
import type { RunMenu } from "./RunContextMenu";

interface StudioRunsProps extends RunSelectionProps {
  plan: PlanView;
  region: StudioRegionState;
  /** The runs shown are of an earlier revision while the shown revision's plan is read. */
  pending?: boolean;
  pivotActive?: boolean;
  upstreamDetail?: ReactNode;
  runMenu?: RunMenu;
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
                  <Button
                    variant="ghost" aria-label={studioAccessibleName("select", `${step.label} ${summary}`)}
                    onClick={() => onChoose(step.id)} className="w-full justify-between"
                  >
                    <span className="min-w-0 truncate">{step.label}</span>{" "}
                    <span className="shrink-0 tabular-nums text-shell-muted">{summary}</span>
                  </Button>
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

function StudioRunPivot({ step, state, onSelectionChange, rows, related, runMenu }: RunSelectionProps & {
  rows: readonly PlanRunView[];
  related: RunSetsByStep;
  runMenu?: RunMenu;
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
        <label className="flex items-center gap-2">
          {studioName("rows")}
          <Select aria-label={studioName("rows")} value={y.id} onValueChange={setRowId}
            items={step.sweeps.map(({ id, label }) => ({ value: id, label }))} />
        </label>
        <label className="flex items-center gap-2">
          {studioName("columns")}
          <Select aria-label={studioName("columns")} value={x.id} onValueChange={setColumnId}
            items={step.sweeps.filter(({ id }) => id !== y.id).map(({ id, label }) => ({ value: id, label }))} />
        </label>
      </div>
      <div className="max-h-96 overflow-auto rounded-surface border border-shell-divider">
        <Table aria-label={studioName("pivot")}>
          <thead><tr>
            <TableHeaderCell scope="col">{y.label}</TableHeaderCell>
            {x.values.map((value) => <TableHeaderCell key={value.id} scope="col">{value.label}</TableHeaderCell>)}
          </tr></thead>
          <tbody>
            {y.values.map((row) => (
              <tr key={row.id}>
                <TableRowHeader>{row.label}</TableRowHeader>
                {x.values.map((column) => {
                  const cell: readonly PlanRunView[] = pivot.groups.get(row.id)?.get(column.id) ?? [];
                  const count = new Map<StudioResultState, number>();
                  for (const run of cell) count.set(run.state, (count.get(run.state) ?? 0) + 1);
                  const labels = [...count].map(([status, amount]) => cell.length > 1
                    ? `${amount} ${studioRunStatus(status)}`
                    : studioRunStatus(status));
                  return (
                    <TableCell key={column.id}>
                      {cell.length > 0 && (
                        <Button
                          variant="ghost"
                          aria-label={studioAccessibleName("select", [`${y.label} ${row.label}`, `${x.label} ${column.label}`, ...labels].join(" · "))}
                          pressed={cell.every(({ id }) => selected.has(id))}
                          className="-ml-control-inline gap-2"
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
                        </Button>
                      )}
                    </TableCell>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </Table>
      </div>
      {pivot.unplaced.length > 0 && (
        <StudioRunTable step={step} state={state} onSelectionChange={onSelectionChange} rows={pivot.unplaced} related={related}
          runMenu={runMenu} />
      )}
    </div>
  );
}

export function StudioRunDetails({ step, state, detail }: Pick<RunSelectionProps, "step" | "state"> & {
  detail?: ReactNode;
}) {
  const selected = useMemo(() => selectedRuns(step, state.selection), [step, state.selection]);
  if (!selected.length) return null;
  const single = selected.length === 1;
  const counts = new Map<StudioResultState, number>();
  const reasons = new Set<string>();
  const openingErrors = new Set(selected.flatMap((run) => run.openingErrors ?? []));
  const validity = new Map<string, { state: StudioResultState; rows: StudioValidityRows }>();
  for (const run of selected) {
    counts.set(run.state, (counts.get(run.state) ?? 0) + 1);
    if (run.reason && !run.results.some(({ validity }) => validity.reasons?.length)) reasons.add(run.reason);
    for (const { validity: { state, reasons: typed = [], environment } } of run.results) {
      const mark = state === "current" ? "succeeded" : state;
      for (const reason of typed) {
        const rows = studioValidityReason(reason);
        validity.set(JSON.stringify(rows), { state: mark, rows });
      }
      if (environment) {
        const rows = studioEnvironmentNotice(environment);
        validity.set(JSON.stringify(rows), { state: "environment", rows });
      }
    }
  }
  return <section className="space-y-2 text-sm" aria-label={studioName("inspector")}>
    {single && <h3 className="font-semibold">{selected[0].label}</h3>}
    <ul className="flex flex-wrap gap-x-3 gap-y-1">
      {[...counts].map(([status, count]) => <li key={status} className="inline-flex items-center gap-1">
        {!single && <span className="tabular-nums">{count}</span>}
        <StudioRunStatus status={status} />
      </li>)}
    </ul>
    {validity.size > 0 && <ul aria-label={studioName("reason")} className="divide-y divide-shell-divider border-y border-shell-divider">
      {[...validity].map(([key, { state, rows: { title, changes } }]) => <li key={key} className="py-1 text-xs">
        <div className="flex min-h-row items-center gap-1.5">
          <StatusMark status={state} label={studioRunStatus(state)} />
          <span className="font-medium">{title}</span>
        </div>
        {changes.length > 0 && <ul className="ps-[calc(1em+0.375rem)] text-shell-muted">
          {changes.map((change) => <li key={change} className="break-words">{change}</li>)}
        </ul>}
      </li>)}
    </ul>}
    {single && detail ? detail : reasons.size > 0 && <ul className="space-y-1 text-xs text-shell-muted">
      {[...reasons].map((reason) => <li key={reason}>{studioReason("observed", reason)}</li>)}
    </ul>}
    {openingErrors.size > 0 && <ul className="space-y-1 font-mono text-xs text-shell-muted">
      {[...openingErrors].map((line) => <li key={line}>{line}</li>)}
    </ul>}
  </section>;
}

const lowRunCommands = new Set<StudioCommandId>(["openRun", "pivot"]);

function glyphCommand(command: StudioCommand): GlyphCommand<StudioCommand> {
  if (!command.glyph) throw new Error(`Pinned command ${command.id} has no glyph`);
  return command as GlyphCommand<StudioCommand>;
}

/** The Runs header: Run and Stop are pinned, the Machine picker is normal, Open Run and Pivot are low. */
export function StudioRunControls({ commands = [], machine, pivotCommand, pivotActive, reason }: {
  commands?: readonly StudioCommand[];
  machine?: BarControl;
  pivotCommand?: StudioCommand;
  pivotActive?: boolean;
  reason?: string;
}) {
  const run = commands.find(({ id }) => id === "run");
  const stop = commands.find(({ id }) => id === "stop");
  const rest = [...commands.filter((command) => command !== run && command !== stop), ...pivotCommand ? [pivotCommand] : []];
  const entries: BarEntry<StudioCommand>[] = [
    ...machine ? [{ control: machine, priority: "normal" as const }] : [],
    ...rest.map((command) => ({ command, priority: lowRunCommands.has(command.id) ? "low" as const : "normal" as const })),
  ];
  return <CommandBar moreLabel={studioName("more")} entries={entries}
    pinned={run ? { action: glyphCommand(run), counterpart: stop && glyphCommand(stop) } : undefined}
    itemProps={(command) => ({
      ...commandAttributes(command),
      ...(command === pivotCommand ? { pressed: pivotActive } : {}),
      ...(command.id === "run" ? {
        focusableWhenDisabled: true,
        tabIndex: command.enabled ? undefined : -1,
        "aria-description": command.enabled ? undefined : reason,
      } : {}),
    })} />;
}

function RunsContent({ plan, step, state, onSelectionChange, pivotActive, upstreamDetail, runMenu }: StudioRunsProps) {
  const lineage = useMemo(() => indexPlanLineage(plan.edges), [plan.edges]);
  const related = useMemo(() => lineage.related(state.focus), [lineage, state.focus]);
  const selected = useMemo(() => selectedRuns(step, state.selection), [step, state.selection]);
  const roots = useMemo(() => selected.map(({ id }) => ({ stepId: step.id, runId: id })), [selected, step.id]);
  const upstream = useMemo(() => lineage.upstream(roots), [lineage, roots]);
  const downstream = useMemo(() => lineage.downstream(roots), [lineage, roots]);
  const selection = { step, state, onSelectionChange };
  return <div className="flex h-full min-h-0 flex-col">
    <div className="min-h-0 flex-1 overflow-auto">
      {pivotActive
        ? <StudioRunPivot {...selection} rows={step.runs} related={related} runMenu={runMenu} />
        : <StudioRunTable {...selection} rows={step.runs} related={related} runMenu={runMenu} fillSlot />}
    </div>
    {(upstream.size > 0 || downstream.size > 0) && <div className="max-h-36 shrink-0 overflow-auto px-2 pb-2 text-sm">
      <StudioRunLineage plan={plan} upstream={upstream} downstream={downstream} single={selected.length === 1}
        upstreamDetail={upstreamDetail} onChoose={(id) => onSelectionChange(selectStep(state, id))} />
    </div>}
  </div>;
}

export function StudioRuns(props: StudioRunsProps) {
  const region = useRegionPending(props.pending ?? false);
  return (
    <section aria-label={studioName("runs")} {...region} className="h-full min-h-0">
      <StudioRegionBoundary name="runs">
        <StudioRegion
          state={props.region}
          skeleton={<div className="h-28 motion-safe:animate-pulse rounded-surface bg-shell-divider" />}
          empty={<div className="h-12 rounded-surface border border-dashed border-shell-divider" />}
        >
          <RunsContent key={props.step.id} {...props} />
        </StudioRegion>
      </StudioRegionBoundary>
    </section>
  );
}
