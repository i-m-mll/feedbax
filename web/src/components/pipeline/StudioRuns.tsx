import { useMemo, useState, type ReactNode } from "react";
import { StudioRegion, StudioRegionBoundary, useRegionPending, type StudioRegionState } from "@/components/activity/StudioRegion";
import { StudioRunStatus } from "@/components/activity/StudioRunStatus";
import {
  studioAccessibleName, studioChangeDetail, studioCountOf, studioEnvironmentNotice, studioName, studioPlaceReason, studioProblemDetail,
  studioReason, studioRunStatus, studioValidityReason, type StudioPlace, type StudioResultState, type StudioValidityRows,
} from "@/components/activity/studioVocabulary";
import { StatusMark } from "@/components/ui/StatusMark";
import { StepKindMark } from "./StepKindMark";
import { commandAttributes, type StudioCommand, type StudioCommandId } from "@/features/projectWindow/studioCommands";
import type { PlanRunView, PlanStepView, PlanView } from "@/features/pipeline/planView";
import { MentionRow } from "@/features/projectWindow/StudioProblems";
import type { StudioProjectWindowMapProblem } from "@/generated/studioContracts";
import {
  selectionButton,
  indexPlanLineage,
  selectedRuns,
  selectRun,
  selectSweepGroup,
  type RunSetsByStep,
} from "@/features/pipeline/selection";
import { Button } from "@/components/ui/Button";
import { ScrollFade } from "@/components/ui/ScrollFade";
import { CommandBar, withGlyph, type BarControl, type BarEntry } from "@/components/ui/CommandBar";
import { Select } from "@/components/ui/Select";
import { Table, TableCell, TableHeaderCell, TableRowHeader } from "@/components/ui/Table";
import { StudioRunTable, type RunSelectionProps } from "./StudioRunTable";
import { focusRing } from "@/components/ui/styles";
import { CommandContextMenu, type RunMenu, type StepMenu } from "./CommandContextMenu";

interface StudioRunsProps extends RunSelectionProps {
  plan: PlanView;
  region: StudioRegionState;
  /** The runs shown are of an earlier revision while the shown revision's plan is read. */
  pending?: boolean;
  pivotActive?: boolean;
  runMenu?: RunMenu;
}

/** The steps the chosen runs were made from and are used by, one row each; pressed, a row selects its step. */
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
  return <>
    {([["madeFrom", upstream], ["usedBy", downstream]] as const).map(([name, runs]) => plan.steps.map((step) => {
      const members = runs.get(step.id);
      if (!members?.size) return null;
      const summary = studioCountOf(members.size, step.runs.length);
      return <li key={`${name}:${step.id}`}>
        <InspectorRow aria-label={studioAccessibleName("select", `${studioName(name)} ${step.label} ${summary}`)}
          onClick={() => onChoose(step.id)}>
          <span className="text-shell-muted">{studioName(name)}</span>{" · "}{step.label}{" · "}
          <span className="tabular-nums text-shell-muted">{summary}</span>
        </InspectorRow>
        {single && step.runs.length > 1 && <ul className="ps-4 text-shell-muted">
          {step.runs.filter(({ id }) => members.has(id)).map(({ id, label }) => <li key={id}>{label}</li>)}
        </ul>}
        {name === "madeFrom" && single && upstreamDetail}
      </li>;
    }))}
  </>;
}

/** One row under the inspector's hairline: an input or a step the runs relate to. */
export function InspectorRow({ children, ...props }: { children: ReactNode; onClick?: () => void; "aria-label"?: string }) {
  return props.onClick
    ? <Button variant="ghost" size="content" className="block w-full truncate rounded-control px-control-inline py-1 text-left" {...props}>
      {children}
    </Button>
    : <div className="truncate px-control-inline py-1">{children}</div>;
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
      <ScrollFade axis="both" className="max-h-96 rounded-surface border border-shell-divider [--fade-inset-top:var(--row-height)]">
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
      </ScrollFade>
      {pivot.unplaced.length > 0 && (
        <StudioRunTable step={step} state={state} onSelectionChange={onSelectionChange} rows={pivot.unplaced} related={related}
          runMenu={runMenu} />
      )}
    </div>
  );
}

/** What tells a run apart within its step: its sweep values, or its own name when it has none. */
function runName(step: PlanStepView, run: PlanRunView) {
  const values = step.sweeps.flatMap((sweep) => {
    const value = run.values[sweep.id];
    return value === undefined || sweep.values.length < 2 ? []
      : [`${sweep.label} ${sweep.values.find(({ id }) => id === value)?.label ?? value}`];
  });
  return values.length ? values.join(" · ") : run.label;
}

/** A reason a run names, which its row reveals. */
export interface RunMention {
  readonly place: StudioPlace | null;
  readonly detail: string;
  readonly problem?: StudioProjectWindowMapProblem;
}

/**
 * The Runs view's inspector: the step or chosen runs on a title line, the runs' reasons, the step's fields resolved for
 * them, then under a hairline the rows of what attaches to the step (exceptions, inputs) and the steps the runs relate to.
 */
export function StudioStepInspector({
  plan, step, state, detail, fields, attachments, revealed, upstreamDetail, onReveal, onChooseStep, stepMenu, stepKey,
}: Pick<RunSelectionProps, "step" | "state"> & {
  plan: PlanView;
  stepMenu?: StepMenu;
  /** The step's key under its name. */
  stepKey?: ReactNode;
  /** What a single chosen run's result adds to its reasons. */
  detail?: ReactNode;
  fields?: ReactNode;
  attachments?: ReactNode;
  /** A reason revealed at this step, drawn among its reasons. */
  revealed?: RunMention;
  upstreamDetail?: ReactNode;
  onReveal?: (run: PlanRunView, mention: RunMention) => void;
  onChooseStep: (stepId: string) => void;
}) {
  const selected = useMemo(() => selectedRuns(step, state.selection), [step, state.selection]);
  const lineage = useMemo(() => indexPlanLineage(plan.edges), [plan.edges]);
  const roots = useMemo(() => selected.map(({ id }) => ({ stepId: step.id, runId: id })), [selected, step.id]);
  const upstream = useMemo(() => lineage.upstream(roots), [lineage, roots]);
  const downstream = useMemo(() => lineage.downstream(roots), [lineage, roots]);
  const chosen = state.selection?.kind !== "steps" && selected.length > 0;
  const single = selected.length === 1;
  const counts = new Map<StudioResultState, number>();
  const reasons = new Set<string>();
  const mentions = new Map<string, { run?: PlanRunView; mention: RunMention }>();
  const openingErrors = new Set(chosen ? selected.flatMap((run) => run.openingErrors ?? []) : []);
  const validity = new Map<string, { state: StudioResultState; rows: StudioValidityRows }>();
  if (revealed) mentions.set(studioPlaceReason(revealed.place, revealed.detail), { mention: revealed });
  for (const run of chosen ? selected : []) {
    counts.set(run.state, (counts.get(run.state) ?? 0) + 1);
    const named: RunMention[] = [
      ...(run.problems ?? []).map((problem) => ({ place: problem.place, detail: studioProblemDetail(problem), problem })),
      ...run.change ? [{ place: run.change.place, detail: studioChangeDetail(run.change) }] : [],
    ];
    for (const mention of named) mentions.set(studioPlaceReason(mention.place, mention.detail), { run, mention });
    if (run.reason && !named.length && !run.results.some(({ validity }) => validity.reasons?.length)) reasons.add(run.reason);
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
  const title = chosen && single && step.runs.length > 1 ? `${step.label} · ${runName(step, selected[0])}` : step.label;
  const related = upstream.size > 0 || downstream.size > 0;
  return <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-2 text-sm [overflow-wrap:anywhere]">
    <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
      <CommandContextMenu commands={stepMenu && (() => stepMenu(step))} trigger={
        <span className={`flex min-w-0 items-center gap-1.5 rounded-sm ${focusRing}`} tabIndex={stepMenu ? 0 : undefined}>
          <StepKindMark kind={step.kind} /><h3 className="min-w-0 break-words font-semibold">{title}</h3>
        </span>} />
      {chosen ? <ul className="contents">
        {[...counts].map(([status, count]) => <li key={status} className="inline-flex items-center gap-1">
          {!single && <span className="tabular-nums">{count}</span>}
          <StudioRunStatus status={status} />
        </li>)}
      </ul> : null}
    </div>
    {stepKey}
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
    {chosen && single && detail ? detail : reasons.size > 0 && <ul className="space-y-1 text-xs text-shell-muted">
      {[...reasons].map((reason) => <li key={reason}>{studioReason("observed", reason)}</li>)}
    </ul>}
    {mentions.size > 0 && <ul className="text-xs"
      aria-label={studioName([...mentions.values()].some(({ mention }) => mention.problem) ? "problems" : "changes")}>
      {[...mentions].map(([key, { run, mention }]) => <li key={key} className="break-words">
        <MentionRow place={mention.place} detail={mention.detail} placeShown={!namedByStep(mention.place)}
          onReveal={run && onReveal && (() => onReveal(run, mention))} />
      </li>)}
    </ul>}
    {openingErrors.size > 0 && <ul className="space-y-1 font-mono text-xs text-shell-muted">
      {[...openingErrors].map((line) => <li key={line}>{line}</li>)}
    </ul>}
    {fields}
    {(attachments || related) && <div className="grid gap-px border-t border-shell-divider pt-1">
      {attachments}
      {related && <ul>
        <StudioRunLineage plan={plan} upstream={upstream} downstream={downstream} single={single}
          upstreamDetail={upstreamDetail} onChoose={onChooseStep} />
      </ul>}
    </div>}
  </div>;
}

/** Whether a place is the step itself, which the inspector's title line already names. */
function namedByStep(place: StudioPlace | null) {
  return place === null || place.part === "step" && !place.field;
}

const lowRunCommands = new Set<StudioCommandId>(["openRun", "pivot"]);

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
  const leading: BarEntry<StudioCommand>[] = machine ? [{ control: machine, priority: "normal" }] : [];
  const entries: BarEntry<StudioCommand>[] = [
    ...rest.map((command) => ({ command: withGlyph(command), priority: lowRunCommands.has(command.id) ? "low" as const : "normal" as const })),
  ];
  return <CommandBar moreLabel={studioName("more")} leading={leading} entries={entries}
    pinned={run ? { action: withGlyph(run), counterpart: stop && withGlyph(stop) } : undefined}
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

function RunsContent({ plan, step, state, onSelectionChange, pivotActive, runMenu }: StudioRunsProps) {
  const lineage = useMemo(() => indexPlanLineage(plan.edges), [plan.edges]);
  const related = useMemo(() => lineage.related(state.focus), [lineage, state.focus]);
  const selection = { step, state, onSelectionChange };
  return <ScrollFade axis="both" className="h-full min-h-0">
    {pivotActive
      ? <StudioRunPivot {...selection} rows={step.runs} related={related} runMenu={runMenu} />
      : <StudioRunTable {...selection} rows={step.runs} related={related} runMenu={runMenu} fillSlot />}
  </ScrollFade>;
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
