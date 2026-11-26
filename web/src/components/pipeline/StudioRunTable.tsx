import { appTip } from "@window-shell/browser/tooltip";
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { StudioCollectionProgress, StudioRunProgress, StudioRunStatus } from "@/components/activity/StudioRunStatus";
import { studioBytes, studioName, studioRunReason, studioRunStatus } from "@/components/activity/studioVocabulary";
import { StatusMark, statusCounted, uncountedRun } from "@/components/ui/StatusMark";
import type { PlanRunView, PlanStepView, PlanSweepView } from "@/features/pipeline/planView";
import {
  selectionButton,
  selectedRuns,
  selectRun,
  selectStep,
  type PipelineSelectionState,
  type RunSetsByStep,
} from "@/features/pipeline/selection";
import { Button } from "@/components/ui/Button";
import { Table, TableCell, TableHeaderCell, TableRow, TableRowHeader } from "@/components/ui/Table";
import { rulePickSurface } from "@/features/projectWindow/elementSelection";
import { CommandContextMenu, type RunMenu } from "./CommandContextMenu";
import type { AxisMenu } from "./PipelineRunMarks";
import { DocumentPicker, type DocumentEditing } from "@/features/projectWindow/DocumentPicker";

const gutter = "pl-1 pr-2 text-right";

/** The controls inside a run's row; a press anywhere else on the row chooses the run. */
const rowControls = "button, a, input, select, textarea, [role=button], [role=menuitem], [contenteditable=true]";


export interface RunSelectionProps {
  step: PlanStepView;
  state: PipelineSelectionState;
  onSelectionChange: (state: PipelineSelectionState) => void;
}

/** The step's drawn axes whose values differ between the shown rows, the fields that tell those runs apart. */
export function runColumns(step: PlanStepView, rows: readonly PlanRunView[]) {
  return step.sweeps.filter((sweep) => new Set(rows.map(({ values }) => values[sweep.id])).size > 1);
}

/** Whether the shown columns leave two rows with the same cells, so the runs' own names must be drawn. */
function needsNames(rows: readonly PlanRunView[], sweeps: readonly PlanSweepView[], showVariant: boolean) {
  const seen = new Set<string>();
  for (const run of rows) {
    const cells = JSON.stringify([showVariant ? run.variant : null, ...sweeps.map(({ id }) => run.values[id] ?? null)]);
    if (seen.has(cells)) return true;
    seen.add(cells);
  }
  return false;
}

function useValueLabels(sweeps: readonly PlanSweepView[]) {
  return useMemo(() => new Map(sweeps.map((sweep) => [
    sweep.id, new Map(sweep.values.map((value) => [value.id, value.valueLabel])),
  ])), [sweeps]);
}

interface TableProps extends RunSelectionProps {
  rows: readonly PlanRunView[];
  related: RunSetsByStep;
  fillSlot?: boolean;
  runMenu?: RunMenu;
  axisMenu?: AxisMenu;
  /** While runs are picked for a rule: those picked, drawn as selected, and the toggle a press on a row takes. */
  picking?: { readonly runs: ReadonlySet<string>; toggle(runs: readonly string[]): void };
}

/** A sweep's column head: the axis, the set it reads, and the axis's context menu. */
function AxisHead({ step, sweep, axisMenu }: { step: string; sweep: PlanSweepView; axisMenu?: AxisMenu }) {
  const [editing, setEditing] = useState<DocumentEditing>();
  const authored = axisMenu?.(step, sweep.id);
  const head = <TableHeaderCell scope="col">
    {authored?.picker ? <DocumentPicker model={authored.picker} editing={editing} setEditing={setEditing} /> : sweep.label}
    {authored?.refusal && <span className="inline-flex" aria-description={authored.refusal}
      {...appTip(studioRunReason({ reason: authored.refusal }))}>
      <StatusMark status="cannot_run" label={studioRunStatus('cannot_run')} />
    </span>}
  </TableHeaderCell>;
  return authored?.commands.length ? <CommandContextMenu commands={() => authored.commands} trigger={head} /> : head;
}

export function StudioRunTable({ step, state, onSelectionChange, rows, related, fillSlot, runMenu, axisMenu, picking }: TableProps) {
  const scroll = useRef<HTMLDivElement>(null);
  const mountedRows = useRef(new Map<string, HTMLTableRowElement>());
  const pendingKeyboardFocus = useRef<{ id: string; column: number } | null>(null);
  const getScrollElement = useCallback(() => scroll.current, []);
  const itemKey = useCallback((index: number) => rows[index].id, [rows]);
  const virtual = useVirtualizer({
    count: rows.length,
    getScrollElement,
    estimateSize: () => 44,
    getItemKey: itemKey,
    overscan: 6,
    initialRect: { width: 0, height: 360 },
  });
  const showVariant = new Set(rows.map(({ variant }) => variant)).size > 1;
  const sweeps = useMemo(() => runColumns(step, rows), [step, rows]);
  const named = useMemo(() => needsNames(rows, sweeps, showVariant), [rows, sweeps, showVariant]);
  const labels = useValueLabels(sweeps);
  const ordinals = useMemo(() => new Map(step.runs.map(({ id }, index) => [id, index + 1])), [step.runs]);
  const selected = new Set(selectedRuns(step, state.selection).map(({ id }) => id));
  const focus = state.focus?.stepId === step.id ? state.focus.runId : null;
  const focusedIndex = focus === null ? -1 : rows.findIndex(({ id }) => id === focus);
  useEffect(() => {
    if (focusedIndex >= 0) virtual.scrollToIndex(focusedIndex);
  }, [focusedIndex, virtual]);

  /** Chrome's focus scrolls only a cell that is out of view, so a wide cell partly in view is scrolled to here. */
  function focusCell(row: HTMLTableRowElement, column: number) {
    const target = column === 0 ? row.cells[0]?.querySelector('button') : row.cells[column];
    target?.focus({ preventScroll: true });
    target?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }

  function navigate(event: KeyboardEvent<HTMLTableRowElement>, index: number) {
    const gridCell = (event.target as HTMLElement).closest('th,td') as HTMLTableCellElement | null;
    if (gridCell?.parentElement !== event.currentTarget) return;
    const { cellIndex } = gridCell;
    let destination: number;
    switch (event.key) {
      case "ArrowLeft":
      case "ArrowRight": {
        event.preventDefault();
        const column = cellIndex + (event.key === "ArrowLeft" ? -1 : 1);
        focusCell(event.currentTarget, column);
        return;
      }
      case "ArrowUp": destination = Math.max(0, index - 1); break;
      case "ArrowDown": destination = Math.min(rows.length - 1, index + 1); break;
      case "Home": destination = 0; break;
      case "End": destination = rows.length - 1; break;
      default: return;
    }
    event.preventDefault();
    const id = rows[destination].id;
    const row = mountedRows.current.get(id);
    pendingKeyboardFocus.current = row ? null : { id, column: cellIndex };
    virtual.scrollToIndex(destination);
    if (row) focusCell(row, cellIndex);
  }
  const relatedIds = related.get(step.id);
  const items = virtual.getVirtualItems();
  const measure = step.kind === "evaluate" ? "size" : "progress";
  const width = sweeps.length + (showVariant ? 1 : 0) + (named ? 1 : 0) + 3;
  const cellFocus = "outline-none focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-shell-accent";
  const first = items[0];
  const last = items[items.length - 1];
  const top = first?.start ?? 0;
  const bottom = last ? virtual.getTotalSize() - last.end : 0;

  return (
      <Table aria-label={studioName("runs")} aria-rowcount={rows.length + 1} {...picking && { [rulePickSurface]: "" }}
        columns={[
          { least: "2.5rem", greatest: "2.5rem" },
          ...named ? [{ least: "6rem", greatest: "16rem" }] : [],
          ...showVariant ? [{ least: "6rem" }] : [],
          ...sweeps.map(() => ({ least: "6rem" })),
          { least: "8rem", greatest: "28rem" }, { least: "8rem" },
        ]}
        held={named ? 2 : 1}
        box={{ ref: scroll, axis: "both", className: `${fillSlot ? "h-full" : "max-h-96"} [--fade-inset-top:var(--row-height)]` }}>
        <thead className="sticky top-0 z-10">
          <tr aria-rowindex={1}>
            <TableHeaderCell aria-hidden="true" className={gutter} />
            {named && <TableHeaderCell scope="col">{studioName("run")}</TableHeaderCell>}
            {showVariant && <TableHeaderCell scope="col">{studioName("variant")}</TableHeaderCell>}
            {sweeps.map((sweep) => <AxisHead key={sweep.id} step={step.id} sweep={sweep} axisMenu={axisMenu} />)}
            <TableHeaderCell scope="col">{studioName("status")}</TableHeaderCell>
            <TableHeaderCell scope="col">{studioName(measure)}</TableHeaderCell>
          </tr>
        </thead>
        <tbody>
          {top > 0 && <tr aria-hidden="true"><td colSpan={width} style={{ height: top }} /></tr>}
          {items.map((item) => {
            const run = rows[item.index];
            const lit = picking ? picking.runs.has(run.id) : selected.has(run.id);
            const reason = run.results.some(({ validity }) => validity.reasons?.length) ? undefined : run.reason;
            const shownReason = studioRunReason({ ...run, reason });
            const selectRow = picking ? { onClick: () => picking.toggle([run.id]), onKeyDown: undefined }
              : selectionButton((extend) => onSelectionChange(selectRun(state, { stepId: step.id, runId: run.id }, extend)));
            return (
              <CommandContextMenu key={run.id} onOpen={() => { if (!lit) onSelectionChange(selectRun(state, { stepId: step.id, runId: run.id })); }}
                commands={!picking && runMenu ? (anchor) => runMenu([run], anchor) : undefined} trigger={<TableRow
                data-result-run={run.id}
                ref={(node) => {
                  virtual.measureElement(node);
                  if (!node) {
                    mountedRows.current.delete(run.id);
                    return;
                  }
                  mountedRows.current.set(run.id, node);
                  if (pendingKeyboardFocus.current?.id === run.id) {
                    const { column } = pendingKeyboardFocus.current;
                    pendingKeyboardFocus.current = null;
                    focusCell(node, column);
                  }
                }}
                data-index={item.index}
                data-run={run.id}
                selected={lit}
                related={relatedIds?.has(run.id)}
                aria-rowindex={item.index + 2}
                onKeyDown={(event) => { if (!event.defaultPrevented) navigate(event, item.index); }}
                onMouseDown={picking ? (event) => event.preventDefault() : undefined}
                onClick={(event) => {
                  const control = (event.target as Element).closest(rowControls);
                  if (!event.defaultPrevented && (!control || !event.currentTarget.contains(control))) selectRow.onClick(event);
                }}
                className={statusCounted(run.state) ? "studio-run-row" : `studio-run-row ${uncountedRun}`}
              >
                <TableRowHeader className={gutter}>
                  <Button
                    variant="bare" size="content"
                    pressed={state.selection?.kind === "rows" && lit}
                    aria-current={focus === run.id ? true : undefined}
                    onClick={selectRow.onClick}
                    onKeyDown={selectRow.onKeyDown}
                    className="w-full justify-end rounded-row tabular-nums text-shell-muted underline-offset-2 aria-[current=true]:underline"
                  >
                    <span data-run-ordinal>{ordinals.get(run.id)}</span>{' '}
                    <span className="sr-only">{run.label}</span>
                  </Button>
                </TableRowHeader>
                {named && <TableCell tabIndex={-1} className={cellFocus}>{run.label}</TableCell>}
                {showVariant && <TableCell tabIndex={-1} className={cellFocus}>{run.variant}</TableCell>}
                {sweeps.map((sweep) => (
                  <TableCell key={sweep.id} tabIndex={-1} className={`${cellFocus} tabular-nums`}>
                    {labels.get(sweep.id)?.get(run.values[sweep.id])}
                  </TableCell>
                ))}
                <TableCell tabIndex={-1} className={cellFocus} aria-description={shownReason}>
                  <span className="flex min-w-0 items-center gap-1.5">
                    <StudioRunStatus status={run.state} {...appTip(studioRunReason({ ...run, reason }))} />
                    {run.state === "failed" && shownReason && <span data-run-reason className="min-w-0 truncate text-sm"
                      {...appTip(studioRunReason({ ...run, reason }))}>{shownReason.split("\n")[0]}</span>}
                    {run.blockedBy && <Button variant="bare" size="content" className="underline-offset-2 hover:underline"
                      onClick={() => onSelectionChange(selectStep(state, run.blockedBy!.stepId))}>{run.blockedBy.label}</Button>}
                  </span>
                </TableCell>
                <TableCell tabIndex={-1} className={cellFocus}>
                  {(run.status === "running" || run.status === "stopped") && run.progress &&
                    <StudioRunProgress progress={run.progress} etaSeconds={run.status === "running" ? run.etaSeconds : undefined} />}
                  {run.status === "collecting" && <StudioCollectionProgress progress={run.collectionProgress} />}
                  {run.sizeBytes !== undefined && studioBytes(run.sizeBytes)}
                </TableCell>
              </TableRow>} />
            );
          })}
          {bottom > 0 && <tr aria-hidden="true"><td colSpan={width} style={{ height: bottom }} /></tr>}
        </tbody>
      </Table>
  );
}
