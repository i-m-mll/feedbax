import { useCallback, useEffect, useMemo, useRef, type KeyboardEvent } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { StudioRunProgress, StudioRunStatus } from "@/components/activity/StudioRunStatus";
import { studioCountOf, studioName, studioReason } from "@/components/activity/studioVocabulary";
import type { PlanRunView, PlanStepView, PlanSweepView } from "@/features/pipeline/planView";
import {
  selectionButton,
  selectedRuns,
  selectRun,
  selectSweepValue,
  selectionTerms,
  type PipelineSelectionState,
  type RunSetsByStep,
} from "@/features/pipeline/selection";

const byteSize = new Intl.NumberFormat(undefined, { style: "unit", unit: "byte", unitDisplay: "narrow" });

export interface RunSelectionProps {
  step: PlanStepView;
  state: PipelineSelectionState;
  onSelectionChange: (state: PipelineSelectionState) => void;
}

export function StudioSweepValues({ step, state, onSelectionChange }: RunSelectionProps) {
  const selection = state.selection;
  const terms = selection?.kind === "values" && selection.stepId === step.id ? selection.terms : [];
  return (
    <div className="space-y-1">
      {step.sweeps.map((sweep) => {
        const selected = terms.find(({ sweepId }) => sweepId === sweep.id);
        return (
          <fieldset key={sweep.id} className="flex flex-wrap items-baseline gap-1.5">
            <legend className="font-medium text-slate-800">{sweep.label}</legend>
            {sweep.values.map((value) => (
              <button
                key={value.id}
                type="button"
                aria-pressed={selected?.valueIds.includes(value.id) ?? false}
                {...selectionButton((extend) => onSelectionChange(selectSweepValue(state, step.id, sweep.id, value.id, extend)))}
                className="rounded border border-slate-300 px-2 py-0.5 tabular-nums aria-pressed:border-violet-600 aria-pressed:bg-violet-100"
              >
                {value.label}
              </button>
            ))}
          </fieldset>
        );
      })}
    </div>
  );
}

function variedSweeps(step: PlanStepView) {
  return step.sweeps.filter((sweep) => sweep.values.length > 1);
}

function useValueLabels(sweeps: readonly PlanSweepView[]) {
  return useMemo(() => new Map(sweeps.map((sweep) => [
    sweep.id, new Map(sweep.values.map((value) => [value.id, value.label])),
  ])), [sweeps]);
}

interface TableProps extends RunSelectionProps {
  rows: readonly PlanRunView[];
  related: RunSetsByStep;
}

export function StudioRunTable({ step, state, onSelectionChange, rows, related }: TableProps) {
  const scroll = useRef<HTMLDivElement>(null);
  const rowButtons = useRef(new Map<string, HTMLButtonElement>());
  const pendingKeyboardFocus = useRef<string | null>(null);
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
  const sweeps = useMemo(() => variedSweeps(step), [step]);
  const labels = useValueLabels(sweeps);
  const showVariant = new Set(step.runs.map(({ variant }) => variant)).size > 1;
  const selected = new Set(selectedRuns(step, state.selection).map(({ id }) => id));
  const focus = state.focus?.stepId === step.id ? state.focus.runId : null;
  const focusedIndex = focus === null ? -1 : rows.findIndex(({ id }) => id === focus);
  useEffect(() => {
    if (focusedIndex >= 0) virtual.scrollToIndex(focusedIndex);
  }, [focusedIndex, virtual]);

  function navigate(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let destination: number;
    switch (event.key) {
      case "ArrowUp": destination = Math.max(0, index - 1); break;
      case "ArrowDown": destination = Math.min(rows.length - 1, index + 1); break;
      case "Home": destination = 0; break;
      case "End": destination = rows.length - 1; break;
      default: return;
    }
    event.preventDefault();
    const id = rows[destination].id;
    const button = rowButtons.current.get(id);
    pendingKeyboardFocus.current = button ? null : id;
    virtual.scrollToIndex(destination);
    button?.focus({ preventScroll: true });
  }
  const relatedIds = related.get(step.id);
  const items = virtual.getVirtualItems();
  const measure = step.kind === "evaluate" ? "size" : "progress";
  const width = sweeps.length + (showVariant ? 4 : 3);
  const first = items[0];
  const last = items[items.length - 1];
  const top = first?.start ?? 0;
  const bottom = last ? virtual.getTotalSize() - last.end : 0;

  return (
    <div ref={scroll} className="max-h-96 overflow-auto rounded-md border border-slate-200">
      <table className="w-full border-separate border-spacing-0 text-left text-sm" aria-label={studioName("runs")} aria-rowcount={rows.length + 1}>
        <thead className="sticky top-0 z-10 bg-slate-50">
          <tr aria-rowindex={1}>
            <th scope="col" className="p-2">{studioName("run")}</th>
            {showVariant && <th scope="col" className="p-2">{studioName("variant")}</th>}
            {sweeps.map((sweep) => <th key={sweep.id} scope="col" className="p-2">{sweep.label}</th>)}
            <th scope="col" className="p-2">{studioName("status")}</th>
            <th scope="col" className="p-2">{studioName(measure)}</th>
          </tr>
        </thead>
        <tbody>
          {top > 0 && <tr aria-hidden="true"><td colSpan={width} style={{ height: top }} /></tr>}
          {items.map((item) => {
            const run = rows[item.index];
            const lit = selected.has(run.id);
            const selectRow = selectionButton((extend) => onSelectionChange(selectRun(state, { stepId: step.id, runId: run.id }, extend)));
            return (
              <tr
                key={run.id}
                ref={virtual.measureElement}
                data-index={item.index}
                data-selected={lit || undefined}
                data-related={relatedIds?.has(run.id) || undefined}
                aria-rowindex={item.index + 2}
                className="studio-run-row data-[related=true]:bg-slate-100 data-[selected=true]:bg-violet-50"
              >
                <th scope="row" className="border-t border-slate-200 p-2 font-medium">
                  <button
                    type="button"
                    ref={(button) => {
                      if (!button) {
                        rowButtons.current.delete(run.id);
                        return;
                      }
                      rowButtons.current.set(run.id, button);
                      if (pendingKeyboardFocus.current === run.id) {
                        pendingKeyboardFocus.current = null;
                        button.focus({ preventScroll: true });
                      }
                    }}
                    aria-pressed={state.selection?.kind === "rows" && lit}
                    aria-current={focus === run.id ? true : undefined}
                    onClick={selectRow.onClick}
                    onKeyDown={(event) => {
                      selectRow.onKeyDown(event);
                      if (!event.defaultPrevented) navigate(event, item.index);
                    }}
                    className="rounded text-left underline-offset-2 hover:underline aria-[current=true]:underline"
                  >
                    {run.label}
                  </button>
                </th>
                {showVariant && <td className="border-t border-slate-200 p-2">{run.variant}</td>}
                {sweeps.map((sweep) => (
                  <td key={sweep.id} className="border-t border-slate-200 p-2 tabular-nums">
                    {labels.get(sweep.id)?.get(run.values[sweep.id])}
                  </td>
                ))}
                <td className="border-t border-slate-200 p-2">
                  <StudioRunStatus status={run.status} />
                  {run.reason && <p className="text-xs text-slate-700">{studioReason("observed", run.reason)}</p>}
                </td>
                <td className="border-t border-slate-200 p-2">
                  {run.progress && <StudioRunProgress progress={run.progress} etaSeconds={run.etaSeconds} />}
                  {run.sizeBytes !== undefined && byteSize.format(run.sizeBytes)}
                </td>
              </tr>
            );
          })}
          {bottom > 0 && <tr aria-hidden="true"><td colSpan={width} style={{ height: bottom }} /></tr>}
        </tbody>
      </table>
    </div>
  );
}

export function StudioSelectionSummary({ step, state }: Pick<RunSelectionProps, "step" | "state">) {
  const { selection } = state;
  const active = selection?.kind === "steps" ? selection.stepIds.includes(step.id) : selection?.stepId === step.id;
  if (!active) return null;
  const selected = selectedRuns(step, selection);
  const terms = selectionTerms(step, selection);
  const count = studioCountOf(selected.length, step.runs.length);
  return <p aria-live="polite" className="text-sm text-slate-700">{[...terms, count].join(" · ")}</p>;
}
