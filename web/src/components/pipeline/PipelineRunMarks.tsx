import { memo, useMemo, type KeyboardEvent } from 'react';
import { studioRunMarkClass } from '@/components/activity/StudioRunStatus';
import {
  studioAccessibleName,
  studioCountOf,
  studioName,
  studioRunStatus,
  type StudioRunStatusId,
} from '@/components/activity/studioVocabulary';
import type { PlanStepView, PlanSweepView } from '@/features/pipeline/planView';
import {
  selectionButton,
  selectRun,
  selectSweepGroup,
  selectSweepValue,
  type PipelineSelection,
  type PipelineSelectionState,
} from '@/features/pipeline/selection';
import { mapRunCells, type RunMark } from './pipelineMapLayout';

export type ApplyMapSelection = (change: (state: PipelineSelectionState) => PipelineSelectionState) => void;

export function MapSweepValue({
  sweep,
  value,
  stepId,
  selection,
  applySelection,
}: {
  sweep: PlanSweepView;
  value: PlanSweepView['values'][number];
  stepId: string;
  selection: PipelineSelection | null;
  applySelection: ApplyMapSelection;
}) {
  const selected = selection?.kind === 'values' && selection.terms.some(
    (term) => term.sweepId === sweep.id && term.valueIds.includes(value.id),
  );
  return (
    <button
      type="button"
      className="rounded px-1 text-xs text-slate-700 hover:bg-slate-100 aria-pressed:bg-slate-200 aria-pressed:font-semibold"
      aria-label={studioAccessibleName('select', `${sweep.label} ${value.label}`)}
      aria-pressed={selected}
      {...selectionButton((extend) => applySelection((state) => selectSweepValue(state, stepId, sweep.id, value.id, extend)))}
    >
      {value.label}
    </button>
  );
}

interface MarkPresentation {
  mark: RunMark;
  label: string;
  statuses: readonly [StudioRunStatusId, number][];
}

function MarkButton({
  item,
  stepId,
  selected,
  selectionKind,
  applySelection,
}: {
  item: MarkPresentation;
  stepId: string;
  selected: ReadonlySet<string>;
  selectionKind?: PipelineSelection['kind'];
  applySelection: ApplyMapSelection;
}) {
  const { mark, label, statuses } = item;
  const runs = mark.kind === 'group' ? mark.runs : [mark.run];
  const selectedCount = runs.reduce((count, run) => count + Number(selected.has(run.id)), 0);
  const status = mark.kind === 'run'
    ? studioRunStatus(mark.run.status)
    : statuses.map(([state, count]) => `${count} ${studioRunStatus(state)}`).join(' · ');
  const share = selectedCount && mark.kind === 'group'
    ? studioCountOf(selectedCount, runs.length)
    : null;
  const name = [label, status, share].filter(Boolean).join(' · ');
  const ring = selectedCount && (selectionKind === 'rows'
    ? 'outline outline-2 outline-slate-900'
    : 'outline outline-1 outline-slate-700');
  return (
    <button
      type="button"
      aria-label={studioAccessibleName('select', name)}
      aria-pressed={!selectedCount ? false : selectedCount === runs.length ? true : 'mixed'}
      className={`relative h-3 w-3 shrink-0 rounded-[2px] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 ${ring || ''}`}
      {...selectionButton((extend) => {
        if (mark.kind === 'group') applySelection(() => selectSweepGroup(stepId, mark.terms));
        else applySelection((state) => selectRun(state, { stepId, runId: mark.run.id }, extend));
      })}
    >
      <span aria-hidden="true" className="absolute inset-0 flex overflow-hidden rounded-[2px]">
        {statuses.map(([state, count]) => (
          <span key={state} className={studioRunMarkClass(state)} style={{ flexGrow: count }} />
        ))}
      </span>
      {selectedCount > 0 && mark.kind === 'group' && (
        <span
          aria-hidden="true"
          className="absolute inset-x-0 bottom-0 h-1 rounded-bl-[2px] bg-slate-900"
          style={{ width: `${100 * selectedCount / runs.length}%` }}
        />
      )}
    </button>
  );
}

function navigateMarks(event: KeyboardEvent<HTMLDivElement>) {
  const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'];
  if (event.defaultPrevented || !keys.includes(event.key) || !(event.target instanceof HTMLElement)) return;
  const cell = event.target.closest<HTMLElement>('[aria-colindex]');
  const row = cell?.closest<HTMLElement>('[aria-rowindex]');
  if (!cell || !row) return;
  const horizontal = event.key === 'ArrowLeft' || event.key === 'ArrowRight';
  const direction = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
  const buttons = [...cell.querySelectorAll<HTMLButtonElement>('button')];
  const adjacent = horizontal ? buttons[buttons.indexOf(event.target as HTMLButtonElement) + direction] : undefined;
  const selector = '[aria-colindex]';
  const column = cell.getAttribute('aria-colindex');
  let destination = adjacent;
  if (!destination && horizontal) {
    const cells = [...row.querySelectorAll<HTMLElement>(selector)];
    const next = cells[cells.indexOf(cell) + direction];
    const controls = next && [...next.querySelectorAll<HTMLButtonElement>('button')];
    destination = direction < 0 ? controls?.at(-1) : controls?.[0];
  } else if (!horizontal) {
    const rows = [...event.currentTarget.querySelectorAll<HTMLElement>('[aria-rowindex]')];
    for (let index = rows.indexOf(row) + direction; rows[index] && !destination; index += direction) {
      destination = rows[index].querySelector<HTMLButtonElement>(`[aria-colindex="${column}"] button`) ?? undefined;
    }
  }
  event.preventDefault();
  destination?.focus();
}

export const PipelineRunMarks = memo(function PipelineRunMarks({
  step,
  sweeps,
  folded,
  selection,
  selected,
  applySelection,
}: {
  step: PlanStepView;
  sweeps: readonly PlanSweepView[];
  folded: PlanSweepView | undefined;
  selection: PipelineSelection | null;
  selected: ReadonlySet<string>;
  applySelection: ApplyMapSelection;
}) {
  const grid = useMemo(() => {
    const { rowSweep, columnSweep, cells } = mapRunCells(step.runs, sweeps, folded);
    const valuesBySweep = new Map(sweeps.map(
      (sweep) => [sweep.id, new Map(sweep.values.map(({ id, label }) => [id, label]))],
    ));
    const positions = (sweep?: PlanSweepView) => new Map(sweep?.values.map(({ id }, index) => [id, index]));
    const rows = positions(rowSweep);
    const columns = positions(columnSweep);
    const projectedCells = [...cells].map(([key, cell]) => ({
      key,
      row: (rows.get(cell.row) ?? rows.size) + (columnSweep ? 2 : 1),
      column: (columns.get(cell.column) ?? columns.size) + (rowSweep ? 2 : 1),
      marks: [...cell.marks].map(([id, mark]) => {
        if (mark.kind === 'run') {
          const coordinates = sweeps.flatMap((sweep) => {
            const value = valuesBySweep.get(sweep.id)?.get(mark.run.values[sweep.id]);
            return value === undefined ? [] : [`${sweep.label} ${value}`];
          });
          const label = coordinates.join(' · ') || mark.run.label;
          return { id, mark, label, statuses: [[mark.run.status, 1]] } satisfies MarkPresentation & { id: string };
        }
        const scope = sweeps.flatMap((sweep) => {
          const term = mark.terms.find(({ sweepId }) => sweepId === sweep.id);
          if (!term || sweep.id === folded?.id) return [];
          const values = term.valueIds.map((valueId) => valuesBySweep.get(sweep.id)?.get(valueId) ?? valueId);
          return [`${sweep.label} ${values.join(', ')}`];
        });
        const counts = new Map<StudioRunStatusId, number>();
        for (const run of mark.runs) counts.set(run.status, (counts.get(run.status) ?? 0) + 1);
        const label = [...scope, `${mark.runs.length} ${studioName('runs')}`].join(' · ');
        return { id, mark, label, statuses: [...counts] };
      }),
    }));
    const cellsByRow = new Map<number, typeof projectedCells>();
    for (const cell of projectedCells) {
      const row = cellsByRow.get(cell.row);
      if (row) row.push(cell);
      else cellsByRow.set(cell.row, [cell]);
    }
    for (const cells of cellsByRow.values()) cells.sort((a, b) => a.column - b.column);
    const extraColumn = projectedCells.some((cell) => cell.column > columns.size + Number(Boolean(rowSweep)));
    return {
      rowSweep, columnSweep, cellsByRow,
      rowCount: Math.max(0, ...cellsByRow.keys()),
      columnCount: Math.max(1, columns.size + Number(extraColumn)),
    };
  }, [step.runs, sweeps, folded]);
  if (!grid.rowCount) return null;
  const headerRows = grid.columnSweep ? 1 : 0;
  return (
    <div className="max-h-56 overflow-auto p-1">
      <div role="grid" aria-label={studioName('runs')} aria-rowcount={grid.rowCount}
        aria-colcount={grid.columnCount + Number(!!grid.rowSweep)} onKeyDown={navigateMarks}
        className="inline-grid items-center gap-1"
        style={{ gridTemplateColumns: `${grid.rowSweep ? 'max-content ' : ''}repeat(${grid.columnCount}, max-content)` }}>
        {grid.columnSweep && (
          <div role="row" aria-rowindex={1} className="contents">
            {grid.columnSweep.values.map((value, index) => (
              <div role="columnheader" aria-colindex={index + 2} key={value.id} className="text-center"
                style={{ gridRow: 1, gridColumn: index + 2 }}>
                <MapSweepValue sweep={grid.columnSweep!} value={value} stepId={step.id}
                  selection={selection} applySelection={applySelection} />
              </div>
            ))}
          </div>
        )}
        {Array.from({ length: grid.rowCount - headerRows }, (_, index) => {
          const row = index + headerRows + 1;
          const heading = grid.rowSweep?.values[index];
          return (
            <div role="row" aria-rowindex={row} key={row} className="contents">
              {grid.rowSweep && heading && (
                <div role="rowheader" aria-colindex={1} className="text-right" style={{ gridColumn: 1, gridRow: row }}>
                  <MapSweepValue sweep={grid.rowSweep} value={heading} stepId={step.id}
                    selection={selection} applySelection={applySelection} />
                </div>
              )}
              {grid.cellsByRow.get(row)?.map((cell) => (
                <div role="gridcell" aria-colindex={cell.column} key={cell.key}
                  className="flex flex-wrap items-center justify-center gap-1"
                  style={{ gridColumn: cell.column, gridRow: row, maxWidth: 192 }}>
                  {cell.marks.map((item) => (
                    <MarkButton key={item.id} item={item} stepId={step.id} selected={selected}
                      selectionKind={selection?.kind} applySelection={applySelection} />
                  ))}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
});
