import clsx from 'clsx';
import { Check } from 'lucide-react';
import { memo, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent, type PointerEvent } from 'react';
import { appTip } from '@window-shell/browser/tooltip';
import { StatusMark, statusCounted, statusTone, uncountedRun } from '@/components/ui/StatusMark';
import {
  studioAccessibleName,
  studioCountOf,
  studioFormat,
  studioFormatParts,
  studioName,
  studioRunReason,
  studioRunStatus,
  type StudioResultState,
} from '@/components/activity/studioVocabulary';
import type { PlanStepView, PlanSweepView } from '@/features/pipeline/planView';
import {
  extendsSelection,
  selectionButton,
  selectRun,
  selectSweepGroup,
  selectSweepValue,
  type PipelineSelection,
  type PipelineSelectionState,
} from '@/features/pipeline/selection';
import {
  exchangeMapSweeps,
  mapAxes,
  mapMarkSize,
  mapRunCells,
  type MapMarkSize,
  type MarkCell,
  type RunMark,
} from './pipelineMapLayout';
import { CommandContextMenu, CommandMenuItems, type RunMenu } from './CommandContextMenu';
import type { DocumentPickerModel } from '@/features/projectWindow/DocumentPicker';
import { useCommandSource } from '@/features/projectWindow/heldGestures';
import type { StudioCommand } from '@/features/projectWindow/studioCommands';
import { Button } from '@/components/ui/Button';
import { Menu, MenuContent, MenuItem, MenuRadioGroup, MenuRadioItem, MenuSeparator, MenuSubmenu, MenuTrigger } from '@/components/ui/Menu';
import { followPress, type Point } from '@/components/ui/pointerPress';
import { ScrollFade } from '@/components/ui/ScrollFade';

export type ApplyMapSelection = (change: (state: PipelineSelectionState) => PipelineSelectionState) => void;

/** What a step's authored axis offers: the set it reads, as a document picker, and the commands that take the axis. */
export type AxisMenu = (stepId: string, axis: string) => {
  readonly picker?: DocumentPickerModel;
  readonly commands: readonly StudioCommand[];
  /** The command that removes one of the axis's values, for an axis whose values are authored. */
  readonly removeValue?: (value: string, label: string) => StudioCommand | undefined;
  /** Why the axis's last command was refused, or else the first Problem the authored axis has. */
  readonly refusal?: string;
};

type Place = 'rows' | 'columns' | 'folded';
const placeFormat = { rows: 'mapRowsAxis', columns: 'mapColumnsAxis', folded: 'mapFoldedSweep' } as const;
/** An axis names its count only when there are more values than can be counted at a glance; no mark shows a folded sweep's. */
const countedValues = 7;

function MapAxisLabel({
  sweep,
  place,
  step,
  selection,
  applySelection,
  placeAt,
  grab,
  dragging,
  axisMenu,
  pickRuns,
}: {
  axisMenu?: AxisMenu;
  pickRuns?: PickRuns;
  sweep: PlanSweepView;
  place: Place;
  step: PlanStepView;
  selection: PipelineSelection | null;
  applySelection: ApplyMapSelection;
  placeAt?: (place: 'rows' | 'columns') => void;
  /** Starts dragging the axis onto another's place; `close` closes its menu once the press is a drag. */
  grab?: (event: PointerEvent, sweep: string, close: () => void) => void;
  dragging: boolean;
}) {
  const [open, setOpen] = useState(false);
  const count = sweep.values.length;
  const folded = place === 'folded';
  const shownCount = folded || count > countedValues ? studioFormat('mapValueCount', { count }) : '';
  // Only the name's track shrinks, to its first letter, so a shortened label keeps its glyph and count.
  const parts = studioFormatParts<string | null>(placeFormat[place], { name: null, count: shownCount });
  const nameAt = parts.indexOf(null);
  const chosen = new Set(selection?.kind === 'values' && selection.stepId === step.id
    ? selection.terms.find(({ sweepId }) => sweepId === sweep.id)?.valueIds : []);
  // A key's modifiers do not reach the click a menu item synthesizes from it.
  const extendByKey = useRef(false);
  const authored = axisMenu?.(step.id, sweep.id);
  useCommandSource(axisMenu && (() => axisMenu(step.id, sweep.id)));
  const picker = authored?.picker;
  const removals = authored?.removeValue ? sweep.values.flatMap((value) => {
    const command = authored.removeValue!(value.id, value.label);
    return command ? [{ value, command }] : [];
  }) : [];
  return (
    <Menu open={open} onOpenChange={(next) => setOpen(next)}>
      <MenuTrigger render={
        <Button
          variant="ghost"
          size="mark"
          // The line never wraps: folded names shorten first, their weight keeping an axis name's share of the
          // shrink below one layout unit until they are at their least.
          className={clsx('studio-map-axis !shrink aria-expanded:bg-shell-selected data-[dragging]:bg-shell-selected',
            folded && '![flex-shrink:1000000]')}
          data-place={place}
          data-sweep={sweep.id}
          data-dragging={dragging || undefined}
          aria-label={studioAccessibleName('sweep', `${sweep.label} ${count}`)}
          aria-description={authored?.refusal}
          {...appTip(authored?.refusal ? studioRunReason({ reason: authored.refusal })
            : studioAccessibleName('sweep', `${sweep.label} ${count}`))}
          onPointerDown={grab && ((event) => grab(event, sweep.id, () => setOpen(false)))}
        >
          <span className="grid grid-cols-[auto_minmax(1.5em,max-content)_auto] whitespace-pre">
            <span>{parts.slice(0, nameAt)}</span>
            <span className="truncate">{sweep.label}</span>
            <span>{parts.slice(nameAt + 1)}</span>
          </span>
          {authored?.refusal && <StatusMark status="cannot_run" label={studioRunStatus('cannot_run')} />}
        </Button>
      } />
      <MenuContent aria-label={studioAccessibleName('sweep', sweep.label)}>
        <ScrollFade axis="y" className="nowheel max-h-60">
          {sweep.values.map((value) => (
            <MenuItem key={value.id}
              glyph={<span className="flex size-[1em] shrink-0 items-center justify-center">{chosen.has(value.id) && <Check />}</span>}
              onKeyDown={(event) => { extendByKey.current = extendsSelection(event); }}
              onClick={(event) => {
                const extend = extendsSelection(event) || extendByKey.current;
                extendByKey.current = false;
                if (pickRuns) pickRuns(step.runs.filter(({ values }) => values[sweep.id] === value.id).map(({ id }) => id));
                else applySelection((state) => selectSweepValue(state, step.id, sweep.id, value.id, extend));
              }}>
              {value.label}
            </MenuItem>
          ))}
        </ScrollFade>
        {picker && <>
          <MenuSeparator />
          <MenuRadioGroup value={picker.document}
            onValueChange={(next: string) => { if (next !== picker.document) picker.choose?.(next); }}>
            {picker.documents.map(({ key, name, depth }) => <MenuRadioItem key={key} value={key} disabled={!picker.choose}>
              <span style={depth ? { paddingInlineStart: `${depth}rem` } : undefined}>{name}</span>
            </MenuRadioItem>)}
          </MenuRadioGroup>
          <MenuItem disabled={!picker.duplicate} onClick={() => { void picker.duplicate?.(); }}>{studioName('duplicate')}</MenuItem>
          {picker.detach && <MenuItem onClick={picker.detach}>{studioName('detach')}</MenuItem>}
        </>}
        {(!!authored?.commands.length || !!removals.length) && <MenuSeparator />}
        {!!removals.length && <MenuSubmenu label={studioName(removals[0].command.name)}>
          {removals.map(({ value, command }) => <MenuItem key={value.id} disabled={!command.enabled} onClick={command.invoke}>
            {value.label}
          </MenuItem>)}
        </MenuSubmenu>}
        {!!authored?.commands.length && <CommandMenuItems commands={() => authored.commands} />}
        {placeAt && <>
          <MenuSeparator />
          <MenuRadioGroup value={folded ? '' : place}
            onValueChange={(next: string) => placeAt(next as 'rows' | 'columns')}>
            <MenuRadioItem value="rows">{studioName('rows')}</MenuRadioItem>
            <MenuRadioItem value="columns">{studioName('columns')}</MenuRadioItem>
          </MenuRadioGroup>
        </>}
      </MenuContent>
    </Menu>
  );
}

interface MarkPresentation {
  mark: RunMark;
  label: string;
  statuses: readonly [StudioResultState, number][];
}

/** Toggles runs among those picked for a rule, in place of selecting them. */
export type PickRuns = (runs: readonly string[]) => void;

const markRadius: Record<MapMarkSize['id'], string> = { full: 'rounded-sm', small: 'rounded-[2px]', smallest: 'rounded-[1px]' };

function MarkButton({
  item,
  size,
  stepId,
  selected,
  selectionKind,
  boxed,
  upstream,
  downstream,
  applySelection,
  runMenu,
  pickRuns,
}: {
  item: MarkPresentation;
  pickRuns?: PickRuns;
  size: MapMarkSize;
  stepId: string;
  selected: ReadonlySet<string>;
  selectionKind?: PipelineSelection['kind'];
  /** The runs a rule being written chooses, boxed as a selection of runs is. */
  boxed?: ReadonlySet<string>;
  upstream?: ReadonlySet<string>;
  downstream?: ReadonlySet<string>;
  applySelection: ApplyMapSelection;
  runMenu?: RunMenu;
}) {
  const { mark, label, statuses } = item;
  const runs = mark.kind === 'group' ? mark.runs : [mark.run];
  const selectedCount = runs.reduce((count, run) => count + Number(selected.has(run.id)), 0);
  let upstreamCount = 0;
  let downstreamCount = 0;
  let relatedCount = 0;
  if (upstream || downstream) for (const { id } of runs) {
    const before = upstream?.has(id);
    const after = downstream?.has(id);
    upstreamCount += Number(!!before);
    downstreamCount += Number(!!after);
    relatedCount += Number(!!before || !!after);
  }
  const lineage = upstreamCount ? downstreamCount ? 'both' : 'upstream' : downstreamCount ? 'downstream' : undefined;
  const lineageDescription = [
    upstreamCount ? studioName('madeFrom') : '',
    downstreamCount ? studioName('usedBy') : '',
    relatedCount && mark.kind === 'group' ? studioCountOf(relatedCount, runs.length) : '',
  ].filter(Boolean).join(' · ');
  const status = mark.kind === 'run'
    ? studioRunStatus(mark.run.state)
    : statuses.map(([state, count]) => `${count} ${studioRunStatus(state)}`).join(' · ');
  const share = selectedCount && mark.kind === 'group'
    ? studioCountOf(selectedCount, runs.length)
    : null;
  const name = [label, status, share].filter(Boolean).join(' · ');
  const failure = mark.kind === 'run' && mark.run.state === 'failed' ? studioRunReason(mark.run) : undefined;
  const shaped = size.id === 'full' && statuses.length === 1;
  const chooseFirst = () => {
    if (selectedCount === runs.length) return;
    if (mark.kind === 'run') applySelection((state) => selectRun(state, { stepId, runId: mark.run.id }));
    else if (mark.terms) applySelection(() => selectSweepGroup(stepId, mark.terms!));
    else applySelection(() => ({ selection: { kind: 'rows', stepId, runIds: runs.map(({ id }) => id) }, focus: null }));
  };
  const choose = pickRuns
    ? { onClick: () => pickRuns(runs.map(({ id }) => id)), onMouseDown: (event: MouseEvent) => event.preventDefault() }
    : selectionButton((extend) => {
      if (mark.kind === 'run') applySelection((state) => selectRun(state, { stepId, runId: mark.run.id }, extend));
      else if (mark.terms) applySelection(() => selectSweepGroup(stepId, mark.terms!));
      else applySelection(() => ({ selection: { kind: 'rows', stepId, runIds: runs.map(({ id }) => id) }, focus: null }));
    });
  return <CommandContextMenu onOpen={chooseFirst} commands={!pickRuns && runMenu ? (anchor) => runMenu(runs, anchor) : undefined} trigger={
    <Button
      variant="bare"
      size="content"
      data-result-run={mark.kind === 'run' ? mark.run.id : undefined}
      aria-label={studioAccessibleName('select', name)}
      aria-pressed={!selectedCount ? false : selectedCount === runs.length ? true : 'mixed'}
      aria-description={[failure, lineageDescription].filter(Boolean).join(' · ') || undefined}
      {...appTip(studioAccessibleName('select', [name, failure].filter(Boolean).join(' · ')))}
      data-map-focus={lineage}
      data-boxed={boxed ? runs.some(({ id }) => boxed.has(id)) ? '' : undefined
        : selectionKind === 'rows' && selectedCount > 0 ? '' : undefined}
      data-group-member={selectionKind === 'values' && selectedCount > 0 && mark.kind === 'run' ? '' : undefined}
      className={clsx('studio-map-run-mark relative flex shrink-0', markRadius[size.id],
        !runs.some(({ state }) => statusCounted(state)) && uncountedRun)}
      style={{ width: size.mark, height: size.mark }}
      {...choose}
    >
      {shaped ? <StatusMark status={statuses[0][0]} /> : (
        <span aria-hidden="true" className={clsx('flex size-full overflow-hidden', markRadius[size.id])}>
          {statuses.map(([state, count]) => (
            <span key={state} className={clsx('h-full min-w-0 basis-0 bg-current', statusTone(state))} style={{ flexGrow: count }} />
          ))}
        </span>
      )}
      {selectedCount > 0 && mark.kind === 'group' && (
        <span
          aria-hidden="true"
          className="absolute bottom-0 left-0 bg-shell-foreground"
          style={{ width: `${100 * selectedCount / runs.length}%`, height: Math.max(1, size.mark / 4) }}
        />
      )}
      {relatedCount > 0 && <span aria-hidden="true"
        className={mark.kind === 'group'
          ? 'absolute left-0 top-0 h-0 border-t-2 border-dotted border-shell-foreground'
          : 'absolute -inset-[2px] rounded-[3px] border border-dashed border-shell-foreground'}
        style={mark.kind === 'group' ? { width: `${100 * relatedCount / runs.length}%` } : undefined} />}
    </Button>
  } />;
}

const arrowSteps: Record<string, readonly [number, number]> = {
  ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0],
};

/** Arrow keys move to the nearest mark in their direction, past empty cells. */
function navigateMarks(event: KeyboardEvent<HTMLDivElement>) {
  const direction = arrowSteps[event.key];
  if (event.defaultPrevented || !direction || !(event.target instanceof HTMLElement)) return;
  const cell = event.target.closest<HTMLElement>('[aria-colindex]');
  const row = cell?.closest<HTMLElement>('[aria-rowindex]');
  if (!cell || !row) return;
  event.preventDefault();
  const grid = event.currentTarget;
  const rows = Number(grid.getAttribute('aria-rowcount'));
  const columns = Number(grid.getAttribute('aria-colcount'));
  const [down, across] = direction;
  for (
    let r = Number(row.getAttribute('aria-rowindex')) + down, c = Number(cell.getAttribute('aria-colindex')) + across;
    r >= 1 && r <= rows && c >= 1 && c <= columns;
    r += down, c += across
  ) {
    const next = grid.querySelector<HTMLButtonElement>(`[aria-rowindex="${r}"] > [aria-colindex="${c}"] button`);
    if (next) {
      next.focus();
      return;
    }
  }
}

/** A step's runs on its card: the label line naming each sweep's place, then one mark per cell on a uniform pitch. */
export const PipelineRunMarks = memo(function PipelineRunMarks({
  step,
  sweeps: stored,
  selection,
  selected,
  boxed,
  upstream,
  downstream,
  applySelection,
  runMenu,
  axisMenu,
  arrange,
  pickRuns,
}: {
  axisMenu?: AxisMenu;
  pickRuns?: PickRuns;
  step: PlanStepView;
  sweeps: readonly PlanSweepView[];
  selection: PipelineSelection | null;
  selected: ReadonlySet<string>;
  boxed?: ReadonlySet<string>;
  upstream?: ReadonlySet<string>;
  downstream?: ReadonlySet<string>;
  applySelection: ApplyMapSelection;
  runMenu?: RunMenu;
  /** Saves a new sweep order; absent where the card cannot be arranged. */
  arrange?: (order: string[]) => void | Promise<unknown>;
}) {
  const axesLine = useRef<HTMLDivElement>(null);
  /** An axis dragged onto another's place, `to`, over the order it was dragged in; it is drawn exchanged until saved. */
  const [swap, setSwap] = useState<{ base: readonly PlanSweepView[]; from: string; to: string | null }>();
  const swapped = swap?.base === stored && swap.to ? exchangeMapSweeps(stored, swap.from, swap.to) : undefined;
  const swappedKey = swapped?.join('\n');
  const sweeps = useMemo(() => swapped ? swapped.map((id) => stored.find((sweep) => sweep.id === id)!) : stored,
    [stored, swappedKey]);
  const axes = useMemo(() => mapAxes(sweeps), [sweeps]);
  const grid = useMemo(() => {
    const { cells, rowCount, columnCount } = mapRunCells(step.runs, axes);
    const valuesBySweep = new Map(sweeps.map(
      (sweep) => [sweep.id, new Map(sweep.values.map(({ id, label }) => [id, label]))],
    ));
    const present = ({ mark }: MarkCell): MarkPresentation => {
      if (mark.kind === 'run') return { mark, label: mark.run.label, statuses: [[mark.run.state, 1]] };
      const scope = (mark.terms ?? []).map(({ sweepId, valueIds }) =>
        valueIds.map((id) => valuesBySweep.get(sweepId)?.get(id) ?? id).join(', '));
      const counts = new Map<StudioResultState, number>();
      for (const run of mark.runs) counts.set(run.state, (counts.get(run.state) ?? 0) + 1);
      return { mark, label: [...scope, `${mark.runs.length} ${studioName('runs')}`].join(' · '), statuses: [...counts] };
    };
    const byRow = Array.from({ length: rowCount }, () => [] as (MarkCell & { item: MarkPresentation })[]);
    for (const cell of cells) byRow[cell.row].push({ ...cell, item: present(cell) });
    for (const row of byRow) row.sort((a, b) => a.column - b.column);
    return { byRow, rowCount, columnCount, size: mapMarkSize(columnCount, rowCount) };
  }, [step.runs, sweeps, axes]);
  // A step with no runs still draws its axes, such as one whose required Set has no entries, to mark and repair them.
  if (!grid.rowCount && !sweeps.length) return null;
  const { rows, columns, folded } = axes;
  const { size } = grid;
  const terms = selection?.kind === 'values' && selection.stepId === step.id ? selection.terms : [];
  const chosen = (sweep: PlanSweepView | undefined) => terms.find(({ sweepId }) => sweepId === sweep?.id)?.valueIds;
  const chosenRows = chosen(rows);
  const chosenColumns = chosen(columns);
  const boxRows = terms.length === 1 && chosenRows !== undefined;
  const boxColumns = terms.length === 1 && chosenColumns !== undefined;
  const boxIntersections = terms.length === 2 && chosenRows !== undefined && chosenColumns !== undefined;
  const exchange = arrange && stored.length > 1
    ? (from: string, to: string) => {
      const order = exchangeMapSweeps(stored, from, to);
      return order && arrange(order);
    }
    : undefined;
  const grab = exchange && ((event: PointerEvent, from: string, close: () => void) => followPress(event, () => {
    close();
    const base = stored;
    const places = [...axesLine.current?.querySelectorAll<HTMLElement>('[data-sweep]') ?? []]
      .map((element) => ({ id: element.dataset.sweep!, box: element.getBoundingClientRect() }));
    // The places are those the axes had when the drag began, so an exchange drawn under the pointer does not move them.
    const over = ({ x, y }: Point) => {
      const to = places.find(({ box }) => x >= box.left && x <= box.right && y >= box.top - box.height && y <= box.bottom + box.height)?.id;
      return to === from ? null : to ?? null;
    };
    return {
      move: (at) => setSwap({ base, from, to: over(at) }),
      release(at) {
        const to = over(at);
        if (!to) return setSwap(undefined);
        const held = { base, from, to };
        setSwap(held);
        void Promise.resolve(exchange(from, to)).finally(() => requestAnimationFrame(() => requestAnimationFrame(() =>
          setSwap((current) => current === held ? undefined : current))));
      },
      cancel: () => setSwap(undefined),
    };
  }));
  const label = (sweep: PlanSweepView, place: Place) => (
    <MapAxisLabel key={sweep.id} sweep={sweep} place={place} step={step} selection={selection}
      applySelection={applySelection} grab={grab} dragging={swap?.from === sweep.id} axisMenu={axisMenu} pickRuns={pickRuns}
      placeAt={exchange && rows && columns ? (target) => exchange(sweep.id, (target === 'rows' ? rows : columns).id) : undefined} />
  );
  return (
    <div className="grid gap-1">
      {sweeps.length > 0 && (
        <div ref={axesLine} className="studio-map-axes flex min-w-0 items-center gap-1 overflow-hidden text-xs text-shell-muted">
          {rows && label(rows, 'rows')}
          {columns && label(columns, 'columns')}
          {folded.map((sweep) => label(sweep, 'folded'))}
        </div>
      )}
      {grid.rowCount > 0 && <ScrollFade axis="both" className="nowheel max-h-56 p-1">
        <div role="grid" aria-label={studioName('runs')} aria-rowcount={grid.rowCount} aria-colcount={grid.columnCount}
          data-mark-size={size.id} onKeyDown={navigateMarks}
          className="studio-map-run-grid relative grid w-max"
          style={{
            gridTemplateColumns: `repeat(${grid.columnCount}, ${size.mark}px)`,
            gridAutoRows: `${size.mark}px`,
            gap: size.gap,
          }}>
          {boxRows && rows?.values.map((value, index) => chosenRows?.includes(value.id) && (
            <span key={`row:${value.id}`} aria-hidden="true" className="studio-map-group-box"
              style={{ gridRow: `${index + 1} / span 1`, gridColumn: `1 / ${grid.columnCount + 1}` }} />
          ))}
          {boxColumns && columns?.values.map((value, index) => chosenColumns?.includes(value.id) && (
            <span key={`column:${value.id}`} aria-hidden="true" className="studio-map-group-box"
              style={{ gridRow: `1 / ${grid.rowCount + 1}`, gridColumn: `${index + 1} / span 1` }} />
          ))}
          {grid.byRow.map((cells, row) => (
            <div role="row" aria-rowindex={row + 1} key={row} className="contents">
              {cells.map((cell) => (
                <div role="gridcell" aria-colindex={cell.column + 1} key={cell.column}
                  data-map-group-cell={boxIntersections && chosenRows.includes(rows?.values[cell.row]?.id ?? '')
                    && chosenColumns.includes(columns?.values[cell.column]?.id ?? '') ? '' : undefined}
                  className="relative flex"
                  style={{ gridRow: cell.row + 1, gridColumn: cell.column + 1 }}>
                  <MarkButton item={cell.item} size={size} stepId={step.id} selected={selected}
                    selectionKind={selection?.kind} boxed={boxed} upstream={upstream} downstream={downstream}
                    applySelection={applySelection} runMenu={runMenu} pickRuns={pickRuns} />
                </div>
              ))}
            </div>
          ))}
        </div>
      </ScrollFade>}
    </div>
  );
});
