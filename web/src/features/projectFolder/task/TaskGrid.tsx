import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { GripHorizontal, GripVertical, MoreHorizontal, Plus } from 'lucide-react';
import { useEffect, useRef, useState, type DragEvent, type ReactElement, type ReactNode, type SyntheticEvent } from 'react';
import { studioAccessibleName, studioName, studioValueType, type StudioNameId } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { FieldProblem } from '@/components/ui/FieldProblem';
import { InlineEdit } from '@/components/ui/InlineEdit';
import {
  ContextMenu, ContextMenuContent, ContextMenuTrigger, Menu, MenuCheckboxItem, MenuContent, MenuItem, MenuTrigger,
} from '@/components/ui/Menu';
import { Table, TableAddColumn, TableFieldCell, TableHeaderCell, TableRowHeader } from '@/components/ui/Table';
import type { FieldModel } from '@/components/values/fieldModel';
import { ValueField } from '@/components/values/ValueField';
import { withAttention } from './attention';
import {
  sameSelection, type TaskCell, type TaskEpoch, type TaskRepeat, type TaskSelection, type TaskSignal,
} from './taskModel';

/** A refusal of an edit, or nothing once it is made. */
export type Made = Promise<string | undefined> | void;

export interface GridEdits {
  addEpoch?(name: string, index: number): Made;
  renameEpoch?(id: string, name: string): Made;
  moveEpoch?(id: string, index: number): Made;
  removeEpochs?(ids: readonly string[]): Made;
  addSignal?(name: string): Made;
  renameSignal?(id: string, name: string): Made;
  moveSignal?(id: string, index: number): Made;
  removeSignal?(id: string): Made;
  /** The cell spans `first` through `last`, and the signal's other cells give up the epochs it takes. */
  merge?(signal: string, cell: string, first: string, last: string): Made;
  /** The cell spans its first epoch alone; the others it spanned are absent. */
  split?(signal: string, cell: TaskCell): Made;
  absent?(signal: string, cells: readonly string[]): Made;
  /** A cell over each of `epochs`, which no cell of the signal spans. */
  present?(signal: string, epochs: readonly string[]): Made;
}

/**
 * What a drag or Shift-click has extended the selection over: adjacent cells of one signal, or adjacent columns, as
 * epoch indexes. It holds while the selection is still the one it extended (`from`).
 */
type Extent =
  | { readonly kind: 'cells'; readonly from: TaskSelection; readonly signal: string; readonly anchor: number; readonly head: number }
  | { readonly kind: 'epochs'; readonly from: TaskSelection; readonly anchor: number; readonly head: number };

/** The cells of a signal that meet the epochs from `a` to `b`, and the span they cover together. */
function cellsOver(signal: TaskSignal, epochs: readonly TaskEpoch[], a: number, b: number) {
  const index = (id: string) => epochs.findIndex((epoch) => epoch.id === id);
  const [low, high] = a <= b ? [a, b] : [b, a];
  const cells = signal.cells.filter((cell) => index(cell.last) >= low && index(cell.first) <= high);
  return { cells, first: index(cells[0].first), last: index(cells.at(-1)!.last) };
}

const epochDrag = 'application/x-feedbax-task-epoch';
const signalDrag = 'application/x-feedbax-task-signal';

/** The index a dragged member lands at, dropped on the one at `index`: before it in its leading half, else after. */
function dropIndex(event: DragEvent<HTMLElement>, index: number, axis: 'x' | 'y') {
  const box = event.currentTarget.getBoundingClientRect();
  const before = axis === 'x' ? event.clientX < box.left + box.width / 2 : event.clientY < box.top + box.height / 2;
  return before ? index : index + 1;
}

/** Drop handlers that move the member dragged with `type` among `ids` to where it is dropped on the one at `index`. */
function dropTarget(type: string, ids: readonly string[], index: number, axis: 'x' | 'y', move?: (id: string, at: number) => void) {
  return {
    onDragOver: (event: DragEvent<HTMLElement>) => {
      if (!move || !event.dataTransfer.types.includes(type)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
    },
    onDrop: (event: DragEvent<HTMLElement>) => {
      const id = event.dataTransfer.getData(type);
      if (!id || !move) return;
      event.preventDefault();
      const from = ids.indexOf(id);
      const to = dropIndex(event, index, axis);
      const at = to > from ? to - 1 : to;
      if (at !== from) move(id, at);
    },
  };
}

function Grip({ type, id, name, axis }: { type: string; id: string; name: string; axis: 'x' | 'y' }) {
  const Glyph = axis === 'x' ? GripHorizontal : GripVertical;
  return <span draggable role="img" aria-label={studioAccessibleName('move', name)}
    className={clsx('inline-flex shrink-0 cursor-grab text-shell-muted [&_svg]:size-[1em]', withAttention)}
    onPointerDown={(event) => event.stopPropagation()}
    onDragStart={(event) => {
      event.dataTransfer.setData(type, id);
      event.dataTransfer.effectAllowed = 'move';
    }}>
    <Glyph aria-hidden="true" />
  </span>;
}

/** An absent cell: the signal's absent value, muted over the hatch, which fills the cell. */
function AbsentCell({ text }: { text: string }) {
  return <span role="img" aria-label={studioAccessibleName('absent', text)}
    className="flex h-[calc(var(--row-height)-1px)] min-w-0 items-center truncate px-row-inline font-mono text-xs text-shell-muted bg-[repeating-linear-gradient(135deg,var(--studio-control-border)_0_1px,transparent_1px_5px)]">
    {text}
  </span>;
}

const stop = (event: SyntheticEvent) => event.stopPropagation();

/** A present cell's field, whose text committed empty makes the cell absent. */
function emptiedAbsent(field: FieldModel, absent: (() => Made) | undefined): FieldModel {
  const { commit } = field;
  if (!commit || !absent) return field;
  return {
    ...field,
    commit: async (edit) => edit.kind === 'text' && !edit.text.trim() ? (await absent()) || undefined : commit(edit),
  };
}

/**
 * A press on one of a cell's own buttons (its menu, the insert control) acts on that control, not on the selection;
 * nor does an event React carries to the cell from a menu it opened, which is outside the cell's own elements.
 */
const onControl = (event: SyntheticEvent<Element>) => !(event.target instanceof Element) ||
  !event.currentTarget.contains(event.target) || !!event.target.closest('button');

/** The commands on what is selected, as the items of its menu and its context menu. */
interface Commands {
  /** What the commands are on: a kind, and the name of the thing of that kind. */
  readonly on: StudioNameId;
  readonly of: string;
  readonly items: ReactNode;
}

function SelectionMenu({ commands, className }: { commands: Commands; className?: string }) {
  return <Menu>
    <MenuTrigger render={<IconButton variant="ghost" size="mark" className={className} label={studioAccessibleName(commands.on, commands.of)}
      icon={<MoreHorizontal aria-hidden="true" />} />} onPointerDown={stop} />
    <MenuContent aria-label={studioAccessibleName(commands.on, commands.of)}>{commands.items}</MenuContent>
  </Menu>;
}

/** `trigger` with the selection's context menu; `onOpen` selects what it opened on, when that is outside the selection. */
function WithContextMenu({ commands, onOpen, trigger }: { commands: Commands; onOpen(): void; trigger: ReactElement }) {
  return <ContextMenu onOpenChange={(open) => { if (open) onOpen(); }}>
    <ContextMenuTrigger render={trigger} />
    <ContextMenuContent>
      <div className="contents" onClick={stop} onPointerDown={stop}>{commands.items}</div>
    </ContextMenuContent>
  </ContextMenu>;
}

/**
 * Epochs as columns of equal width and signals as rows, under a lane of repeats. The End row holds each epoch's End;
 * a cell is a field over adjacent epochs, or absent. A click selects a cell, a row or a column, and a drag or
 * Shift-click extends a cell or column over its neighbours; the selection's menu and its context menu hold its
 * commands. A column or a row is added at the table's add controls, or a column between two others at their boundary,
 * and is written once it is named.
 */
export function TaskGrid({ epochs, repeats, signals, selected, select, edits }: {
  epochs: readonly TaskEpoch[]; repeats: readonly TaskRepeat[]; signals: readonly TaskSignal[];
  selected?: TaskSelection; select(selection: TaskSelection): void; edits: GridEdits;
}) {
  const [extent, setExtent] = useState<Extent>();
  const [dragging, setDragging] = useState(false);
  /** A press has chosen the cell it went down on, so the focus it gives the cell's field chooses nothing again. */
  const pressed = useRef(false);
  const [adding, setAdding] = useState<{ readonly kind: 'epoch'; readonly at: number } | { readonly kind: 'signal' }>();
  const [refusal, setRefusal] = useState<{ readonly at: string; readonly reason: string }>();
  useEffect(() => {
    if (!dragging) return;
    const end = () => setDragging(false);
    window.addEventListener('pointerup', end);
    return () => window.removeEventListener('pointerup', end);
  }, [dragging]);
  const settle = (at: string, made: Made) => {
    void Promise.resolve(made).then((reason) => setRefusal(reason ? { at, reason } : undefined));
  };
  const index = (id: string) => epochs.findIndex((epoch) => epoch.id === id);
  const ids = epochs.map(({ id }) => id);

  /** The selection's extent: what a drag or Shift-click extended it over, or what it selects alone. */
  const shown: Extent | undefined = extent && sameSelection(extent.from, selected) ? extent
    : selected?.kind === 'epoch' && index(selected.id) >= 0
      ? { kind: 'epochs', from: selected, anchor: index(selected.id), head: index(selected.id) }
      : selected?.kind === 'cell' ? (() => {
        const cell = signals.find(({ id }) => id === selected.signal)?.cells.find(({ id }) => id === selected.id);
        return cell && { kind: 'cells', from: selected, signal: selected.signal, anchor: index(cell.first), head: index(cell.last) } as const;
      })() : undefined;
  const columns = shown?.kind === 'epochs'
    ? { first: Math.min(shown.anchor, shown.head), last: Math.max(shown.anchor, shown.head) } : undefined;
  const span = shown?.kind === 'cells' ? (() => {
    const signal = signals.find(({ id }) => id === shown.signal);
    return signal && { signal, ...cellsOver(signal, epochs, shown.anchor, shown.head) };
  })() : undefined;

  const chooseCell = (signal: TaskSignal, cell: TaskCell, extend: boolean) => {
    const at = index(cell.first);
    if (extend && shown?.kind === 'cells' && shown.signal === signal.id) {
      setExtent({ ...shown, head: at < shown.anchor ? at : index(cell.last) });
      return;
    }
    const selection = { kind: 'cell', signal: signal.id, id: cell.id } as const;
    setExtent({ kind: 'cells', from: selection, signal: signal.id, anchor: at, head: index(cell.last) });
    if (!sameSelection(selection, selected)) select(selection);
  };
  const chooseEpoch = (epoch: TaskEpoch, extend: boolean) => {
    const at = index(epoch.id);
    if (extend && shown?.kind === 'epochs') {
      setExtent({ ...shown, head: at });
      return;
    }
    const selection = { kind: 'epoch', id: epoch.id } as const;
    setExtent({ kind: 'epochs', from: selection, anchor: at, head: at });
    if (!sameSelection(selection, selected)) select(selection);
  };

  /** The commands on the cells of `signal` from `first` through `last`. */
  const cellCommands = (signal: TaskSignal, a: number, b: number): Commands => {
    const { cells, first, last } = cellsOver(signal, epochs, a, b);
    const present = cells.filter((cell): cell is TaskCell & { kind: 'value' } => cell.kind === 'value');
    const at = cells[0].id;
    const merge = edits.merge && cells.length > 1 && present.length > 0
      ? () => settle(at, edits.merge!(signal.id, present[0].id, ids[first], ids[last])) : undefined;
    const lone = cells.length === 1 && cells[0].kind === 'value' && cells[0].first !== cells[0].last ? cells[0] : undefined;
    const split = edits.split && lone ? () => settle(at, edits.split!(signal.id, lone)) : undefined;
    const allAbsent = present.length === 0;
    const toggle = allAbsent
      ? edits.present && (() => settle(at, edits.present!(signal.id, cells.map(({ first }) => first))))
      : edits.absent && (() => settle(at, edits.absent!(signal.id, present.map(({ id }) => id))));
    return {
      on: 'cells', of: signal.name,
      items: <>
        <MenuItem disabled={!merge} onClick={merge}>{studioName('merge')}</MenuItem>
        <MenuItem disabled={!split} onClick={split}>{studioName('split')}</MenuItem>
        <MenuCheckboxItem checked={allAbsent} disabled={!toggle} onCheckedChange={() => toggle?.()}>{studioName('absent')}</MenuCheckboxItem>
      </>,
    };
  };
  const columnCommands = (epoch: TaskEpoch, rename: () => void): Commands => {
    const chosen = columns && index(epoch.id) >= columns.first && index(epoch.id) <= columns.last
      ? ids.slice(columns.first, columns.last + 1) : [epoch.id];
    const remove = edits.removeEpochs && chosen.length < epochs.length
      ? () => settle(chosen[0], edits.removeEpochs!(chosen)) : undefined;
    return {
      on: 'epoch', of: epoch.name,
      items: <>
        <MenuItem disabled={chosen.length > 1 || !edits.renameEpoch} onClick={rename}>{studioName('rename')}</MenuItem>
        <MenuItem disabled={!edits.addEpoch} onClick={() => setAdding({ kind: 'epoch', at: index(epoch.id) })}>
          {studioName('insertEpoch')}
        </MenuItem>
        <MenuItem disabled={!remove} onClick={remove}>{studioName('remove')}</MenuItem>
      </>,
    };
  };
  const signalCommands = (signal: TaskSignal, row: number, rename: () => void): Commands => ({
    on: 'signal', of: signal.name,
    items: <>
      <MenuItem disabled={!edits.renameSignal} onClick={rename}>{studioName('rename')}</MenuItem>
      <MenuItem disabled={row === 0 || !edits.moveSignal}
        onClick={() => settle(signal.id, edits.moveSignal?.(signal.id, row - 1))}>{studioName('moveEarlier')}</MenuItem>
      <MenuItem disabled={row === signals.length - 1 || !edits.moveSignal}
        onClick={() => settle(signal.id, edits.moveSignal?.(signal.id, row + 1))}>{studioName('moveLater')}</MenuItem>
      <MenuItem disabled={!edits.removeSignal} onClick={() => settle(signal.id, edits.removeSignal?.(signal.id))}>
        {studioName('remove')}
      </MenuItem>
    </>,
  });

  const pending = adding?.kind === 'epoch' ? adding.at : undefined;
  const doneAdding = () => setAdding(undefined);
  const newEpoch = pending !== undefined && edits.addEpoch && <NewMember key="new-epoch" header name="addEpoch"
    add={(name) => edits.addEpoch!(name, pending)} done={doneAdding} />;
  /** A row's cells, with the column being added drawn where it will be: inside a cell that spans its place, else empty. */
  const withPending = (cells: readonly { readonly first: number; readonly last: number; readonly node: (extra: number) => ReactNode }[]) =>
    cells.flatMap(({ first, last, node }) => {
      if (pending === undefined || pending < first || pending > last) return [node(0)];
      if (pending > first) return [node(1)];
      return [<td key="new-epoch" className="h-row border-b border-l border-shell-divider" />, node(0)];
    }).concat(pending === epochs.length ? [<td key="new-epoch" className="h-row border-b border-l border-shell-divider" />] : []);

  return <Table aria-label={studioName('epochs')} held={1}
    columns={[{ least: '10rem', fixed: true }, ...epochs.map(() => ({ least: '8rem' })), ...pending !== undefined ? [{ least: '8rem' }] : []]}
    addColumn={edits.addEpoch && { label: studioName('addEpoch'), onAdd: () => setAdding({ kind: 'epoch', at: epochs.length }),
      disabled: !!adding }}
    addRow={edits.addSignal && { label: studioName('addSignal'), onAdd: () => setAdding({ kind: 'signal' }), disabled: !!adding }}>
    <thead>
      {repeats.length > 0 && <RepeatLane epochs={epochs} repeats={repeats} selected={selected} select={select} />}
      <tr>
        <TableHeaderCell aria-hidden="true" />
        {epochs.flatMap((epoch, column) => [
          ...column === pending ? [newEpoch] : [],
          <EpochHeader key={epoch.id} epoch={epoch} column={column} ids={ids} edits={edits}
            chosen={!!columns && column >= columns.first && column <= columns.last}
            menu={!!columns && column === columns.last} commands={(rename) => columnCommands(epoch, rename)}
            refusal={refusal?.at === epoch.id ? refusal.reason : undefined}
            choose={(extend) => chooseEpoch(epoch, extend)} insert={edits.addEpoch && !adding && column > 0
              ? () => setAdding({ kind: 'epoch', at: column }) : undefined}
            rename={(name) => settle(epoch.id, edits.renameEpoch?.(epoch.id, name))}
            move={edits.moveEpoch && ((id, at) => settle(id, edits.moveEpoch!(id, at)))} />,
        ])}
        {pending === epochs.length && newEpoch}
        {edits.addEpoch && <TableAddColumn />}
      </tr>
    </thead>
    <tbody>
      <tr>
        <TableRowHeader className="text-shell-muted">{studioName('end')}</TableRowHeader>
        {withPending(epochs.map((epoch, column) => ({
          first: column, last: column, node: () => {
            const selection = { kind: 'end', id: epoch.id } as const;
            const chosen = sameSelection(selection, selected);
            return <TableFieldCell key={epoch.id} selected={chosen} className="border-l"
              onPointerDown={() => { if (!chosen) select(selection); }} onFocusCapture={() => { if (!chosen) select(selection); }}>
              <ValueField form="cell" model={epoch.end} />
            </TableFieldCell>;
          },
        })))}
      </tr>
      {signals.map((signal, row) => <SignalRow key={signal.id} signal={signal} row={row} signals={signals}
        selected={selected} select={select} edits={edits} commands={(rename) => signalCommands(signal, row, rename)}
        refusal={refusal?.at === signal.id ? refusal.reason : undefined}
        rename={(name) => settle(signal.id, edits.renameSignal?.(signal.id, name))}
        move={edits.moveSignal && ((id, at) => settle(id, edits.moveSignal!(id, at)))}>
        {withPending(signal.cells.map((cell) => ({
          first: index(cell.first), last: index(cell.last), node: (extra: number) => {
            const inSpan = span?.signal.id === signal.id && span.cells.includes(cell);
            const commands = inSpan ? cellCommands(signal, span.first, span.last)
              : cellCommands(signal, index(cell.first), index(cell.last));
            const choose = (extend: boolean) => chooseCell(signal, cell, extend);
            return <WithContextMenu key={cell.id} commands={commands} onOpen={() => { if (!inSpan) choose(false); }} trigger={
              <TableFieldCell colSpan={index(cell.last) - index(cell.first) + 1 + extra} selected={inSpan}
                data-cell={cell.id} className="group/owner relative border-l"
                onPointerDown={(event) => {
                  if (event.button !== 0 || onControl(event)) return;
                  pressed.current = true;
                  choose(event.shiftKey);
                  setDragging(true);
                }}
                onPointerEnter={() => {
                  if (dragging && shown?.kind === 'cells' && shown.signal === signal.id) chooseCell(signal, cell, true);
                }}
                onFocusCapture={(event) => {
                  if (onControl(event)) return;
                  if (!pressed.current && !inSpan) choose(false);
                  pressed.current = false;
                }}>
                {cell.kind === 'absent' ? <AbsentCell text={signal.absent.text ?? ''} /> : <ValueField form="cell" model={emptiedAbsent(cell.value,
                  edits.absent && (() => edits.absent!(signal.id, [cell.id])))} />}
                {refusal?.at === cell.id && <span className="absolute inset-y-0 right-6 flex items-center"><FieldProblem reason={refusal.reason} /></span>}
                {inSpan && cell === span.cells.at(-1) && <SelectionMenu commands={commands}
                  className="absolute right-0.5 top-1/2 z-10 -translate-y-1/2" />}
              </TableFieldCell>} />;
          },
        })))}
      </SignalRow>)}
      {adding?.kind === 'signal' && edits.addSignal && <tr>
        <NewMember name="addSignal" add={edits.addSignal} done={doneAdding} />
        <td colSpan={epochs.length} />
      </tr>}
    </tbody>
  </Table>;
}

function EpochHeader({ epoch, column, ids, chosen, menu, commands, refusal, choose, insert, rename, move, edits }: {
  epoch: TaskEpoch; column: number; ids: readonly string[]; chosen: boolean; menu: boolean;
  commands(rename: () => void): Commands; refusal?: string; choose(extend: boolean): void; insert?: () => void;
  rename(name: string): void; move?: (id: string, at: number) => void; edits: GridEdits;
}) {
  const [renaming, setRenaming] = useState(false);
  const own = commands(() => setRenaming(true));
  return <WithContextMenu commands={own} onOpen={() => { if (!chosen) choose(false); }} trigger={
    <TableHeaderCell scope="col" data-selected={chosen || undefined} data-epoch={epoch.id}
      className={clsx('studio-table-field-cell group/owner relative border-l px-0 overflow-visible',
        chosen && 'bg-shell-row-selected text-shell-foreground')}
      onPointerDown={(event) => { if (event.button === 0 && !onControl(event)) choose(event.shiftKey); }}
      {...dropTarget(epochDrag, ids, column, 'x', move)}>
      <span className="flex h-row min-w-0 items-center gap-0.5 pl-1 pr-1">
        {move && <Grip type={epochDrag} id={epoch.id} axis="x" name={epoch.name} />}
        <InlineEdit form="cell" value={epoch.name} editing={renaming} onEditingChange={setRenaming}
          className="min-w-0 flex-1 font-mono text-shell-foreground"
          aria-label={studioAccessibleName('rename', epoch.name)} onCommit={edits.renameEpoch ? rename : () => false} />
        {refusal && <FieldProblem reason={refusal} />}
        <SelectionMenu commands={own} className={clsx('ml-auto shrink-0', !menu && withAttention)} />
      </span>
      {insert && <IconButton variant="ghost" size="mark" label={studioName('insertEpoch')} icon={<Plus aria-hidden="true" />}
        onPointerDown={stop} onClick={insert}
        className={clsx('absolute -left-2.5 top-1/2 z-10 -translate-y-1/2 rounded-full bg-shell-surface', withAttention)} />}
    </TableHeaderCell>} />;
}

function SignalRow({ signal, row, signals, selected, select, edits, commands, refusal, rename, move, children }: {
  signal: TaskSignal; row: number; signals: readonly TaskSignal[]; selected?: TaskSelection; select(selection: TaskSelection): void;
  edits: GridEdits; commands(rename: () => void): Commands; refusal?: string; rename(name: string): void;
  move?: (id: string, at: number) => void; children: ReactNode;
}) {
  const [renaming, setRenaming] = useState(false);
  const selection = { kind: 'signal', id: signal.id } as const;
  const chosen = sameSelection(selection, selected);
  const choose = () => { if (!chosen) select(selection); };
  const own = commands(() => setRenaming(true));
  return <tr className="group/owner" data-selected={chosen || undefined}
    {...dropTarget(signalDrag, signals.map(({ id }) => id), row, 'y', move)}>
    <WithContextMenu commands={own} onOpen={choose} trigger={
      <TableFieldCell header selected={chosen} className={clsx(chosen && 'font-medium')} {...appTip(studioValueType(signal.valueType, signal.type))}
        onPointerDown={choose} onFocusCapture={choose}>
        <span className="flex h-full min-w-0 items-center gap-0.5 pl-1">
          {move && <Grip type={signalDrag} id={signal.id} axis="y" name={signal.name} />}
          <InlineEdit form="cell" value={signal.name} editing={renaming} onEditingChange={setRenaming}
            className="min-w-0 flex-1 font-mono" aria-label={studioAccessibleName('rename', signal.name)}
            onCommit={edits.renameSignal ? rename : () => false} />
          {refusal && <FieldProblem reason={refusal} />}
          <SelectionMenu commands={own} className={clsx('mr-1 shrink-0', !chosen && withAttention)} />
        </span>
      </TableFieldCell>} />
    {children}
  </tr>;
}

/** A column or row being added: its name, which writes it when it is given. */
function NewMember({ name, add, done, header }: { name: StudioNameId; add(name: string): Made; done(): void; header?: boolean }) {
  const [refusal, setRefusal] = useState<string>();
  const input = <span className="flex h-full min-w-0 items-center">
    <InlineEdit form="cell" value="" editing className="min-w-0 flex-1 font-mono" aria-label={studioName(name)}
      onEditingChange={(editing) => { if (!editing && !refusal) done(); }}
      onCommit={(given) => {
        void Promise.resolve(add(given)).then((reason) => { setRefusal(reason || undefined); if (!reason) done(); });
      }} />
    {refusal && <FieldProblem reason={refusal} />}
  </span>;
  return header
    ? <TableHeaderCell scope="col" className="studio-table-field-cell border-l px-0 overflow-visible">{input}</TableHeaderCell>
    : <TableFieldCell header>{input}</TableFieldCell>;
}

/** The repeats over the columns, as a lane above them: a bracket over each, its count a field at its end. */
function RepeatLane({ epochs, repeats, selected, select }: {
  epochs: readonly TaskEpoch[]; repeats: readonly TaskRepeat[]; selected?: TaskSelection; select(selection: TaskSelection): void;
}) {
  const cells: ReactNode[] = [];
  for (let index = 0; index < epochs.length;) {
    const repeat = repeats.find(({ first }) => first === epochs[index].id);
    if (!repeat) {
      cells.push(<th key={epochs[index].id} aria-hidden="true" />);
      index += 1;
      continue;
    }
    const last = epochs.findIndex(({ id }) => id === repeat.last);
    const span = last - index + 1;
    const selection = { kind: 'repeat', id: repeat.id } as const;
    const chosen = sameSelection(selection, selected);
    cells.push(<th key={repeat.id} colSpan={span} scope="colgroup" aria-label={studioName('repeat')}
      className="px-1 pb-0.5 pt-1 font-normal" onPointerDown={() => { if (!chosen) select(selection); }}>
      <span className={clsx('flex items-end gap-1', chosen && 'text-shell-foreground')}>
        <span aria-hidden="true" className={clsx('h-2 grow border-x border-t', chosen ? 'border-shell-foreground' : 'border-shell-muted')} />
        <span className="flex w-16 shrink-0 items-center gap-0.5 text-xs text-shell-muted">×<ValueField form="cell" model={repeat.count} /></span>
      </span>
    </th>);
    index = last + 1;
  }
  return <tr><th aria-hidden="true" />{cells}</tr>;
}
